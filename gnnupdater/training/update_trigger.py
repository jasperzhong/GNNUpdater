import logging
import time
from abc import ABC, abstractmethod
from collections import deque

import dgl
import numpy as np
import pandas as pd
import torch
from alibi_detect.cd.mmd import MMDDrift
from skmultiflow.drift_detection import (ADWIN, DDM, EDDM, HDDM_A, HDDM_W,
                                         KSWIN, PageHinkley)
from torch_geometric.nn import LabelPropagation
from torch_geometric.utils import (from_scipy_sparse_matrix,
                                   to_scipy_sparse_matrix)
from torch_sparse import SparseTensor


class UpdateTrigger(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def update_check(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass


class NotTrigger(UpdateTrigger):
    def update_check(self, *args, **kwargs):
        return False


class IntervalTrigger(UpdateTrigger):
    """
    Update trigger that fires every `interval` day.
    """

    def __init__(self, interval: int = 7):
        """
        Args:
            interval (int): Interval of days to trigger update.

        Note:
            When interval=1, it triggers every day.
        """
        self.interval = interval
        self.day_counter = 0

    def update_check(self, *args, **kwargs):
        self.day_counter += 1
        if self.day_counter % self.interval == 0:
            return True
        return False


class DelayedAccuracyDropTrigger(UpdateTrigger):
    """
    Update trigger that fires when the true accuracy (delayed) drops below a certain threshold.
    """

    def __init__(self, slo_threshold: float,
                 drop_threshold: float = 0.1,
                 window_size: int = 7,
                 delay: int = 7):
        """
        Args:
            slo_threshold (float): SLO threshold.
            drop_threshold (float): Drop threshold.
            window_size (int): Window size for calculating the moving average.
            delay (int): Delay in days.
        """
        self.threshold = slo_threshold * (1 - drop_threshold)
        self.window_size = window_size
        self.delay = delay
        self.pred_accuracy_history = deque(maxlen=window_size+delay)
        logging.info(
            f"Accuracy Drop Threshold: {self.threshold:.4f}, Window Size: {window_size}, Delay: {delay}")

    def update_check(self, *args, **kwargs):
        """
        Check if the predicted accuracy drops below the threshold.

        Args:
            true_accuracy (float): Predicted accuracy.

        Returns:
            bool: Whether to trigger the update.
        """
        true_accuracy = np.mean(kwargs['true_accuracy'])

        self.pred_accuracy_history.append(true_accuracy)

        if len(self.pred_accuracy_history) < self.pred_accuracy_history.maxlen:
            return False

        # Convert deque to list for slicing
        accuracy_list = list(self.pred_accuracy_history)
        rolling_mean_true_accuracy = np.mean(accuracy_list[:self.window_size])

        if rolling_mean_true_accuracy < self.threshold:
            logging.info(
                f"Rolling Mean Accuracy: {rolling_mean_true_accuracy:.4f} drops below threshold {self.threshold:.4f}")
            return True

        return False

    def reset(self, *args, **kwargs):
        n = self.window_size
        while n:
            self.pred_accuracy_history.popleft()
            n -= 1


class AccuracyDropTrigger(UpdateTrigger):
    """
    Update trigger that fires when the accuracy drops below a certain threshold.
    """

    def __init__(self, slo_threshold: float,
                 drop_threshold: float = 0.1,
                 window_size: int = 7):
        """
        Args:
            slo_threshold (float): SLO threshold.
            window_size (int): Window size for calculating the moving average.
        """
        self.threshold = slo_threshold * (1 - drop_threshold)
        self.pred_accuracy_history = deque(maxlen=window_size)
        logging.info(
            f"Accuracy Drop Threshold: {self.threshold:.4f}, Window Size: {window_size}")

    def update_check(self, *args, **kwargs):
        """
        Check if the predicted accuracy drops below the threshold.

        Args:
            pred_accuracy (float): Predicted accuracy.

        Returns:
            bool: Whether to trigger the update.
        """
        pred_accuracy = kwargs['pred_accuracy']

        self.pred_accuracy_history.append(pred_accuracy)

        if len(self.pred_accuracy_history) < self.pred_accuracy_history.maxlen:
            return False

        rolling_mean_pred_accuracy = np.mean(self.pred_accuracy_history)

        if rolling_mean_pred_accuracy < self.threshold:
            logging.info(
                f"Rolling Mean Accuracy: {rolling_mean_pred_accuracy:.4f} drops below threshold {self.threshold:.4f}")
            return True

        return False

    def reset(self, *args, **kwargs):
        self.pred_accuracy_history.clear()


class LabelPropagationUpdateTrigger(UpdateTrigger):
    """
    \mathbf{S} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}
    \mathbf{Y} = \alpha\mathbf{S}\mathbf{Y} + (1-\alpha)\mathbf{Y}
    """

    def __init__(self, slo_threshold: float,
                 problem_ratio: float = 0.5,
                 propagation_weight: float = 0.9,
                 propagation_layers: int = 1,
                 threshold: float = 0.5,
                 max_node_id: int = 10000,
                 init_df: pd.DataFrame = None):
        """
        Args:
            slo_threshold (float): SLO threshold.
            problem_ratio (float): Ratio of problematic nodes.
            propagation_weight (float): Propagation weight.
            propagation_layers (int): Number of propagation layers.
        """
        self.slo_threshold = slo_threshold
        self.propagation_weight = propagation_weight
        self.problem_ratio_threshold = problem_ratio
        self.problem_ratio = 0
        self.threshold = threshold

        self.Y = None
        self.model = LabelPropagation(num_layers=propagation_layers,
                                      alpha=propagation_weight)

        if init_df is not None:
            self.Y = torch.zeros((max_node_id+1, 1))

            edge_index = init_df[['src', 'dst']].values.T
            # 直接构建 Adj
            self.adj = torch.zeros((max_node_id+1, max_node_id+1))
            self.adj[edge_index[0], edge_index[1]] = 1
            self.adj[edge_index[1], edge_index[0]] = 1

    @torch.no_grad()
    def update_check(self, *args, **kwargs):
        pred_accuracy = kwargs['pred_accuracy']
        max_node_id = kwargs['max_node_id']
        num_nodes = kwargs['num_nodes']
        batch_nodes = kwargs['batch_nodes']
        df = kwargs['df']
        batch_df = kwargs['batch_df']

        # resize
        if self.Y is None:
            self.Y = torch.zeros((max_node_id+1, 1))
        elif max_node_id >= self.Y.shape[0]:
            # 直接创建新的张量并复制旧数据
            new_Y = torch.zeros((max_node_id+1, 1))
            new_Y[:self.Y.shape[0]] = self.Y
            self.Y = new_Y

        if pred_accuracy < self.slo_threshold:
            label = 1.0
        else:
            label = 0.0
        self.Y[batch_nodes, 0] = label

        new_edges = batch_df[['src', 'dst']].values.T
        self.adj[new_edges[0], new_edges[1]] = 1
        self.adj[new_edges[1], new_edges[0]] = 1

        sp_adj = SparseTensor.from_dense(
            self.adj[:max_node_id+1, :max_node_id+1])
        sp_adj = sp_adj.set_value_(None)

        # 使用更新后的邻接矩阵进行传播
        out = self.model(self.Y[:max_node_id+1], sp_adj)
        y_pred = out > self.threshold

        self.problem_ratio = y_pred.sum().item() / num_nodes
        logging.info(
            f"Label: {label}, Problem Ratio: {self.problem_ratio:.4f}")

        # check
        if self.problem_ratio > self.problem_ratio_threshold:
            return True

        return False

    def reset(self, *args, **kwargs):
        self.Y.zero_()


class LabelNoPropagationTrueAccuracyUpdateTrigger(UpdateTrigger):
    """
    \mathbf{S} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}
    \mathbf{Y} = \alpha\mathbf{S}\mathbf{Y} + (1-\alpha)\mathbf{Y}
    """

    def __init__(self, slo_threshold: float,
                 problem_ratio: float = 0.5,
                 propagation_weight: float = 0.9,
                 propagation_layers: int = 1,
                 threshold: float = 0.5,
                 max_node_id: int = 10000,
                 init_df: pd.DataFrame = None):
        """
        Args:
            slo_threshold (float): SLO threshold.
            problem_ratio (float): Ratio of problematic nodes.
            propagation_weight (float): Propagation weight.
            propagation_layers (int): Number of propagation layers.
        """
        self.slo_threshold = slo_threshold
        self.propagation_weight = propagation_weight
        self.problem_ratio_threshold = problem_ratio
        self.problem_ratio = 0
        self.threshold = threshold

        self.Y = None

        if init_df is not None:
            self.Y = torch.zeros((max_node_id+1, 1))

    @torch.no_grad()
    def update_check(self, *args, **kwargs):
        true_accuracy = kwargs['true_accuracy']
        max_node_id = kwargs['max_node_id']
        num_nodes = kwargs['num_nodes']
        batch_nodes = kwargs['batch_nodes']
        df = kwargs['df']
        batch_df = kwargs['batch_df']

        # resize
        if self.Y is None:
            self.Y = torch.zeros((max_node_id+1, 1))
        elif max_node_id >= self.Y.shape[0]:
            # 直接创建新的张量并复制旧数据
            new_Y = torch.zeros((max_node_id+1, 1))
            new_Y[:self.Y.shape[0]] = self.Y
            self.Y = new_Y

        label = (true_accuracy < self.slo_threshold).astype(np.float32)
        self.Y[batch_nodes, 0] = torch.from_numpy(label)

        # NB: no label propagation

        self.problem_ratio = self.Y.sum().item() / num_nodes
        logging.info(
            f"Problem Ratio: {self.problem_ratio:.4f}")

        # check
        if self.problem_ratio > self.problem_ratio_threshold:
            return True

        return False

    def reset(self, *args, **kwargs):
        self.Y.zero_()


class LabelPropagationDelayedTrueAccuracyUpdateTrigger(UpdateTrigger):
    """
    \mathbf{S} = \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}
    \mathbf{Y} = \alpha\mathbf{S}\mathbf{Y} + (1-\alpha)\mathbf{Y}
    """

    def __init__(self, slo_threshold: float,
                 problem_ratio: float = 0.5,
                 propagation_weight: float = 0.9,
                 propagation_layers: int = 1,
                 threshold: float = 0.5,
                 max_node_id: int = 10000,
                 init_df: pd.DataFrame = None):
        """
        Args:
            slo_threshold (float): SLO threshold.
            problem_ratio (float): Ratio of problematic nodes.
            propagation_weight (float): Propagation weight.
            propagation_layers (int): Number of propagation layers.
        """
        self.slo_threshold = slo_threshold
        self.propagation_weight = propagation_weight
        self.problem_ratio_threshold = problem_ratio
        self.problem_ratio = 0
        self.threshold = threshold

        self.queue = deque(maxlen=7)

        self.Y = None
        self.model = LabelPropagation(num_layers=propagation_layers,
                                      alpha=propagation_weight)

        if init_df is not None:
            self.Y = torch.zeros((max_node_id+1, 1))

            edge_index = init_df[['src', 'dst']].values.T
            # 直接构建 Adj
            self.adj = torch.zeros((max_node_id+1, max_node_id+1))
            self.adj[edge_index[0], edge_index[1]] = 1
            self.adj[edge_index[1], edge_index[0]] = 1

    @torch.no_grad()
    def update_check(self, *args, **kwargs):
        true_accuracy = kwargs['true_accuracy']
        max_node_id = kwargs['max_node_id']
        num_nodes = kwargs['num_nodes']
        batch_nodes = kwargs['batch_nodes']
        df = kwargs['df']
        batch_df = kwargs['batch_df']

        self.queue.append((batch_nodes, true_accuracy))
        if len(self.queue) < self.queue.maxlen:
            return False

        batch_nodes, true_accuracy = self.queue[0]

        # resize
        if self.Y is None:
            self.Y = torch.zeros((max_node_id+1, 1))
        elif max_node_id >= self.Y.shape[0]:
            # 直接创建新的张量并复制旧数据
            new_Y = torch.zeros((max_node_id+1, 1))
            new_Y[:self.Y.shape[0]] = self.Y
            self.Y = new_Y

        label = (true_accuracy < self.slo_threshold).astype(np.float32)
        self.Y[batch_nodes, 0] = torch.from_numpy(label)

        new_edges = batch_df[['src', 'dst']].values.T
        self.adj[new_edges[0], new_edges[1]] = 1
        self.adj[new_edges[1], new_edges[0]] = 1

        sp_adj = SparseTensor.from_dense(
            self.adj[:max_node_id+1, :max_node_id+1])
        sp_adj = sp_adj.set_value_(None)

        # 使用更新后的邻接矩阵进行传播
        out = self.model(self.Y[:max_node_id+1], sp_adj)
        y_pred = out > self.threshold

        self.problem_ratio = y_pred.sum().item() / num_nodes
        logging.info(
            f"Problem Ratio: {self.problem_ratio:.4f}")

        # check
        if self.problem_ratio > self.problem_ratio_threshold:
            return True

        return False

    def reset(self, *args, **kwargs):
        self.Y.zero_()


class DriftDetectionUpdateTrigger(UpdateTrigger):
    """
    Update trigger that monitors the accuracy drop using Drift Detection.
    """
    CTOR = {
        'ddm': DDM,
        'adwin': ADWIN,
        'eddm': EDDM,
        'hddm_a': HDDM_A,
        'hddm_w': HDDM_W,
        'kswin': KSWIN,
        'pagehinkley': PageHinkley
    }

    def __init__(self, name='DDM', **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.create_ddm()

    def create_ddm(self):
        if self.name == 'adwin':
            delta = self.kwargs.get('delta', 0.002)
            logging.info(f"Using ADWIN with delta: {delta}")
            self.ddm = DriftDetectionUpdateTrigger.CTOR[self.name](delta=delta)
        elif self.name == 'kswin':
            alpha = self.kwargs.get('alpha', 0.005)
            logging.info(f"Using KSWIN with alpha: {alpha}")
            self.ddm = DriftDetectionUpdateTrigger.CTOR[self.name](alpha=alpha,
                                                                   window_size=15,
                                                                   stat_size=5)
        else:
            self.ddm = DriftDetectionUpdateTrigger.CTOR[self.name]()

    def update_check(self, *args, **kwargs):
        pred_accuracy = kwargs['pred_accuracy']
        self.ddm.add_element(pred_accuracy)
        if self.ddm.detected_change():
            logging.info(f"{self.name} detected change")
            return True
        return False

    def reset(self, *args, **kwargs):
        pass


class DriftDetectionTrueAccuracyUpdateTrigger(UpdateTrigger):
    """
    Update trigger that monitors the accuracy drop using Drift Detection.
    """
    CTOR = {}
    for name, ctor in DriftDetectionUpdateTrigger.CTOR.items():
        CTOR[name + '_delayed_accuracy'] = ctor

    def __init__(self, name='DDM', **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.queue = deque(maxlen=7)
        self.create_ddm()

    def create_ddm(self):
        if self.name == 'adwin_delayed_accuracy':
            delta = self.kwargs.get('delta', 0.002)
            logging.info(f"Using ADWIN (delayed accuracy) with delta: {delta}")
            self.ddm = DriftDetectionTrueAccuracyUpdateTrigger.CTOR[self.name](
                delta=delta)
        elif self.name == 'kswin_delayed_accuracy':
            alpha = self.kwargs.get('alpha', 0.005)
            logging.info(f"Using KSWIN (delayed accuracy) with alpha: {alpha}")
            self.ddm = DriftDetectionTrueAccuracyUpdateTrigger.CTOR[self.name](alpha=alpha,
                                                                               window_size=15,
                                                                               stat_size=5)
        else:
            self.ddm = DriftDetectionTrueAccuracyUpdateTrigger.CTOR[self.name](
            )

    def update_check(self, *args, **kwargs):
        true_accuracy = np.mean(kwargs['true_accuracy'])
        self.queue.append(true_accuracy)

        if len(self.queue) < self.queue.maxlen:
            return False

        true_accuracy = self.queue[0]
        self.ddm.add_element(true_accuracy)
        if self.ddm.detected_change():
            logging.info(f"{self.name} detected change")
            return True
        return False

    def reset(self, *args, **kwargs):
        pass


class MMDUpdateTrigger(UpdateTrigger):
    """
    using MMD on graph node embeddings to detect drift
    """

    def __init__(self, x_ref, device, backend='pytorch', distance_threshold=0.0):
        """
        Args:
            x_ref (np.ndarray): Reference node embeddings
            backend (str): Backend framework. Default is 'pytorch'.
            distance_threshold (float): Threshold for the MMD distance
        """
        self.distance_threshold = distance_threshold
        self.backend = backend
        self.device = device
        if isinstance(x_ref, torch.Tensor):
            x_ref = x_ref.cpu().numpy()

        self.mmd_drift_detector = MMDDrift(x_ref, backend=backend,
                                           n_permutations=1,
                                           device=device)

    @torch.no_grad()
    def update_check(self, *args, **kwargs):
        x = kwargs['x']
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        result = self.mmd_drift_detector.predict(x)
        # logging.info(
        #     f"mmd: {result['data']['distance']: .4f}, p-value: {result['data']['p_val']: .4f},"
        #     f"is_drift: {result['data']['is_drift']}")

        mmd = result['data']['distance']
        logging.info(f"MMD: {mmd:.4f}")

        return mmd > self.distance_threshold

    def reset(self, *args, **kwargs):
        x_ref = kwargs['x_ref']
        if isinstance(x_ref, torch.Tensor):
            x_ref = x_ref.cpu().numpy()
        self.mmd_drift_detector = MMDDrift(x_ref, backend=self.backend,
                                           n_permutations=1,
                                           device=self.device)


def get_update_trigger(trigger_type: str, **kwargs):
    if trigger_type == 'none':
        return NotTrigger()
    elif trigger_type == 'interval':
        interval = kwargs.get('interval', 90)
        logging.info(f"Using interval: {interval}")
        return IntervalTrigger(interval)
    elif trigger_type == 'accuracy_drop':
        slo_threshold = kwargs["slo_threshold"]
        drop_threshold = kwargs.get("accuracy_drop_threshold", 0.1)
        window_size = kwargs.get("window_size", 7)
        logging.info(
            f"Using SLO threshold: {slo_threshold}, Window Size: {window_size}")
        return AccuracyDropTrigger(slo_threshold, drop_threshold, window_size)
    elif trigger_type == 'delayed_accuracy_drop':
        slo_threshold = kwargs["slo_threshold"]
        drop_threshold = kwargs.get("accuracy_drop_threshold", 0.1)
        window_size = kwargs.get("window_size", 7)
        delay = kwargs.get("delay", 7)
        logging.info(
            f"Using SLO threshold: {slo_threshold}, Drop Threshold: {drop_threshold}, Window Size: {window_size}, Delay: {delay}")
        return DelayedAccuracyDropTrigger(slo_threshold, drop_threshold, window_size, delay)
    elif trigger_type == 'label_propagation':
        slo_threshold = kwargs["slo_threshold"]
        problem_ratio = kwargs.get("problem_ratio", 0.1)
        propagation_weight = kwargs.get("propagation_weight", 0.1)
        propagation_layers = kwargs.get("propagation_layers", 2)
        threshold = kwargs.get("threshold", 0.5)
        max_node_id = kwargs.get("max_node_id", 10000)
        init_df = kwargs.get("init_df", None)
        logging.info(
            f"Using SLO threshold: {slo_threshold}, Threshold: {threshold}"
            f"Problem Ratio: {problem_ratio}, Propagation weight: {propagation_weight}, Propagation layers: {propagation_layers}")
        return LabelPropagationUpdateTrigger(slo_threshold, problem_ratio, propagation_weight, propagation_layers,
                                             threshold, max_node_id, init_df)
    elif trigger_type == 'label_no_propagation_true_accuracy':
        slo_threshold = kwargs["slo_threshold"]
        problem_ratio = kwargs.get("problem_ratio", 0.1)
        propagation_weight = kwargs.get("propagation_weight", 0.1)
        propagation_layers = kwargs.get("propagation_layers", 2)
        threshold = kwargs.get("threshold", 0.5)
        max_node_id = kwargs.get("max_node_id", 10000)
        init_df = kwargs.get("init_df", None)
        logging.info(
            f"Using SLO threshold: {slo_threshold}, Threshold: {threshold}"
            f"Problem Ratio: {problem_ratio}, Propagation weight: {propagation_weight}, Propagation layers: {propagation_layers}")
        return LabelNoPropagationTrueAccuracyUpdateTrigger(slo_threshold, problem_ratio, propagation_weight, propagation_layers,
                                                           threshold, max_node_id, init_df)
    elif trigger_type == 'label_propagation_delayed_true_accuracy':
        slo_threshold = kwargs["slo_threshold"]
        problem_ratio = kwargs.get("problem_ratio", 0.1)
        propagation_weight = kwargs.get("propagation_weight", 0.1)
        propagation_layers = kwargs.get("propagation_layers", 2)
        threshold = kwargs.get("threshold", 0.5)
        max_node_id = kwargs.get("max_node_id", 10000)
        init_df = kwargs.get("init_df", None)
        logging.info(
            f"Using SLO threshold: {slo_threshold}, Threshold: {threshold}"
            f"Problem Ratio: {problem_ratio}, Propagation weight: {propagation_weight}, Propagation layers: {propagation_layers}")
        return LabelPropagationDelayedTrueAccuracyUpdateTrigger(slo_threshold, problem_ratio, propagation_weight, propagation_layers,
                                                                threshold, max_node_id, init_df)
    elif trigger_type == 'mmd':
        if 'x_ref' not in kwargs:
            raise ValueError("x_ref must be provided for MMD trigger")
        x_ref = kwargs['x_ref']
        device = kwargs.get('device', 'cuda')
        backend = kwargs.get('backend', 'pytorch')
        distance_threshold = kwargs.get('distance_threshold', 0.0)
        logging.info(
            f"Using MMD with distance threshold: {distance_threshold}")
        return MMDUpdateTrigger(x_ref, device, backend, distance_threshold)
    else:
        if trigger_type in DriftDetectionUpdateTrigger.CTOR:
            logging.info(f"Using Drift Detection: {trigger_type}")
            return DriftDetectionUpdateTrigger(trigger_type, **kwargs)
        elif trigger_type in DriftDetectionTrueAccuracyUpdateTrigger.CTOR:
            logging.info(f"Using Drift Detection: {trigger_type}")
            return DriftDetectionTrueAccuracyUpdateTrigger(trigger_type, **kwargs)
        else:
            raise ValueError(f"Unknown trigger type: {trigger_type}")
