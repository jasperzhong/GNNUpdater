import logging
import os
import random
import time
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from gnnupdater.cache import get_cache
from gnnupdater.data.load_data import load_data
from gnnupdater.data.time_series_data_loader import TimeSeriesDataLoader
from gnnupdater.dynamic_graph import build_dynamic_graph
from gnnupdater.models import get_model
from gnnupdater.temporal_sampler import TemporalSampler
from gnnupdater.training.evaluator import Evaluator
from gnnupdater.training.mmd import calculate_embeddings_distribution_shift
from gnnupdater.training.train_perf_predictor import (predict_perf,
                                                   train_perf_predictor)
from gnnupdater.training.update_trigger import get_update_trigger
from gnnupdater.training.utils import WindowActivityTracker
from gnnupdater.utils import PROJ_ROOT, EarlyStopMonitor, mfgs_to_cuda


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ContinuousGraphLearner:
    def __init__(self, dataset: str, model: str, phase: str, device, args,
                 build_graph: bool = True, seed: int = 42):
        self.args = args
        self.model_name = model
        self.dataset = dataset
        self.phase = phase

        df, offline_df, streaming_df, config, split_points, node_label_dict, node_feat, edge_feat = load_data(
            dataset)
        self.df, self.offline_df, self.streaming_df = df, offline_df, streaming_df
        self.split_points = split_points
        self.node_label_dict = node_label_dict

        self.num_nodes, self.num_edges = config['num_nodes'], config['num_edges']
        self.node_feat, self.edge_feat = node_feat, edge_feat
        self.config = config
        self.device = device

        set_seed(seed)
        if build_graph:
            self.build_graph_sampler_cache(df, config, device)

    def build_new_model(self):
        config = self.config
        device = self.device

        task = config['task']
        node_feat_dim, edge_feat_dim = config['node_feat_dim'], config['edge_feat_dim']
        model, model_config = get_model(
            self.model_name, task, device, num_nodes=self.num_nodes, node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim,
            dataset_config=config)

        return model, model_config

    def build_graph_sampler_cache(self, build_graph_df, config, device):
        self.dgraph = build_dynamic_graph(
            **config['data'], device=device, dataset_df=build_graph_df)

        self.model, self.model_config = self.build_new_model()

        self.sampler = TemporalSampler(self.dgraph, **self.model_config)
        model_config = deepcopy(self.model_config)
        model_config['fanouts'] = [100]
        # NB: only for inference to ensure the same results
        model_config['sample_strategy'] = 'recent'
        self.full_sampler = TemporalSampler(self.dgraph, **model_config)
        model_config['fanouts'] = [100, 100]
        self.full_sampler2 = TemporalSampler(self.dgraph, **model_config)
        self.cache = get_cache(cache_name=config['cache']['name'], node_feat=self.node_feat, edge_feat=self.edge_feat,
                               edge_cache_ratio=config['cache']['edge_cache_ratio'], node_cache_ratio=config['cache']['node_cache_ratio'],
                               num_nodes=self.num_nodes, num_edges=self.num_edges, device=device, model_config=self.model_config, batch_size=config[
                                   'batch_size'],
                               sampler=self.sampler, train_df=build_graph_df)

        log_prefix = f'{self.phase}_{self.config["name"]}_{self.model_config["name"]}'
        self.log_path = os.path.join(PROJ_ROOT, 'logs', f'{log_prefix}.log')
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler()
            ]
        )

    def offline_learning(self, use_initial_offline: bool = True, start_idx: Optional[int] = None,
                         end_idx: Optional[int] = None):
        if start_idx is not None and end_idx is not None:
            # 使用指定的数据进行离线学习
            logging.info(
                f"Offline learning with specified data from idx {start_idx} to {end_idx}")
            ckpt_prefix = f'{self.prefix}_specified{start_idx}_{end_idx}'
            self.ckpt_path = os.path.join(
                PROJ_ROOT, 'checkpoints', f'{ckpt_prefix}.pth')
            os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)

            train_dataloader = TimeSeriesDataLoader(
                self.df.iloc[start_idx:end_idx], self.node_label_dict, self.split_points,
                start_idx=start_idx, end_idx=end_idx)
            val_dataloader = None
            num_epochs = self.config['finetune_num_epochs']
            lr = 1e-3
        else:
            # 初始离线学习逻辑
            offline_ratio = self.config['initial_offline_ratio'] if use_initial_offline else self.config['offline_ratio']
            logging.info(
                f"Offline learning with offline_ratio={offline_ratio}")
            ckpt_prefix = f'{self.phase}_{self.config["name"]}_{self.model_config["name"]}_initial{offline_ratio}'
            self.ckpt_path = os.path.join(
                PROJ_ROOT, 'checkpoints', f'{ckpt_prefix}.pth')
            os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)

            test_ratio = self.config['test_ratio']
            train_dataloader = TimeSeriesDataLoader(
                self.df, self.node_label_dict, self.split_points,
                start_ts_quantile=0, end_ts_quantile=offline_ratio*(1-test_ratio))
            val_dataloader = TimeSeriesDataLoader(
                self.df, self.node_label_dict, self.split_points,
                start_ts_quantile=offline_ratio*(1-test_ratio), end_ts_quantile=offline_ratio)
            num_epochs = self.config['num_epochs']
            lr = self.args.lr

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        criterion = torch.nn.CrossEntropyLoss()
        early_stop = EarlyStopMonitor(higher_better=False, max_round=5)
        eval_metric = self.config['eval_metric']
        best_val = float('inf')
        for epoch in range(num_epochs):
            start = time.time()
            train_dict = self._train_one_epoch(
                train_dataloader, optimizer, criterion, eval_metric)
            epoch_time = time.time() - start
            # torch.cuda.empty_cache()
            if val_dataloader:
                start = time.time()
                val_dict = self._eval(val_dataloader, eval_metric)
                val_time = time.time() - start
                logging.info(
                    f'Epoch {epoch}: Train Loss: {train_dict["ce"]:.4f}, '
                    f'Val Loss: {val_dict["ce"]:.4f}, '
                    f'Epoch Time: {epoch_time:.2f}s, Val Time: {val_time:.2f}s')

                if val_dict['ce'] < best_val:
                    best_val = val_dict['ce']
                    torch.save({
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'config': self.config,
                        'model_config': self.model_config,
                    }, self.ckpt_path)

                if early_stop.early_stop_check(val_dict['ce']):
                    logging.info(f'Early stopping at epoch {epoch}')
                    break
            else:
                logging.info(
                    f'Epoch {epoch}: Train Loss: {train_dict["ce"]:.4f}, Epoch Time: {epoch_time:.2f}s')

        if not val_dataloader:
            torch.save({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'config': self.config,
                'model_config': self.model_config,
            }, self.ckpt_path)

        return epoch

    def _train_one_epoch(self, dataloader: TimeSeriesDataLoader,
                         optimizer, criterion):
        total_batch_time = 0
        total_data_time = 0
        total_sample_time = 0
        total_feature_time = 0
        total_forward_time = 0
        total_backward_time = 0
        total_eval_time = 0

        self.model.train()

        max_batch_size = self.config['batch_size']

        total_loss = 0
        num_label_ts = 0
        # 初始训练逻辑
        for (batch, labels) in dataloader:
            batch_start = time.time()
            # src, dst, ts, eid = batch
            label_src, label_ts, node_labels = labels
            node_labels_cuda = node_labels.to(self.device).to_dense()
            node_labels = node_labels.to_dense()
            total_data_time += time.time() - batch_start

            # in case of large batch size, split the batch into smaller ones
            for start in range(0, len(label_src), max_batch_size):
                end = min(start + max_batch_size, len(label_src))
                label_src_tmp = label_src[start:end]
                label_ts_tmp = label_ts[start:end]
                node_labels_cuda_tmp = node_labels_cuda[start:end]

                n_id = label_src_tmp

                sample_start = time.time()
                mfgs = self.sampler.sample(n_id, label_ts_tmp)
                total_sample_time += time.time() - sample_start

                feature_start = time.time()
                mfgs_to_cuda(mfgs, self.device)
                mfgs = self.cache.fetch_feature(mfgs)
                total_feature_time += time.time() - feature_start

                # Get updated memory of all nodes involved in the computation.
                if "memory" in self.model:
                    # TODO: Implement memory module
                    z, last_update = self.model['memory'](n_id)

                forward_start = time.time()
                z = self.model['encoder'](mfgs)

                if "memory" in self.model:
                    pass
                    # self.model['memory'].update_state(src, pos_dst, t, msg)

                pred = self.model['decoder'](z)
                loss = criterion(pred, node_labels_cuda_tmp)
                total_forward_time += time.time() - forward_start

                backward_start = time.time()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_backward_time += time.time() - backward_start

                if "memory" in self.model:
                    self.model['memory'].detach()

                total_loss += float(loss)

            eval_start = time.time()

            num_label_ts += 1
            total_eval_time += time.time() - eval_start

            total_batch_time += time.time() - batch_start

        metric_dict = {
            'ce': total_loss / num_label_ts
        }
        return metric_dict

    @torch.no_grad()
    def _eval(self, dataloader: TimeSeriesDataLoader):
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        # total_score = 0
        num_label_ts = 0
        total_loss = 0
        nodes = self.dgraph.nodes()
        for (batch, labels) in dataloader:
            # src, dst, ts, eid = batch
            label_src, label_ts, node_labels = labels
            node_labels_cuda = node_labels.to(self.device).to_dense()
            node_labels = node_labels.to_dense()

            # in case label_src's node not in the graph
            mask = np.isin(label_src, nodes)
            label_src = label_src[mask]
            label_ts = label_ts[mask]
            node_labels = node_labels[mask]

            for start in range(0, len(label_src), self.config['batch_size']):
                end = min(start + self.config['batch_size'], len(label_src))
                label_src_tmp = label_src[start:end]
                label_ts_tmp = label_ts[start:end]
                node_labels_cuda_tmp = node_labels_cuda[start:end]

                n_id = label_src_tmp

                mfgs = self.sampler.sample(n_id, label_ts_tmp)
                mfgs_to_cuda(mfgs, self.device)
                mfgs = self.cache.fetch_feature(mfgs)

                z = self.model['encoder'](mfgs)

                pred = self.model['decoder'](z)

                loss = criterion(pred, node_labels_cuda_tmp)
                total_loss += float(loss)

            num_label_ts += 1

        metric_dict = {
            'ce': total_loss / num_label_ts
        }
        return metric_dict

    def _calculate_embeddings(self, all_nodes, all_ts):
        infer_batch_size = 2000
        torch.cuda.empty_cache()
        self.model.eval()

        with torch.no_grad():
            # TODO: Implement memory module
            embeds = self.model['encoder'].inference(
                all_nodes, all_ts, self.full_sampler, infer_batch_size, self.cache, self.device)

        return embeds

    def collecting_data_from_offline(self, update_freqs: List[int] = np.arange(0, 31)):
        """collecting training data for accuracy predictor from offline data

        Args:
            initial_offline_ratio: the ratio of initial offline data to use
            offline_ratio: the ratio of offline data to use
            update_freqs: the frequency of updating the accuracy predictor by day (0 means no update)
        """
        print("Collecting data from offline")
        torch.cuda.empty_cache()
        initial_offline_ratio = self.config['initial_offline_ratio']
        offline_ratio = self.config['offline_ratio']

        dataloader = TimeSeriesDataLoader(
            self.df, self.node_label_dict, self.split_points,
            start_ts_quantile=initial_offline_ratio, end_ts_quantile=offline_ratio)

        # 初始化图
        split_point = dataloader.initial_start_idx
        dataset_df = self.df.iloc[:split_point]
        self.build_graph_sampler_cache(dataset_df, self.config, self.device)

        # 初始化模型
        ckpt_prefix = f'offline_{self.config["name"]}_{self.model_config["name"]}_initial{initial_offline_ratio}'
        ckpt_path = os.path.join(
            PROJ_ROOT, 'checkpoints', f"{ckpt_prefix}.pth")
        prefix = f'offline_{self.config["name"]}_{self.model_config["name"]}'
        self.training_pairs_path = os.path.join(
            PROJ_ROOT, 'training_pairs', f'{prefix}_training_pairs.feather')
        os.makedirs(os.path.dirname(self.training_pairs_path), exist_ok=True)

        self.model.load_state_dict(torch.load(
            ckpt_path, weights_only=False, map_location=torch.device(f"cuda:{self.device}"))['state_dict'])

        # 初始化reference node embeddings
        start_ts = dataloader.get_label_ts()
        old_nodes = self.dgraph.nodes()
        old_ts = np.full(len(old_nodes), start_ts)

        old_embeds = self._calculate_embeddings(old_nodes, old_ts)
        print("Initial embeddings calculated")

        evaluator = Evaluator(name=self.config['name'])
        eval_metric = self.config['eval_metric']

        nodes_set = set(self.dgraph.nodes())
        num_edges = self.dgraph.num_edges()
        training_pairs = []

        update_freq = update_freqs[0]

        logging.info(
            f"Start collecting data from offline data with update_freq={update_freq}")

        num_batches = 0
        node_activity_tracker = WindowActivityTracker(window_size=14)
        for (batch, labels) in tqdm(dataloader):
            # 1. 更新图
            src, dst, ts, eid = batch
            label_src, label_ts, node_labels = labels
            node_labels = node_labels.to_dense()

            self.dgraph.add_edges(src, dst, ts, eid, add_reverse=True)
            cur_nodes = self.dgraph.nodes()
            node_to_index = {node: idx for idx, node in enumerate(cur_nodes)}
            mask = np.isin(label_src, cur_nodes)
            label_src = label_src[mask]
            label_ts = label_ts[mask]
            node_labels = node_labels[mask]

            activity = node_activity_tracker.update_and_get(label_src)

            # 2. 计算所有节点的embeddings
            cur_ts = np.full(len(cur_nodes), label_ts[0], dtype=np.float32)
            cur_embeds = self._calculate_embeddings(cur_nodes, cur_ts)

            # 3. 计算embedding的分布变化
            shift_results = calculate_embeddings_distribution_shift(
                old_nodes, old_embeds, cur_nodes, cur_embeds, methods=['cmd', 'mmd2'])
            full_mmd2, full_cmd = shift_results['mmd2'], shift_results['cmd']
            full_mmd2_time, full_cmd_time = shift_results['mmd2_time'], shift_results['cmd_time']

            # 4. 计算evaluation metric for label_src
            label_src_indices = np.array(
                [node_to_index[node] for node in label_src])
            label_src_embeds = cur_embeds[label_src_indices]

            shift_results = calculate_embeddings_distribution_shift(
                old_nodes, old_embeds, label_src, label_src_embeds, methods=['mmd2', 'cmd', 'mse', 'mae'])
            mmd2, cmd, mse, mae = shift_results['mmd2'], shift_results[
                'cmd'], shift_results['mse'], shift_results['mae']
            mmd2_time, cmd_time, mse_time, mae_time = shift_results['mmd2_time'], shift_results[
                'cmd_time'], shift_results['mse_time'], shift_results['mae_time']

            mfg = self.full_sampler.sample(label_src, label_ts)[0][0]
            label_src_neighbors = mfg.srcdata['ID'][mfg.num_dst_nodes():]
            label_src_neighbors = np.unique(label_src_neighbors.cpu().numpy())
            label_src_neighbors_indices = np.array(
                [node_to_index[node] for node in label_src_neighbors])
            label_src_neighbors_embeds = cur_embeds[label_src_neighbors_indices]

            label_src_with_neighbors = np.concatenate(
                [label_src, label_src_neighbors], axis=0)
            label_src_with_neighbors_embeds = torch.cat(
                [label_src_embeds, label_src_neighbors_embeds], dim=0)
            shift_results = calculate_embeddings_distribution_shift(
                old_nodes, old_embeds, label_src_with_neighbors, label_src_with_neighbors_embeds, methods=['mmd2', 'cmd', 'mse', 'mae'])
            neighbor_mmd2, neighbor_cmd, neighbor_mse, neighbor_mae = shift_results['mmd2'], shift_results[
                'cmd'], shift_results['mse'], shift_results['mae']
            neighbor_mmd2_time, neighbor_cmd_time, neighbor_mse_time, neighbor_mae_time = shift_results[
                'mmd2_time'], shift_results['cmd_time'], shift_results['mse_time'], shift_results['mae_time']

            mfg2 = self.full_sampler2.sample(label_src, label_ts)[0][0]
            label_src_neighbors_2hop = mfg2.srcdata['ID'][len(label_src):]
            label_src_neighbors_2hop = np.unique(
                label_src_neighbors_2hop.cpu().numpy())
            label_src_neighbors_indices = np.array(
                [node_to_index[node] for node in label_src_neighbors_2hop])
            label_src_neighbors_embeds = cur_embeds[label_src_neighbors_indices]

            label_src_with_neighbors = np.concatenate(
                [label_src, label_src_neighbors_2hop], axis=0)
            label_src_with_neighbors_embeds = torch.cat(
                [label_src_embeds, label_src_neighbors_embeds], dim=0)
            shift_results = calculate_embeddings_distribution_shift(
                old_nodes, old_embeds, label_src_with_neighbors, label_src_with_neighbors_embeds, methods=['mmd2', 'cmd', 'mse', 'mae'])
            neighbor_2hop_mmd2, neighbor_2hop_cmd, neighbor_2hop_mse, neighbor_2hop_mae = shift_results['mmd2'], shift_results[
                'cmd'], shift_results['mse'], shift_results['mae']
            neighbor_2hop_mmd2_time, neighbor_2hop_cmd_time, neighbor_2hop_mse_time, neighbor_2hop_mae_time = shift_results[
                'mmd2_time'], shift_results['cmd_time'], shift_results['mse_time'], shift_results['mae_time']

            pred = self.model['decoder'](label_src_embeds)
            np_pred = pred.detach().cpu().numpy()
            np_true = node_labels

            input_dict = {
                "y_true": np_true,
                "y_pred": np_pred,
                "eval_metric": [eval_metric]
            }
            result_dict = evaluator.eval(input_dict)

            # 5. 记录数据
            label_src_node_degrees = self.dgraph.out_degree(label_src)
            data = {
                'full_mmd2': full_mmd2,
                'full_cmd': full_cmd,
                'mmd2': mmd2,
                'cmd': cmd,
                'mse': mse,
                'mae': mae,
                'neighbor_mmd2': neighbor_mmd2,
                'neighbor_cmd': neighbor_cmd,
                'neighbor_mse': neighbor_mse,
                'neighbor_mae': neighbor_mae,
                'neighbor_num_nodes': len(label_src_neighbors),
                'neighbor_2hop_mmd2': neighbor_2hop_mmd2,
                'neighbor_2hop_cmd': neighbor_2hop_cmd,
                'neighbor_2hop_mse': neighbor_2hop_mse,
                'neighbor_2hop_mae': neighbor_2hop_mae,
                'neighbor_2hop_num_nodes': len(label_src_neighbors_2hop),
                'metric': result_dict[eval_metric],
                'num_edges': num_edges,
                'num_nodes': len(cur_nodes),
                'num_new_nodes': len(set(src) - nodes_set),
                'num_new_edges': len(src),
                'num_label_nodes': len(label_src),
                'label_nodes_mean_degree': label_src_node_degrees.mean(),
                'label_nodes_mean_activity': activity.mean(),
                'ts': label_ts[0],
                'update_freq': update_freq,
                'full_mmd2_time': full_mmd2_time,
                'full_cmd_time': full_cmd_time,
                'mmd2_time': mmd2_time,
                'cmd_time': cmd_time,
                'mse_time': mse_time,
                'mae_time': mae_time,
                'neighbor_mmd2_time': neighbor_mmd2_time,
                'neighbor_cmd_time': neighbor_cmd_time,
                'neighbor_mse_time': neighbor_mse_time,
                'neighbor_mae_time': neighbor_mae_time,
                'neighbor_2hop_mmd2_time': neighbor_2hop_mmd2_time,
                'neighbor_2hop_cmd_time': neighbor_2hop_cmd_time,
                'neighbor_2hop_mse_time': neighbor_2hop_mse_time,
                'neighbor_2hop_mae_time': neighbor_2hop_mae_time,
            }
            training_pairs.append(data)

            logging.info(data)

            df = pd.DataFrame(training_pairs)
            df.to_feather(self.training_pairs_path)

            # 6. 更新其他指标
            nodes_set.update(src)
            nodes_set.update(dst)
            num_edges += len(src)

            num_batches += 1
            if update_freq > 0 and num_batches % update_freq == 0:
                # TODO: 更新模型!
                pass

        # 保存数据
        df = pd.DataFrame(training_pairs)
        print(df.head())
        df.to_feather(self.training_pairs_path)

    def continuous_learning(self, offline_ratio: float = 0.3,
                            trigger_type: str = 'label_propagation',
                            sliding_window: int = 365,
                            warmup: int = 0,
                            cooldown: int = 0,
                            delay: int = 7,
                            **kwargs):
        """continuous learning

        Args:
            offline_ratio (float): the ratio of offline data to use
            trigger_type (str): the type of trigger for updating the model
            sliding_window (int): the length of sliding window (day) for retraining the model (default: 30)
            warmup (int): the number of days for warmup (default: 5)
            cooldown (int): the number of days for cooldown (default: 0)
            delay (int): the number of days for labels to be available (default: 7)
        """
        # 持续学习逻辑
        print("Continuous learning")
        torch.cuda.empty_cache()

        dataloader = TimeSeriesDataLoader(
            self.df, self.node_label_dict, self.split_points,
            start_ts_quantile=offline_ratio, end_ts_quantile=1)

        # 初始化图
        split_point = dataloader.initial_start_idx
        dataset_df = self.df.iloc[:split_point]
        self.build_graph_sampler_cache(dataset_df, self.config, self.device)

        # 初始化模型
        ckpt_prefix = f'offline_{self.config["name"]}_{self.model_config["name"]}_initial{offline_ratio}'
        ckpt_path = os.path.join(
            PROJ_ROOT, 'checkpoints', f"{ckpt_prefix}.pth")
        prefix = f'offline_{self.config["name"]}_{self.model_config["name"]}'
        offline_training_pairs_path = os.path.join(
            PROJ_ROOT, 'training_pairs', f'{prefix}_training_pairs.feather')
        os.makedirs(os.path.dirname(
            offline_training_pairs_path), exist_ok=True)

        prefix = f'streaming_{self.config["name"]}_{self.model_config["name"]}_trigger_{trigger_type}'
        if trigger_type == 'interval':
            prefix += f'{kwargs["interval"]}'
        elif trigger_type == 'accuracy_drop':
            prefix += f'{kwargs["accuracy_drop_threshold"]}'
        elif trigger_type == 'delayed_accuracy_drop':
            prefix += f'{kwargs["accuracy_drop_threshold"]}'
        elif trigger_type == 'label_propagation':
            prefix += f'{kwargs["problem_ratio"]}'
        elif trigger_type == 'label_no_propagation_true_accuracy':
            prefix += f'{kwargs["problem_ratio"]}'
        elif trigger_type == 'label_propagation_delayed_true_accuracy':
            prefix += f'{kwargs["problem_ratio"]}'
        elif trigger_type == 'mmd':
            prefix += f'{kwargs["distance_threshold"]}'
        elif trigger_type == 'adwin':
            prefix += f'{kwargs["delta"]}'
        elif trigger_type == 'kswin':
            prefix += f'{kwargs["alpha"]}'
        elif trigger_type == 'adwin_delayed_accuracy':
            prefix += f'{kwargs["delta"]}'
        elif trigger_type == 'kswin_delayed_accuracy':
            prefix += f'{kwargs["alpha"]}'

        prefix += f'_sliding_window_{sliding_window}'
        self.prefix = prefix
        self.training_pairs_path = os.path.join(
            PROJ_ROOT, 'training_pairs', f'{self.prefix}_training_pairs.feather')

        model_ckpt = torch.load(ckpt_path, map_location=torch.device(
            f"cuda:{self.device}"), weights_only=False)
        self.model.load_state_dict(model_ckpt['state_dict'])

        # NB: 训练accuracy predictor
        df_training_pairs = pd.read_feather(offline_training_pairs_path)
        perf_predictor, std_scaler, results = train_perf_predictor(
            df_training_pairs)
        print(results)

        # 初始化reference node embeddings
        start_ts = dataloader.get_label_ts()
        old_nodes = self.dgraph.nodes()
        old_ts = np.full(len(old_nodes), start_ts, dtype=np.float32)
        old_embeds = self._calculate_embeddings(old_nodes, old_ts)
        print("Initial embeddings calculated")

        slo_threshold = self.config['slo_threshold']
        init_df = dataset_df
        max_node_id = max(self.df['src'].max(), self.df['dst'].max())
        logging.info(f"Using SLO threshold: {slo_threshold}")
        threshold = 0.5

        logging.info(f"Using cooldown: {cooldown}")

        update_trigger = get_update_trigger(
            trigger_type, slo_threshold=slo_threshold,
            max_node_id=max_node_id, init_df=init_df,
            x_ref=old_embeds, threshold=threshold,
            propagation_weight=kwargs.get('weight', 0.5),
            propagation_layers=kwargs.get('k', 2),
            device=torch.device(f"cuda:{self.device}"), **kwargs)

        evaluator = Evaluator(name=self.config['name'])
        eval_metric = self.config['eval_metric']

        nodes_set = set(self.dgraph.nodes())
        num_edges = self.dgraph.num_edges()
        training_pairs = []

        ####### 实验数据 #######
        batch_size_list = []
        graph_update_time = []
        inference_time = []
        collect_features_time = []
        accuracy_predict_time = []
        train_accuracy_predictor_time = []
        compute_true_accuracy_time = []
        update_trigger_time = []
        retrain_model_time = []
        pred_accuracy_list = []
        true_accuracy_list = []
        loss_list = []
        training_samples = []
        ####### 实验数据 #######

        criterion = torch.nn.CrossEntropyLoss()

        num_batches = 0
        last_retraining_batch_idx = dataloader.get_cur_idx()
        node_activity_tracker = WindowActivityTracker(window_size=14)
        logging.info("Start continuous learning")
        for (batch, labels) in tqdm(dataloader):
            # 1. 更新图
            src, dst, ts, eid = batch
            label_src, label_ts, node_labels = labels
            node_labels_cuda = node_labels.to(self.device).to_dense()
            node_labels = node_labels.to_dense()

            # 1.1 记录实验数据
            batch_size_list.append(len(src))

            start_time = time.time()
            self.dgraph.add_edges(src, dst, ts, eid, add_reverse=True)
            graph_update_time.append(time.time() - start_time)

            cur_nodes = self.dgraph.nodes()
            node_to_index = {node: idx for idx, node in enumerate(cur_nodes)}
            mask = np.isin(label_src, cur_nodes)
            label_src = label_src[mask]
            label_ts = label_ts[mask]
            node_labels = node_labels[mask]
            node_labels_cuda = node_labels_cuda[mask]

            # 2. 计算所有节点的embeddings and 推理
            start_time = time.time()
            cur_ts = np.full(len(cur_nodes), label_ts[0], dtype=np.float32)
            cur_embeds = self._calculate_embeddings(cur_nodes, cur_ts)
            label_src_indices = np.array(
                [node_to_index[node] for node in label_src])
            label_src_embeds = cur_embeds[label_src_indices]

            pred = self.model['decoder'](label_src_embeds)
            inference_time.append(time.time() - start_time)

            start = time.time()

            loss = criterion(pred, node_labels_cuda)
            loss_list.append(loss.item())

            np_pred = pred.detach().cpu().numpy()
            np_true = node_labels

            input_dict = {
                "y_true": np_true,
                "y_pred": np_pred,
                "eval_metric": [eval_metric]
            }
            result_dict = evaluator.eval(input_dict, average=False)
            true_accuracy_list.append(result_dict[eval_metric].mean())
            compute_true_accuracy_time.append(time.time() - start)

            # 4. 计算evaluation metric for label_src
            start_time = time.time()
            activity = node_activity_tracker.update_and_get(label_src)

            mfg = self.full_sampler.sample(label_src, label_ts)[0][0]
            label_src_neighbors = mfg.srcdata['ID'][mfg.num_dst_nodes():]
            label_src_neighbors = np.unique(label_src_neighbors.cpu().numpy())
            label_src_neighbors_indices = np.array(
                [node_to_index[node] for node in label_src_neighbors])
            label_src_neighbors_embeds = cur_embeds[label_src_neighbors_indices]

            label_src_with_neighbors = np.concatenate(
                [label_src, label_src_neighbors], axis=0)
            label_src_with_neighbors_embeds = torch.cat(
                [label_src_embeds, label_src_neighbors_embeds], dim=0)
            shift_results = calculate_embeddings_distribution_shift(
                old_nodes, old_embeds, label_src_with_neighbors, label_src_with_neighbors_embeds, methods=['mae'])
            neighbor_mae = shift_results['mae']

            # 5. 记录数据
            label_src_node_degrees = self.dgraph.out_degree(label_src)
            data = {
                'full_mmd2': -1,  # NB: full_mmd2 is not used
                'mmd2': -1,  # NB: mmd2 is not used
                'full_cmd': -1,  # NB: full_cmd is not used
                'cmd': -1,  # NB: cmd is not used
                'mae': -1,
                'mse': -1,  # NB: mse is not used
                'neighbor_mmd2': -1,  # NB: neighbor_mmd2 is not used
                'neighbor_cmd': -1,  # NB: neighbor_mmd2 is not used
                'neighbor_mse': -1,  # NB: neighbor_mse is not used
                'neighbor_mae': neighbor_mae,
                'neighbor_num_nodes': len(label_src_neighbors),
                'metric': result_dict[eval_metric].mean(),
                'num_edges': num_edges,
                'num_nodes': len(cur_nodes),
                'num_new_nodes': len(set(src) - nodes_set),
                'num_new_edges': len(src),
                'num_label_nodes': len(label_src),
                'label_nodes_mean_degree': label_src_node_degrees.mean(),
                'label_nodes_mean_activity': activity.mean(),
                'ts': label_ts[0],
                'update_freq': -1  # NB: update_freq is not used
            }
            collect_features_time.append(time.time() - start_time)

            # 6. 更新其他指标
            nodes_set.update(src)
            num_edges += len(src)

            # 7. accuracy predictor预测
            start_time = time.time()
            pred_perf = predict_perf(perf_predictor, std_scaler, data)
            accuracy_predict_time.append(time.time() - start_time)

            error = (abs(data['metric'] - pred_perf) / data['metric'])*100
            data['pred_perf'] = pred_perf
            pred_accuracy_list.append(pred_perf)

            training_pairs.append(data)

            start_time = time.time()
            df = pd.DataFrame(training_pairs)
            df_training_pairs_all = pd.concat(
                [df_training_pairs, df], ignore_index=True)
            perf_predictor, std_scaler, results = train_perf_predictor(
                df_training_pairs_all)
            train_accuracy_predictor_time.append(time.time() - start_time)
            df_training_pairs_all.to_feather(self.training_pairs_path)

            num_batches += 1

            # 8. 检查是否需要更新模型
            start_time = time.time()
            prev_split_point = dataloader.split_points[dataloader.label_ts_idx-2]
            cur_split_point = dataloader.split_points[dataloader.label_ts_idx-1]
            seen_df = self.df.iloc[:cur_split_point]
            if num_batches == 1:
                prev_split_point = split_point
            batch_df = self.df.iloc[prev_split_point:cur_split_point]

            if num_batches < warmup or num_batches > len(dataloader) - cooldown:
                # NB: skip the first N batches for warm-up and the last N batches for cooldown
                need_update = False
            else:
                need_update = update_trigger.update_check(
                    # NB: use pred_perf instead of data['metric'] here
                    pred_accuracy=pred_perf,
                    true_accuracy=result_dict[eval_metric],
                    max_node_id=self.dgraph.max_vertex_id(),
                    num_nodes=len(cur_nodes),
                    batch_nodes=label_src,
                    df=seen_df,
                    batch_df=batch_df,
                    x=cur_embeds,
                )

            update_trigger_time.append(time.time() - start_time)

            cur_idx = dataloader.get_cur_idx()
            cur_idx -= delay  # NB: delay for labels to be available
            start_idx = max(0, cur_idx - sliding_window)

            if need_update:
                # 更新模型
                start_time = time.time()
                logging.info(f"Updating model at batch {num_batches}")

                num_epochs = self.offline_learning(
                    start_idx=start_idx, end_idx=cur_idx)
                num_samples_per_epoch = self.split_points[cur_idx] - \
                    self.split_points[start_idx]
                training_samples.append(num_epochs * num_samples_per_epoch)
                logging.info(
                    f"Finished model retraining at batch {num_batches}")
                # 重新加载模型
                self.model.load_state_dict(torch.load(
                    self.ckpt_path, weights_only=False, map_location=torch.device(f"cuda:{self.device}"))['state_dict'])
                logging.info(f"Model reloaded at batch {num_batches}")

                # 更新reference node embeddings
                old_nodes = cur_nodes
                old_ts = np.full(len(old_nodes), label_ts[0], dtype=np.float32)
                old_embeds = self._calculate_embeddings(old_nodes, old_ts)
                logging.info(
                    f"Reference node embeddings updated at batch {num_batches}")

                # 重置trigger
                update_trigger.reset(x_ref=old_embeds)

                retrain_model_time.append(time.time() - start_time)

                # 设置cooldown
                warmup = num_batches + cooldown
            else:
                retrain_model_time.append(0)
                training_samples.append(0)

            # print profile
            logging.info(f"Batch [{num_batches}]: batch_size: {batch_size_list[-1]}, "
                         f"Graph update time: {graph_update_time[-1]:.2f}s, "
                         f"Inference time: {inference_time[-1]:.2f}s, "
                         f"Collect features time: {collect_features_time[-1]:.2f}s, "
                         f"Accuracy predict time: {accuracy_predict_time[-1]:.2f}s, "
                         f"Train accuracy predictor time: {train_accuracy_predictor_time[-1]:.2f}s, "
                         f"Compute true accuracy time: {compute_true_accuracy_time[-1]:.2f}s, "
                         f"Update trigger time: {update_trigger_time[-1]:.2f}s, "
                         f"Retrain model time: {retrain_model_time[-1]:.2f}s, "
                         f"Pred accuracy: {pred_accuracy_list[-1]:.4f}, "
                         f"True accuracy: {true_accuracy_list[-1]:.4f}, "
                         f"Loss: {loss_list[-1]:.4f}, "
                         f"Training samples: {training_samples[-1]}, "
                         f"Error: {error:.2f}%")

            print(f"len(batch_size_list): {len(batch_size_list)}")

            # 随时保存实验数据
            df_exp = pd.DataFrame({
                'batch_size': batch_size_list,
                'graph_update_time': graph_update_time,
                'inference_time': inference_time,
                'collect_features_time': collect_features_time,
                'accuracy_predict_time': accuracy_predict_time,
                'train_accuracy_predictor_time': train_accuracy_predictor_time,
                'compute_true_accuracy_time': compute_true_accuracy_time,
                'update_trigger_time': update_trigger_time,
                'retrain_model_time': retrain_model_time,
                'pred_accuracy': pred_accuracy_list,
                'true_accuracy': true_accuracy_list,
                'loss': loss_list,
                'training_samples': training_samples,
                # NB: expected_len is only for checking
                'expected_len': [len(dataloader) for _ in range(len(batch_size_list))]
            })
            df_exp_path = os.path.join(
                PROJ_ROOT, 'exp', f'{self.prefix}_exp.feather')
            os.makedirs(os.path.dirname(df_exp_path), exist_ok=True)
            df_exp.to_feather(df_exp_path)

        # 保存数据
        df = pd.DataFrame(training_pairs)
        df_training_pairs_all = pd.concat(
            [df_training_pairs, df], ignore_index=True)
        print(df.tail())
        df_training_pairs_all.to_feather(self.training_pairs_path)

        # 保存实验数据
        df_exp = pd.DataFrame({
            'batch_size': batch_size_list,
            'graph_update_time': graph_update_time,
            'inference_time': inference_time,
            'collect_features_time': collect_features_time,
            'accuracy_predict_time': accuracy_predict_time,
            'train_accuracy_predictor_time': train_accuracy_predictor_time,
            'compute_true_accuracy_time': compute_true_accuracy_time,
            'update_trigger_time': update_trigger_time,
            'retrain_model_time': retrain_model_time,
            'pred_accuracy': pred_accuracy_list,
            'true_accuracy': true_accuracy_list,
            'loss': loss_list,
            'training_samples': training_samples,
            # NB: expected_len is only for checking
            'expected_len': [len(dataloader) for _ in range(len(batch_size_list))]
        })
        df_exp_path = os.path.join(
            PROJ_ROOT, 'exp', f'{self.prefix}_exp.feather')
        os.makedirs(os.path.dirname(df_exp_path), exist_ok=True)
        df_exp.to_feather(df_exp_path)

    def benchmark_pyg_dgl_continuous_learning(self, offline_ratio: float = 0.3):
        import dgl
        import sys

        dataloader = TimeSeriesDataLoader(
            self.df, self.node_label_dict, self.split_points,
            start_ts_quantile=offline_ratio, end_ts_quantile=1)

        # 初始化图
        split_point = dataloader.initial_start_idx
        dataset_df = self.df.iloc[:split_point]
        self.build_graph_sampler_cache(dataset_df, self.config, self.device)

        self.prefix = f'benchmark_{self.config["name"]}_{self.model_config["name"]}'

        # DGL build grpah time

        dgl_build_graph_time = []
        our_build_graph_time = []

        dgl_sample_time = []
        our_sample_time = []

        dgl_feature_fetch_time = []
        our_feature_fetch_time = []

        node_feats, edge_feats = self.cache.node_feats, self.cache.edge_feats
        j = 0
        for (batch, labels) in tqdm(dataloader):
            prev_split_point = dataloader.split_points[dataloader.label_ts_idx-2]
            cur_split_point = dataloader.split_points[dataloader.label_ts_idx-1]
            seen_df = self.df.iloc[:cur_split_point]

            # 1. build graph
            start_time = time.time()
            src, dst = seen_df['src'].values, seen_df['dst'].values
            g = dgl.graph((src, dst))
            g.ndata['feat'] = node_feats[:g.num_nodes()].clone()
            g.edata['feat'] = edge_feats[:g.num_edges()].clone()
            g = dgl.add_reverse_edges(g, copy_edata=True)
            csr_g = g.formats('csr')
            end_time = time.time()
            dgl_build_graph_time.append(end_time - start_time)

            src, dst, ts, eid = batch
            label_src, label_ts, node_labels = labels
            mask = np.isin(label_src, g.nodes())
            label_src = label_src[mask]
            label_ts = label_ts[mask]

            start_time = time.time()
            self.dgraph.add_edges(src, dst, ts, eid, add_reverse=True)
            end_time = time.time()
            our_build_graph_time.append(end_time - start_time)

            # 2. sample

            sampler = dgl.dataloading.NeighborSampler([10, 10])
            old_func = sampler.sample_blocks

            dgl_sample_time_tmp = []

            def func(*args, **kwargs):
                nonlocal dgl_sample_time_tmp
                start = time.time()
                ret = old_func(*args, **kwargs)
                dgl_sample_time_tmp.append(time.time() - start)
                return ret
            sampler.sample_blocks = func

            batch_size = 1000

            dgl_feat_time_tmp = 0
            our_sample_time_tmp = 0
            our_feat_time_tmp = 0
            for i in range(0, len(label_src), batch_size):
                label_src_tmp = label_src[i:i+batch_size]
                label_ts_tmp = label_ts[i:i+batch_size]

                start_time = time.time()
                dgl_dataloader = dgl.dataloading.DataLoader(
                    g, label_src_tmp, sampler,
                    batch_size=len(label_src_tmp), shuffle=True, drop_last=False, num_workers=0,
                    use_uva=True, gpu_cache={'node': {'feat': 1e4}, 'edge': {'feat': 1e7}})

                for input_nodes, output_nodes, blocks in dgl_dataloader:
                    pass
                end_time = time.time()
                tot_time = end_time - start_time
                feat_time = tot_time - dgl_sample_time_tmp[-1]
                dgl_feat_time_tmp += feat_time

                start_time = time.time()
                our_mfgs = self.sampler.sample(label_src_tmp, label_ts_tmp)
                end_time = time.time()
                our_sample_time_tmp += end_time - start_time

                start_time = time.time()
                mfgs_to_cuda(our_mfgs, self.device)
                mfgs = self.cache.fetch_feature(our_mfgs)
                end_time = time.time()
                our_feat_time_tmp += end_time - start_time

            dgl_sample_time.append(np.sum(dgl_sample_time_tmp))
            dgl_feature_fetch_time.append(dgl_feat_time_tmp)

            our_sample_time.append(our_sample_time_tmp)
            our_feature_fetch_time.append(our_feat_time_tmp)

            j += 1
            if (j % 100) == 0:
                df = pd.DataFrame({
                    'dgl_build_graph_time': dgl_build_graph_time,
                    'our_build_graph_time': our_build_graph_time,
                    'dgl_sample_time': dgl_sample_time,
                    'our_sample_time': our_sample_time,
                    'dgl_feature_fetch_time': dgl_feature_fetch_time,
                    'our_feature_fetch_time': our_feature_fetch_time,
                })
                MB = 1 << 20
                print('avg_linked_list_length: {:.2f},  graph mem usage: {:.2f}MB, metadata (on GPU) mem usage: {:.2f}MB'.format(
                    self.dgraph.avg_linked_list_length(),
                    self.dgraph.get_graph_memory_usage() / MB,
                    self.dgraph.get_metadata_memory_usage() / MB))
                print(df.mean() * 1000)

                df.to_feather(f"../exp/{self.prefix}_system_benchmark.feather")

        df = pd.DataFrame({
            'dgl_build_graph_time': dgl_build_graph_time,
            'our_build_graph_time': our_build_graph_time,
            'dgl_sample_time': dgl_sample_time,
            'our_sample_time': our_sample_time,
            'dgl_feature_fetch_time': dgl_feature_fetch_time,
            'our_feature_fetch_time': our_feature_fetch_time,
        })

        df.to_feather(f"../exp/{self.prefix}_system_benchmark.feather")
