import logging
import os
import random
import time
from multiprocessing import shared_memory
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.distributed
from dgl.heterograph import DGLBlock

from .dynamic_graph import DynamicGraph

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

NODE_FEATS = None

shms = []


def create_shared_mem_array(name, shape, dtype):
    d_size = np.dtype(dtype).itemsize * np.prod(shape)

    shm = shared_memory.SharedMemory(create=True, size=d_size, name=name)
    shms.append(shm)  # keep track of shared memory blocks
    # numpy array on shared memory buffer
    dst = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
    print(f'create shared memory array {name} with size {dst.size}')
    return dst


def get_shared_mem_array(name, shape, dtype):
    shm = shared_memory.SharedMemory(name=name)
    shms.append(shm)  # keep track of shared memory blocks
    dst = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
    print(f'get shared memory array {name} with size {dst.size}')
    return dst


def get_node_feats():
    global NODE_FEATS
    return NODE_FEATS


def local_world_size():
    return int(os.environ["LOCAL_WORLD_SIZE"])


def local_rank():
    return int(os.environ["LOCAL_RANK"])


def rank():
    return torch.distributed.get_rank()


def get_project_root_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dataset(dataset: str, data_dir: Optional[str] = None) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the dataset and returns the dataframes for the train, validation, test and
    whole dataset.


    Args:
        dataset: the name of the dataset.
        data_dir: the directory where the dataset is stored.

    Returns:
        train_data: the dataframe for the train dataset.
        val_data: the dataframe for the validation dataset.
        test_data: the dataframe for the test dataset.
        full_data: the dataframe for the whole dataset.
    """
    if data_dir is None:
        data_dir = os.path.join(get_project_root_dir(), "data")

    path = os.path.join(data_dir, dataset, 'edges.feather')
    if not os.path.exists(path):
        raise ValueError('{} does not exist'.format(path))

    full_data = pd.read_feather(path)
    assert isinstance(full_data, pd.DataFrame)

    # if 'Unnamed: 0' in full_data.columns:
    full_data.rename(columns={'Unnamed: 0': 'eid'}, inplace=True)

    train_end = full_data['ext_roll'].values.searchsorted(1)
    val_end = full_data['ext_roll'].values.searchsorted(2)
    train_data = full_data[:train_end]
    val_data = full_data[train_end:val_end]
    test_data = full_data[val_end:]
    return train_data, val_data, test_data, full_data


def load_partition_table(dataset: str):
    """
    Loads the dataset and returns the dataframes for the train, validation, test and
    whole dataset.


    Args:
        dataset: the name of the dataset.

    Returns:
        pt: partition_table of the first 60% data of the dataset
    """

    data_dir = os.path.join(get_project_root_dir(), "partition_data")

    path = os.path.join(data_dir, dataset + '_metis_partition.pt')

    if not os.path.exists(path):
        logging.info(
            "Didn't find Partition table under path: {}, using default partition algorithm to partition...".format(path))
        return None

    logging.info(
        "Find corresponding file under path {}. Using this file to skip the initial partition phase!".format(path))
    pt = torch.load(path)
    return pt


def load_partition_table(dataset: str):
    """
    Loads the dataset and returns the dataframes for the train, validation, test and
    whole dataset.
    Args:
        dataset: the name of the dataset.
    Returns:
        pt: partition_table of the first 60% data of the dataset
    """

    data_dir = os.path.join(get_project_root_dir(), "partition_data")

    path = os.path.join(data_dir, dataset + '_metis_partition.pt')

    if not os.path.exists(path):
        logging.info(
            "Didn't find Partition table under path: {}, using default partition algorithm to partition...".format(path))
        return None

    logging.info(
        "Find corresponding file under path {}. Using this file to skip the initial partition phase!".format(path))
    pt = torch.load(path).to(torch.int8)
    return pt


def load_dataset_in_chunks(dataset: str, data_dir: Optional[str] = None, chunksize: int = 100000000):
    """
    Loads the dataset and returns an iterator of the whole dataset

    Args:
        dataset: the name of the dataset.
        data_dir: the directory where the dataset is stored.
        chunksize: the size of the chunk to be loaded at a time

    Returns:
        iterator: the iterator of the whole dataset
    """
    if data_dir is None:
        data_dir = os.path.join(get_project_root_dir(), "data")

    path = os.path.join(data_dir, dataset, 'edges.csv')
    if not os.path.exists(path):
        raise ValueError('{} does not exist'.format(path))

    # NB: pyarrow is not support with chunksize
    return pd.read_csv(path, chunksize=chunksize, usecols=['src', 'dst', 'time', 'Unnamed: 0', 'ext_roll'])


def load_partitioned_dataset(dataset: str, data_dir: Optional[str] = None, rank: int = 0, world_size: int = 1, partition_train_data: bool = False):
    """
    Loads the partitioned dataset and returns the dataframes for the train, validation, test.

    Args:
        dataset: the name of the dataset.
        data_dir: the directory where the dataset is stored.

    Returns:
        train_data: the dataframe for the train dataset.
        val_data: the dataframe for the validation dataset.
    """
    if data_dir is None:
        data_dir = os.path.join(get_project_root_dir(), "data")

    train_path = os.path.join(
        data_dir, dataset, 'edges_train_{}_{}.csv'.format(str(world_size), str(rank)))
    val_path = os.path.join(
        data_dir, dataset, 'edges_val_{}_{}.csv'.format(str(world_size), str(rank)))
    test_path = os.path.join(
        data_dir, dataset, 'edges_test_{}_{}.csv'.format(str(world_size), str(rank)))

    train_data = None
    if not partition_train_data:
        train_data = pd.read_csv(train_path)
        assert isinstance(train_data, pd.DataFrame)
    val_data = pd.read_csv(val_path)
    assert isinstance(val_data, pd.DataFrame)
    test_data = pd.read_csv(test_path)
    assert isinstance(test_data, pd.DataFrame)

    return train_data, val_data, test_data


def load_node_feat(dataset: str, data_dir: Optional[str] = None):
    """
    Loads the node features for the dataset.

    Args:
        dataset: the name of the dataset.
        data_dir: the directory where the dataset is stored.
    """
    global NODE_FEATS
    if data_dir is None:
        data_dir = os.path.join(get_project_root_dir(), "data")

    dataset_path = os.path.join(data_dir, dataset)
    start = time.time()
    if dataset == 'MAG':
        # each local_rank == 0 worker read a part of the node features
        machine_rank = rank() // local_world_size()
        num_machines = torch.distributed.get_world_size() // local_world_size()
        path = os.path.join(
            dataset_path, 'node_features_{}.npy'.format(machine_rank))
        if not os.path.exists(path):
            raise ValueError('{} does not exist'.format(path))
        node_feat = np.load(path, allow_pickle=True)
        node_feat = torch.from_numpy(node_feat)
        logging.info("Rank: {}: Loaded node feature part {} in {:.2f} seconds.".format(
            rank(), rank(), time.time() - start))

        # send to rank == 0's worker using send/recv
        if rank() == 0:
            node_feat_list = [node_feat]
            for i in range(1, num_machines):
                shape = torch.empty(2, dtype=torch.int64)
                torch.distributed.recv(shape, i * local_world_size())
                node_feat_part = torch.empty(
                    shape.tolist(), dtype=torch.float16)
                torch.distributed.recv(node_feat_part, i * local_world_size())
                node_feat_list.append(node_feat_part)
            node_feat = torch.cat(node_feat_list, dim=0)
            del node_feat_list
            NODE_FEATS = node_feat
        else:
            shape = torch.tensor(node_feat.shape, dtype=torch.int64)
            torch.distributed.send(shape, 0)
            torch.distributed.send(node_feat, 0)
            logging.info("Rank: {}: Sent node feature part {} in {:.2f} seconds.".format(
                rank(), rank(), time.time() - start))
            del node_feat
    else:
        if rank() == 0:
            path = os.path.join(dataset_path, 'node_features.npy')
            if not os.path.exists(path):
                raise ValueError('{} does not exist'.format(path))

            node_feats = np.load(path, allow_pickle=False)
            NODE_FEATS = torch.from_numpy(node_feats)

    if rank() == 0:
        logging.info("Loaded node feature in %f seconds.", time.time() - start)


def load_feat(dataset: str, data_dir: Optional[str] = None,
              shared_memory: bool = False, local_rank: int = 0, local_world_size: int = 1,
              memmap: bool = False, load_node: bool = True, load_edge: bool = True):
    """
    Loads the node and edge features for the given dataset.

    NB: either node_feats or edge_feats can be None, but not both.

    Args:
        dataset: the name of the dataset.
        data_dir: the directory where the dataset is stored.
        shared_memory: whether to use shared memory.
        local_rank: the local rank of the process.
        local_world_size: the local world size of the process.
        memmap (bool): whether to use memmap.
        load_node (bool): whether to load node features.
        load_edge (bool): whether to load edge features.

    Returns:
        node_feats: the node features. (None if not available)
        edge_feats: the edge features. (None if not available)
    """
    if data_dir is None:
        data_dir = os.path.join(get_project_root_dir(), "data")

    dataset_path = os.path.join(data_dir, dataset)
    node_feat_path = os.path.join(dataset_path, 'node_features.npy')
    edge_feat_path = os.path.join(dataset_path, 'edge_features.npy')

    if not os.path.exists(node_feat_path) and \
            not os.path.exists(edge_feat_path):
        raise ValueError("Both {} and {} do not exist".format(
            node_feat_path, edge_feat_path))

    mmap_mode = "r+" if memmap else None

    node_feats = None
    edge_feats = None
    if not shared_memory or (shared_memory and local_rank == 0):
        if os.path.exists(node_feat_path) and load_node:
            node_feats = np.load(
                node_feat_path, mmap_mode=mmap_mode, allow_pickle=False)
            # if not memmap:
            #     node_feats = torch.from_numpy(node_feats)

        if os.path.exists(edge_feat_path) and load_edge:
            edge_feats = np.load(
                edge_feat_path, mmap_mode=mmap_mode, allow_pickle=False)
            # if not memmap:
            #     edge_feats = torch.from_numpy(edge_feats)

    if shared_memory:
        node_feats_shm, edge_feats_shm = None, None
        if local_rank == 0:
            if node_feats is not None:
                # node_feats = node_feats.astype(np.float32)
                node_feats_shm = create_shared_mem_array(
                    'node_feats', node_feats.shape, node_feats.dtype)
                # node_feats_shm[:] = node_feats[:]
                np.copyto(node_feats_shm, node_feats)
            if edge_feats is not None:
                # edge_feats = edge_feats.astype(np.float32)
                edge_feats_shm = create_shared_mem_array(
                    'edge_feats', edge_feats.shape, edge_feats.dtype)
                print("edge_feats_shm.shape: ",
                      edge_feats_shm.shape, flush=True)
                # edge_feats_shm[:] = edge_feats[:]
                np.copyto(edge_feats_shm, edge_feats)
                print("edge_feats assigned", flush=True)
            # broadcast the shape and dtype of the features
            node_feats_shape = node_feats.shape if node_feats is not None else None
            edge_feats_shape = edge_feats.shape if edge_feats is not None else None
            node_feats_dtype = node_feats.dtype if node_feats is not None else None
            edge_feats_dtype = edge_feats.dtype if edge_feats is not None else None
            torch.distributed.broadcast_object_list(
                [node_feats_shape, edge_feats_shape], src=0)
            torch.distributed.broadcast_object_list(
                [node_feats_dtype, edge_feats_dtype], src=0)
            del node_feats, edge_feats

        if local_rank != 0:
            shapes = [None, None]
            dtypes = [None, None]
            torch.distributed.broadcast_object_list(
                shapes, src=0)
            torch.distributed.broadcast_object_list(
                dtypes, src=0)
            node_feats_shape, edge_feats_shape = shapes
            node_feats_dtype, edge_feats_dtype = dtypes
            if node_feats_shape is not None:
                node_feats_shm = get_shared_mem_array(
                    'node_feats', node_feats_shape, node_feats_dtype)
            if edge_feats_shape is not None:
                edge_feats_shm = get_shared_mem_array(
                    'edge_feats', edge_feats_shape, edge_feats_dtype)

        torch.distributed.barrier()
        if node_feats_shm is not None:
            logging.info("rank {} node_feats_shm shape {}".format(
                local_rank, node_feats_shm.shape))
            node_feats_shm = torch.from_numpy(node_feats_shm)

        if edge_feats_shm is not None:
            logging.info("rank {} edge_feats_shm shape {}".format(
                local_rank, edge_feats_shm.shape))
            edge_feats_shm = torch.from_numpy(edge_feats_shm)

        return node_feats_shm, edge_feats_shm

    if not memmap:
        if node_feats is not None:
            node_feats = torch.from_numpy(node_feats)
        if edge_feats is not None:
            edge_feats = torch.from_numpy(edge_feats)
    return node_feats, edge_feats


class DstRandEdgeSampler:
    """
    Samples random edges from the graph.
    """

    def __init__(self, dst_list, seed=42):
        self.seed = seed
        self.dst_list = np.unique(dst_list)

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        if self.seed is None:
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)

    def add_dst_list(self, dst):
        self.dst_list = np.unique(np.concatenate((self.dst_list, dst)))

    def save_state(self):
        """
        Save the current state of the random state generator.
        """
        return self.random_state.get_state()

    def restore_state(self, state):
        """
        Restore the random state generator from a saved state.
        """
        self.random_state.set_state(state)


def get_batch(df: pd.DataFrame, batch_size: int, num_chunks: int,
              rand_edge_sampler: DstRandEdgeSampler, world_size: int = 1):
    if num_chunks == 0:
        random_size = 0
    else:
        randint = torch.randint(
            0, num_chunks, size=(1,), device="cuda:{}".format(local_rank()))
        if world_size > 1:
            torch.distributed.broadcast(randint, src=0)
        random_size = int(randint) * batch_size // num_chunks

    indices = np.array(df.index // batch_size)[random_size:]
    df = df.iloc[random_size:]
    for _, rows in df.groupby(indices):
        neg_batch = rand_edge_sampler.sample(len(rows.src.values))
        target_nodes = np.concatenate(
            [rows.src.values, rows.dst.values, neg_batch]).astype(
            np.int64)
        ts = np.concatenate(
            [rows.time.values, rows.time.values, rows.time.values]).astype(
            np.float32)

        eid = rows['eid'].values

        yield target_nodes, ts, eid


def get_batch_no_neg(df: pd.DataFrame, batch_size: int):
    indices = np.array(df.index // batch_size)
    for _, rows in df.groupby(indices):
        target_nodes = np.concatenate(
            [rows.src.values, rows.dst.values]).astype(
            np.int64)
        ts = np.concatenate(
            [rows.time.values, rows.time.values]).astype(
            np.float32)

        eid = rows['eid'].values

        yield target_nodes, ts, eid




def prepare_input(mfgs, node_feats: Optional[torch.Tensor], edge_feats: Optional[torch.Tensor]):
    if node_feats is not None:
        src_ids = torch.cat([b.srcdata['ID'] for b in mfgs[0]])
        unique_src_ids, inverse_indices = torch.unique(src_ids, return_inverse=True)

        unique_features = node_feats[unique_src_ids]

        all_features = unique_features[inverse_indices]

        start_idx = 0
        for b in mfgs[0]:
            num_nodes = b.srcdata['ID'].size(0)
            b.srcdata['h'] = all_features[start_idx:start_idx+num_nodes]
            start_idx += num_nodes

    if edge_feats is not None:
        for mfg in mfgs:
            # 对当前消息流图中所有块的边ID进行合并和去重
            edge_ids = torch.cat([b.edata['ID'] for b in mfg])
            unique_edge_ids, inverse_indices = torch.unique(edge_ids, return_inverse=True)
            
            # 只查询一次边特征
            id_device, feat_device = unique_edge_ids.device, edge_feats.device
            if id_device != feat_device:
                unique_edge_ids = unique_edge_ids.to(feat_device)
            unique_edge_features = edge_feats[unique_edge_ids].float()
            if id_device != feat_device:
                unique_edge_features = unique_edge_features.to(id_device)
            
            # 使用inverse_indices重建完整的特征张量
            all_edge_features = unique_edge_features[inverse_indices]
            
            # 将特征分配回每个块
            start_idx = 0
            for b in mfg:
                num_edges = b.edata['ID'].size(0)
                b.edata['f'] = all_edge_features[start_idx:start_idx + num_edges]
                start_idx += num_edges
    return mfgs


def mfgs_to_cuda(mfgs: List[List[DGLBlock]], device: Union[str, torch.device]):
    for mfg in mfgs:
        for i in range(len(mfg)):
            mfg[i] = mfg[i].to(device)
    return mfgs



def get_pinned_buffers(
        fanouts, sample_history, batch_size, dim_node, dim_edge, node_dtype, edge_dtype):
    pinned_nfeat_buffs = list()
    pinned_efeat_buffs = list()
    limit = int(batch_size * 3.3)
    if dim_node and node_dtype:
        for i in fanouts:
            limit *= i
            if dim_edge != 0:
                for _ in range(sample_history):
                    pinned_efeat_buffs.insert(0, torch.zeros(
                        (limit, dim_edge), pin_memory=True, dtype=edge_dtype))

    if dim_edge and edge_dtype:
        if dim_node != 0:
            for _ in range(sample_history):
                pinned_nfeat_buffs.insert(0, torch.zeros(
                    (limit, dim_node), pin_memory=True, dtype=node_dtype))

    return pinned_nfeat_buffs, pinned_efeat_buffs


class RandEdgeSampler:
    """
    Samples random edges from the graph.
    """

    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:

            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


class EarlyStopMonitor:
    """
    Monitor the early stopping criteria.
    """

    def __init__(self, max_round=5, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round


# credit to: https://zhuanlan.zhihu.com/p/163839117
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0) # 合并在一起
    L2_distance = torch.cdist(total, total) ** 2 # 计算L2范数

    # 计算多核中每个核的bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val) # 将多个核合并在一起

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n] 
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1,-1)  # K_ss矩阵，Source<->Source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1) # K_st矩阵，Source<->Target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1) # K_ts矩阵,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)  # K_tt矩阵,Target<->Target
    	
    loss = (XX + XY).sum() + (YX + YY).sum()
    return loss