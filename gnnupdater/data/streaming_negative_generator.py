import os
import pickle
from collections import defaultdict
from typing import Dict, Optional

import numba as nb
import numpy as np
from numba import jit
from tqdm import tqdm

from gnnupdater.dynamic_graph import build_dynamic_graph


@nb.njit('int64[:](int64[:], int64[:])')
def setdiff1d_nb(arr1, arr2):
    delta = set(arr2)

    # : build the result
    result = np.empty(len(arr1), dtype=arr1.dtype)
    j = 0
    for i in range(arr1.shape[0]):
        if arr1[i] not in delta:
            result[j] = arr1[i]
            j += 1
    return result[:j]


@jit(nopython=True)
def _filter_valid_samples(candidates, invalid_markers, num_needed):
    """使用numba加速的过滤函数"""
    valid_samples = []
    for c in candidates:
        if not invalid_markers[c]:
            valid_samples.append(c)
            if len(valid_samples) >= num_needed:
                break
    return valid_samples


def save_pkl(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


class StreamingNegativeEdgeGenerator:
    def __init__(
        self,
        dataset_name: str,
        num_neg_e: int = 100,
        hist_ratio: float = 0.5,
    ):
        self.dataset_name = dataset_name
        self.num_neg_e = num_neg_e
        self.hist_ratio = hist_ratio

        # 动态维护的历史信息
        self.hist_edge_set_per_node = defaultdict(
            set)  # src -> list of historical dsts

        self.min_dst_idx = float('inf')
        self.max_dst_idx = float('-inf')

        self.full_evaluation_set = {}

    def add_batch(self, src, dst, t) -> None:
        """更新历史信息"""
        # 添加新的边
        for s, d, time in zip(src, dst, t):
            s, d = int(s), int(d)
            self.hist_edge_set_per_node[s].add(d)

    def generate_negative_samples(self, pos_src, pos_dst, pos_t):
        """为batch生成negative samples"""
        self.min_dst_idx = min(pos_dst.min(), self.min_dst_idx)
        self.max_dst_idx = max(pos_dst.max(), self.max_dst_idx)
        current_edges = defaultdict(list)  # {src: [current_dsts]}

        # 构建当前边的查找表
        for s, d in zip(pos_src, pos_dst):
            current_edges[s].append(d)

        historical_dsts_lookup = defaultdict(set)
        pos_e_dst_same_src_lookup = defaultdict(set)
        unique_pos_src = np.unique(pos_src)
        for s in unique_pos_src:
            historical_dsts_lookup[s] = np.array(
                list(self.hist_edge_set_per_node[s]), dtype=np.int64)
            pos_e_dst_same_src_lookup[s] = np.array(
                current_edges[s], dtype=np.int64)

        total_nodes = self.max_dst_idx - self.min_dst_idx + 1
        invalid_markers = np.zeros(total_nodes, dtype=np.bool_)

        self._cache_filtered_dsts = {}
        self._cache_valid_dsts = {}

        # all_dsts = np.arange(self.min_dst_idx, self.max_dst_idx + 1)
        for pos_s, pos_d, pos_time in zip(pos_src, pos_dst, pos_t):
            historical_dsts = historical_dsts_lookup[pos_s]
            # 当前源节点的所有目标节点
            pos_e_dst_same_src = pos_e_dst_same_src_lookup[pos_s]

            # 采样historical negatives
            num_hist_neg_e = int(self.num_neg_e * self.hist_ratio)
            neg_hist_dsts = self._sample_historical_negatives(
                pos_s,
                pos_e_dst_same_src,
                historical_dsts,
                num_hist_neg_e
            )

            # 采样random negatives
            num_rnd_neg_e = self.num_neg_e - len(neg_hist_dsts)
            neg_rnd_dsts = self._sample_random_negatives(
                pos_s,
                pos_e_dst_same_src,
                historical_dsts,
                num_rnd_neg_e,
                invalid_markers
            )

            # 合并两种negatives
            neg_dst_arr = np.concatenate((neg_hist_dsts, neg_rnd_dsts))
            if len(neg_dst_arr) < self.num_neg_e:
                continue
            elif len(neg_dst_arr) > self.num_neg_e:
                print("WARNING: More negative samples than expected!")

            # reduce memory usage
            pos_s = pos_s.astype(np.int32)
            pos_d = pos_d.astype(np.int32)
            pos_time = pos_time.astype(np.float32)
            neg_dst_arr = neg_dst_arr.astype(np.int32)
            self.full_evaluation_set[(pos_s, pos_d, pos_time)] = neg_dst_arr

    def process_batches(self, data_loader, split_mode: str):
        """处理多个batch"""
        # 重置
        self.full_evaluation_set = {}
        print(
            f"INFO: Generating negative samples for {self.dataset_name} {split_mode} evaluation!")

        # 使用tqdm显示处理进度
        i = 0
        for (src, dst, t, _) in tqdm(data_loader, desc=f"Processing {self.dataset_name} {split_mode} batches"):
            if i > 0:
                # 生成当前batch的negative samples
                self.generate_negative_samples(src, dst, t)

            # 更新历史信息
            self.add_batch(src, dst, t)
            i += 1

    def _sample_historical_negatives(self, src, current_dsts: np.ndarray, historical_dsts: np.ndarray, num_samples):
        """从历史边中采样"""
        if src in self._cache_filtered_dsts:
            filtered_dsts = self._cache_filtered_dsts[src]
        else:
            filtered_dsts = setdiff1d_nb(historical_dsts, current_dsts)

        if len(filtered_dsts) == 0:
            return np.array([], dtype=np.int64)

        num_samples = min(num_samples, len(filtered_dsts))
        return np.random.choice(filtered_dsts, num_samples, replace=False)

    def _sample_random_negatives(self, src, current_dsts: np.ndarray, historical_dsts: np.ndarray, num_samples,
                                 invalid_markers: np.ndarray):
        """随机采样 - 进一步优化版本"""
        # 计算总节点数
        total_nodes = self.max_dst_idx - self.min_dst_idx + 1

        # 创建布尔数组标记无效节点
        invalid_markers.fill(False)
        invalid_markers[current_dsts - self.min_dst_idx] = True
        invalid_markers[historical_dsts - self.min_dst_idx] = True

        # 一次生成足够多的随机数
        batch_size = min(num_samples * 2, total_nodes // 2)
        sampled = set()
        max_attempts = 3  # 最多尝试3轮

        for _ in range(max_attempts):
            candidates = np.random.randint(
                self.min_dst_idx, self.max_dst_idx + 1, size=batch_size)
            valid_samples = _filter_valid_samples(
                candidates, invalid_markers, num_samples - len(sampled))
            sampled.update(valid_samples)

            if len(sampled) >= num_samples:
                break

        # 如果还不够，fallback到原始方法
        if len(sampled) < num_samples:
            return self._sample_random_negatives_original(src, current_dsts, historical_dsts, num_samples)

        return np.array(list(sampled), dtype=np.int64)

    def _sample_random_negatives_original(self, src, current_dsts: np.ndarray, historical_dsts: np.ndarray, num_samples):
        """Original implementation as fallback"""
        if src in self._cache_valid_dsts:
            valid_dsts = self._cache_valid_dsts[src]
        else:
            all_dsts = np.arange(self.min_dst_idx, self.max_dst_idx + 1)
            invalid_dsts = np.union1d(current_dsts, historical_dsts)
            valid_dsts = setdiff1d_nb(all_dsts, invalid_dsts)

        num_samples = min(num_samples, len(valid_dsts))
        return np.random.choice(valid_dsts, num_samples, replace=False)

    def save_negatives(self, split_mode: str, path: str = "."):
        # use pickle to save the negative samples
        filename = os.path.join(
            path, f"{self.dataset_name}_{split_mode}_ns.pkl")

        # if exists delete
        if os.path.exists(filename):
            os.remove(filename)
            print(f"INFO: Old negative samples removed: {filename}")

        save_pkl(self.full_evaluation_set, filename)
        print(f"INFO: Negative samples saved to {filename}")


if __name__ == '__main__':
    import os
    import random

    import numpy as np

    from gnnupdater.data.load_data import load_data
    from gnnupdater.data.streaming_negative_generator import \
        StreamingNegativeEdgeGenerator
    from gnnupdater.data.time_series_data_loader import TimeSeriesDataLoader
    from gnnupdater.utils import PROJ_ROOT

    dataset = 'tgbl_coin'
    num_neg_e = 100
    hist_ratio = 0.5

    offline_df, streaming_df, config, _, _ = load_data(
        dataset, load_feat=False)

    neg_gen = StreamingNegativeEdgeGenerator(
        dataset_name=dataset,
        num_neg_e=num_neg_e,
        hist_ratio=hist_ratio
    )

    # 遍历所有batch
    data_loader1 = TimeSeriesDataLoader(
        df=offline_df,
        config=config,
        phase='streaming',
        phase="phase1"
    )

    path = os.path.join(PROJ_ROOT, "datasets", dataset)

    neg_gen.process_batches(data_loader1, split_mode="phase1")
    neg_gen.save_negatives(split_mode="phase1", path=path)

    offset = len(offline_df)
    del data_loader1, offline_df
    data_loader2 = TimeSeriesDataLoader(
        df=streaming_df,
        config=config,
        phase='streaming',
        phase="phase2",
        phase1_offset=offset
    )

    neg_gen.process_batches(data_loader2, split_mode="phase2")
    neg_gen.save_negatives(split_mode="phase2", path=path)

    print(f"INFO: {dataset} negative samples saved!")
