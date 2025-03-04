import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.sparse
from scipy.sparse import dok_array

from gnnupdater.utils import PROJ_ROOT


def dok_array_to_torch_sparse(dok_list: List[dok_array]) -> torch.sparse.Tensor:
    """
    Convert list of dok arrays to a torch sparse tensor
    """
    n_row, n_col = len(dok_list), dok_list[0].shape[0]
    rows, cols = [], []
    vals = []
    for row, dok in enumerate(dok_list):
        rows.extend([row] * len(dok.keys()))
        cols.extend(list(dok.keys()))
        vals.extend(list(dok.values()))

    rows = torch.tensor(rows, dtype=torch.long)
    cols = torch.tensor(cols, dtype=torch.long)
    vals = torch.tensor(vals, dtype=torch.float32)
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n_row, n_col))


class TimeSeriesDataLoader:
    def __init__(self,
                 df: pd.DataFrame,
                 node_label_dict: Dict,
                 split_points: np.ndarray,
                 start_ts_quantile: Optional[float] = 0.0,
                 end_ts_quantile: Optional[float] = 1.0,
                 start_idx: Optional[int] = None,
                 end_idx: Optional[int] = None,
                 ):
        """
        Args:
            df: dataframe
            node_label_dict: dictionary of node labels
            split_points: split points for each batch
            start_ts_quantile: start quantile for time series (0.0 to 1.0)
            end_ts_quantile: end quantile for time series (0.0 to 1.0)
            start_idx: start index for time series
            end_idx: end index for time series

        Note: If start_idx and end_idx are provided, start_ts_quantile and 
            end_ts_quantile are ignored.
        """
        self.df = df
        self.node_label_dict = node_label_dict
        self.label_ts = sorted(list(node_label_dict.keys()))

        if not start_idx and not end_idx:
            # Validate and process quantiles
            if start_ts_quantile is None:
                start_ts_quantile = 0.0
            if end_ts_quantile is None:
                end_ts_quantile = 1.0

            if not (0.0 <= start_ts_quantile <= 1.0 and 0.0 <= end_ts_quantile <= 1.0):
                raise ValueError("Quantiles must be between 0.0 and 1.0")
            if start_ts_quantile >= end_ts_quantile:
                raise ValueError(
                    "start_ts_quantile must be less than end_ts_quantile")

            # Calculate start and end indices
            start_idx = int(len(self.label_ts) * start_ts_quantile)
            end_idx = int(len(self.label_ts) * end_ts_quantile)

        if start_idx >= len(self.label_ts):
            raise ValueError(
                f"start_ts_quantile too high: {start_ts_quantile}")

        self.start_idx = start_idx
        self.end_idx = end_idx
        # Subset the label timestamps and corresponding split points
        self.label_ts = self.label_ts[start_idx:end_idx]
        self.split_points = split_points[start_idx:end_idx]
        self.label_ts_idx = 0
        self.initial_start_idx = 0 if start_idx == 0 else split_points[start_idx-1]

    def get_label_ts(self):
        return self.label_ts[self.label_ts_idx]

    def get_cur_idx(self):
        return self.label_ts_idx + self.start_idx

    def __iter__(self):
        self.label_ts_idx = 0

        def _process_batch(batch):
            src, dst, ts, eid = batch['src'].values, batch['dst'].values, batch['ts'].values, batch.index.values
            src, dst, ts, eid = src.astype(np.int64), dst.astype(
                np.int64), ts.astype(np.float32), eid.astype(np.int64)
            return src, dst, ts, eid

        def _process_label(ts):
            node_ids = np.array(list(self.node_label_dict[ts].keys()))
            node_labels = []
            for node_id in node_ids:
                node_labels.append(self.node_label_dict[ts][node_id])
            if isinstance(node_labels[0], np.ndarray):
                node_labels_torch = torch.from_numpy(np.stack(node_labels))
            else:
                node_labels_torch = dok_array_to_torch_sparse(node_labels)
            label_ts = np.full(len(node_labels), ts, dtype=np.float32)
            return node_ids, label_ts, node_labels_torch

        start_idx = self.initial_start_idx
        for end_idx in self.split_points:
            batch = self.df.loc[start_idx:end_idx-1]
            if self.label_ts_idx >= len(self.label_ts):
                # No more labels
                return
            ts = self.label_ts[self.label_ts_idx]
            self.label_ts_idx += 1
            yield _process_batch(batch), _process_label(ts)
            start_idx = end_idx

        if start_idx < len(self.df):
            batch = self.df.loc[start_idx:]
            if self.label_ts_idx >= len(self.label_ts):
                return
            ts = self.label_ts[self.label_ts_idx]
            self.label_ts_idx += 1
            yield _process_batch(batch), _process_label(ts)

    def __len__(self):
        if not len(self.split_points):
            return 0
        return len(self.split_points) + (1 if len(self.df) > self.split_points[-1] else 0)


if __name__ == "__main__":
    import time

    from tqdm import tqdm

    from gnnupdater.data.load_data import load_data
    dataset = 'tgbn_genre'

    df, config, split_points, node_label_dict, node_feat, edge_feat = load_data(
        dataset, load_feat=False)

    # 1. test offline and streaming dataloader
    # dataloader = TimeSeriesDataLoader(
    #     df, node_label_dict, split_points)
    # num_batches = 0
    # last_label_ts = 0
    # for i, (batch, labels) in tqdm(enumerate(dataloader)):
    #     batch_ts = batch[2].max()
    #     label_ts = labels[1].max()
    #     assert batch_ts <= label_ts and batch_ts > last_label_ts, \
    #         f'Batch ts: {batch_ts}, Label ts: {label_ts}, Last label ts: {last_label_ts}'
    #     last_label_ts = label_ts
    #     num_batches += 1

    # print(f'Number of batches: {num_batches}')

    # 2. test different quantile

    # dataloader = TimeSeriesDataLoader(
    #     df, node_label_dict, split_points,
    #     start_ts_quantile=0, end_ts_quantile=0.3)
    # num_batches1 = 0
    # last_label_ts = 0
    # for i, (batch, labels) in tqdm(enumerate(dataloader)):
    #     batch_ts = batch[2].max()
    #     label_ts = labels[1].max()
    #     assert batch_ts <= label_ts and batch_ts > last_label_ts, \
    #         f'Batch ts: {batch_ts}, Label ts: {label_ts}, Last label ts: {last_label_ts}'
    #     last_label_ts = label_ts
    #     num_batches1 += 1

    # print(f'Number of batches: {num_batches1}')

    # dataloader = TimeSeriesDataLoader(
    #     df, node_label_dict, split_points,
    #     start_ts_quantile=0.3, end_ts_quantile=1)
    # num_batches2 = 0
    # last_label_ts = 0
    # for i, (batch, labels) in tqdm(enumerate(dataloader)):
    #     batch_ts = batch[2].max()
    #     label_ts = labels[1].max()
    #     assert batch_ts <= label_ts and batch_ts > last_label_ts, \
    #         f'Batch ts: {batch_ts}, Label ts: {label_ts}, Last label ts: {last_label_ts}'
    #     last_label_ts = label_ts
    #     num_batches2 += 1

    # print(f'Number of batches: {num_batches2}')

    # print(f'Total Number of batches: {num_batches1 + num_batches2}')

    # 3. test 0 to 0.1 quantile
    dataloader = TimeSeriesDataLoader(
        df, node_label_dict, split_points,
        start_ts_quantile=0, end_ts_quantile=0.1)
    num_batches1 = 0
    last_label_ts = 0
    for i, (batch, labels) in tqdm(enumerate(dataloader)):
        batch_ts = batch[2].max()
        label_ts = labels[1].max()
        assert batch_ts <= label_ts and batch_ts > last_label_ts, \
            f'Batch ts: {batch_ts}, Label ts: {label_ts}, Last label ts: {last_label_ts}'
        last_label_ts = label_ts
        num_batches1 += 1

    print(f'Number of batches: {num_batches1}')
