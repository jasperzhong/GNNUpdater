import os
import pickle

import numpy as np

from gnnupdater.utils import PROJ_ROOT


class NegativeSampler:
    """负样本采样器"""

    def __init__(self, dataset, phase, num_neg_e):
        self.filename = os.path.join(
            PROJ_ROOT, 'datasets', dataset, f'{dataset}_{phase}_ns.pkl')
        self.file = None
        self.num_neg_e = num_neg_e
        self.dset = pickle.load(open(self.filename, 'rb'))

    def query_batch(self, pos_src, pos_dst, pos_ts) -> np.ndarray:
        pos_src = pos_src.astype(np.int32)
        pos_dst = pos_dst.astype(np.int32)
        pos_ts = pos_ts.astype(np.float32)

        mask = np.zeros(len(pos_src), dtype=np.bool_)

        result = np.zeros((len(pos_src), self.num_neg_e), dtype=np.int32)
        for i, (src, dst, ts) in enumerate(zip(pos_src, pos_dst, pos_ts)):
            key = (src, dst, ts)
            if key not in self.dset:
                mask[i] = True
                continue
            neg_edges = self.dset[key][:]
            result[i, :] = neg_edges

        pos_src = pos_src[~mask]
        pos_dst = pos_dst[~mask]
        pos_ts = pos_ts[~mask]
        result = result[~mask]
        return pos_src, pos_dst, pos_ts, result
