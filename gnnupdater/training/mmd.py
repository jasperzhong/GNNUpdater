import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from alibi_detect.utils.pytorch.distance import mmd2_from_kernel_matrix
from alibi_detect.utils.pytorch.kernels import GaussianRBF


@torch.no_grad()
def kernel_matrix(x: torch.Tensor, y: torch.Tensor, infer_sigma: bool) -> torch.Tensor:
    """ Compute and return full kernel matrix between arrays x and y. """
    device = x.device
    kernel = GaussianRBF().to(device)
    k_xy = kernel(x, y, infer_sigma)
    k_xx = kernel(x, x, infer_sigma)
    k_yy = kernel(y, y, infer_sigma)
    kernel_mat = torch.cat(
        [torch.cat([k_xx, k_xy], 1), torch.cat([k_xy.T, k_yy], 1)], 0)
    return kernel_mat


@torch.no_grad()
def compute_mmd(x_ref: torch.Tensor, x: torch.Tensor, infer_sigma: bool = True) -> torch.Tensor:
    assert x_ref.device == x.device

    n = x.shape[0]
    kernel_mat = kernel_matrix(x_ref, x, infer_sigma)
    kernel_mat = kernel_mat - torch.diag(torch.diag(kernel_mat))
    mmd2 = mmd2_from_kernel_matrix(
        kernel_mat, n, permute=False, zero_diag=False)
    return mmd2


def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    return torch.norm(x1 - x2, p=2)


def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    ss1 = (sx1**k).mean(0)
    ss2 = (sx2**k).mean(0)
    return l2diff(ss1, ss2)


def compute_cmd(x1, x2, n_moments=5):
    mx1 = x1.mean(dim=0)
    mx2 = x2.mean(dim=0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1, mx2)
    scms = dm

    for i in range(n_moments-1):
        # moment diff of centralized samples
        scms += moment_diff(sx1, sx2, i+2)
    return scms


def calculate_embeddings_distribution_shift(old_nodes: np.ndarray, old_node_embeds: torch.Tensor,
                                            new_nodes: np.ndarray, new_node_embeds: torch.Tensor,
                                            methods=['mmd2']) -> Dict[str, float]:
    torch.cuda.empty_cache()
    common_nodes = np.intersect1d(old_nodes, new_nodes)

    # 使用字典做映射,避免重复查找
    old_node_to_idx = {node: idx for idx, node in enumerate(old_nodes)}
    new_node_to_idx = {node: idx for idx, node in enumerate(new_nodes)}

    old_indices = np.array([old_node_to_idx[node] for node in common_nodes])
    new_indices = np.array([new_node_to_idx[node] for node in common_nodes])

    old_node_embeds = old_node_embeds[old_indices]
    new_node_embeds = new_node_embeds[new_indices]

    results = {}
    with torch.no_grad():
        if 'mmd2' in methods:
            start = time.time()
            if len(common_nodes) < 2:
                print(f"WARNING: Number of common nodes is less than 2, "
                      f"setting MMD2 to nan.")
                results['mmd2'] = np.nan
            else:
                results['mmd2'] = compute_mmd(
                    old_node_embeds, new_node_embeds).item()
            end = time.time()
            results['mmd2_time'] = end - start
        if 'cmd' in methods:
            start = time.time()
            results['cmd'] = compute_cmd(
                old_node_embeds, new_node_embeds).item()
            end = time.time()
            results['cmd_time'] = end - start
        if 'mse' in methods:
            start = time.time()
            results['mse'] = F.mse_loss(
                old_node_embeds, new_node_embeds).item()
            end = time.time()
            results['mse_time'] = end - start
        if 'mae' in methods:
            start = time.time()
            results['mae'] = F.l1_loss(old_node_embeds, new_node_embeds).item()
            end = time.time()
            results['mae_time'] = end - start

    return results
