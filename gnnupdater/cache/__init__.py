from typing import Optional

import numpy as np

from gnnupdater.utils import get_pinned_buffers

from .fifo_cache import FIFOCache
from .gnnlab_static_cache import GNNLabStaticCache
from .lfu_cache import LFUCache
from .lru_cache import LRUCache

CACHE_DICT = {
    'fifo': FIFOCache,
    'gnnlab_static': GNNLabStaticCache,
    'lfu': LFUCache,
    'lru': LRUCache,
}

def get_cache(cache_name: str, node_feat: Optional[np.ndarray], edge_feat: Optional[np.ndarray],
              edge_cache_ratio, node_cache_ratio, num_nodes, num_edges, device, model_config, batch_size, 
              sampler=None, train_df=None):
    dim_node = node_feat.shape[1] if node_feat is not None else 0
    dim_edge = edge_feat.shape[1] if edge_feat is not None else 0
    node_dtype = node_feat.dtype if node_feat is not None else None
    edge_dtype = edge_feat.dtype if edge_feat is not None else None
    pinned_nfeat_buffs, pinned_efeat_buffs = get_pinned_buffers(
        model_config['fanouts'], model_config['num_snapshots'], batch_size,
        dim_node, dim_edge, node_dtype, edge_dtype)
    
    cache = CACHE_DICT[cache_name](
        edge_cache_ratio, node_cache_ratio, num_nodes, num_edges,
        device, node_feat, edge_feat, dim_node, dim_edge,
        pinned_nfeat_buffs, pinned_efeat_buffs
    )

    if cache_name == 'gnnlab_static':
        cache.init_cache(sampler=sampler, train_df=train_df)
    else:
        cache.init_cache()
    
    return cache
