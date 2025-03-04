from typing import Optional, Union, List

import torch
import numpy as np

from gnnupdater.cache.cache import Cache
from gnnupdater.distributed.kvstore import KVStoreClient
from gnnupdater.utils import EarlyStopMonitor



class EMACache(Cache):
    """
    Use a predictive model to predict the future access frequency of nodes and edges
    """

    def __init__(self, edge_cache_ratio: int, node_cache_ratio: int,
                 num_nodes: int, num_edges: int,
                 device: Union[str, torch.device],
                 node_feats: Optional[torch.Tensor] = None,
                 edge_feats: Optional[torch.Tensor] = None,
                 dim_node_feat: Optional[int] = 0,
                 dim_edge_feat: Optional[int] = 0,
                 pinned_nfeat_buffs: Optional[torch.Tensor] = None,
                 pinned_efeat_buffs: Optional[torch.Tensor] = None,
                 kvstore_client: Optional[KVStoreClient] = None,
                 distributed: Optional[bool] = False,
                 neg_sample_ratio: Optional[int] = 1, 
                 **kwargs):
        """
        Initialize the cache

        Args:
            edge_cache_ratio: The edge ratio of the cache size to the total number of nodes or edges
                    range: [0, 1].
            node_cache_ratio: The node ratio of the cache size to the total number of nodes or edges
                    range: [0, 1].
            num_nodes: The number of nodes in the graph
            num_edges: The number of edges in the graph
            device: The device to use
            node_feats: The node features
            edge_feats: The edge features
            dim_node_feat: The dimension of node features
            dim_edge_feat: The dimension of edge features
            pinned_nfeat_buffs: The pinned memory buffers for node features
            pinned_efeat_buffs: The pinned memory buffers for edge features
            kvstore_client: The KVStore_Client for fetching features when using distributed
                    training
            distributed: Whether to use distributed training
            neg_sample_ratio: The ratio of negative samples to positive samples
        """
        super(EMACache, self).__init__(edge_cache_ratio, node_cache_ratio, num_nodes,
                                       num_edges, device, node_feats,
                                       edge_feats, dim_node_feat,
                                       dim_edge_feat, pinned_nfeat_buffs,
                                       pinned_efeat_buffs, kvstore_client,
                                       distributed, neg_sample_ratio)
        self.name = 'ema_cache'

        self.node_popularity_predictor_update_interval = kwargs.get(
            'node_popularity_predictor_update_interval', 1)
        self.edge_popularity_predictor_update_interval = kwargs.get(
            'edge_popularity_predictor_update_interval', 1)

        self.node_popularity_window = kwargs.get(
            'node_popularity_window', 9)
        self.edge_popularity_window = kwargs.get(
            'edge_popularity_window', 9)

        self.node_ema_alpha = 2/(self.node_popularity_window + 1)
        self.edge_ema_alpha = 2/(self.edge_popularity_window + 1)

        
        self.node_batch = 0 
        self.edge_batch = 0 


        if self.dim_node_feat != 0:
            self.node_popularity = torch.zeros(
                self.num_nodes, dtype=torch.float32, device=self.device)
            self.last_ema_node_popularity = torch.zeros(
                self.num_nodes, dtype=torch.float32, device=self.device)
            self.node_popularity_in_cache = torch.zeros(
                self.node_capacity, dtype=torch.float32, device=self.device)

            
        if self.dim_edge_feat != 0:
            self.edge_popularity = torch.zeros(
                self.num_edges, dtype=torch.float32, device=self.device)
            self.last_ema_edge_popularity = torch.zeros(
                self.num_edges, dtype=torch.float32, device=self.device)
            self.edge_popularity_in_cache = torch.zeros(
                self.edge_capacity, dtype=torch.float32, device=self.device)

    def get_mem_size(self) -> int:
        """
        Get the memory size of the cache in bytes
        """
        mem_size = super(EMACache, self).get_mem_size()
        if self.dim_node_feat != 0:
            mem_size += self.node_popularity.element_size() * self.node_popularity.nelement()
        if self.dim_edge_feat != 0:
            mem_size += self.edge_popularity.element_size() * self.edge_popularity.nelement()
        return mem_size


    def resize(self, new_num_nodes: int, new_num_edges: int):
        """
        Resize the cache

        Args:
            new_num_nodes: The new number of nodes
            new_num_edges: The new number of edges
        """
        super(EMACache, self).resize(new_num_nodes, new_num_edges)
        if self.dim_node_feat != 0:
            self.node_popularity.resize_(self.node_capacity)
        if self.dim_edge_feat != 0:
            self.edge_popularity.resize_(self.edge_capacity)

    def update_node_cache(self, cached_node_index: torch.Tensor,
                          uncached_node_id: torch.Tensor,
                          uncached_node_feature: torch.Tensor,
                          **kwargs):
        """
        Update the node cache

        Args:
            cached_node_index: The index of the cached nodes
            uncached_node_id: The id of the uncached nodes
            uncached_node_feature: The features of the uncached nodes
        """
        all_nodes = kwargs.get('all_nodes', None)
        cached_nodes = kwargs.get('cached_nodes', None)

        if all_nodes is not None:
            # update current popularity
            self.node_popularity[all_nodes] += 1  

        # update cache hit's predict popularity value
        if cached_nodes is not None:
            last_ema_node_popularity = self.last_ema_node_popularity[cached_nodes] 
            current_popularity = self.node_popularity[cached_nodes] 
            yhat = self.node_ema_alpha * current_popularity + (1 - self.node_ema_alpha) * last_ema_node_popularity
            self.node_popularity_in_cache[cached_node_index] = yhat

        # EMA_t = \alpha * f_t + (1 - \alpha) * EMA_{t-1}
        # cache EMA_t >= min EMA in cache
        if len(uncached_node_id) > 0:
            # should incorporate the current popularity
            last_ema_node_popularity = self.last_ema_node_popularity[uncached_node_id] 
            current_popularity = self.node_popularity[uncached_node_id] 
            yhat = self.node_ema_alpha * current_popularity + (1 - self.node_ema_alpha) * last_ema_node_popularity

            mask = yhat >= self.node_popularity_in_cache.min()
            # print(f"mean popularity: {self.node_popularity_in_cache.min()}, yhat max: {yhat.max()}")
            num_nodes_to_cache = mask.sum()
            if num_nodes_to_cache > self.node_capacity:
                num_nodes_to_cache = self.node_capacity
                # largest N 
                mask = torch.topk(yhat, k=self.node_capacity, largest=True).indices
            print(f"filter nodes: {len(uncached_node_id)} -> {num_nodes_to_cache}")

            uncached_node_id = uncached_node_id[mask]
            uncached_node_feature = uncached_node_feature[mask]
            yhat = yhat[mask]
        else:
            num_nodes_to_cache = len(uncached_node_id)
        
        if num_nodes_to_cache > 0:
            if len(uncached_node_id) > self.node_capacity:
                num_nodes_to_cache = self.node_capacity
                node_id_to_cache = uncached_node_id[:num_nodes_to_cache]
                node_feature_to_cache = uncached_node_feature[:num_nodes_to_cache]
            else:
                node_id_to_cache = uncached_node_id
                node_feature_to_cache = uncached_node_feature
        
            # get the k node id with the least water level
            removing_cache_index = torch.topk(
                self.node_popularity_in_cache, k=num_nodes_to_cache, largest=False).indices

            assert len(removing_cache_index) == len(
                node_id_to_cache) == len(node_feature_to_cache)
            removing_node_id = self.cache_index_to_node_id[removing_cache_index]

            # update cache attributes
            self.cache_node_buffer[removing_cache_index] = node_feature_to_cache
            self.cache_node_flag[removing_node_id] = False
            self.cache_node_flag[node_id_to_cache] = True
            self.cache_node_map[removing_node_id] = -1
            self.cache_node_map[node_id_to_cache] = removing_cache_index
            self.cache_index_to_node_id[removing_cache_index] = node_id_to_cache
            self.node_popularity_in_cache[removing_cache_index] = yhat

        self.node_batch += 1

        if self.node_batch % self.node_popularity_predictor_update_interval == 0:
            # EMA_t = \alpha * f_t + (1 - \alpha) * EMA_{t-1}
            self.last_ema_node_popularity = self.node_ema_alpha * self.node_popularity + (1 - self.node_ema_alpha) * self.last_ema_node_popularity
            self.node_popularity.zero_()
            # update the cache popularity
            cached_node_index = torch.arange(self.node_capacity, device=self.device)
            cached_nodes = self.cache_index_to_node_id[cached_node_index]
            self.node_popularity_in_cache[cached_node_index] = self.last_ema_node_popularity[cached_nodes]



    def update_edge_cache(self, cached_edge_index: torch.Tensor,
                          uncached_edge_id: torch.Tensor,
                          uncached_edge_feature: torch.Tensor,
                          **kwargs):
        """
        Update the edge cache

        Args:
            cached_edge_index: The index of the cached edges
            uncached_edge_id: The id of the uncached edges
            uncached_edge_feature: The features of the uncached edges
        """
        all_edges = kwargs.get('all_edges', None)
        cached_edges = kwargs.get('cached_edges', None)
        if all_edges is not None:
            # update current popularity
            self.edge_popularity[all_edges] += 1
        
        # update cache hit's predict popularity value 
        if cached_edges is not None:
            last_ema_edge_popularity = self.last_ema_edge_popularity[cached_edges]
            current_popularity = self.edge_popularity[cached_edges]
            yhat = self.edge_ema_alpha * current_popularity + (1 - self.edge_ema_alpha) * last_ema_edge_popularity
            self.edge_popularity_in_cache[cached_edge_index] = yhat
            
        # EMA_t = \alpha * f_t + (1 - \alpha) * EMA_{t-1}
        # cache EMA_t >= min EMA in cache 
        if len(uncached_edge_id) > 0:
            # should incorporate the current popularity
            last_ema_edge_popularity = self.last_ema_edge_popularity[uncached_edge_id]
            current_popularity = self.edge_popularity[uncached_edge_id]
            yhat = self.edge_ema_alpha * current_popularity + (1 - self.edge_ema_alpha) * last_ema_edge_popularity

            mask = yhat >= self.edge_popularity_in_cache.min()
            num_edges_to_cache = mask.sum()
            if num_edges_to_cache > self.edge_capacity:
                num_edges_to_cache = self.edge_capacity
                # largest N 
                mask = torch.topk(yhat, k=self.edge_capacity, largest=True).indices
            print(f"filter edges: {len(uncached_edge_id)} -> {num_edges_to_cache}")
            uncached_edge_id = uncached_edge_id[mask]
            uncached_edge_feature = uncached_edge_feature[mask]
            yhat = yhat[mask]
        else:
            num_edges_to_cache = len(uncached_edge_id)
        
        if num_edges_to_cache > 0:
            if len(uncached_edge_id) > self.edge_capacity:
                num_edges_to_cache = self.edge_capacity
                edge_id_to_cache = uncached_edge_id[:num_edges_to_cache]
                edge_feature_to_cache = uncached_edge_feature[:num_edges_to_cache]
            else:
                edge_id_to_cache = uncached_edge_id
                edge_feature_to_cache = uncached_edge_feature
        
            # get the k edge id with the least water level
            removing_cache_index = torch.topk(
                self.edge_popularity_in_cache, k=num_edges_to_cache, largest=False).indices
        
            assert len(removing_cache_index) == len(
                edge_id_to_cache) == len(edge_feature_to_cache)
            removing_edge_id = self.cache_index_to_edge_id[removing_cache_index]
        
            # update cache attributes
            self.cache_edge_buffer[removing_cache_index] = edge_feature_to_cache
            self.cache_edge_flag[removing_edge_id] = False
            self.cache_edge_flag[edge_id_to_cache] = True
            self.cache_edge_map[removing_edge_id] = -1
            self.cache_edge_map[edge_id_to_cache] = removing_cache_index
            self.cache_index_to_edge_id[removing_cache_index] = edge_id_to_cache
            self.edge_popularity_in_cache[removing_cache_index] = yhat

        self.edge_batch += 1

        if self.edge_batch % self.edge_popularity_predictor_update_interval == 0:
            # EMA_t = \alpha * f_t + (1 - \alpha) * EMA_{t-1}
            self.last_ema_edge_popularity = self.edge_ema_alpha * self.edge_popularity + (1 - self.edge_ema_alpha) * self.last_ema_edge_popularity
            self.edge_popularity.zero_()

            # update the cache popularity
            cached_edge_index = torch.arange(self.edge_capacity, device=self.device)
            cached_edges = self.cache_index_to_edge_id[cached_edge_index]
            self.edge_popularity_in_cache[cached_edge_index] = self.last_ema_edge_popularity[cached_edges]
       
