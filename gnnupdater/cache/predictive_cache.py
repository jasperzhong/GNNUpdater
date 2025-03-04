from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from gnnupdater.cache.cache import Cache
from gnnupdater.distributed.kvstore import KVStoreClient
from gnnupdater.utils import EarlyStopMonitor


class Predictor(nn.Module):
    """2-layer MLP with leaky relu; 16 hidden dim"""
    def __init__(self, input_dim, hid_dim):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, 1)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze()

class StandardScaler:
    """for pytorch"""
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X):
        self.mean = X.mean(dim=0)
        self.std = X.std(dim=0)
    
    def transform(self, X):
        return (X - self.mean) / self.std


class PredictiveCache(Cache):
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
        super(PredictiveCache, self).__init__(edge_cache_ratio, node_cache_ratio, num_nodes,
                                       num_edges, device, node_feats,
                                       edge_feats, dim_node_feat,
                                       dim_edge_feat, pinned_nfeat_buffs,
                                       pinned_efeat_buffs, kvstore_client,
                                       distributed, neg_sample_ratio)
        self.name = 'predictive_cache'

        self.node_popularity_predictor = None
        self.node_popularity_optimizer = None
        self.edge_popularity_predictor = None
        self.edge_popularity_optimizer = None

        self.node_popularity_std_scaler = StandardScaler()
        self.edge_popularity_std_scaler = StandardScaler()

        self.node_popularity_predictor_update_interval = kwargs.get(
            'node_popularity_predictor_update_interval', 50)
        self.edge_popularity_predictor_update_interval = kwargs.get(
            'edge_popularity_predictor_update_interval', 100)

        self.node_popularity_list = []
        self.edge_popularity_list = []

        self.node_popularity_list_max_len = kwargs.get('node_popularity_list_max_len', 10)
        self.edge_popularity_list_max_len = kwargs.get('edge_popularity_list_max_len', 10)
        
        self.node_batch = 0 
        self.edge_batch = 0 

        self.node_val_list = []
        self.edge_val_list = []

        if self.dim_node_feat != 0:
            self.node_popularity = torch.zeros(
                self.num_nodes, dtype=torch.float32, device=self.device)
            self.predict_node_popularity = torch.zeros(
                self.node_capacity, dtype=torch.float32, device=self.device)
            
        if self.dim_edge_feat != 0:
            self.edge_popularity = torch.zeros(
                self.num_edges, dtype=torch.float32, device=self.device)
            self.predict_edge_popularity = torch.zeros(
                self.edge_capacity, dtype=torch.float32, device=self.device)

    def get_mem_size(self) -> int:
        """
        Get the memory size of the cache in bytes
        """
        mem_size = super(PredictiveCache, self).get_mem_size()
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
        super(PredictiveCache, self).resize(new_num_nodes, new_num_edges)
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
        if cached_nodes is not None and self.node_popularity_predictor:
            yhat = self.predict(self.node_popularity_predictor, self.node_popularity_std_scaler, 
                            cached_nodes, self.node_popularity_list, self.node_popularity)
            self.predict_node_popularity[cached_node_index] = yhat
        
        # y_hat <= predictor([p_{K-1}, ..., p])
        # cache y_hat >= predict_node_popularity
        if len(uncached_node_id) > 0 and self.node_popularity_predictor is not None:
            yhat = self.predict(self.node_popularity_predictor, self.node_popularity_std_scaler, 
                            uncached_node_id, self.node_popularity_list, self.node_popularity)

            kth_popularity = torch.min(self.predict_node_popularity)
            # filter yhat >= kth_popularity
            mask = yhat >= kth_popularity
            num_nodes_to_cache = mask.sum()
            if num_nodes_to_cache > self.node_capacity:
                num_nodes_to_cache = self.node_capacity
                # largest N 
                mask = torch.topk(yhat, k=self.node_capacity, largest=True).indices
            # print(f"filter nodes: {len(uncached_node_id)} -> {num_nodes_to_cache}")
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
                self.predict_node_popularity, k=num_nodes_to_cache, largest=False).indices

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
            if self.node_popularity_predictor is not None:
                self.predict_node_popularity[removing_cache_index] = yhat

        self.node_batch += 1

        if self.node_batch % self.node_popularity_predictor_update_interval == 0:
            if len(self.node_popularity_list) < self.node_popularity_list_max_len:
                self.node_popularity_list.append(self.node_popularity.clone().detach())
                self.node_popularity.zero_()
                return

            if self.node_popularity_predictor is None:
                self.node_popularity_predictor = Predictor(
                    self.node_popularity_list_max_len, 16).to(self.device)
                self.node_popularity_optimizer = optim.Adam(
                    self.node_popularity_predictor.parameters(), lr=1e-3)
            X = torch.stack(self.node_popularity_list, dim=0).T
            y = self.node_popularity

            self.node_popularity_std_scaler.fit(X)
            X = self.node_popularity_std_scaler.transform(X)

            self.train_predictor(self.node_popularity_predictor, self.node_popularity_optimizer, X, y, 'node')

            yhat = self.train_predictor(X)

            self.node_popularity_list.append(self.node_popularity.clone().detach())
            if len(self.node_popularity_list) > self.node_popularity_list_max_len:
                self.node_popularity_list.pop(0)
            
            self.node_popularity.zero_()
        

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
        if cached_edges is not None and self.edge_popularity_predictor:
            yhat = self.predict(self.edge_popularity_predictor, self.edge_popularity_std_scaler, 
                            cached_edges, self.edge_popularity_list, self.edge_popularity)
            self.predict_edge_popularity[cached_edge_index] = yhat
        
        if len(uncached_edge_id) > 0 and self.edge_popularity_predictor is not None:
            yhat = self.predict(self.edge_popularity_predictor, self.edge_popularity_std_scaler, 
                            uncached_edge_id, self.edge_popularity_list, self.edge_popularity)

            kth_popularity = torch.min(self.predict_edge_popularity)
            # filter yhat >= kth_popularity
            mask = yhat >= kth_popularity
            num_edges_to_cache = mask.sum()
            if num_edges_to_cache > self.edge_capacity:
                num_edges_to_cache = self.edge_capacity
                # largest N 
                mask = torch.topk(yhat, k=self.edge_capacity, largest=True).indices

            # print(f"filter edges: {len(uncached_edge_id)} -> {num_edges_to_cache}")
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
                self.predict_edge_popularity, k=num_edges_to_cache, largest=False).indices
        
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
            if self.edge_popularity_predictor is not None:
                self.predict_edge_popularity[removing_cache_index] = yhat

        self.edge_batch += 1

        if self.edge_batch % self.edge_popularity_predictor_update_interval == 0:
            if len(self.edge_popularity_list) < self.edge_popularity_list_max_len:
                self.edge_popularity_list.append(self.edge_popularity.clone().detach())
                self.edge_popularity.zero_()
                return

            if self.edge_popularity_predictor is None:
                self.edge_popularity_predictor = Predictor(
                    self.edge_popularity_list_max_len, 16).to(self.device)
                self.edge_popularity_optimizer = optim.Adam(
                    self.edge_popularity_predictor.parameters(), lr=1e-3)
            X = torch.stack(self.edge_popularity_list, dim=0).T
            y = self.edge_popularity

            self.edge_popularity_std_scaler.fit(X)
            X = self.edge_popularity_std_scaler.transform(X)

            self.train_predictor(self.edge_popularity_predictor, self.edge_popularity_optimizer, X, y, 'edge')

            self.edge_popularity_list.append(self.edge_popularity.clone().detach())
            if len(self.edge_popularity_list) > self.edge_popularity_list_max_len:
                self.edge_popularity_list.pop(0)
            
            self.edge_popularity.zero_()


    def train_predictor(self, predictor: nn.Module, optimizer, X: torch.Tensor, y: torch.Tensor, label: str):
        """
        Train the predictor
        """ 
        # split train/val; random split
        train_size = int(0.8 * len(X))
        val_size = len(X) - train_size

        train_idx = np.random.choice(len(X), train_size, replace=False)
        val_idx = np.setdiff1d(np.arange(len(X)), train_idx)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        loss_func = nn.MSELoss()

        early_stop_monitor = EarlyStopMonitor(higher_better=False)
        min_val_loss = float('inf')
        best_epoch = 0
        for epoch in range(300):
            optimizer.zero_grad()
            y_pred = predictor(X_train)
            loss = loss_func(y_pred, y_train)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                val_loss = loss_func(predictor(X_val), y_val).item()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}: train loss: {loss.item()}, val loss: {val_loss}")

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_epoch = epoch
                torch.save(predictor.state_dict(), f'{label}_predictor.pth')

            if early_stop_monitor.early_stop_check(val_loss):
                break

        print(f"Best epoch: {best_epoch}, best val: {min_val_loss}")
        if label == 'node':
            self.node_val_list.append(min_val_loss)
        else:
            self.edge_val_list.append(min_val_loss)
        predictor.load_state_dict(torch.load(f'{label}_predictor.pth'))
    
    def predict(self, predictor: nn.Module, scaler, items: torch.Tensor, popularity_list: List[torch.Tensor], popularity: torch.Tensor):
        """
        Predict
        """
        X_list = []
        for p in popularity_list[1:]:
            X_list.append(p[items])
        
        X_list.append(popularity[items])

        X = torch.stack(X_list, dim=0).T

        X = scaler.transform(X)

        with torch.no_grad():
            yhat = predictor(X) 
        return yhat

