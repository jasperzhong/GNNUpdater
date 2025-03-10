"""
Time Encoding Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""


import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.heterograph import DGLBlock
from torch import Tensor
from torch.nn import Linear


class TimeEncoder(torch.nn.Module):
    def __init__(self, dim):
        super(TimeEncoder, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(1, dim)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(
            1 / 10 ** np.linspace(0, 9, dim, dtype=np.float32))).reshape(dim, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output


class LinkPredictor(torch.nn.Module):
    """
    Reference:
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
    """

    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h).sigmoid()


class NodePredictor(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_node = Linear(in_dim, in_dim)
        self.out = Linear(in_dim, out_dim)

    def forward(self, node_embed):
        h = self.lin_node(node_embed)
        h = h.relu()
        h = self.out(h)
        # h = F.log_softmax(h, dim=-1)
        return h


class TransfomerAttentionLayer(torch.nn.Module):
    """
    Transfomer attention layer
    """

    def __init__(self, dim_node: int, dim_edge: int, dim_time: int,
                 dim_out: int, num_head: int, dropout: float, att_dropout: float,
                 use_relu: bool = True):
        """
        Args:
            dim_node: dimension of node features/embeddings
            dim_edge: dimension of edge features
            dim_time: dimension of time features
            dim_out: dimension of output embeddings
            num_head: number of heads
            dropout: dropout rate
            att_dropout: dropout rate for attention
            use_relu: whether to use ReLU activation
        """
        super(TransfomerAttentionLayer, self).__init__()
        # assert dim_node > 0 or dim_edge > 0, \
        #     "either dim_node or dim_edge should be positive"

        self.use_node_feat = dim_node > 0
        self.use_edge_feat = dim_edge > 0
        self.use_time_enc = dim_time > 0

        self.dim_node = dim_node
        self.num_head = num_head
        self.dim_time = dim_time
        self.dim_out = dim_out
        self.dropout = torch.nn.Dropout(dropout)
        self.att_dropout = torch.nn.Dropout(att_dropout)
        self.att_act = torch.nn.LeakyReLU(0.2)

        if self.use_time_enc:
            self.time_enc = TimeEncoder(dim_time)

        if self.use_node_feat or self.use_time_enc:
            self.w_q = torch.nn.Linear(dim_node + dim_time, dim_out)
        else:
            self.w_q = torch.nn.Identity()

        self.w_k = torch.nn.Linear(
            dim_node + dim_edge + dim_time, dim_out)
        self.w_v = torch.nn.Linear(
            dim_node + dim_edge + dim_time, dim_out)

        self.w_out = torch.nn.Linear(dim_node + dim_out, dim_out)

        self.layer_norm = torch.nn.LayerNorm(dim_out)

        self.use_relu = use_relu

    def forward(self, b: DGLBlock):
        """
        Args:
            b: sampled message flow graph (mfg), where
                `b.num_dst_nodes()` is the number of target nodes to sample,
                `b.srcdata['h']` is the embedding of all nodes,
                `b.edge['f']` is the edge features of sampled edges, and
                `b.edata['dt']` is the delta time of sampled edges.

        Returns:
            output: output embedding of target nodes (shape: (num_dst_nodes, dim_out))
        """
        num_edges = b.num_edges()
        num_dst_nodes = b.num_dst_nodes()
        device = b.device

        # sample nothing (no neighbors)
        if num_edges == 0:
            return torch.zeros((num_dst_nodes, self.dim_out), device=device)

        if self.use_node_feat:
            target_node_embeddings = b.srcdata['h'][:num_dst_nodes]
            source_node_embeddings = b.srcdata['h'][num_dst_nodes:]
        else:
            # dummy node embeddings
            if self.use_time_enc:
                target_node_embeddings = torch.zeros(
                    (num_dst_nodes, 0), device=device)
            else:
                target_node_embeddings = torch.ones(
                    (num_dst_nodes, self.dim_out), device=device)

            source_node_embeddings = torch.zeros(
                (num_edges, 0), device=device)

        if self.use_edge_feat:
            edge_feats = b.edata['f']
        else:
            # dummy edge features
            edge_feats = torch.zeros((num_edges, 0), device=device)

        if self.use_time_enc:
            delta_time = b.edata['dt']
            time_feats = self.time_enc(delta_time)
            zero_time_feats = self.time_enc(torch.zeros(
                num_dst_nodes, dtype=torch.float32, device=device))
        else:
            # dummy time features
            time_feats = torch.zeros((num_edges, 0), device=device)
            zero_time_feats = torch.zeros((num_dst_nodes, 0), device=device)

        assert isinstance(edge_feats, torch.Tensor)
        Q = torch.cat([target_node_embeddings, zero_time_feats], dim=1)
        K = torch.cat([source_node_embeddings, edge_feats, time_feats], dim=1)
        V = torch.cat([source_node_embeddings, edge_feats, time_feats], dim=1)

        Q = self.w_q(Q)[b.edges()[1]]
        K = self.w_k(K)
        V = self.w_v(V)

        Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
        K = torch.reshape(K, (K.shape[0], self.num_head, -1))
        V = torch.reshape(V, (V.shape[0], self.num_head, -1))

        # compute attention scores
        att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
        att = self.att_dropout(att)
        V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))

        b.srcdata['v'] = torch.cat((torch.zeros(
            (num_dst_nodes, V.shape[1]), device=device), V), dim=0)
        b.update_all(fn.copy_u('v', 'm'), fn.sum('m', 'h'))

        if self.use_node_feat:
            rst = torch.cat((b.dstdata['h'], target_node_embeddings), dim=1)
        else:
            rst = b.dstdata['h']

        rst = self.w_out(rst)
        if self.use_relu:
            rst = F.relu(self.dropout(rst))
        return self.layer_norm(rst)
