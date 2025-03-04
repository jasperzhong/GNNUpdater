"""
This code is based on the implementation of TGL's model module.

Implementation at:
    https://github.com/amazon-research/tgl/blob/main/modules.py
"""
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from dgl.heterograph import DGLBlock

from gnnupdater.distributed.kvstore import KVStoreClient
from gnnupdater.dynamic_graph import DynamicGraph
from gnnupdater.models.layers import TransfomerAttentionLayer
from gnnupdater.temporal_sampler import TemporalSampler
from gnnupdater.utils import mfgs_to_cuda, prepare_input


class TGNN(torch.nn.Module):
    """
    Temporal Graph Neural Model (TGNN)
    """

    def __init__(self, dim_node: int, dim_edge: int, dim_time: int,
                 dim_embed: int, num_layers: int, num_snapshots: int,
                 att_head: int, dropout: float, att_dropout: float):
        """
        Args:
            dim_node: dimension of node features/embeddings
            dim_edge: dimension of edge features
            dim_time: dimension of time features
            dim_embed: dimension of output embeddings
            num_layers: number of layers
            num_snapshots: number of snapshots
            att_head: number of heads for attention
            dropout: dropout rate
            att_dropout: dropout rate for attention
        """
        super(TGNN, self).__init__()
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge
        self.dim_time = dim_time
        self.dim_embed = dim_embed
        self.num_layers = num_layers
        self.num_snapshots = num_snapshots
        self.att_head = att_head
        self.dropout = dropout
        self.att_dropout = att_dropout

        self.layers = torch.nn.ModuleDict()
        for l in range(num_layers):
            for h in range(num_snapshots):
                if l == 0:
                    dim_node_input = dim_node
                else:
                    dim_node_input = dim_embed

                key = 'l' + str(l) + 'h' + str(h)
                is_last_layer = l == num_layers - 1
                self.layers[key] = TransfomerAttentionLayer(dim_node_input,
                                                            dim_edge,
                                                            dim_time,
                                                            dim_embed,
                                                            att_head,
                                                            dropout,
                                                            att_dropout,
                                                            use_relu=not is_last_layer)

        if self.num_snapshots > 1:
            self.combiner = torch.nn.RNN(
                dim_embed, dim_embed)

    def forward(self, mfgs: List[List[DGLBlock]]):
        """
        Args:
            mfgs: list of list of DGLBlocks
            neg_sample_ratio: negative sampling ratio
        """
        out = list()
        for l in range(self.num_layers):
            for h in range(self.num_snapshots):
                key = 'l' + str(l) + 'h' + str(h)
                rst = self.layers[key](mfgs[l][h])
                if l != self.num_layers - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)

        if self.num_snapshots == 1:
            embed = out[0]
        else:
            embed = torch.stack(out, dim=0)
            embed = self.combiner(embed)[0][-1, :, :]

        return embed

    @torch.no_grad()
    def inference(self, all_nodes, all_ts, sampler: TemporalSampler,
                  batch_size: int, cache, device):
        # Compute representations layer by layer
        x = None
        for l in range(self.num_layers):
            embeds = torch.zeros(
                (all_nodes.max()+1, self.dim_embed), device=device)
            for i in range(0, len(all_nodes), batch_size):
                nodes = all_nodes[i:i+batch_size]
                timestamps = all_ts[i:i+batch_size]
                mfgs = sampler.sample(nodes, timestamps)
                mfgs_to_cuda(mfgs, device)
                if l == 0:
                    mfgs = cache.fetch_feature(mfgs, None)
                else:
                    prepare_input(mfgs, x, cache.edge_feats)

                for h in range(self.num_snapshots):
                    key = 'l' + str(l) + 'h' + str(h)
                    rst = self.layers[key](mfgs[0][h])
                    embeds[nodes] = rst

            x = embeds

        return embeds[all_nodes]
