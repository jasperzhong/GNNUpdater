from typing import List, Optional

import dgl
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.heterograph import DGLBlock

from gnnupdater.dynamic_graph import DynamicGraph
from gnnupdater.temporal_sampler import TemporalSampler
from gnnupdater.utils import mfgs_to_cuda, prepare_input


class SAGE(nn.Module):
    def __init__(self, dim_node: int, dim_embed: int,
                 num_layers: int = 2,
                 aggregator: Optional[str] = 'mean',
                 dropout: float = 0.1):

        if aggregator not in ['mean', 'gcn', 'pool', 'lstm']:
            raise ValueError(
                "aggregator {} is not in ['mean', 'gcn', 'pool', 'lstm']".format(aggregator))
        super().__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleDict()
        for l in range(num_layers):
            # static graph doesn't have snapshot
            key = 'l' + str(l) + 'h' + str(0)
            if l == 0:
                self.layers[key] = dglnn.SAGEConv(
                    dim_node, dim_embed, aggregator)
            else:
                self.layers[key] = dglnn.SAGEConv(
                    dim_embed, dim_embed, aggregator)

        self.layer_norm = nn.LayerNorm(dim_embed)
        self.dropout = nn.Dropout(dropout)
        self.dim_embed = dim_embed
        self.num_snapshots = 1

    def forward(self, mfgs: List[List[DGLBlock]]):
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
        for l in range(self.num_layers):
            key = 'l' + str(l) + 'h' + str(0)
            h = self.layers[key](mfgs[l][0], mfgs[l][0].srcdata['h'])
            if l != self.num_layers - 1:
                h = F.relu(self.dropout(h))

            h = self.layer_norm(h)

            if l != self.num_layers - 1:
                mfgs[l + 1][0].srcdata['h'] = h

        return h

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
                    prepare_input(mfgs, x, None)

                for h in range(self.num_snapshots):
                    key = 'l' + str(l) + 'h' + str(h)
                    rst = self.layers[key](mfgs[0][h], mfgs[0][h].srcdata['h'])
                    if l != self.num_layers - 1:
                        rst = F.relu(self.dropout(rst))

                    rst = self.layer_norm(rst)

                    embeds[nodes] = rst

            x = embeds

        return embeds[all_nodes]
