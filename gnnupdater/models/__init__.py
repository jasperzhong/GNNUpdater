from typing import Dict

import torch
import torch.nn as nn
import yaml

from gnnupdater.models.gat import GAT
from gnnupdater.models.graphsage import SAGE
from gnnupdater.models.layers import LinkPredictor, NodePredictor
from gnnupdater.models.memory_module import (IdentityMessage, LastAggregator,
                                          MeanAggregator, TGNMemory)
from gnnupdater.models.tgnn import TGNN
from gnnupdater.utils import PROJ_ROOT


def get_model(model_name: str, task: str, device, **kwargs) -> Dict:
    if model_name not in ['TGN', 'TGAT', 'DySAT', 'GraphSAGE', 'GAT']:
        raise ValueError(f'Invalid model name: {model_name}')

    if task not in ['link_prediction', 'node_classification']:
        raise ValueError(f'Invalid task: {task}')

    config = yaml.safe_load(
        open(f'{PROJ_ROOT}/configs/models/{model_name}.yml'))

    model = nn.ModuleDict()
    if config['use_memory']:
        model['memory'] = TGNMemory(
            num_nodes=kwargs['num_nodes'],
            raw_msg_dim=kwargs['edge_feat_dim'],
            memory_dim=config['dim_memory'],
            time_dim=config['dim_time'],
            message_module=IdentityMessage(
                kwargs['edge_feat_dim'], config['dim_memory'], config['dim_time']),
            aggregator_module=LastAggregator()
        ).to(device)

    if model_name in ['TGN', 'TGAT', 'DySAT', 'GAT']:
        model['encoder'] = TGNN(
            dim_node=kwargs['node_feat_dim'],
            dim_edge=kwargs['edge_feat_dim'],
            dim_time=config['dim_time'],
            dim_embed=config['dim_embed'],
            num_layers=config['num_layers'],
            num_snapshots=config['num_snapshots'],
            att_head=config['att_head'],
            dropout=config['dropout'],
            att_dropout=config['att_dropout']
        ).to(device)
    elif model_name == 'GraphSAGE':
        model['encoder'] = SAGE(
            dim_node=kwargs['node_feat_dim'],
            dim_embed=config['dim_embed'],
            num_layers=config['num_layers'],
            aggregator=config['aggregator']
        ).to(device)

    if task == "link_prediction":
        model['decoder'] = LinkPredictor(config['dim_embed']).to(device)
    elif task == "node_classification":
        dataset_config = kwargs['dataset_config']
        model['decoder'] = NodePredictor(
            config['dim_embed'], dataset_config['num_classes']).to(device)

    return model, config
