from typing import Dict

from gnnupdater.models.dgnn import DGNN
from gnnupdater.models.graphsage import SAGE


def get_model(model: str, node_feat_dim: int, edge_feat_dim: int, model_config: Dict):
    if model == "GRAPHSAGE":
        model = SAGE(node_feat_dim, model_config['dim_embed'])
    elif model == 'GAT':
        model = DGNN(node_feat_dim, edge_feat_dim, **model_config)
    else:
        model = DGNN(node_feat_dim, edge_feat_dim, **model_config, num_nodes=num_nodes,
                     memory_device=device, memory_shared=distributed)
    
    return model