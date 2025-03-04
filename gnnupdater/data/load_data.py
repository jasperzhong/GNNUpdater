import os
import pickle

import numpy as np
import pandas as pd
import torch
import yaml

from gnnupdater.utils import PROJ_ROOT


def load_data(dataset: str, load_feat=True, data_dir='datasets'):
    df = pd.read_feather(os.path.join(
        PROJ_ROOT, data_dir, dataset, "edges.feather"))
    offline_df = pd.read_feather(os.path.join(
        PROJ_ROOT, data_dir, dataset, "offline_edges.feather"))
    streaming_df = pd.read_feather(os.path.join(
        PROJ_ROOT, data_dir, dataset, "streaming_edges.feather"))

    config = yaml.safe_load(
        open(os.path.join(PROJ_ROOT, 'configs', f'{dataset}.yml')))

    split_points = np.load(os.path.join(
        PROJ_ROOT, data_dir, dataset, 'split_points.npy'))

    dataset_name = dataset.replace('_', '-')
    node_label_dict = pickle.load(open(os.path.join(
        PROJ_ROOT, data_dir, dataset, f'ml_{dataset_name}_node.pkl'), "rb"))

    node_feat, edge_feat = None, None
    if load_feat:
        node_feat_path = os.path.join(
            PROJ_ROOT, data_dir, dataset, 'node_features.npy')
        if os.path.exists(node_feat_path):
            node_feat = np.load(node_feat_path)
        else:
            node_feat = np.random.randn(
                config['num_nodes'], config['node_feat_dim'])
            np.save(node_feat_path, node_feat)
            print(
                f"Node features are randomly initialized and saved to {node_feat_path}")
        node_feat = torch.from_numpy(node_feat)

        edge_feat_path = os.path.join(
            PROJ_ROOT, data_dir, dataset, 'edge_features.npy')
        if os.path.exists(edge_feat_path):
            edge_feat = torch.from_numpy(np.load(edge_feat_path))

    return df, offline_df, streaming_df, config, split_points, node_label_dict, node_feat, edge_feat


if __name__ == '__main__':
    import time
    start = time.time()
    for dataset in ['tgbn_genre', 'tgbn_reddit', 'tgbn_token']:
        df, offline_df, streaming_df, config, split_points, node_label_dict, node_feat, edge_feat = load_data(
            dataset, load_feat=False)
        print(dataset)
        import pdb
        pdb.set_trace()
    # print(f'Time: {time.time() - start:.2f}s')
