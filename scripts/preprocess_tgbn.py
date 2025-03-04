import csv
import os
import os.path as osp
import pickle

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm


def load_label_dict(fname: str, node_ids: dict, rd_dict: dict) -> dict:
    """
    load node labels into a nested dictionary instead of pandas dataobject
    {ts: {node_id: label_vec}}
    Parameters:
        fname: str, name of the input file
        node_ids: dictionary of user names mapped to integer node ids
        rd_dict: dictionary of subreddit names mapped to integer node ids
    """
    if not osp.exists(fname):
        raise FileNotFoundError(f"File not found at {fname}")

    # day, user_idx, label_vec
    label_size = len(rd_dict)
    node_label_dict = {}  # {ts: {node_id: label_vec}}

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # ['ts', 'src', 'dst', 'w']
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
            else:
                u = node_ids[row[1]]
                ts = int(row[0])
                v = int(rd_dict[row[2]])
                weight = float(row[3])
                if (ts not in node_label_dict):
                    node_label_dict[ts] = {}

                if (u not in node_label_dict[ts]):
                    node_label_dict[ts][u] = sparse.dok_array(
                        (label_size, ), dtype=np.float32)

                node_label_dict[ts][u][v] = weight
                idx += 1
        return node_label_dict


def convert_to_sparse_format(node_label_dict):
    """
    Convert the node_label_dict from dense numpy arrays to sparse format
    Modifies the dictionary in-place to save memory

    Parameters:
        node_label_dict: dict, {ts: {node_id: np.array}}
    """
    new_node_label_dict = {}
    for ts in node_label_dict:
        if ts not in new_node_label_dict:
            new_node_label_dict[ts] = {}

        for node_id in node_label_dict[ts]:
            # 获取原来的dense vector
            dense_vec = node_label_dict[ts][node_id]
            # 转换成dok_matrix
            sparse_vec = sparse.dok_array((dense_vec.shape[0],))
            # 只复制非零元素
            nonzero_indices = dense_vec.nonzero()[0]
            for idx in nonzero_indices:
                sparse_vec[idx] = dense_vec[idx]

            # 替换原来的vector
            new_node_label_dict[ts][node_id] = sparse_vec

    return new_node_label_dict


def verify_conversion(old_dict, new_dict):
    """
    Verify the conversion is correct by comparing values
    """
    # 如果想要验证转换的正确性，可以用这个函数
    for ts in old_dict:
        if ts not in new_dict:
            return False
        for node_id in old_dict[ts]:
            if node_id not in new_dict[ts]:
                return False
            old_vec = old_dict[ts][node_id]
            new_vec = new_dict[ts][node_id].toarray().flatten()
            if not np.allclose(old_vec, new_vec):
                return False
    return True


if __name__ == '__main__':
    # datasets = ['tgbn_genre', 'tgbn_reddit', 'tgbn_token']
    datasets = ['tgbn_token']
    for dataset in datasets:
        print(f"Processing {dataset}...")
        # Load edge data
        dataset_name = dataset.replace('_', '-')
        with open(f'{dataset}/ml_{dataset_name}.pkl', 'rb') as f:
            df = pickle.load(f)

        df.drop(columns=['idx'], inplace=True)
        if 'label' in df.columns:
            df.drop(columns=['label'], inplace=True)
        df.columns = ['src', 'dst', 'ts', 'w']
        df['src'] = df['src'].astype(np.uint32)
        df['dst'] = df['dst'].astype(np.uint32)
        df['ts'] = df['ts'].astype(np.uint64)
        if dataset == 'tgbn_token':
            df['w'] = np.log(df['w'])
        df['w'] = df['w'].astype(np.float32)
        import pdb
        pdb.set_trace()

        df.to_feather(f"{dataset}/edges.feather")

        # edge_feat_path = f"{dataset}/ml_{dataset_name}_edge.pkl"
        # if os.path.exists(edge_feat_path):
        #     with open(edge_feat_path, 'rb') as f:
        #         edge_feat = pickle.load(f)
        #     if dataset == 'tgbn_token':
        #         print("applying log transformation to edge features")
        #         edge_feat[:, 0] = np.log(edge_feat[:, 0])
        #         import pdb
        #         pdb.set_trace()
        #     edge_feat = edge_feat.astype(np.float32)
        #     np.save(f"{dataset}/edge_features.npy", edge_feat)
        # else:
        #     print(f"Edge feature file not found: {edge_feat_path}")

        # print(f"Processing {dataset}...")
        # dataset_name = dataset.replace('_', '-')
        # node_path = f"{dataset}/ml_{dataset_name}_node.pkl"
        # if dataset == 'tgbn_genre':
        #     node_label_dict = pickle.load(open(node_path, 'rb'))
        #     import pdb
        #     pdb.set_trace()
        #     # new_node_label_dict = convert_to_sparse_format(node_label_dict)
        #     # assert verify_conversion(node_label_dict, new_node_label_dict)
        #     # with open(node_path, 'wb') as f:
        #     #     pickle.dump(new_node_label_dict, f)
        # else:
        #     # node_ids = pickle.load(open(node_path, 'rb'))
        #     rd_dict = pickle.load(
        #         open(f"{dataset}/ml_{dataset_name}_label.pkl", 'rb'))
        #     import pdb
        #     pdb.set_trace()
        #     # node_label_dict = load_label_dict(
        #     #     f"{dataset}/{dataset_name}_node_labels.csv", node_ids, rd_dict)
        #     # with open(node_path, 'wb') as f:
        #     #     pickle.dump(node_label_dict, f)
