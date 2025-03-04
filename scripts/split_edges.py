import pickle

import numpy as np
import pandas as pd

# datasets = ['tgbl_review', 'tgbl_coin',
#             'tgbl_comment', 'tgbl_flight', 'GDELT', 'MAG']
datasets = ['tgbn_genre', 'tgbn_reddit', 'tgbn_token']
# datasets = ['tgbn_token']

cutoff_config = {
    'tgbl_review': '2014-01-01',
    'tgbl_coin': '2022-05-01',
    'tgbl_comment': '2010-01-01',
    'tgbl_flight': '2020-01-01',
    'GDELT': '1971-07-01',
    'MAG': '1979-01-01',
}

ingestion_window = {
    'tgbl_review': '1d',
    'tgbl_coin': '1h',
    'tgbl_comment': '1d',
    'tgbl_flight': '1d',
    'GDELT': '1d',
    'MAG': '1M',
    'tgbn_genre': '1d',
    'tgbn_reddit': '1d',
    'tgbn_token': '1d',
}

WINDOW_FUNCS = {
    '1d': lambda x: x.dt.date,
    '1h': lambda x: x.dt.floor('H'),
    '1M': lambda x: x.dt.to_period('M')
}


for dataset in datasets:
    print(f'Processing {dataset}...')
    df = pd.read_feather(f'{dataset}/edges.feather')
    dataset_name = dataset.replace('_', '-')

    node_label_dict = pickle.load(
        open(f"{dataset}/ml_{dataset_name}_node.pkl", "rb"))

    all_keys = node_label_dict.keys()
    ts_array = df.ts.values

    split_points = np.searchsorted(ts_array, list(all_keys))
    print(f"Number of splits: {len(split_points) + 1}")
    max_num_samples = np.diff(split_points).max()
    print(f"Max number of samples: {max_num_samples}")
    np.save(f'{dataset}/split_points.npy', split_points)

    # offline ratio 0.3
    ratio = 0.3

    split_idx = int(len(split_points) * ratio)
    split_points = split_points[:split_idx]
    split_point = split_points[split_idx]
    offline_ratio = (split_point / len(df)) * 100
    print(
        f"offline ratio: {offline_ratio:.2f}%, streaming ratio: {100 - offline_ratio:.2f}%")
    offline_df = df.iloc[:split_point].reset_index(drop=True)
    streaming_df = df.iloc[split_point:].reset_index(drop=True)
    # import pdb
    # pdb.set_trace()

    offline_df.to_feather(f'{dataset}/offline_edges.feather')
    streaming_df.to_feather(f'{dataset}/streaming_edges.feather')
