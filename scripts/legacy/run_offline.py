import itertools
import os

import numpy as np

models = ['TGN', 'TGAT']
datasets = ['GDELT']
cache_ratio = [0, 0.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for model, dataset, cache_ratio in itertools.product(models, datasets, cache_ratio):
    cmd = f"CUDA_VISIBLE_DEVICES=3 python offline_edge_prediction.py --model {model} --data {dataset} \
    --cache LRUCache --edge-cache-ratio {cache_ratio} \
    --node-cache-ratio {cache_ratio} \
    --ingestion-batch-size 100000000 > logs/{model}_{dataset}_LRUCache_{cache_ratio}.log 2>&1"

    print(cmd)
    os.system(cmd)
