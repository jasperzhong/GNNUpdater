#!/bin/bash

MODEL=$1
DATA=$2
CACHE="${3:-LRUCache}"
EDGE_CACHE_RATIO="${4:-0.2}" # default 20% of cache
NODE_CACHE_RATIO="${5:-0.2}" # default 20% of cache
NPROC_PER_NODE="${6:-1}"
REPLAY="${7:-0}"
PHASE1_RATIO="${8:-0.3}" # default 50% of replay
UPDATE_PER_N_MINUTE="${9:-1440}" 
RETRAIN="${10:-1}"


if [[ $NPROC_PER_NODE -gt 1 ]] ; then
    cmd="torchrun \
        --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \
        --standalone \
        online_edge_prediction.py --model $MODEL --data $DATA \
        --cache $CACHE --edge-cache-ratio $EDGE_CACHE_RATIO --node-cache-ratio $NODE_CACHE_RATIO
        --replay-ratio $REPLAY --phase1-ratio $PHASE1_RATIO --update-per-n-minute $UPDATE_PER_N_MINUTE"
else
    cmd="python online_edge_prediction.py --model $MODEL --data $DATA \
        --cache $CACHE --edge-cache-ratio $EDGE_CACHE_RATIO --node-cache-ratio $NODE_CACHE_RATIO
        --replay-ratio $REPLAY --phase1-ratio $PHASE1_RATIO --update-per-n-minute $UPDATE_PER_N_MINUTE"
fi

if [[ $RETRAIN -eq 1 ]] ; then
    cmd="$cmd --retrain"
fi

rm -rf /dev/shm/rmm_*
echo $cmd
OMP_NUM_THREADS=8 exec $cmd 
