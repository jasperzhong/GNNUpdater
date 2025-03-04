import itertools
import multiprocessing
import os
import time
from multiprocessing import Manager, Pool

models = ['TGN']
datasets = ['REDDIT', 'WIKI']
update_intervals = [1000, 10000, 100000]
num_epochs = [1, 2, 4, 5] #, 10]
replay_ratios = [0, 0.01, 0.02, 0.04, 0.05] # , 0.1]

def run_command(cmd, gpu_id):
    full_cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd}"
    print(f"Running on GPU {gpu_id}: {full_cmd}")
    os.system(full_cmd)

def worker(task, gpu_queue):
    gpu_id = gpu_queue.get()
    try:
        run_command(task, gpu_id)
    finally:
        gpu_queue.put(gpu_id)

def run_tasks_in_parallel(tasks, num_gpus=2):
    manager = Manager()
    gpu_queue = manager.Queue()
    for i in range(num_gpus):
        gpu_queue.put(i)

    with Pool(num_gpus) as pool:
        pool.starmap(worker, [(task, gpu_queue) for task in tasks])

if __name__ == "__main__":
    # First for loop tasks
    # tasks1 = []
    # for model, dataset in itertools.product(models, datasets):
    #     cmd = f"python online_edge_prediction.py --model {model} --data {dataset} " \
    #           f"--cache LRUCache --edge-cache-ratio 0.2 --node-cache-ratio 0.2 " \
    #           f"--replay-ratio 0 --phase1-ratio 0.3 > logs2/{model}_{dataset}_noretrain.log 2>&1"
    #     tasks1.append(cmd)

    # # Run first set of tasks
    # print("Running first set of tasks...")
    # run_tasks_in_parallel(tasks1)

    # # Wait for all tasks to complete
    # print("First set of tasks completed. Waiting for 60 seconds before starting second set...")
    # time.sleep(60)

    # Second for loop tasks
    tasks2 = []
    for model, dataset, update_interval, num_epoch, replay_ratio in itertools.product(models, datasets, update_intervals, num_epochs, replay_ratios):
        cmd = f"python online_edge_prediction.py --model {model} --data {dataset} " \
              f"--cache LRUCache --edge-cache-ratio 1 --node-cache-ratio 1 " \
              f"--replay-ratio {replay_ratio} --phase1-ratio 0.3 --update-interval {update_interval} --retrain --epoch {num_epoch} " \
              f"> logs2/{model}_{dataset}_{update_interval}_{num_epoch}_{replay_ratio}.log 2>&1"
        print(cmd)
        tasks2.append(cmd)

    # Run second set of tasks
    print("Running second set of tasks...")
    run_tasks_in_parallel(tasks2)

    print("All tasks completed.")
