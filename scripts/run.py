import os
import subprocess
import threading
import time
from itertools import product
from queue import Queue

import pandas as pd

# 配置
models = ['TGAT']
datasets = ['tgbn_genre']
gpu_queue = Queue()
for i in range(0, 4):  # 添加4个GPU到队列
    gpu_queue.put(i)


dataset_params = {
    'tgbn_genre': {
        'accuracy_drop_thresholds': [0.005, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.08, 0.1, 0.15],
        'intervals': [14, 30, 60, 90, 180, 360],
        'problem_ratios': [0.15, 0.18, 0.2, 0.22, 0.25, 0.3],
        'delta': [1.0, 2.0, 5.0, 5.5, 6.0, 6.5, 7.0, 10.0],
        'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9],
        'distance_threshold': [-0.0003, -0.0005, -0.0006, -0.0007, -0.00075, -0.0008, -0.00085, -0.0009, -0.001],
    },
    'tgbn_reddit': {
        'accuracy_drop_thresholds': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        'intervals': [14, 30, 60, 90, 180, 360],
        'problem_ratios': [0.3, 0.4, 0.45],
        'delta': [1.0, 2.0, 5.0, 5.5, 6.0, 6.5, 7.0, 10.0],
        'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9],
        'distance_threshold': [0.0015, 0.0012, 0.001, 0.0007, 0.0005, 0.0003],
    },
    'tgbn_token': {
        'accuracy_drop_thresholds': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        'intervals': [7, 14, 30, 60, 90, 180, 360],
        'problem_ratios': [0.3, 0.4, 0.5],
        'delta': [1.0, 2.0, 5.0, 5.5, 6.0, 6.5, 7.0, 10.0],
        'alpha': [0.005, 0.007, 0.008, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9],
        'distance_threshold': [0.2, 0.15, 0.13, 0.1, 0.08, 0.07, 0.05],
    }
}


def check_whether_to_run(dataset, model, phase, trigger_type, param,
                         sliding_window=365):
    """if the task has been run, return False"""
    if trigger_type == 'interval' and param == 0:
        trigger_type = 'none'

    prefix = f'streaming_{dataset}_{model}_trigger_{trigger_type}'
    prefix += f'{param}'
    prefix += f'_sliding_window_{sliding_window}'

    path = f'../exp/{prefix}_exp.feather'
    if not os.path.exists(path):
        print(f"File {path} does not exist, run the task")
        return False

    df = pd.read_feather(path)
    if df['expected_len'].iloc[0] > len(df) + 10:
        print(
            f"File {path} has not finished, expected_len={df['expected_len'].iloc[0]}, actual_len={len(df)}")
        return False

    # # Check the last modified time
    # last_modified_time = os.path.getmtime(path)
    # one_day_ago = time.time() - 86400*5  # 86400*2 seconds in a day
    # if last_modified_time < one_day_ago:
    #     print(
    #         f"File {path} was modified more than five days ago, rerun the task")
    #     return False

    print(f"Skip {path} since it has been run")
    return True


def run_task(cmd):
    # 从队列获取一个GPU
    gpu_id = gpu_queue.get()
    try:
        cmd += ['--device', str(gpu_id)]
        time.sleep(10)
        print(f'Starting {cmd}')
        process = subprocess.run(
            cmd,
            check=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env={**os.environ, 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
        )
        print(process.stdout)
        if process.stderr:
            print(f"Errors for {cmd}")
            print(process.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Errors for {cmd}")
        print(e.stderr)
    finally:
        # 任务完成后，将GPU放回队列
        gpu_queue.put(gpu_id)


# 创建并启动所有线程
threads = []

phase = 'streaming'

# # adwin
# for model, dataset in product(models, datasets):
#     for delta in dataset_params[dataset]['delta']:
#         if check_whether_to_run(dataset, model, phase, 'adwin', delta):
#             continue
#         cmd = ['python', 'continuous_learning.py',
#                '--dataset', dataset,
#                '--model', model,
#                '--phase', phase,
#                '--trigger_type', 'adwin',
#                '--delta', str(delta)]
#         thread = threading.Thread(target=run_task, args=(cmd,))
#         threads.append(thread)
#         thread.start()
#         # 短暂睡眠以确保命令行输出不会太混乱
#         time.sleep(0.1)
#
# # adwin delayed accuracy
# for model, dataset in product(models, datasets):
#     for delta in dataset_params[dataset]['delta']:
#         if check_whether_to_run(dataset, model, phase, 'adwin_delayed_accuracy', delta):
#             continue
#         cmd = ['python', 'continuous_learning.py',
#                '--dataset', dataset,
#                '--model', model,
#                '--phase', phase,
#                '--trigger_type', 'adwin_delayed_accuracy',
#                '--delta', str(delta)]
#         thread = threading.Thread(target=run_task, args=(cmd,))
#         threads.append(thread)
#         thread.start()
#         # 短暂睡眠以确保命令行输出不会太混乱
#         time.sleep(0.1)
#
# #
# # kswin
# for model, dataset in product(models, datasets):
#     if dataset != 'tgbn_token':
#         continue
#
#     for alpha in dataset_params[dataset]['alpha']:
#         if check_whether_to_run(dataset, model, phase, 'kswin', alpha):
#             continue
#         cmd = ['python', 'continuous_learning.py',
#                '--dataset', dataset,
#                '--model', model,
#                '--phase', phase,
#                '--trigger_type', 'kswin',
#                '--alpha', str(alpha)]
#         thread = threading.Thread(target=run_task, args=(cmd,))
#         threads.append(thread)
#         thread.start()
#         # 短暂睡眠以确保命令行输出不会太混乱
#         time.sleep(0.1)
#
# # kswin delayed accuracy
# for model, dataset in product(models, datasets):
#     if dataset != 'tgbn_token':
#         continue
#
#     for alpha in dataset_params[dataset]['alpha']:
#         if check_whether_to_run(dataset, model, phase, 'kswin_delayed_accuracy', alpha):
#             continue
#         cmd = ['python', 'continuous_learning.py',
#                '--dataset', dataset,
#                '--model', model,
#                '--phase', phase,
#                '--trigger_type', 'kswin_delayed_accuracy',
#                '--alpha', str(alpha)]
#         thread = threading.Thread(target=run_task, args=(cmd,))
#         threads.append(thread)
#         thread.start()
#         # 短暂睡眠以确保命令行输出不会太混乱
#         time.sleep(0.1)

# for model, dataset in product(models, datasets):
#     if dataset == 'tgbn_token':
#         continue
#
#     for distance_threshold in dataset_params[dataset]['distance_threshold']:
#         # if check_whether_to_run(dataset, model, phase, 'mmd', distance_threshold):
#         #     continue
#         cmd = ['python', 'continuous_learning.py',
#                '--dataset', dataset,
#                '--model', model,
#                '--phase', phase,
#                '--trigger_type', 'mmd',
#                '--distance_threshold', str(distance_threshold)]
#         thread = threading.Thread(target=run_task, args=(cmd,))
#         threads.append(thread)
#         thread.start()
#         # 短暂睡眠以确保命令行输出不会太混乱
#         time.sleep(0.1)
#
# for model, dataset in product(models, datasets):
#     interval_list = dataset_params[dataset]['intervals']
#     for interval in interval_list:
#         if check_whether_to_run(dataset, model, phase, 'interval', interval):
#             continue
#
#         cmd = ['python', 'continuous_learning.py',
#                '--dataset', dataset,
#                '--model', model,
#                '--phase', phase]
#         if interval != 0:
#             cmd += ['--trigger_type', 'interval', '--interval', str(interval)]
#         else:
#             cmd += ['--trigger_type', 'none']
#         thread = threading.Thread(target=run_task, args=(cmd,))
#         threads.append(thread)
#         thread.start()
#         # 短暂睡眠以确保命令行输出不会太混乱
#         time.sleep(0.1)


# for model, dataset in product(models, datasets):
#     threshold_list = dataset_params[dataset]['accuracy_drop_thresholds']
#     for threshold in threshold_list:
#         if check_whether_to_run(dataset, model, phase, 'accuracy_drop', threshold):
#             continue
#
#         cmd = ['python', 'continuous_learning.py',
#                '--dataset', dataset,
#                '--model', model,
#                '--phase', phase,
#                '--trigger_type', 'accuracy_drop',
#                '--accuracy_drop_threshold', str(threshold)]
#         thread = threading.Thread(target=run_task, args=(cmd,))
#         threads.append(thread)
#         thread.start()
#         # 短暂睡眠以确保命令行输出不会太混乱
#         time.sleep(0.1)
#
#
# for model, dataset in product(models, datasets):
#     threshold_list = dataset_params[dataset]['accuracy_drop_thresholds']
#     for threshold in threshold_list:
#         if check_whether_to_run(dataset, model, phase, 'delayed_accuracy_drop', threshold):
#             continue
#
#         cmd = ['python', 'continuous_learning.py',
#                '--dataset', dataset,
#                '--model', model,
#                '--phase', phase,
#                '--trigger_type', 'delayed_accuracy_drop',
#                '--accuracy_drop_threshold', str(threshold)]
#         thread = threading.Thread(target=run_task, args=(cmd,))
#         threads.append(thread)
#         thread.start()
#         # 短暂睡眠以确保命令行输出不会太混乱
#         time.sleep(0.1)
#

for model, dataset in product(models, datasets):
    problem_ratio_list = dataset_params[dataset]['problem_ratios']
    for problem_ratio in problem_ratio_list:
        # if check_whether_to_run(dataset, model, phase, 'label_propagation', problem_ratio):
        #     continue

        cmd = ['python', 'continuous_learning.py',
               '--dataset', dataset,
               '--model', model,
               '--phase', phase,
               '--trigger_type', 'label_propagation_true_accuracy',
               '--k', '3',
               '--weight', '0.05',
               '--problem_ratio', str(problem_ratio)]

        thread = threading.Thread(target=run_task, args=(cmd,))
        threads.append(thread)
        thread.start()
        # 短暂睡眠以确保命令行输出不会太混乱
        time.sleep(0.1)

for model, dataset in product(models, datasets):
    problem_ratio_list = dataset_params[dataset]['problem_ratios']
    for problem_ratio in problem_ratio_list:
        # if check_whether_to_run(dataset, model, phase, 'label_propagation', problem_ratio):
        #     continue

        cmd = ['python', 'continuous_learning.py',
               '--dataset', dataset,
               '--model', model,
               '--phase', phase,
               '--trigger_type', 'label_propagation_true_accuracy',
               '--k', '1',
               '--weight', '0.2',
               '--problem_ratio', str(problem_ratio)]

        thread = threading.Thread(target=run_task, args=(cmd,))
        threads.append(thread)
        thread.start()
        # 短暂睡眠以确保命令行输出不会太混乱
        time.sleep(0.1)

#     problem_ratio_list = dataset_params[dataset]['problem_ratios']
#     for problem_ratio in problem_ratio_list:
#         if check_whether_to_run(dataset, model, phase, 'label_propagation_true_accuracy', problem_ratio):
#             continue
#
#         cmd = ['python', 'continuous_learning.py',
#                '--dataset', dataset,
#                '--model', model,
#                '--phase', phase,
#                '--trigger_type', 'label_propagation_true_accuracy',
#                '--problem_ratio', str(problem_ratio)]
#
#         thread = threading.Thread(target=run_task, args=(cmd,))
#         threads.append(thread)
#         thread.start()
#         # 短暂睡眠以确保命令行输出不会太混乱
#         time.sleep(0.1)

print(f"Started {len(threads)} tasks")

for thread in threads:
    thread.join()


print("All tasks completed!")
