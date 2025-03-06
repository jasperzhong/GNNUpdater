import argparse

from gnnupdater.training.continuous_graph_learner import ContinuousGraphLearner

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='tgbn_genre')
parser.add_argument('--model', type=str, default='TGAT')
parser.add_argument('--phase', type=str, default='offline')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--use_initial_offline', action='store_true',
                    help='Use initial offline learning')
parser.add_argument('--sliding_window', type=int, default=365,
                    help='Size of sliding window')
parser.add_argument('--trigger_type', type=str, default='interval',
                    help='Type of update trigger')
parser.add_argument('--interval', type=int, default=90,
                    help='Interval for interval trigger')
parser.add_argument('--accuracy_drop_window', type=int, default=7,
                    help='Window size for accuracy drop trigger')
parser.add_argument('--accuracy_drop_threshold', type=float, default=0.1,
                    help='Window size for accuracy drop trigger')
parser.add_argument('--problem_ratio', type=float, default=0.1,
                    help='Ratio of problem nodes in label propagation')
parser.add_argument('--k', type=int, default=2,
                    help='layers of the label propagation')
parser.add_argument('--weight', type=float, default=0.1,
                    help='weight of the label propagation')
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--distance_threshold', type=float, default=0.0,
                    help='The distance threshold for MMD.')
parser.add_argument('--delta', type=float, default=0.002,
                    help='The delta parameter for the ADWIN algorithm.')
parser.add_argument('--alpha', type=float, default=0.005,
                    help='Probability for the test statistic of the Kolmogorov-Smirnov-Test The alpha parameter is very sensitive, therefore should be set below 0.01')
parser.add_argument('--delay', type=int, default=7,
                    help='Label delay in days')
args = parser.parse_args()

if __name__ == '__main__':
    import time
    start = time.time()
    learner = ContinuousGraphLearner(
        args.dataset, args.model, args.phase, args.device, args,
        build_graph=True if args.phase == 'offline' else False)
    end = time.time()
    print(f"Time to build the graph: {end - start:.2f}s")

    if args.phase == 'offline':
        learner.offline_learning(use_initial_offline=args.use_initial_offline)
    elif args.phase == 'streaming':
        learner.continuous_learning(
            trigger_type=args.trigger_type, interval=args.interval,
            accuracy_drop_threshold=args.accuracy_drop_threshold,
            accuracy_drop_window=args.accuracy_drop_window,
            problem_ratio=args.problem_ratio, k=args.k, weight=args.weight,
            delta=args.delta, alpha=args.alpha, distance_threshold=args.distance_threshold,
            sliding_window=args.sliding_window, delay=args.delay)
    elif args.phase == 'collect':
        learner.collecting_data_from_offline()
    elif args.phase == 'benchmark':
        learner.benchmark_pyg_dgl_continuous_learning()
