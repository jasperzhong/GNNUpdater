from collections import Counter, deque

import numpy as np


class WindowActivityTracker:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.current_scores = {}  # 使用普通dict而不是defaultdict

    def update_and_get(self, label_src):
        # 计算新批次的计数
        batch_counts = Counter(label_src)

        # 如果窗口已满，先减去最老的数据
        if len(self.history) == self.window_size:
            oldest_counts = self.history[0]
            for node, count in oldest_counts.items():
                self.current_scores[node] -= count
                # 如果计数变为0，删除该节点
                if self.current_scores[node] == 0:
                    del self.current_scores[node]

        # 添加新的计数
        for node, count in batch_counts.items():
            self.current_scores[node] = self.current_scores.get(
                node, 0) + count

        # 更新历史
        self.history.append(batch_counts)

        # 返回当前窗口的计数 for label_src
        scores = [self.current_scores.get(node, 0) for node in label_src]
        return np.array(scores)
