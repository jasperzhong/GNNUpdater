from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class NonPositiveLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1e-4):
        self.coef_ = None
        self.intercept_ = 0.0
        self.alpha = alpha  # 添加正则化参数

    def fit(self, X, y):
        # 中心化y
        self.y_mean_ = np.mean(y)
        y_centered = y - self.y_mean_

        # 添加L2正则化项到NNLS问题
        n_samples, n_features = X.shape
        X_reg = np.vstack([X, np.sqrt(self.alpha) * np.eye(n_features)])
        y_reg = np.concatenate([y_centered, np.zeros(n_features)])

        # 求解正则化后的NNLS
        neg_coef, _ = nnls(-X_reg, y_reg)
        self.coef_ = -neg_coef

        # 计算intercept
        self.intercept_ = self.y_mean_ - np.mean(X @ self.coef_)

        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


X_cols = ['neighbor_mae', 'neighbor_num_nodes', 'log_num_nodes', 'log_num_edges', 'ts',
          'log_num_label_nodes', 'label_nodes_mean_degree']


def train_perf_predictor(input_df: pd.DataFrame, predictor_type='RandomForest', test_ratio=0.2, seed=42):
    """
    训练准确度预测器

    Args:
        df: 数据集
        predictor: 预测器 (RandomForest)
        test_ratio: 测试集比例
        seed: 随机种子
    """
    df = input_df.copy()
    df = df.iloc[1:]  # 去掉第一行
    df.drop(columns=['pred_perf'], errors='ignore', inplace=True)
    df['log_num_nodes'] = np.log(df['num_nodes'] + 1)
    df['log_num_edges'] = np.log(df['num_edges'] + 1)
    df['log_num_new_nodes'] = np.log(df['num_new_nodes'] + 1)
    df['log_num_new_edges'] = np.log(df['num_new_edges'] + 1)
    df['log_num_label_nodes'] = np.log(df['num_label_nodes'] + 1)
    df.dropna(inplace=True, subset=X_cols + ['metric'])

    train_df, test_df = train_test_split(
        df, test_size=test_ratio, shuffle=True, random_state=seed)
    y_col = 'metric'
    X_train = train_df[X_cols]
    X_test = test_df[X_cols]
    y_train = train_df[y_col]
    y_test = test_df[y_col]

    std_scaler = StandardScaler()
    X_train = std_scaler.fit_transform(X_train)
    X_test = std_scaler.transform(X_test)

    if predictor_type == 'LinearRegression':
        perf_predictor = NonPositiveLinearRegression(alpha=1e-4)
    elif predictor_type == 'RandomForest':
        perf_predictor = RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42)
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")

    perf_predictor.fit(X_train, y_train)
    y_pred = perf_predictor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    corr = np.corrcoef(y_test, y_pred)[0, 1]
    error = (np.abs(y_test - y_pred) / y_test).mean()

    # 特征重要性分析
    importance = perf_predictor.feature_importances_ if predictor_type == 'RandomForest' else abs(
        perf_predictor.coef_)
    feature_importance = pd.DataFrame({
        'feature': X_cols,
        'importance': importance
    })

    top3_features = feature_importance.sort_values(
        'importance', ascending=False).head(3)['feature'].values
    top3_features_weights = feature_importance.sort_values(
        'importance', ascending=False).head(3)['importance'].values
    results = {
        'mse': mse,
        'rmse': rmse,
        'corr': corr,
        'error': error,
        'top3_features': top3_features,
        'top3_features_weights': top3_features_weights
    }

    return perf_predictor, std_scaler, results


def predict_perf(perf_predictor, std_scaler, data: Dict):
    """
    预测准确度

    Args:
        perf_predictor: 准确度预测器
        std_scaler: 标准化器
        data: 原始数据

        data = {
            'full_mmd2': full_mmd2,
            'mmd2': mmd2,
            'mse': mse,
            'mae': mae,
            'metric': result_dict[eval_metric],
            'num_edges': num_edges,
            'num_nodes': len(cur_nodes),
            'num_new_nodes': len(set(src) - nodes_set),
            'num_new_edges': len(src),
            'num_label_nodes': len(label_src),
            'label_nodes_mean_degree': label_src_node_degrees.mean(),
            'label_nodes_mean_activity': activity.mean(),
            'ts': label_ts[0],
        }
    """
    df = pd.DataFrame(data, index=[0])
    df['log_num_nodes'] = np.log(df['num_nodes'] + 1)
    df['log_num_edges'] = np.log(df['num_edges'] + 1)
    df['log_num_new_nodes'] = np.log(df['num_new_nodes'] + 1)
    df['log_num_new_edges'] = np.log(df['num_new_edges'] + 1)
    df['log_num_label_nodes'] = np.log(df['num_label_nodes'] + 1)

    X = df[X_cols]
    X = std_scaler.transform(X)
    return float(perf_predictor.predict(X))
