"""
辅助函数模块
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def ensure_dir(directory: str) -> None:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")


def save_json(data: Dict, filepath: str) -> None:
    """
    保存数据为JSON文件
    
    Args:
        data: 要保存的数据
        filepath: 文件路径
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: str) -> Dict:
    """
    加载JSON文件
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的数据
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded JSON from {filepath}")
    return data


def save_pickle(data: Any, filepath: str) -> None:
    """
    保存数据为pickle文件
    
    Args:
        data: 要保存的数据
        filepath: 文件路径
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Saved pickle to {filepath}")


def load_pickle(filepath: str) -> Any:
    """
    加载pickle文件
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的数据
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    logger.info(f"Loaded pickle from {filepath}")
    return data


def sliding_window(data: np.ndarray, window_size: int, step: int = 1) -> List[np.ndarray]:
    """
    创建滑动窗口
    
    Args:
        data: 输入数据
        window_size: 窗口大小
        step: 步长
        
    Returns:
        窗口列表
    """
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i:i + window_size])
    return windows


def normalize_data(data: np.ndarray, method: str = 'minmax') -> tuple:
    """
    数据归一化
    
    Args:
        data: 输入数据
        method: 归一化方法 ('minmax' or 'standard')
        
    Returns:
        归一化后的数据和参数
    """
    if method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        params = {'min': min_val, 'max': max_val}
    elif method == 'standard':
        mean = np.mean(data)
        std = np.std(data)
        normalized = (data - mean) / (std + 1e-8)
        params = {'mean': mean, 'std': std}
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def denormalize_data(data: np.ndarray, params: Dict, method: str = 'minmax') -> np.ndarray:
    """
    数据反归一化
    
    Args:
        data: 归一化的数据
        params: 归一化参数
        method: 归一化方法
        
    Returns:
        反归一化后的数据
    """
    if method == 'minmax':
        denormalized = data * (params['max'] - params['min']) + params['min']
    elif method == 'standard':
        denormalized = data * params['std'] + params['mean']
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return denormalized


def create_lagged_features(data: pd.DataFrame, 
                          target_col: str, 
                          lag_steps: List[int]) -> pd.DataFrame:
    """
    创建滞后特征
    
    Args:
        data: 输入DataFrame
        target_col: 目标列名
        lag_steps: 滞后步数列表
        
    Returns:
        包含滞后特征的DataFrame
    """
    df = data.copy()
    for lag in lag_steps:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df


def plot_training_history(history: Dict, save_path: Optional[str] = None) -> None:
    """
    绘制训练历史
    
    Args:
        history: 训练历史字典
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制损失
    axes[0].plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History - Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 绘制指标
    if 'metrics' in history:
        for metric_name, values in history['metrics'].items():
            axes[1].plot(values, label=metric_name)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric Value')
        axes[1].set_title('Training History - Metrics')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()


def check_gpu_availability() -> bool:
    """
    检查GPU是否可用
    
    Returns:
        GPU是否可用
    """
    import torch
    
    if torch.cuda.is_available():
        logger.info(f"GPU is available. Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        return True
    else:
        logger.info("GPU is not available. Using CPU.")
        return False


def set_random_seed(seed: int = 42) -> None:
    """
    设置随机种子以确保可重复性
    
    Args:
        seed: 随机种子
    """
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Random seed set to {seed}")

def filter_metrics(metrics, select={"MAE_Coverage", "mean_wQuantileLoss", "Coverage[0.9]", "MSE", "RMSE","NRMSE", "ND"}):
    res = {}
    for m in select:
        if m == "Coverage[0.9]":
            res["Coverage Diff"] = np.abs(metrics[m].item() - 0.9)
        res[m] = metrics[m].item()
    return res