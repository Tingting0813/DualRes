"""
数据加载和处理模块
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.util import to_pandas
from gluonts.dataset.common import ListDataset
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据加载器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.dataset_name = config['dataset']['name']
        self.freq = config['dataset']['freq']
        self.dataset = None
        self.train_data = None
        self.test_data = None
        
    def load_dataset(self) -> None:
        """加载数据集"""
        logger.info(f"Loading dataset: {self.dataset_name}")
        self.dataset = get_dataset(self.dataset_name)
        self.train_data = list(self.dataset.train)
        self.test_data = list(self.dataset.test)
        logger.info(f"Dataset loaded. Train size: {len(self.train_data)}, Test size: {len(self.test_data)}")
        
    def get_dataset_info(self) -> Dict:
        """获取数据集信息"""
        if self.dataset is None:
            self.load_dataset()
            
        train_entry = self.train_data[0]
        test_entry = self.test_data[0]
        
        info = {
            'metadata': self.dataset.metadata,
            'train_size': len(self.train_data),
            'test_size': len(self.test_data),
            'train_length': len(train_entry["target"]),
            'test_length': len(test_entry["target"]),
            'freq': self.freq
        }
        return info
    
    def prepare_sliding_windows(self, data: List[Dict], context_length: int) -> List[np.ndarray]:
        """
        为滑动窗口预测准备数据
        
        Args:
            data: 数据列表
            context_length: 上下文长度
            
        Returns:
            预测结果列表
        """
        all_predictions = []
        
        for ts_entry in data:
            target = ts_entry["target"]
            predictions = []
            
            for i in range(len(target) - context_length):
                context = target[i:i+context_length]
                predictions.append(context)
                
            all_predictions.append(predictions)
            
        return all_predictions
    
    def create_loss_dataset(self, log_mse_data: pd.DataFrame) -> ListDataset:
        """
        创建用于log model训练的损失数据集
        
        Args:
            log_mse_data: 包含log MSE的DataFrame
            
        Returns:
            GluonTS ListDataset
        """
        gluonts_data = []
        
        for series_id, group in log_mse_data.groupby('series_id'):
            group_sorted = group.sort_values("timestamp")
            start = pd.to_datetime(group_sorted.iloc[0]["timestamp"])
            target = group_sorted["log_mse"].values.astype(np.float32)
            
            gluonts_data.append({
                "start": start,
                "target": target,
                "item_id": str(series_id)
            })
        
        loss_dataset = ListDataset(gluonts_data, freq=self.freq)
        logger.info(f"Created loss dataset with {len(gluonts_data)} time series")
        
        return loss_dataset
    
    def save_predictions(self, predictions: np.ndarray, filename: str) -> None:
        """
        保存预测结果
        
        Args:
            predictions: 预测数组
            filename: 文件名
        """
        save_path = os.path.join(self.config['paths']['predictions_dir'], filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, predictions)
        logger.info(f"Predictions saved to {save_path}")
    
    def load_predictions(self, filename: str) -> np.ndarray:
        """
        加载预测结果
        
        Args:
            filename: 文件名
            
        Returns:
            预测数组
        """
        load_path = os.path.join(self.config['paths']['predictions_dir'], filename)
        predictions = np.load(load_path, allow_pickle=True)
        logger.info(f"Predictions loaded from {load_path}")
        return predictions