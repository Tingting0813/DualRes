"""
数据加载和处理模块
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Iterator, Optional
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.util import to_pandas
from gluonts.dataset.common import Dataset, MetaData, ListDataset
import logging
from pandas import Period
from pathlib import Path

logger = logging.getLogger(__name__)

class ETTh1Dataset:
    """ETTh1数据集类，兼容GluonTS格式"""
    
    def __init__(self, csv_path: str, dataset_name: str = 'etth1', prediction_length: int = 48):
        self.csv_path = csv_path
        self.dataset_name = dataset_name
        self.prediction_length = prediction_length
        self.freq = 'h'
        
        # 加载和处理数据
        self.raw_data = self._load_data()
        self.train_entries, self.test_entries = self._prepare_datasets()
        
        # 创建元数据
        self.metadata = MetaData(
            freq=self.freq,
            target_dim=1,  # 单变量
            prediction_length=prediction_length,
            feat_static_cat_cardinalities=[self.raw_data.shape[1]],  # 7个不同的时间序列
            feat_static_real_shape=()
        )
    
    def _load_data(self) -> pd.DataFrame:
        """加载ETTh1数据"""
        df = pd.read_csv(self.csv_path)
        
        # ETTh1格式: date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
        # 处理日期列
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # 保留7个数值列: HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
        expected_columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        
        # 检查是否有所有期望的列
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing columns in {self.dataset_name}: {missing_columns}")
            # 使用实际存在的数值列
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            df = df[numeric_columns]
        else:
            # 使用期望的7列
            df = df[expected_columns]
        
        logger.info(f"Loaded {self.dataset_name} data: shape={df.shape}, columns={list(df.columns)}")
        logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
        return df
    
    def _prepare_datasets(self):
        """准备训练和测试数据集"""
        # 95%作为训练数据
        train_size = int(len(self.raw_data) * 0.95)
        test_start_idx = train_size
        remaining_length = len(self.raw_data) - train_size
        
        # 计算测试组数 - 每组预测prediction_length个点
        num_test_groups = remaining_length // self.prediction_length
        
        logger.info(f"Data split - Total: {len(self.raw_data)}, Train: {train_size}, "
                   f"Test: {remaining_length}, Test groups: {num_test_groups}")
        logger.info(f"Total test cases: {num_test_groups * len(self.raw_data.columns)} "
                   f"({num_test_groups} groups × {len(self.raw_data.columns)} columns)")
        
        train_entries = []
        test_entries = []
        
        # 为每个列创建训练数据条目
        for col_idx, col_name in enumerate(self.raw_data.columns):
            ts_data = self.raw_data[col_name].values.astype(np.float32)
            
            # 训练数据条目：只包含训练期间的数据
            train_entry = {
                'target': ts_data[:train_size],
                'start': Period(self.raw_data.index[0], freq=self.freq),
                'feat_static_cat': np.array([col_idx], dtype=np.int32),  # 第0个时间序列=0, 第1个=1, ...
                'item_id': f"{self.dataset_name}_col_{col_idx}"
            }
            train_entries.append(train_entry)
        
        # 为测试数据创建条目 - 按测试组排序，而不是按时间序列排序
        for group_idx in range(num_test_groups):
            # 当前测试组结束的索引
            test_end_idx = test_start_idx + (group_idx + 1) * self.prediction_length
            
            # 为当前测试组的每个时间序列创建测试条目
            for col_idx, col_name in enumerate(self.raw_data.columns):
                ts_data = self.raw_data[col_name].values.astype(np.float32)
                
                # 测试目标：从时间序列开始(0)到当前测试组结束的所有数据
                test_entry = {
                    'target': ts_data[:test_end_idx],  # 包含从0到test_end_idx-1的所有数据
                    'start': Period(self.raw_data.index[0], freq=self.freq),
                    'feat_static_cat': np.array([col_idx], dtype=np.int32),  # 同一个时间序列保持相同的标识符
                    'item_id': f"{self.dataset_name}_col_{col_idx}_group_{group_idx}"
                }
                test_entries.append(test_entry)
                
                logger.debug(f"Test case group_{group_idx}_col_{col_idx}: target length = {len(test_entry['target'])}, "
                           f"will predict indices {test_end_idx - self.prediction_length} to {test_end_idx - 1}")
        
        return train_entries, test_entries
    
    @property
    def train(self):
        """训练数据集"""
        return ListDataset(self.train_entries, freq=self.freq)
    
    @property
    def test(self):
        """测试数据集"""
        return ListDataset(self.test_entries, freq=self.freq)


        
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
        if self.dataset_name in ['m4_hourly', 'traffic_nips', 'electricity_nips', 'exchange_rate_nips', 'solar_nips']:
            self.dataset = get_dataset(self.dataset_name)
            self.train_data = list(self.dataset.train)
            self.test_data = list(self.dataset.test)
            logger.info(f"Dataset loaded. Train size: {len(self.train_data)}, Test size: {len(self.test_data)}")  
        elif self.dataset_name in ['etth1', 'etth2']:
            # 直接创建ETTh数据集，根据名称选择对应的CSV文件
            current_dir = os.path.dirname(__file__)
            if self.dataset_name == 'etth1':
                a = 'ETTh1.csv'
            elif self.dataset_name == 'etth2':
                a = 'ETTh2.csv'
            csv_path = os.path.join(current_dir, a)
            logger.info(f"Loading {self.dataset_name} from {csv_path}")
            
            self.dataset = ETTh1Dataset(csv_path, self.dataset_name)
            self.train_data = list(self.dataset.train)
            self.test_data = list(self.dataset.test)
            logger.info(f"Dataset loaded. Train size: {len(self.train_data)}, Test size: {len(self.test_data)}")
         
        else:
            raise ValueError(f"Invalid dataset name: {self.dataset_name}")


        
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