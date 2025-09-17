"""
Log Model相关功能
"""
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple
from gluonts.model.predictor import Predictor
from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.dataset.common import ListDataset
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class LogModel:
    """Log Model类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Log Model
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.log_config = config['log_model']
        self.context_length = self.log_config['context_length']
        self.prediction_length = self.log_config['prediction_length']
        self.max_epochs = self.log_config['max_epochs']
        self.model_type = self.log_config.get('model_type', 'SimpleFeedForward')
        self.freq = config['dataset']['freq']
        self.predictor = None
        
        # 模型保存路径
        self.model_name = f'log_{self.model_type}_{self.context_length}_{self.prediction_length}_{self.max_epochs}'
        self.model_path = os.path.join(config['paths']['models_dir'], self.model_name)
    
    def calculate_log_mse(self, true_values: np.ndarray, pred_values: np.ndarray, 
                         mean_context_length: int, dataset_name: str, 
                         timestamps: pd.DatetimeIndex = None, force_recalculate: bool = False) -> pd.DataFrame:
        """
        计算log-MSE损失
        
        Args:
            true_values: 真实值
            pred_values: 预测值
            mean_context_length: mean model的上下文长度
            dataset_name: 数据集名称
            timestamps: 时间戳
            
        Returns:
            包含log-MSE的DataFrame
        """

        # 检查结果文件是否已存在
        save_path = os.path.join(self.config['paths']['results_dir'], 
                                 f'{dataset_name}_log_mse_losses.csv')
        
        if os.path.exists(save_path) and not force_recalculate:
            logger.info(f"Loading existing log_mse from {save_path}")
            return pd.read_csv(save_path)
        
        logger.info("Calculating new log_mse values...")

        all_records = []
        
        for i in range(len(true_values)):
            true_tensor = torch.tensor(true_values[i][mean_context_length:], dtype=torch.float32)
            pred_tensor = torch.tensor(np.array(pred_values[i], dtype=np.float32))
            
            # 计算 log-MSE
            log_mse_loss = torch.log((true_tensor - pred_tensor).pow(2) + 1e-10)
            
            # 添加记录
            for j, loss in enumerate(log_mse_loss):
                record = {
                    "series_id": i,
                    "log_mse": loss.item()
                }
                
                if timestamps is not None:
                    record["timestamp"] = timestamps[i][j]
                
                # 如果有原始值，也保存
                if j < len(pred_tensor):
                    record["mean_true"] = true_tensor[j].item()
                    record["mean_pred"] = pred_tensor[j].item()
                
                all_records.append(record)
        
        df = pd.DataFrame(all_records)
        
        # 保存结果
        df.to_csv(save_path, index=False)
        logger.info(f"Log-MSE losses saved to {save_path}")
        
        return df
    
    def train(self, loss_dataset: ListDataset, force_retrain: bool = False) -> None:
        """
        训练Log Model
        
        Args:
            loss_dataset: 损失数据集
            force_retrain: 是否强制重新训练
        """
        if os.path.exists(self.model_path) and not force_retrain:
            logger.info(f"Loading existing model from {self.model_path}")
            self.predictor = Predictor.deserialize(Path(self.model_path))
        else:
            logger.info(f"Training new {self.model_type} model")
            
            if self.model_type == 'SimpleFeedForward':
                estimator = SimpleFeedForwardEstimator(
                    prediction_length=self.prediction_length,
                    context_length=self.context_length,
                    trainer_kwargs={"max_epochs": self.max_epochs}
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            self.predictor = estimator.train(loss_dataset)
            
            # 保存模型
            os.makedirs(self.model_path, exist_ok=True)
            self.predictor.serialize(Path(self.model_path))
            logger.info(f"Model saved to {self.model_path}")
    
    def predict_log_values(self, loss_dataset: ListDataset, force_recalculate: bool = False) -> pd.DataFrame:
        """
        预测log值
        
        Args:
            loss_dataset: 损失数据集
            
        Returns:
            包含真实值和预测值的DataFrame
        """
        if self.predictor is None:
            raise ValueError("Model not trained yet. Please call train() first.")
        
        # 检查结果文件是否已存在
        save_path = os.path.join(
            self.config['paths']['results_dir'],
            f'log_true_vs_pred_{self.context_length}_{self.prediction_length}_{self.max_epochs}.csv'
        )
        
        if os.path.exists(save_path) and not force_recalculate:
            logger.info(f"Loading existing log_predict_values from {save_path}")
            return pd.read_csv(save_path)
        
        logger.info("Calculating new log_predict_values values...")
        
        all_records = []
        
        for ts_idx, ts_entry in tqdm(enumerate(loss_dataset), 
                                     total=len(loss_dataset), 
                                     desc="Predicting log values"):
            target = ts_entry["target"]
            start = ts_entry["start"]
            
            for i in range(len(target) - self.context_length):
                context = target[i:i + self.context_length]
                true_value = target[i + self.context_length]
                timestamp = start + pd.Timedelta(hours=i + self.context_length)
                
                # 构造预测输入
                input_ds = ListDataset(
                    [{"start": start + pd.Timedelta(hours=i),
                      "target": context}],
                    freq=self.freq
                )
                
                forecast_it = self.predictor.predict(input_ds)
                forecast = next(iter(forecast_it))
                pred_value = forecast.mean[0]
                
                all_records.append({
                    "series_id": ts_idx,
                    "timestamp": timestamp,
                    "true": float(true_value),
                    "pred": float(pred_value)
                })
        
        df = pd.DataFrame(all_records)
        
        # 保存结果
        df.to_csv(save_path, index=False)
        logger.info(f"Log predictions saved to {save_path}")
        
        return df
    
    def calculate_yita(self, mean_data_df: pd.DataFrame, 
                      log_data_df: pd.DataFrame, force_recalculate: bool = False) -> pd.DataFrame:
        """
        计算yita值
        
        Args:
            mean_data_df: mean model的预测数据
            log_data_df: log model的预测数据
            
        Returns:
            yita值DataFrame
        """
        # 检查结果文件是否已存在
        save_path = os.path.join(
            self.config['paths']['results_dir'],
            f'all_yita_array_{self.context_length}_{self.prediction_length}_{self.max_epochs}.csv'
        )
        
        if os.path.exists(save_path) and not force_recalculate:
            logger.info(f"Loading existing yita values from {save_path}")
            return pd.read_csv(save_path)
        
        logger.info("Calculating new yita values...")

        all_yita = []
        
        for series_id, group_df in log_data_df.groupby("series_id"):
            preds = group_df["pred"].values
            df_subset = mean_data_df[mean_data_df["series_id"] == series_id].iloc[self.context_length:].copy()
            error_array = (df_subset["mean_true"] - df_subset["mean_pred"]).to_numpy(dtype=float)
            yita = (error_array / np.sqrt(np.exp(preds)))
            all_yita.append(yita)
            logger.debug(f'Series {series_id}: yita length = {len(yita)}')
        
        df_yita = pd.DataFrame(all_yita)
        
        # 保存结果
        df_yita.to_csv(save_path,  mode='a', index=False, header=False)
        logger.info(f"Yita values saved to {save_path}")
        
        return df_yita
    
    def load_predictor(self) -> None:
        """加载已训练的模型"""
        if os.path.exists(self.model_path):
            self.predictor = Predictor.deserialize(Path(self.model_path))
            logger.info(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")