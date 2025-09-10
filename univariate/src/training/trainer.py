"""
训练管理器
"""
import os
import numpy as np
import pandas as pd
import torch
import copy
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
import logging

logger = logging.getLogger(__name__)


class Trainer:
    """统一的训练管理器"""
    
    def __init__(self, config: Dict[str, Any], data_loader, mean_model, log_model):
        """
        初始化训练管理器
        
        Args:
            config: 配置字典
            data_loader: 数据加载器
            mean_model: Mean模型
            log_model: Log模型
        """
        self.config = config
        self.data_loader = data_loader
        self.mean_model = mean_model
        self.log_model = log_model
        self.num_samples = config['sampling']['num_samples']
        
    def train_mean_model(self, force_retrain: bool = False) -> None:
        """训练Mean Model"""
        logger.info("Starting Mean Model training...")
        
        # 确保数据已加载
        if self.data_loader.train_data is None:
            self.data_loader.load_dataset()
        
        # 训练模型
        self.mean_model.train(self.data_loader.train_data, force_retrain)
        logger.info("Mean Model training completed")
    
    def generate_mean_predictions(self) -> np.ndarray:
        """生成Mean Model的滑动窗口预测"""
        save_path = os.path.join(self.config['paths']['predictions_dir'], 
                                 "mean_predictions_value.npy")
        
        if os.path.exists(save_path) and not self.config.get('force_regenerate', False):
            logger.info(f"Loading existing mean predictions from {save_path}")
            return np.load(save_path, allow_pickle=True)
        
        logger.info("Generating mean model predictions...")
        predictions = self.mean_model.sliding_window_prediction(self.data_loader.train_data)
        
        # 保存预测结果
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, predictions)
        logger.info(f"Mean predictions saved to {save_path}")
        
        return predictions
    
    def prepare_log_model_data(self, mean_predictions: np.ndarray) -> ListDataset:
        """准备Log Model的训练数据"""
        logger.info("Preparing log model training data...")
        
        # 计算log-MSE
        true_values = [entry["target"] for entry in self.data_loader.train_data]
        timestamps = []
        
        for i, entry in enumerate(self.data_loader.train_data):
            start_time = entry["start"].to_timestamp()
            ts = pd.date_range(
                start=start_time + pd.Timedelta(hours=self.mean_model.context_length),
                periods=len(mean_predictions[i]),
                freq="H"
            )
            timestamps.append(ts)
        
        # 计算log-MSE损失
        log_mse_df = self.log_model.calculate_log_mse(
            true_values, 
            mean_predictions,
            self.mean_model.context_length,
            self.config['dataset']['name'],
            timestamps
        )
        
        # 创建损失数据集
        loss_dataset = self.data_loader.create_loss_dataset(log_mse_df)
        
        return loss_dataset
    
    def train_log_model(self, loss_dataset: ListDataset, force_retrain: bool = False) -> None:
        """训练Log Model"""
        logger.info("Starting Log Model training...")
        self.log_model.train(loss_dataset, force_retrain)
        logger.info("Log Model training completed")
    
    def resample_yita(self, df_yita: pd.DataFrame) -> np.ndarray:
        """
        重采样yita值
        
        Args:
            df_yita: yita值DataFrame
            
        Returns:
            重采样后的数组
        """
        logger.info(f"Resampling yita values ({self.num_samples} samples)...")
        
        yita_results = []
        
        for _ in range(self.num_samples):
            sample = []
            for _, row in df_yita.iterrows():
                yita_values = row.dropna().to_numpy(dtype=float)
                s = np.random.choice(yita_values, 
                                   size=self.mean_model.prediction_length, 
                                   replace=True)
                sample.append(s)
            yita_results.append(np.vstack(sample))
        
        all_samples = np.stack(yita_results)
        logger.info(f"Resampled yita shape: {all_samples.shape}")
        
        return all_samples
    
    def tamed_exp(self, x: np.ndarray, t: float = 7) -> np.ndarray:
        """Tamed exponential function"""
        return np.exp(x)  # 可以根据需要修改为更复杂的版本
    
    def rolling_forecast(self, all_samples: np.ndarray) -> np.ndarray:
        """
        执行滚动预测
        
        Args:
            all_samples: 重采样的yita值
            
        Returns:
            最终预测结果
        """
        logger.info("Starting rolling forecast...")
        
        # 准备初始上下文
        all_series = []
        all_starts = []
        
        # 提取初始上下文 (mean model)
        for entry in self.data_loader.test_data:
            full_target = entry["target"]
            start = entry["start"]
            start_offset = len(full_target) - self.mean_model.context_length - self.mean_model.prediction_length
            ctx = list(full_target[-(self.mean_model.context_length + self.mean_model.prediction_length):-self.mean_model.prediction_length])
            all_series.append(ctx)
            all_starts.append(start + pd.Timedelta(hours=start_offset))
        
        # 准备log model的初始上下文
        log_temp_series = []
        log_temp_starts = []
        
        for entry in self.data_loader.test_data:
            this_ts_log_mse = []
            for i in range(self.log_model.context_length):
                full_target = entry["target"]
                start = entry["start"]
                start_offset_log = len(full_target) - self.mean_model.context_length - self.mean_model.prediction_length - self.log_model.context_length + i
                ctx = list(full_target[-(self.mean_model.context_length + self.mean_model.prediction_length + self.log_model.context_length - i):-(self.mean_model.prediction_length + self.log_model.context_length - i)])
                
                input_mean_ds = ListDataset(
                    [{"start": start + pd.Timedelta(hours=start_offset_log),
                      "target": ctx}],
                    freq=self.config['dataset']['freq']
                )
                
                forecast_it = self.mean_model.predictor.predict(input_mean_ds)
                forecast = next(iter(forecast_it))
                mean_pred = torch.tensor(float(forecast.mean[0]))
                true_v = torch.tensor(full_target[-(self.mean_model.prediction_length + self.log_model.context_length - i)])
                
                # 计算log MSE
                this_ts_log_mse.append(torch.log((true_v - mean_pred).pow(2) + 1e-10).item())
            
            log_temp_starts.append(start + pd.Timedelta(hours=len(entry["target"]) - self.mean_model.prediction_length - self.log_model.context_length))
            log_temp_series.append(this_ts_log_mse)
        
        # 执行滚动预测
        final_res = []
        
        for sample_idx in tqdm(range(self.num_samples), desc="Sampling Loop"):
            rolling_preds = [[] for _ in range(len(all_series))]
            all_series_copy = copy.deepcopy(all_series)
            log_temp_series_copy = copy.deepcopy(log_temp_series)
            
            for step in tqdm(range(self.mean_model.prediction_length), 
                           desc=f"Rolling Forecast {sample_idx+1}", 
                           leave=False):
                
                # Mean model预测
                m_input_ds = ListDataset(
                    [
                        {
                            FieldName.START: all_starts[i] + pd.Timedelta(hours=step),
                            FieldName.TARGET: all_series_copy[i][-self.mean_model.context_length:],
                        }
                        for i in range(len(all_series_copy))
                    ],
                    freq=self.config['dataset']['freq']
                )
                forecast_it = self.mean_model.predictor.predict(m_input_ds)
                forecasts = list(forecast_it)
                
                # Log model预测
                l_input_ds = ListDataset(
                    [
                        {
                            FieldName.START: log_temp_starts[i] + pd.Timedelta(hours=step),
                            FieldName.TARGET: log_temp_series_copy[i][-self.log_model.context_length:],
                        }
                        for i in range(len(log_temp_series_copy))
                    ],
                    freq=self.config['dataset']['freq']
                )
                forecast_it_l = self.log_model.predictor.predict(l_input_ds)
                forecasts_l = list(forecast_it_l)
                
                # 组合预测
                for i, forecast in enumerate(forecasts):
                    mean_pred = float(forecast.mean[0])
                    log_pred = float(forecasts_l[i].mean[0])
                    
                    # 计算最终预测
                    err = self.tamed_exp(log_pred / 2.0) * all_samples[sample_idx, i, step]
                    real_pred = mean_pred + err
                    real_log = np.log(np.square(err) + 1e-10)
                    
                    # 更新上下文
                    rolling_preds[i].append(real_pred)
                    all_series_copy[i].append(real_pred)
                    log_temp_series_copy[i].append(real_log)
            
            final_res.append(rolling_preds)
        
        final_res_array = np.array(final_res)
        
        # 保存结果
        save_path = os.path.join(self.config['paths']['predictions_dir'], 
                                 "final_res_tensor.npy")
        np.save(save_path, final_res_array)
        logger.info(f"Final predictions saved to {save_path}")
        
        return final_res_array