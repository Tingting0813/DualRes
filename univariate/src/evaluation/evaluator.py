"""
评估和可视化模块
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Any, List, Tuple, Optional
from gluonts.evaluation import Evaluator
from gluonts.model.forecast import SampleForecast
from statsmodels.graphics.tsaplots import plot_acf
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化评估器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.evaluator = Evaluator()
        
    def evaluate_predictions(self, forecasts: List, tss: List, 
                           num_series: Optional[int] = None) -> Tuple[Dict, pd.DataFrame]:
        """
        评估预测结果
        
        Args:
            forecasts: 预测结果列表
            tss: 真实时间序列列表
            num_series: 时间序列数量
            
        Returns:
            聚合指标和项目指标
        """
        agg_metrics, item_metrics = self.evaluator(tss, forecasts, num_series=num_series)
        
        logger.info("Evaluation metrics:")
        for key in ["Coverage[0.9]", "MSE", "RMSE","NRMSE", "ND", "MAE_Coverage", "mean_wQuantileLoss"]:
            if key in agg_metrics:
                logger.info(f"  {key}: {agg_metrics[key]:.4f}")
        
        return agg_metrics, item_metrics
    
    def evaluate_sample_forecast(self, final_res: np.ndarray, 
                                tss: List, 
                                prediction_length: int) -> Tuple[Dict, pd.DataFrame]:
        """
        评估采样预测结果
        
        Args:
            final_res: 最终预测结果 (num_samples, num_series, prediction_length)
            tts: 预测值
            prediction_length: 预测长度
            
        Returns:
            评估指标
        """
        num_samples, num_series, _ = final_res.shape
        
        # 构造SampleForecast对象
        forecast_list = []
        for i in range(num_series):
            samples_i = final_res[:, i, :]  # shape: (num_samples, prediction_length)
            
            # 获取预测起始时间
            ts = tss[i]
            start_date = ts.index[-prediction_length]
            
            # 构造SampleForecast
            sf = SampleForecast(
                samples=samples_i,
                start_date=start_date,
                item_id=f"series_{i}"
            )
            forecast_list.append(sf)
        
        # 评估
        return self.evaluate_predictions(forecast_list, tss, num_series=num_series)
    
    def plot_prediction(self, forecasts: List, tss: List, 
                       series_idx: int = 0, 
                       context_length: int = 300,
                       save_path: Optional[str] = None) -> None:
        """
        绘制预测结果
        
        Args:
            forecasts: 预测结果
            tss: 真实时间序列
            series_idx: 要绘制的时间序列索引
            context_length: 显示的上下文长度
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 5))
        
        # 绘制真实值
        tss[series_idx][-context_length:].plot(label="True values")
        
        # 绘制预测值
        pred_index = pd.date_range(
            start=forecasts[series_idx].start_date.to_timestamp(),
            periods=len(forecasts[series_idx].mean),
            freq=self.config['dataset']['freq']
        )
        forecast_series = pd.Series(forecasts[series_idx].mean, index=pred_index)
        forecast_series.plot(label="Prediction", color="g", marker='x')
        
        plt.legend()
        plt.title(f"Prediction for Time Series {series_idx}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_sliding_prediction(self, predictions: np.ndarray, 
                              true_data: List[Dict],
                              series_idx: int = 0,
                              window_size: int = 200,
                              context_length: int = 336,
                              save_path: Optional[str] = None) -> None:
        """
        绘制滑动窗口预测结果
        
        Args:
            predictions: 预测数组
            true_data: 真实数据
            series_idx: 时间序列索引
            window_size: 显示窗口大小
            context_length: 上下文长度
            save_path: 保存路径
        """
        true_series = true_data[series_idx]["target"]
        start_time = true_data[series_idx]["start"].to_timestamp()
        
        pred_values = np.array(predictions[series_idx][0:window_size])
        true_values = true_series[context_length:context_length+window_size]
        
        # 时间索引
        true_index = pd.date_range(start=start_time + pd.Timedelta(hours=context_length),
                                  periods=len(true_values), 
                                  freq=self.config['dataset']['freq'])
        pred_index = pd.date_range(start=start_time + pd.Timedelta(hours=context_length),
                                  periods=len(pred_values), 
                                  freq=self.config['dataset']['freq'])
        
        # 画图
        plt.figure(figsize=(12, 5))
        plt.plot(true_index, true_values, label="True values", color="black")
        plt.plot(pred_index, pred_values, label="Predicted values", color="green", linestyle="--")
        plt.title(f"Sliding prediction vs. true series (Series #{series_idx})")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_acf(self, data: pd.DataFrame, 
                start_id: int = 0,
                n_series: int = 8,
                lags: int = 100,
                save_path: Optional[str] = None) -> None:
        """
        绘制ACF图
        
        Args:
            data: 包含log_mse的DataFrame
            start_id: 起始序列ID
            n_series: 要绘制的序列数量
            lags: 滞后数
            save_path: 保存路径
        """
        n_rows, n_cols = 2, 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 8))
        
        for i in range(n_series):
            series_id = start_id + i
            ax = axes[i // n_cols, i % n_cols]
            
            series_data = data[data["series_id"] == series_id].sort_values("timestamp")["log_mse"]
            plot_acf(series_data, lags=lags, ax=ax)
            ax.set_title(f"ACF: Series {series_id}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ACF plot saved to {save_path}")
        
        plt.show()
    
    def plot_final_forecast(self, final_res: np.ndarray,
                           test_data: List,
                           series_idx: int = 0,
                           prediction_length: int = 48,
                           display_length: int = 144,
                           save_path: Optional[str] = None) -> None:
        """
        绘制最终预测结果（包含置信区间）
        
        Args:
            final_res: 最终预测结果
            test_data: 测试数据
            series_idx: 时间序列索引
            prediction_length: 预测长度
            display_length: 显示长度
            save_path: 保存路径
        """
        # 提取数据
        samples = final_res[:, series_idx, :]
        true_entry = test_data[series_idx]
        true_target = true_entry["target"]
        
        # 计算统计量
        mean = samples.mean(axis=0)
        p5 = np.percentile(samples, 5, axis=0)
        p95 = np.percentile(samples, 95, axis=0)
        
        # 准备时间索引
        start_time = true_entry["start"].to_timestamp()
        full_index = pd.date_range(start=start_time, 
                                  periods=len(true_target), 
                                  freq=self.config['dataset']['freq'])
        
        x_time = full_index[-display_length:]
        x_pred = full_index[-prediction_length:]
        true_values = true_target[-display_length:]
        
        # 绘图
        plt.figure(figsize=(12, 6))
        plt.plot(x_time, true_values, color='#004773', label='True values')
        plt.plot(x_pred, mean, color='#ce0300', marker='.', label='Prediction mean')
        plt.fill_between(x_pred, p5, p95, alpha=0.40, label="90% Interval", color='#fa918e')
        
        # 添加预测区间分隔线
        plt.axvline(full_index[-prediction_length], color='black', linestyle='--', linewidth=0.5)
        plt.axvspan(full_index[-prediction_length], full_index[-1], color='grey', alpha=0.1)
        
        # 设置坐标轴
        plt.xlim(full_index[-display_length], full_index[-1])
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.xticks(rotation=45)
        
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(f'Forecast for Time Series {series_idx}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Final forecast plot saved to {save_path}")
        
        plt.show()