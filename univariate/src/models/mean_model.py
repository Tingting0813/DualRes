"""
Mean Model相关功能
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from gluonts.model.predictor import Predictor
from gluonts.evaluation import make_evaluation_predictions
from gluonts.torch.model.patch_tst import PatchTSTEstimator
from gluonts.torch.model.d_linear import DLinearEstimator
from gluonts.torch.model.timemixer import TimeMixerEstimator
from gluonts.torch import DeepAREstimator
from gluonts.dataset.common import ListDataset
from pandas.tseries.offsets import BDay
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class MeanModel:
    """Mean Model类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化Mean Model
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.mean_config = config['mean_model']
        self.context_length = self.mean_config['context_length']
        self.prediction_length = self.mean_config['prediction_length']
        self.max_epochs = self.mean_config['max_epochs']
        self.model_type = self.mean_config.get('model_type')
        self.freq = config['dataset']['freq']
        self.predictor = None
        
        # 模型保存路径
        self.model_name = f'mean_{self.model_type}_{self.context_length}_{self.prediction_length}_{self.max_epochs}'
        self.model_path = os.path.join(config['paths']['models_dir'], self.model_name)
        logger.info(f'wangme test {self.model_name} and {self.model_path}')
        
    def _create_estimator(self):
        """创建估计器"""
        if self.model_type == 'PatchTST':
            return PatchTSTEstimator(
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                trainer_kwargs={"max_epochs": self.max_epochs},
                patch_len=self.mean_config.get('patch_len', 16)
            )
        elif self.model_type == 'DLinear':
            return DLinearEstimator(
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                trainer_kwargs={"max_epochs": self.max_epochs}
            )
        elif self.model_type == 'DeepAR':
            return DeepAREstimator(
                prediction_length=self.prediction_length,
                freq=self.freq,
                context_length=self.context_length,
                trainer_kwargs={"max_epochs": self.max_epochs}
            )
        elif self.model_type == 'TimeMixer':
            return TimeMixerEstimator(
                prediction_length=self.prediction_length,
                # freq=self.freq,
                context_length=self.context_length,
                trainer_kwargs={"max_epochs": self.max_epochs}
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, train_data: List[Dict], force_retrain: bool = False) -> None:
        """
        训练模型
        
        Args:
            train_data: 训练数据
            force_retrain: 是否强制重新训练
        """
        if os.path.exists(self.model_path) and not force_retrain:
            logger.info(f"Loading existing model from {self.model_path}")
            self.predictor = Predictor.deserialize(Path(self.model_path))
        else:
            logger.info(f"Training new {self.model_type} model")
            estimator = self._create_estimator()
            self.predictor = estimator.train(train_data, cache_data=True)
            
            # 保存模型
            os.makedirs(self.model_path, exist_ok=True)
            self.predictor.serialize(Path(self.model_path))
            logger.info(f"Model saved to {self.model_path}")
    
    def predict(self, test_data: List[Dict], num_samples: int = 1) -> Tuple[List, List]:
        """
        进行预测
        
        Args:
            test_data: 测试数据
            num_samples: 采样数量
            
        Returns:
            预测结果和时间序列
        """
        self.load_predictor()
        if self.predictor is None:
            raise ValueError("Model not trained yet. Please call train() first.")
        
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_data,
            predictor=self.predictor,
            num_samples=num_samples
        )
        
        forecasts = list(forecast_it)
        tss = list(ts_it)
        
        return forecasts, tss
    
    def sliding_window_prediction(self, data: List[Dict]) -> np.ndarray:
        """
        滑动窗口预测
        
        Args:
            data: 输入数据
            
        Returns:
            预测结果数组
        """
        if self.predictor is None:
            raise ValueError("Model not trained yet. Please call train() first.")
        
        all_predictions = []
        
        for ts_idx, ts_entry in tqdm(enumerate(data), total=len(data), desc="Sliding window prediction"):
            target = ts_entry["target"]
            start = ts_entry["start"]
            
            predictions = []
            for i in range(len(target) - self.context_length):
                context = target[i:i+self.context_length]
                
                if self.freq == 'h':
                    set_start = start + pd.Timedelta(hours=i)
                elif self.freq == 'B':   
                    set_start = start + BDay(i)
                else:
                    raise ValueError(f"Invalid freq set: {self.freq}")
                input_ds = ListDataset(
                        [{"start": set_start,
                        "target": context}],
                        freq=self.freq
                    )
                forecast_it = self.predictor.predict(input_ds)
                forecast = next(iter(forecast_it))
                
                # 保存预测的第1步
                predictions.append(forecast.mean[0])
            
            all_predictions.append(predictions)
        
        return np.array(all_predictions, dtype=object)
    
    def load_predictor(self) -> None:
        """加载已训练的模型"""
        if os.path.exists(self.model_path):
            self.predictor = Predictor.deserialize(Path(self.model_path))
            logger.info(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")
    
    # def evaluate(self, forecasts: List, tss: List) -> Dict:
    #     """
    #     评估模型性能
        
    #     Args:
    #         forecasts: 预测结果
    #         tss: 真实时间序列
            
    #     Returns:
    #         评估指标
    #     """
    #     from gluonts.evaluation import Evaluator
        
    #     evaluator = Evaluator()
    #     agg_metrics, item_metrics = evaluator(tss, forecasts)
        
    #     return agg_metrics, item_metrics