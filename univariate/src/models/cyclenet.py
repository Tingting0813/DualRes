import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
import pytorch_lightning as pl
import numpy as np

# GluonTS 0.16.2 正确导入
from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.distributions import StudentTOutput

# 在GluonTS 0.15+中，损失函数被移除，我们需要自己实现
import torch

class DistributionLoss:
    """分布损失基类"""
    def __call__(self, input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

class NegativeLogLikelihood(DistributionLoss):
    """负对数似然损失"""
    def __init__(self, beta: float = 0.0):
        self.beta = beta
    
    def __call__(self, input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
        nll = -input.log_prob(target)
        if self.beta > 0.0:
            variance = input.variance
            nll = nll * (variance.detach() ** self.beta)
        return nll.mean()


class ResidualCycleForecasting(nn.Module):
    """Residual Cycle Forecasting (RCF) 核心模块"""
    def __init__(self, cycle_length: int, d_model: int):
        super().__init__()
        self.cycle_length = cycle_length  # W: 周期长度
        self.d_model = d_model  # D: 变量/通道数
        
        # 可学习的循环周期 Q ∈ R^(W×D)，初始化为零
        self.learnable_cycles = nn.Parameter(
            torch.zeros(cycle_length, d_model),  # [W, D]
            requires_grad=True
        )
        
    def forward(self, x: torch.Tensor, cycle_indices: torch.Tensor) -> tuple:
        """
        前向传播
        Args:
            x: 输入时间序列 [batch_size, seq_len, d_model]
            cycle_indices: 周期索引 [batch_size, seq_len] (t mod W)
        Returns:
            cycle_components: 周期分量 [batch_size, seq_len, d_model]
            residuals: 残差分量 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 根据周期索引提取对应的周期分量
        # cycle_indices: [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        cycle_components = self.learnable_cycles[cycle_indices.long()]  # [batch_size, seq_len, d_model]
        
        # 计算残差：原始信号 - 周期分量
        residuals = x - cycle_components  # [batch_size, seq_len, d_model]
        
        return cycle_components, residuals


class CycleNetBackbone(nn.Module):
    """CycleNet主干网络"""
    def __init__(
        self,
        d_model: int,
        cycle_length: int,
        prediction_length: int,
        context_length: int,
        backbone_type: str = "linear",  # "linear" or "mlp"
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.cycle_length = cycle_length
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.backbone_type = backbone_type
        
        # RCF 模块
        self.rcf = ResidualCycleForecasting(cycle_length, d_model)
        
        # 背景网络: Linear 或 MLP
        if backbone_type == "linear":
            # 单层线性模型
            self.backbone = nn.Linear(context_length, prediction_length)
        elif backbone_type == "mlp":
            # 双层MLP模型
            self.backbone = nn.Sequential(
                nn.Linear(context_length, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, prediction_length)
            )
        else:
            raise ValueError("backbone_type must be 'linear' or 'mlp'")
    
    def generate_cycle_indices(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """生成周期索引 (t mod W)"""
        # 为简化实现，假设时间序列是连续的
        # 实际使用中，应该根据真实时间戳计算
        indices = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        cycle_indices = indices % self.cycle_length
        return cycle_indices
    
    def forward(self, past_target: torch.Tensor, future_cycle_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            past_target: [batch_size, context_length, d_model] 
            future_cycle_indices: [batch_size, prediction_length] (可选，预测时的周期索引)
        Returns:
            predictions: [batch_size, prediction_length, d_model]
        """
        batch_size, context_length, d_model = past_target.shape
        device = past_target.device
        
        # 生成历史数据的周期索引
        past_cycle_indices = self.generate_cycle_indices(batch_size, context_length, device)
        
        # 使用RCF分解历史数据
        past_cycle_components, past_residuals = self.rcf(past_target, past_cycle_indices)
        
        # 对每个变量独立进行预测
        predictions = []
        for d in range(d_model):
            # 取出第d个变量的残差 [batch_size, context_length]
            residual_d = past_residuals[:, :, d]  
            
            # 使用backbone预测未来残差 [batch_size, prediction_length]
            future_residual_d = self.backbone(residual_d)
            predictions.append(future_residual_d)
        
        # 堆叠所有变量的预测 [batch_size, prediction_length, d_model]
        future_residuals = torch.stack(predictions, dim=-1)
        
        # 生成未来的周期索引
        if future_cycle_indices is None:
            # 假设未来时间戳是历史数据的延续
            future_start_idx = context_length
            future_indices = torch.arange(
                future_start_idx, 
                future_start_idx + self.prediction_length, 
                device=device
            ).unsqueeze(0).repeat(batch_size, 1)
            future_cycle_indices = future_indices % self.cycle_length
        
        # 获取未来的周期分量
        future_cycle_components = self.rcf.learnable_cycles[future_cycle_indices.long()]
        
        # 最终预测 = 未来周期分量 + 预测的残差
        final_predictions = future_cycle_components + future_residuals
        
        return final_predictions


class CycleNetLightningModule(pl.LightningModule):
    """CycleNet PyTorch Lightning模块"""
    def __init__(
        self,
        d_model: int,
        cycle_length: int,
        prediction_length: int,
        context_length: int,
        backbone_type: str = "linear",
        hidden_dim: int = 512,
        lr: float = 1e-3,
        loss: Optional[DistributionLoss] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = CycleNetBackbone(
            d_model=d_model,
            cycle_length=cycle_length,
            prediction_length=prediction_length,
            context_length=context_length,
            backbone_type=backbone_type,
            hidden_dim=hidden_dim,
        )
        
        self.loss = loss or NegativeLogLikelihood()
        self.lr = lr
        
        # 分布输出层
        self.distr_output = StudentTOutput()
        
    def forward(self, past_target: torch.Tensor, future_cycle_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(past_target, future_cycle_indices)
    
    def training_step(self, batch, batch_idx):
        past_target = batch["past_target"]
        future_target = batch["future_target"]
        
        # 前向传播
        predictions = self(past_target)
        
        # 计算损失
        distr_args = self.distr_output.get_args_proj(predictions)
        distr = self.distr_output.distribution(distr_args)
        loss = self.loss(distr, future_target)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        past_target = batch["past_target"]
        future_target = batch["future_target"]
        
        predictions = self(past_target)
        distr_args = self.distr_output.get_args_proj(predictions)
        distr = self.distr_output.distribution(distr_args)
        loss = self.loss(distr, future_target)
        
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class CycleNetEstimator(PyTorchLightningEstimator):
    """CycleNet估计器"""
    
    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        cycle_length: int,  # 周期长度，需要根据数据特性设定
        freq: str = "H",  # 添加freq参数
        backbone_type: str = "linear",  # "linear" or "mlp"
        hidden_dim: int = 512,  # MLP的隐藏层维度
        lr: float = 1e-3,
        loss: Optional[DistributionLoss] = None,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # 设置默认的trainer参数
        default_trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": 10.0,
        }
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        
        # 只传递PyTorchLightningEstimator接受的参数
        super().__init__(trainer_kwargs=default_trainer_kwargs)
        
        # 保存CycleNet特有的参数
        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.cycle_length = cycle_length
        self.backbone_type = backbone_type
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.loss = loss or NegativeLogLikelihood()
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
    
    def create_transformation(self):
        """创建数据转换管道"""
        from gluonts.transform import (
            Chain,
            SelectFields,
            AddObservedValuesIndicator,
            InstanceSplitter,
            ExpectedNumInstanceSampler,
        )
        
        return Chain([
            SelectFields([
                FieldName.ITEM_ID,
                FieldName.INFO,
                FieldName.START,
                FieldName.TARGET,
            ], allow_missing=True),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
        ])
    
    def create_lightning_module(self) -> pl.LightningModule:
        return CycleNetLightningModule(
            d_model=1,  # 单变量时间序列
            cycle_length=self.cycle_length,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            backbone_type=self.backbone_type,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            loss=self.loss,
        )
    
    def create_training_data_loader(self, data, module, shuffle_buffer_length=None, **kwargs):
        """创建训练数据加载器"""
        from gluonts.dataset.loader import as_stacked_batches
        from gluonts.itertools import Cyclic
        from gluonts.transform import (
            InstanceSplitter,
            ExpectedNumInstanceSampler,
        )
        
        instance_splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_future=self.prediction_length,
            ),
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[FieldName.OBSERVED_VALUES],
        )
        
        transformation = self.create_transformation() + instance_splitter
        
        # 正确的数据流处理方式
        transformed_data = transformation.apply(data, is_train=True)
        
        return as_stacked_batches(
            Cyclic(transformed_data).stream(),
            batch_size=self.batch_size,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )
    
    def create_validation_data_loader(self, data, module, **kwargs):
        """创建验证数据加载器"""
        from gluonts.dataset.loader import as_stacked_batches
        from gluonts.transform import (
            InstanceSplitter,
            ValidationSplitSampler,
        )
        
        instance_splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=ValidationSplitSampler(
                min_future=self.prediction_length,
            ),
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[FieldName.OBSERVED_VALUES],
        )
        
        transformation = self.create_transformation() + instance_splitter
        
        # 正确的数据流处理方式
        transformed_data = transformation.apply(data, is_train=False)
        
        return as_stacked_batches(
            transformed_data,
            batch_size=self.batch_size,
        )
    
    def create_predictor(
        self, 
        transformation,
        trained_network
    ) -> PyTorchPredictor:
        from gluonts.transform import (
            InstanceSplitter,
            TestSplitSampler,
        )
        
        prediction_splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=TestSplitSampler(),
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[FieldName.OBSERVED_VALUES],
        )
        
        prediction_network = trained_network
        
        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            prediction_net=prediction_network,
            prediction_length=self.prediction_length,
            freq=self.freq,
            distr_output=StudentTOutput(),
        )


# 多变量版本的CycleNet
class MultivariateCycleNetEstimator(PyTorchLightningEstimator):
    """多变量CycleNet估计器"""
    
    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        cycle_length: int,
        num_feat_dynamic_real: int = 0,  # 动态特征数量
        backbone_type: str = "linear",
        hidden_dim: int = 512,
        lr: float = 1e-3,
        loss: Optional[DistributionLoss] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # 保存自定义参数
        self.cycle_length = cycle_length
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.backbone_type = backbone_type
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.loss = loss or NegativeLogLikelihood()
        
        trainer_kwargs = trainer_kwargs or {}
        
        # 过滤参数，只传递父类接受的参数
        parent_kwargs = {
            'prediction_length': prediction_length,
            'context_length': context_length,
            'trainer_kwargs': trainer_kwargs,
        }
        
        # 添加其他父类可能接受的标准参数
        standard_params = [
            'freq', 'distr_output', 'batch_size', 'num_batches_per_epoch',
            'train_sampler', 'validation_sampler'
        ]
        
        for param in standard_params:
            if param in kwargs:
                parent_kwargs[param] = kwargs[param]
        
        super().__init__(**parent_kwargs)
    
    def create_lightning_module(self) -> pl.LightningModule:
        # 总输入维度 = 目标变量 + 动态特征
        d_model = 1 + self.num_feat_dynamic_real
        
        return CycleNetLightningModule(
            d_model=d_model,
            cycle_length=self.cycle_length,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            backbone_type=self.backbone_type,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            loss=self.loss,
        )
    
    def create_predictor(
        self, 
        transformation,
        trained_network
    ) -> PyTorchPredictor:
        return PyTorchPredictor(
            input_transform=transformation,
            prediction_net=trained_network,
            prediction_length=self.prediction_length,
            freq=self.freq,
            distr_output=StudentTOutput(),
        )