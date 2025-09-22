import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import lightning.pytorch as L

# GluonTS导入
from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.distributions import StudentTOutput
from gluonts.torch.distributions.distribution_output import DistributionOutput
from gluonts.model.forecast_generator import DistributionForecastGenerator


class ResidualCycleForecasting(nn.Module):
    """Residual Cycle Forecasting (RCF) 核心模块"""
    def __init__(self, cycle_length: int, d_model: int):
        super().__init__()
        self.cycle_length = cycle_length
        self.d_model = d_model
        
        # 可学习的循环周期 Q ∈ R^(W×D)，初始化为零
        self.learnable_cycles = nn.Parameter(
            torch.zeros(cycle_length, d_model),
            requires_grad=True
        )
        
    def forward(self, x: torch.Tensor, cycle_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape
        
        # 根据周期索引提取对应的周期分量
        cycle_components = self.learnable_cycles[cycle_indices.long()]
        
        # 计算残差：原始信号 - 周期分量
        residuals = x - cycle_components
        
        return cycle_components, residuals


class CycleNetModel(nn.Module):
    """CycleNet主干网络 - 改进版，输出分布参数"""
    def __init__(
        self,
        d_model: int,
        cycle_length: int,
        prediction_length: int,
        context_length: int,
        backbone_type: str = "linear",
        hidden_dim: int = 512,
        distr_output: DistributionOutput = StudentTOutput(),
    ):
        super().__init__()
        
        self.d_model = int(d_model)
        self.cycle_length = int(cycle_length)
        self.prediction_length = int(prediction_length)
        self.context_length = int(context_length)
        self.backbone_type = str(backbone_type)
        self.distr_output = distr_output
        
        hidden_dim = int(hidden_dim)
        
        # RCF 模块
        self.rcf = ResidualCycleForecasting(self.cycle_length, self.d_model)
        
        # 背景网络 - 输出分布参数
        args_dim = self.distr_output.args_dim
        self.output_dim = sum(args_dim.values())
        
        if backbone_type == "linear":
            self.backbone = nn.Linear(self.context_length, self.prediction_length * self.output_dim)
        elif backbone_type == "mlp":
            self.backbone = nn.Sequential(
                nn.Linear(self.context_length, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.prediction_length * self.output_dim)
            )
        else:
            raise ValueError("backbone_type must be 'linear' or 'mlp'")
    
    def generate_cycle_indices(self, batch_size: int, seq_len: int, device: torch.device, start_idx: int = 0) -> torch.Tensor:
        indices = torch.arange(start_idx, start_idx + seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        cycle_indices = indices % self.cycle_length
        return cycle_indices
    
    def forward(self, past_target: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # 处理输入维度
        if isinstance(past_target, np.ndarray):
            past_target = torch.from_numpy(past_target).float()
        
        if hasattr(self, 'device'):
            past_target = past_target.to(self.device)
        elif next(self.parameters()).is_cuda:
            past_target = past_target.cuda()
        
        if past_target.dim() == 2:
            past_target = past_target.unsqueeze(-1)
        
        batch_size, context_length, d_model = past_target.shape
        device = past_target.device
        
        # 生成历史数据的周期索引
        past_cycle_indices = self.generate_cycle_indices(batch_size, context_length, device)
        
        # 使用RCF分解历史数据
        past_cycle_components, past_residuals = self.rcf(past_target, past_cycle_indices)
        
        # 对每个变量独立进行预测（获取分布参数）
        predictions = []
        for d in range(d_model):
            residual_d = past_residuals[:, :, d]  # [batch_size, context_length]
            output_d = self.backbone(residual_d)  # [batch_size, prediction_length * output_dim]
            # Reshape to [batch_size, prediction_length, output_dim]
            output_d = output_d.view(batch_size, self.prediction_length, self.output_dim)
            predictions.append(output_d)
        
        # Stack predictions: [batch_size, prediction_length, d_model, output_dim]
        # 对于单变量，简化为 [batch_size, prediction_length, output_dim]
        if d_model == 1:
            distr_params = predictions[0]
        else:
            distr_params = torch.stack(predictions, dim=2)
        
        # 生成未来的周期索引并获取周期分量（用于后处理，但不直接加到分布参数上）
        future_cycle_indices = self.generate_cycle_indices(
            batch_size, self.prediction_length, device, start_idx=context_length
        )
        future_cycle_components = self.rcf.learnable_cycles[future_cycle_indices.long()]
        
        # 将参数分解为分布所需的各个部分
        # 对于StudentT，需要 loc, scale, df
        distr_args = self.distr_output.domain_map(*distr_params.chunk(self.output_dim, dim=-1))
        
        # 将周期分量加到loc参数上
        loc, scale, df = distr_args
        loc = loc.squeeze(-1) if loc.dim() > 2 else loc
        scale = scale.squeeze(-1) if scale.dim() > 2 else scale
        df = df.squeeze(-1) if df.dim() > 2 else df
        
        # 添加周期分量到位置参数
        if future_cycle_components.dim() == 3 and future_cycle_components.shape[-1] == 1:
            future_cycle_components = future_cycle_components.squeeze(-1)
        loc = loc + future_cycle_components
        
        return (df, loc, scale)


class CycleNetLightningModule(L.LightningModule):
    """CycleNet Lightning模块 - 使用分布输出"""
    
    def __init__(
        self,
        d_model: int,
        cycle_length: int,
        prediction_length: int,
        context_length: int,
        backbone_type: str = "linear",
        hidden_dim: int = 512,
        lr: float = 1e-3,
        distr_output: DistributionOutput = StudentTOutput(),
    ):
        super().__init__()
        
        # 保存超参数
        self.save_hyperparameters(ignore=['distr_output'])
        
        # 模型参数
        self.d_model = d_model
        self.cycle_length = cycle_length
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.backbone_type = backbone_type
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.distr_output = distr_output
        
        # 创建模型
        self.model = CycleNetModel(
            d_model=d_model,
            cycle_length=cycle_length,
            prediction_length=prediction_length,
            context_length=context_length,
            backbone_type=backbone_type,
            hidden_dim=hidden_dim,
            distr_output=distr_output,
        )
        
    def forward(self, past_target: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.model(past_target)
    
    def training_step(self, batch, batch_idx):
        past_target = batch["past_target"]
        future_target = batch["future_target"]
        
        # 转换为torch张量（如果是numpy数组）
        if isinstance(past_target, np.ndarray):
            past_target = torch.from_numpy(past_target).float()
        if isinstance(future_target, np.ndarray):
            future_target = torch.from_numpy(future_target).float()
        
        # 确保数据在正确的设备上
        past_target = past_target.to(self.device)
        future_target = future_target.to(self.device)
        
        # 获取分布参数
        distr_args = self(past_target)
        
        # 创建分布
        distr = self.distr_output.distribution(distr_args)
        
        # 计算负对数似然损失
        loss = -distr.log_prob(future_target).mean()
        
        batch_size = past_target.shape[0]
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        past_target = batch["past_target"]
        future_target = batch["future_target"]
        
        # 转换为torch张量（如果是numpy数组）
        if isinstance(past_target, np.ndarray):
            past_target = torch.from_numpy(past_target).float()
        if isinstance(future_target, np.ndarray):
            future_target = torch.from_numpy(future_target).float()
        
        # 确保数据在正确的设备上
        past_target = past_target.to(self.device)
        future_target = future_target.to(self.device)
        
        # 获取分布参数
        distr_args = self(past_target)
        
        # 创建分布
        distr = self.distr_output.distribution(distr_args)
        
        # 计算负对数似然损失
        loss = -distr.log_prob(future_target).mean()
        
        batch_size = past_target.shape[0]
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class CycleNetPredictionNetwork(nn.Module):
    """预测网络包装器 - 处理分布输出"""
    def __init__(self, model, distr_output):
        super().__init__()
        self.model = model
        self.distr_output = distr_output
    
    def forward(self, past_target, past_observed_values=None):
        # 获取分布参数
        distr_args = self.model(past_target)
        
        # 创建分布
        distr = self.distr_output.distribution(distr_args)
        
        # 返回分布（DistributionForecastGenerator会处理采样）
        return distr, distr_args[-1], distr_args[-2] if len(distr_args) > 2 else None


class CycleNetEstimator(PyTorchLightningEstimator):
    """CycleNet估计器 - 标准GluonTS实现"""
    
    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        cycle_length: int,
        freq: str = "H",
        backbone_type: str = "linear",
        hidden_dim: int = 512,
        lr: float = 1e-3,
        distr_output: DistributionOutput = StudentTOutput(),
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # 确保参数类型正确
        self.prediction_length = int(prediction_length)
        self.context_length = int(context_length)
        self.cycle_length = int(cycle_length)
        self.backbone_type = str(backbone_type)
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.num_batches_per_epoch = int(num_batches_per_epoch)
        self.freq = str(freq)
        self.distr_output = distr_output
        
        # 设置默认的trainer参数
        default_trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": 10.0,
            "accelerator": "auto",
            "devices": "auto",
        }
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        
        super().__init__(trainer_kwargs=default_trainer_kwargs)
    
    def create_transformation(self):
        """创建数据转换管道"""
        from gluonts.transform import (
            Chain,
            SelectFields,
            AddObservedValuesIndicator,
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
    
    def create_lightning_module(self):
        """创建Lightning模块"""
        return CycleNetLightningModule(
            d_model=1,  # 单变量
            cycle_length=self.cycle_length,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            backbone_type=self.backbone_type,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            distr_output=self.distr_output,
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
        transformed_data = transformation.apply(data, is_train=False)
        
        return as_stacked_batches(
            transformed_data,
            batch_size=self.batch_size,
        )
    
    def create_predictor(self, transformation, trained_network) -> PyTorchPredictor:
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
        
        # 包装网络以处理分布输出
        prediction_network = CycleNetPredictionNetwork(
            trained_network, 
            self.distr_output
        )
        
        # 使用标准的DistributionForecastGenerator
        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            prediction_net=prediction_network,
            prediction_length=self.prediction_length,
            input_names=["past_target", "past_observed_values"],
            batch_size=self.batch_size,
            forecast_generator=DistributionForecastGenerator(self.distr_output),
            device="auto",
        )