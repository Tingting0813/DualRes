import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Tuple
import pytorch_lightning as pl
import numpy as np

# GluonTS导入
from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.distributions import StudentTOutput


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
    """CycleNet主干网络"""
    def __init__(
        self,
        d_model: int,
        cycle_length: int,
        prediction_length: int,
        context_length: int,
        backbone_type: str = "linear",
        hidden_dim: int = 512,
    ):
        super().__init__()
        
        # 强制转换为int，避免传入tensor或其他类型
        self.d_model = int(d_model)
        self.cycle_length = int(cycle_length)
        self.prediction_length = int(prediction_length)
        self.context_length = int(context_length)
        self.backbone_type = str(backbone_type)
        
        # 打印调试信息
        print(f"CycleNetModel init with:")
        print(f"  context_length: {self.context_length} (type: {type(self.context_length)})")
        print(f"  prediction_length: {self.prediction_length} (type: {type(self.prediction_length)})")
        print(f"  hidden_dim: {hidden_dim} (type: {type(hidden_dim)})")
        print(f"  backbone_type: {self.backbone_type}")
        
        # 确保hidden_dim也是int
        hidden_dim = int(hidden_dim)
        
        # RCF 模块
        self.rcf = ResidualCycleForecasting(self.cycle_length, self.d_model)
        
        # 背景网络 - 小心创建Linear层
        try:
            if backbone_type == "linear":
                print(f"Creating Linear layer: in={self.context_length}, out={self.prediction_length}")
                self.backbone = nn.Linear(self.context_length, self.prediction_length)
            elif backbone_type == "mlp":
                print(f"Creating MLP layers: {self.context_length} -> {hidden_dim} -> {self.prediction_length}")
                self.backbone = nn.Sequential(
                    nn.Linear(self.context_length, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, self.prediction_length)
                )
            else:
                raise ValueError("backbone_type must be 'linear' or 'mlp'")
            print("✅ Backbone created successfully")
        except Exception as e:
            print(f"❌ Error creating backbone: {e}")
            print(f"   context_length: {self.context_length} ({type(self.context_length)})")
            print(f"   prediction_length: {self.prediction_length} ({type(self.prediction_length)})")
            print(f"   hidden_dim: {hidden_dim} ({type(hidden_dim)})")
            raise
    
    def generate_cycle_indices(self, batch_size: int, seq_len: int, device: torch.device, start_idx: int = 0) -> torch.Tensor:
        indices = torch.arange(start_idx, start_idx + seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        cycle_indices = indices % self.cycle_length
        return cycle_indices
    
    def forward(self, past_target: torch.Tensor) -> torch.Tensor:
        # 转换为torch张量（如果是numpy数组）
        if isinstance(past_target, np.ndarray):
            past_target = torch.from_numpy(past_target).float()
        
        # 确保数据在正确的设备上
        if hasattr(self, 'device'):
            past_target = past_target.to(self.device)
        elif next(self.parameters()).is_cuda:
            past_target = past_target.cuda()
        
        # 处理输入维度：GluonTS可能传入2D或3D数据
        if past_target.dim() == 2:
            # [batch_size, seq_len] -> [batch_size, seq_len, 1]
            past_target = past_target.unsqueeze(-1)
        elif past_target.dim() == 3:
            # 已经是3D，直接使用
            pass
        else:
            raise ValueError(f"Expected 2D or 3D input, got {past_target.dim()}D")
            
        batch_size, context_length, d_model = past_target.shape
        device = past_target.device
        
        # 验证输入长度与模型期望是否匹配
        if context_length != self.context_length:
            raise ValueError(f"Input context length {context_length} doesn't match model context length {self.context_length}")
        
        # 生成历史数据的周期索引
        past_cycle_indices = self.generate_cycle_indices(batch_size, context_length, device)
        
        # 使用RCF分解历史数据
        past_cycle_components, past_residuals = self.rcf(past_target, past_cycle_indices)
        
        # 对每个变量独立进行预测
        predictions = []
        for d in range(d_model):
            residual_d = past_residuals[:, :, d]  # [batch_size, context_length]
            future_residual_d = self.backbone(residual_d)  # [batch_size, prediction_length]
            predictions.append(future_residual_d)
        
        future_residuals = torch.stack(predictions, dim=-1)  # [batch_size, prediction_length, d_model]
        
        # 生成未来的周期索引
        future_cycle_indices = self.generate_cycle_indices(
            batch_size, self.prediction_length, device, start_idx=context_length
        )
        
        # 获取未来的周期分量
        future_cycle_components = self.rcf.learnable_cycles[future_cycle_indices.long()]
        
        # 最终预测 = 未来周期分量 + 预测的残差
        final_predictions = future_cycle_components + future_residuals
        
        # 确保输出维度正确：如果输入是2D，输出也应该是2D
        if final_predictions.shape[-1] == 1:
            final_predictions = final_predictions.squeeze(-1)  # [batch_size, pred_len, 1] -> [batch_size, pred_len]
        
        return final_predictions


# 关键修复：创建一个原生的LightningModule，避免导入问题
import lightning.pytorch as L  # 使用原生导入

class CycleNetLightningModule(L.LightningModule):
    """使用原生Lightning导入的CycleNet模块"""
    
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
        
        # 保存超参数
        self.save_hyperparameters()
        
        # 模型参数
        self.d_model = d_model
        self.cycle_length = cycle_length
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.backbone_type = backbone_type
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.loss = loss or NegativeLogLikelihood()
        
        # 创建模型
        self.model = CycleNetModel(
            d_model=d_model,
            cycle_length=cycle_length,
            prediction_length=prediction_length,
            context_length=context_length,
            backbone_type=backbone_type,
            hidden_dim=hidden_dim,
        )
        
        # 分布输出层
        self.distr_output = StudentTOutput()
        
    def forward(self, past_target: torch.Tensor) -> torch.Tensor:
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
        
        # 确保输入维度正确
        if past_target.dim() == 2:
            past_target = past_target.unsqueeze(-1)
        if future_target.dim() == 2:
            future_target = future_target.unsqueeze(-1)
        
        predictions = self(past_target)
        
        # 关键修复：确保predictions的维度正确
        # StudentTOutput期望的是[batch_size, pred_len]的2D张量
        if predictions.dim() == 3:
            predictions = predictions.squeeze(-1)
        elif predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)
        
        # 同样确保future_target的维度匹配
        if future_target.dim() == 3:
            future_target = future_target.squeeze(-1)
        elif future_target.dim() == 1:
            future_target = future_target.unsqueeze(0)
        
        # 打印调试信息
        print(f"Debug: predictions shape: {predictions.shape}, future_target shape: {future_target.shape}")
        
        # 计算损失
        try:
            distr_args = self.distr_output.get_args_proj(predictions)
            distr = self.distr_output.distribution(distr_args)
            loss = self.loss(distr, future_target)
        except Exception as e:
            print(f"Error in loss calculation: {e}")
            print(f"predictions shape: {predictions.shape}")
            print(f"predictions type: {type(predictions)}")
            print(f"future_target shape: {future_target.shape}")
            raise
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
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
        
        # 确保输入维度正确
        if past_target.dim() == 2:
            past_target = past_target.unsqueeze(-1)
        if future_target.dim() == 2:
            future_target = future_target.unsqueeze(-1)
        
        predictions = self(past_target)
        
        # 确保predictions的维度正确
        if predictions.dim() == 3:
            predictions = predictions.squeeze(-1)
        elif predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)
        
        # 同样确保future_target的维度匹配
        if future_target.dim() == 3:
            future_target = future_target.squeeze(-1)
        elif future_target.dim() == 1:
            future_target = future_target.unsqueeze(0)
        
        distr_args = self.distr_output.get_args_proj(predictions)
        distr = self.distr_output.distribution(distr_args)
        loss = self.loss(distr, future_target)
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def predict_step(self, batch, batch_idx):
        past_target = batch["past_target"]
        return self(past_target)


class CycleNetEstimator(PyTorchLightningEstimator):
    """CycleNet估计器 - 兼容性修复版本"""
    
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
        loss: Optional[DistributionLoss] = None,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # 确保所有参数都是正确的类型
        self.prediction_length = int(prediction_length)
        self.context_length = int(context_length)
        self.cycle_length = int(cycle_length)
        self.backbone_type = str(backbone_type)
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.num_batches_per_epoch = int(num_batches_per_epoch)
        self.freq = str(freq)
        self.loss = loss or NegativeLogLikelihood()
        
        # 设置默认的trainer参数
        default_trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": 10.0,
            "accelerator": "auto",
            "devices": "auto",
        }
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        
        # 只传递PyTorchLightningEstimator接受的参数
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
        """创建Lightning模块 - 修复版本"""
        # 打印调试信息
        print(f"Creating Lightning module with:")
        print(f"  d_model=1")
        print(f"  cycle_length={self.cycle_length} (type: {type(self.cycle_length)})")
        print(f"  prediction_length={self.prediction_length} (type: {type(self.prediction_length)})")
        print(f"  context_length={self.context_length} (type: {type(self.context_length)})")
        print(f"  backbone_type={self.backbone_type}")
        print(f"  hidden_dim={self.hidden_dim} (type: {type(self.hidden_dim)})")
        
        # 直接返回我们的模块，不进行额外的类型转换
        module = CycleNetLightningModule(
            d_model=1,
            cycle_length=self.cycle_length,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            backbone_type=self.backbone_type,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            loss=self.loss,
        )
        
        return module
    
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
        
        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            prediction_net=trained_network,
            prediction_length=self.prediction_length,
            freq=self.freq,
            distr_output=StudentTOutput(),
        )


# 使用示例
if __name__ == "__main__":
    # 测试代码
    print("CycleNet 兼容性修复版本已加载")
    
    # 简单测试
    try:
        import lightning.pytorch as L
        
        # 测试模块创建
        module = CycleNetLightningModule(
            d_model=1,
            cycle_length=24,
            prediction_length=24,
            context_length=336
        )
        print(f"✅ Lightning模块创建成功: {type(module)}")
        print(f"✅ 是否为LightningModule: {isinstance(module, L.LightningModule)}")
        
        # 测试估计器创建
        estimator = CycleNetEstimator(
            prediction_length=24,
            context_length=336,
            cycle_length=24,
            freq="H"
        )
        print("✅ CycleNet估计器创建成功")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")