"""
路径管理器 - 统一管理实验路径
"""
import os
import json

from .helpers import filter_metrics

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional



class PathManager:
    """路径管理器"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.res_dir = self.base_dir / "res"
        self.res_dir.mkdir(exist_ok=True)
        
        # 全局结果文件
        self.global_metrics_file = self.res_dir / "final_metrics.json"
        
    def create_experiment_dir(self, config: Dict[str, Any]) -> Path:
        """
        创建实验目录
        
        Args:
            config: 配置字典
            
        Returns:
            实验目录路径
        """
        # 提取模型名称
        mean_model = config['mean_model']
        log_model = config['log_model']
        
        dataset_name = config['dataset']['name']
        mean_name = f"{mean_model['model_type']}_{mean_model['context_length']}_{mean_model['prediction_length']}"
        mean_train_epoch = mean_model['max_epochs']
        log_name = f"log_{log_model['context_length']}_{log_model['prediction_length']}"
        log_train_epoch = log_model['max_epochs']
        
        # # 时间戳
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 实验目录名
        exp_name = f"{dataset_name}_{mean_name}_{log_name}_{mean_train_epoch}_{log_train_epoch}"
        exp_dir = self.res_dir / exp_name
        
        # 创建目录结构
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "models").mkdir(exist_ok=True)
        (exp_dir / "predictions").mkdir(exist_ok=True)
        (exp_dir / "plots").mkdir(exist_ok=True)
        
        # 保存配置
        with open(exp_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        return exp_dir
    
    def save_experiment_results(self, exp_dir: Path, 
                               mean_metrics: Dict[str, Any],
                               final_metrics: Dict[str, Any],
                               config: Dict[str, Any] = None) -> None:
        """
        保存实验结果
        
        Args:
            exp_dir: 实验目录
            mean_metrics: Mean model的评估指标
            final_metrics: 最终模型的评估指标
            config: 实验配置（可选）
        """
        # 组合结果
        results = {
            "mean_model_results": filter_metrics(mean_metrics),
            "final_model_results": filter_metrics(final_metrics),
        }
        print("+++++++++++++++++")
        print(results)
        print("+++++++++++++++++")
        
        # # 计算提升
        # for key in results["final_model_results"]:
        #     if key in results["mean_model_results"]:
        #         improvement = final_metrics[key] - mean_metrics[key]
        #         results["improvement"][f"{key}_improvement_%"] = round(improvement, 2)
        
        # 保存到实验目录
        with open(exp_dir / "metrics.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # 更新全局结果文件
        self.update_global_metrics(exp_dir.name, results, config)
    
    def update_global_metrics(self, exp_name: str, 
                            results: Dict[str, Any],
                            config: Dict[str, Any] = None) -> None:
        """
        更新全局指标文件
        
        Args:
            exp_name: 实验名称
            results: 包含mean和final的结果
            config: 实验配置
        """
        # 读取现有结果
        if self.global_metrics_file.exists():
            with open(self.global_metrics_file, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = {}
        
        # 构建记录
        record = {
            "timestamp": datetime.now().isoformat(),
            "mean_model_results": results.get("mean_model_results", {}),
            "final_model_results": results.get("final_model_results", {}),
        }
        
        # 添加关键配置信息
        if config:
            record["config_summary"] = {
                "mean_model_type": config['mean_model']['model_type'],
                "mean_config": f"{config['mean_model']['context_length']}-{config['mean_model']['prediction_length']}",
                "log_config": f"{config['log_model']['context_length']}-{config['log_model']['prediction_length']}"
            }
        
        all_results[exp_name] = record
        
        # 保存
        with open(self.global_metrics_file, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    def get_latest_experiment(self) -> Optional[Path]:
        """获取最新的实验目录"""
        exp_dirs = [d for d in self.res_dir.iterdir() if d.is_dir()]
        if exp_dirs:
            return max(exp_dirs, key=lambda x: x.stat().st_mtime)
        return None
    
    def list_experiments(self) -> list:
        """列出所有实验"""
        return [d.name for d in self.res_dir.iterdir() if d.is_dir()]


# 使用示例
def setup_experiment_paths(config: Dict[str, Any]) -> Dict[str, str]:
    """
    设置实验路径
    
    Args:
        config: 配置字典
        
    Returns:
        更新后的路径配置
    """
    path_manager = PathManager()
    exp_dir = path_manager.create_experiment_dir(config)
    
    # 更新配置中的路径
    paths = {
        'results_dir': str(exp_dir),
        'models_dir': str(exp_dir / "models"),
        'predictions_dir': str(exp_dir / "predictions"),
        'plots_dir': str(exp_dir / "plots")
    }
    
    return paths