"""
批量实验运行脚本
支持串行和并行运行多个实验
"""
import os
import sys
import yaml
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import ensure_dir, save_json, load_json


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.config_dir = self.base_dir / "config" / "experiments"
        self.results_dir = self.base_dir / "res" / "experiments"
        self.logs_dir = self.base_dir / "logs" / "experiments"
        
        # 创建必要的目录
        ensure_dir(self.results_dir)
        ensure_dir(self.logs_dir)
        
        # 设置日志
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志"""
        log_file = self.logs_dir / f"experiment_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_experiment_config(self, config_file: str) -> Dict:
        """加载实验配置"""
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            # 如果在experiments目录找不到，尝试直接路径
            config_path = Path(config_file)
        
        with open(config_path, 'r') as f:
            exp_config = yaml.safe_load(f)
        
        # 如果有基础配置，加载并合并
        if 'base_config' in exp_config:
            base_path = config_path.parent / exp_config['base_config']
            with open(base_path, 'r') as f:
                base_config = yaml.safe_load(f)
            
            # 深度合并配置
            merged_config = self.merge_configs(base_config, exp_config)
            merged_config['experiment_name'] = exp_config.get('experiment_name', config_file.split('.')[0])
            merged_config['description'] = exp_config.get('description', '')
            return merged_config
        
        return exp_config
    
    def merge_configs(self, base: Dict, override: Dict) -> Dict:
        """深度合并配置"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def create_temp_config(self, config: Dict, exp_name: str) -> str:
        """创建临时配置文件"""
        temp_dir = self.base_dir / "temp_configs"
        ensure_dir(temp_dir)
        
        temp_config_path = temp_dir / f"config_{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return str(temp_config_path)
    
    def run_single_experiment(self, config_file: str, mode: str = "all", 
                            force_retrain: bool = False) -> Dict:
        """运行单个实验"""
        start_time = datetime.now()
        
        try:
            # 加载配置
            config = self.load_experiment_config(config_file)
            exp_name = config.get('experiment_name', config_file.split('.')[0])
            
            self.logger.info(f"Starting experiment: {exp_name}")
            self.logger.info(f"Description: {config.get('description', 'No description')}")
            
            # 修改结果保存路径，添加实验名称
            config['paths']['results_dir'] = str(self.results_dir / exp_name)
            config['paths']['models_dir'] = str(self.results_dir / exp_name / "models")
            config['paths']['predictions_dir'] = str(self.results_dir / exp_name / "predictions")
            
            # 创建临时配置文件
            temp_config = self.create_temp_config(config, exp_name)
            
            # 构建命令
            cmd = [
                sys.executable,
                str(self.base_dir / "main.py"),
                "--config", temp_config,
                "--mode", mode,
                "--log-level", "INFO"
            ]
            
            if force_retrain:
                cmd.append("--force-retrain")
            
            # 运行实验
            log_file = self.logs_dir / f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                process.wait()
            
            # 收集结果
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                'experiment_name': exp_name,
                'config_file': config_file,
                'status': 'success' if process.returncode == 0 else 'failed',
                'return_code': process.returncode,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'log_file': str(log_file),
                'results_dir': config['paths']['results_dir']
            }
            
            # 尝试加载评估指标
            metrics_file = Path(config['paths']['results_dir']) / "final_metrics.json"
            if metrics_file.exists():
                result['metrics'] = load_json(metrics_file)
            
            # 清理临时配置
            os.remove(temp_config)
            
            self.logger.info(f"Experiment {exp_name} completed in {duration:.2f} seconds")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running experiment {config_file}: {str(e)}")
            return {
                'experiment_name': config_file,
                'status': 'error',
                'error': str(e),
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat()
            }
    
    def run_experiments_sequential(self, config_files: List[str], **kwargs) -> List[Dict]:
        """串行运行多个实验"""
        results = []
        
        for i, config_file in enumerate(config_files, 1):
            self.logger.info(f"Running experiment {i}/{len(config_files)}: {config_file}")
            result = self.run_single_experiment(config_file, **kwargs)
            results.append(result)
            
        return results
    
    def run_experiments_parallel(self, config_files: List[str], 
                                max_workers: int = None, **kwargs) -> List[Dict]:
        """并行运行多个实验"""
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count() - 1, len(config_files))
        
        self.logger.info(f"Running {len(config_files)} experiments in parallel with {max_workers} workers")
        
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_config = {
                executor.submit(self.run_single_experiment, config, **kwargs): config 
                for config in config_files
            }
            
            # 收集结果
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in parallel execution for {config}: {str(e)}")
                    results.append({
                        'experiment_name': config,
                        'status': 'error',
                        'error': str(e)
                    })
        
        return results
    
    def generate_report(self, results: List[Dict]) -> pd.DataFrame:
        """生成实验报告"""
        # 创建DataFrame
        df = pd.DataFrame(results)
        
        # 提取关键指标
        if 'metrics' in df.columns:
            # 展开metrics列
            metrics_df = pd.json_normalize(df['metrics'].dropna())
            df = pd.concat([df, metrics_df], axis=1)
        
        # 保存完整报告
        report_file = self.results_dir / f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(report_file, index=False)
        self.logger.info(f"Report saved to {report_file}")
        
        # 打印摘要
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        # 基本信息
        print(f"\nTotal experiments: {len(results)}")
        print(f"Successful: {df[df['status'] == 'success'].shape[0]}")
        print(f"Failed: {df[df['status'] == 'failed'].shape[0]}")
        print(f"Errors: {df[df['status'] == 'error'].shape[0]}")
        
        # 性能比较（如果有指标）
        if 'RMSE' in df.columns:
            print("\nPerformance Comparison:")
            print("-"*40)
            
            # 选择关键列
            perf_cols = ['experiment_name', 'RMSE', 'MAPE', 'CRPS']
            perf_cols = [col for col in perf_cols if col in df.columns]
            
            perf_df = df[df['status'] == 'success'][perf_cols].sort_values('RMSE')
            print(perf_df.to_string(index=False))
            
            # 最佳模型
            best_model = perf_df.iloc[0]
            print(f"\nBest model: {best_model['experiment_name']}")
            print(f"  RMSE: {best_model['RMSE']:.4f}")
        
        print("\n" + "="*80)
        
        return df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Run multiple experiments")
    
    parser.add_argument(
        "--configs",
        nargs="+",
        help="List of experiment config files"
    )
    parser.add_argument(
        "--config-dir",
        help="Directory containing experiment configs (run all .yaml files)"
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["train", "predict", "all"],
        help="Execution mode"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retrain all models"
    )
    
    args = parser.parse_args()
    
    # 初始化运行器
    runner = ExperimentRunner()
    
    # 确定要运行的配置文件
    config_files = []
    
    if args.configs:
        config_files = args.configs
    elif args.config_dir:
        config_dir = Path(args.config_dir)
        config_files = [str(f.name) for f in config_dir.glob("*.yaml")]
    else:
        # 默认运行experiments目录下的所有配置
        config_files = [str(f.name) for f in runner.config_dir.glob("exp_*.yaml")]
    
    if not config_files:
        print("No experiment configs found!")
        return
    
    print(f"Found {len(config_files)} experiment configs:")
    for cf in config_files:
        print(f"  - {cf}")
    
    # 运行实验
    if args.parallel:
        results = runner.run_experiments_parallel(
            config_files,
            max_workers=args.max_workers,
            mode=args.mode,
            force_retrain=args.force_retrain
        )
    else:
        results = runner.run_experiments_sequential(
            config_files,
            mode=args.mode,
            force_retrain=args.force_retrain
        )
    
    # 生成报告
    runner.generate_report(results)
    
    # 保存完整结果
    results_file = runner.results_dir / f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_json(results, results_file)
    print(f"\nAll results saved to {results_file}")


if __name__ == "__main__":
    main()