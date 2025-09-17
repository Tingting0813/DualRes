"""
时间序列预测主程序
"""
import os
import yaml
import argparse
import logging
from pathlib import Path

# 导入自定义模块
from src.data.data_loader import DataLoader
from src.models.mean_model import MeanModel
from src.models.log_model import LogModel
from src.training.trainer import Trainer
from src.evaluation.evaluator import ModelEvaluator
from src.utils.path_manager import PathManager, setup_experiment_paths

# 配置日志
def setup_logging(log_level="INFO"):
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(args):
    """主函数"""
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # 加载配置
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # # 创建必要的目录
    # for path_key in config['paths'].values():
    #     os.makedirs(path_key, exist_ok=True)

    # 创建路径管理器
    path_manager = PathManager()
    # 为这次实验创建目录
    exp_dir = path_manager.create_experiment_dir(config)
    # 更新配置中的路径
    config['paths'] = {
        'results_dir': str(exp_dir),
        'models_dir': str(exp_dir / "models"),
        'predictions_dir': str(exp_dir / "predictions"),
        'plots_dir': str(exp_dir / "plots")
    }
    logger.info(f"Experiment directory: {exp_dir}")
    
    # 初始化组件
    logger.info("Initializing components...")
    data_loader = DataLoader(config)
    mean_model = MeanModel(config)
    log_model = LogModel(config)
    trainer = Trainer(config, data_loader, mean_model, log_model)
    evaluator = ModelEvaluator(config)
    
    # 加载数据
    logger.info("Loading dataset...")
    data_loader.load_dataset()
    dataset_info = data_loader.get_dataset_info()
    logger.info(f"Dataset info: {dataset_info}")
    
    if args.mode == 'train' or args.mode == 'all':
        # 训练Mean Model
        logger.info("=" * 50)
        logger.info("Training Mean Model...")
        trainer.train_mean_model(force_retrain=args.force_retrain)
    
        # 评估Mean Model
        if args.evaluate:
            logger.info("Evaluating Mean Model...")
            forecasts, tss = mean_model.predict(data_loader.test_data, num_samples=config['sampling']['num_samples'])
            mean_agg_metrics, _ = evaluator.evaluate_predictions(forecasts, tss)
            logger.info(f"Mean Model metrics: {mean_agg_metrics}")
            
            # 可视化
            if args.visualize:
                evaluator.plot_prediction(
                    forecasts, tss, 
                    series_idx=0,
                    save_path=os.path.join(config['paths']['plots_dir'], 'mean_model_prediction.png')
                )
            
        # 生成Mean Model预测（用于Log Model训练）
        logger.info("Generating mean model predictions for log model training...")
        mean_predictions = trainer.generate_mean_predictions()
        
        # 准备Log Model数据
        logger.info("Preparing log model data...")
        loss_dataset = trainer.prepare_log_model_data(mean_predictions)
        
        # 训练Log Model
        logger.info("=" * 50)
        logger.info("Training Log Model...")
        trainer.train_log_model(loss_dataset, force_retrain=args.force_retrain)
          
        # 生成Log Model预测
        logger.info("Generating log model predictions...")
        log_predictions_df = log_model.predict_log_values(loss_dataset)
    
        # 计算yita值
        logger.info("Calculating yita values...")
        mean_data_df = pd.read_csv(
            os.path.join(config['paths']['results_dir'], 
                        f"{config['dataset']['name']}_log_mse_losses.csv")
        )
        yita_df = log_model.calculate_yita(mean_data_df, log_predictions_df)
        
        # 可视化ACF
        if args.visualize:
            logger.info("Plotting ACF...")
            evaluator.plot_acf(
                mean_data_df,
                save_path=os.path.join(config['paths']['plots_dir'], 'acf_plot.png')
            )
    
    if args.mode == 'predict' or args.mode == 'all':
        # 确保模型已加载
        logger.info("=" * 50)
        logger.info("Loading models for prediction...")

        if mean_model.predictor is None:
            mean_model.load_predictor()
        if log_model.predictor is None:
            log_model.load_predictor()
        
        
        # 重采样
        logger.info("Resampling yita values...")
        yita_save_path=os.path.join(config['paths']['results_dir'],
                            f"all_yita_array_{log_model.context_length}_{log_model.prediction_length}_{log_model.max_epochs}.csv")
        all_samples = trainer.resample_yita(yita_save_path)
        
        # 执行滚动预测
        logger.info("Performing rolling forecast...")
        final_res = trainer.rolling_forecast(all_samples)
        
        # 评估最终结果
        if args.evaluate:
            logger.info("=" * 50)
            logger.info("Evaluating final predictions...")
            mean_forecasts, tss = mean_model.predict(data_loader.test_data, num_samples=config['sampling']['num_samples'])
            our_agg_metrics, _ = evaluator.evaluate_sample_forecast(
                final_res, 
                tss,
                mean_model.prediction_length
            )
            logger.info(f"Final model metrics: {our_agg_metrics}")
            

            # 保存结果到实验目录和全局文件
            if args.mode == 'predict':
                logger.info(f"Mean Model metrics: ReCalculate...")
                mean_agg_metrics, _ = evaluator.evaluate_predictions(mean_forecasts, tss)
            path_manager.save_experiment_results(exp_dir, mean_agg_metrics, our_agg_metrics, config)
        
        # 可视化最终结果
        if args.visualize:
            logger.info("Visualizing final predictions...")
            evaluator.plot_final_forecast(
                final_res,
                data_loader.test_data,
                series_idx=args.series_idx,
                prediction_length=mean_model.prediction_length,
                save_path=os.path.join(config['paths']['plots_dir'], 'final_forecast.png')
            )
        
    logger.info("=" * 50)
    logger.info("Process completed successfully!")

if __name__ == "__main__":
    import pandas as pd
    from gluonts.model.predictor import Predictor
    
    parser = argparse.ArgumentParser(description="Time Series Forecasting")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to config file")
    parser.add_argument("--mode", type=str, default="all",
                       choices=["train", "predict", "all"],
                       help="Running mode")
    parser.add_argument("--force-retrain", action="store_true",
                       help="Force retrain models")
    parser.add_argument("--evaluate", action="store_true", default=True,
                       help="Evaluate models")
    parser.add_argument("--visualize", action="store_true", default=True,
                       help="Generate visualizations")
    parser.add_argument("--series-idx", type=int, default=0,
                       help="Time series index for visualization")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    main(args)