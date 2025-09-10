"""
测试所有模块导入是否正常
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试所有模块导入"""
    print("Testing module imports...")
    print("=" * 50)
    
    errors = []
    
    # 测试数据模块
    try:
        from src.data.data_loader import DataLoader
        print("✓ DataLoader imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import DataLoader: {e}")
        errors.append(f"DataLoader: {e}")
    
    # 测试模型模块
    try:
        from src.models.mean_model import MeanModel
        print("✓ MeanModel imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MeanModel: {e}")
        errors.append(f"MeanModel: {e}")
    
    try:
        from src.models.log_model import LogModel
        print("✓ LogModel imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import LogModel: {e}")
        errors.append(f"LogModel: {e}")
    
    # 测试训练模块
    try:
        from src.training.trainer import Trainer
        print("✓ Trainer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Trainer: {e}")
        errors.append(f"Trainer: {e}")
    
    # 测试评估模块
    try:
        from src.evaluation.evaluator import ModelEvaluator
        print("✓ ModelEvaluator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ModelEvaluator: {e}")
        errors.append(f"ModelEvaluator: {e}")
    
    # 测试工具模块
    try:
        from src.utils.helpers import ensure_dir
        print("✓ Helper functions imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import helpers: {e}")
        errors.append(f"helpers: {e}")
    
    print("=" * 50)
    
    if errors:
        print(f"\n❌ Found {len(errors)} import errors:")
        for error in errors:
            print(f"  - {error}")
        
        print("\n可能的解决方案:")
        print("1. 确保所有文件都已正确创建")
        print("2. 检查文件路径是否正确")
        print("3. 确保已安装所有依赖: pip install -r requirements.txt")
        print("4. 检查是否有循环导入")
        
        # 检查缺失的依赖
        print("\n检查关键依赖...")
        check_dependencies()
    else:
        print("\n✅ All modules imported successfully!")
        print("\n项目结构验证通过，可以开始使用了！")
        
        # 显示项目结构
        print("\n项目结构:")
        show_project_structure()

def check_dependencies():
    """检查关键依赖是否安装"""
    dependencies = [
        'numpy',
        'pandas',
        'torch',
        'gluonts',
        'matplotlib',
        'yaml',
        'tqdm'
    ]
    
    missing = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep} - NOT INSTALLED")
            missing.append(dep)
    
    if missing:
        print(f"\n请安装缺失的依赖:")
        print(f"pip install {' '.join(missing)}")

def show_project_structure():
    """显示项目结构"""
    structure = """
    项目根目录/
    ├── src/
    │   ├── __init__.py
    │   ├── data/
    │   │   ├── __init__.py
    │   │   └── data_loader.py
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── mean_model.py
    │   │   └── log_model.py
    │   ├── training/
    │   │   ├── __init__.py
    │   │   └── trainer.py
    │   ├── evaluation/
    │   │   ├── __init__.py
    │   │   └── evaluator.py
    │   └── utils/
    │       ├── __init__.py
    │       └── helpers.py
    ├── config/
    │   ├── config.yaml
    │   └── experiments/
    ├── scripts/
    │   ├── run_experiments.py
    │   ├── generate_experiments.py
    │   └── analyze_results.py
    ├── notebooks/
    │   └── experiments.ipynb
    ├── main.py
    └── requirements.txt
    """
    print(structure)

def check_file_exists():
    """检查关键文件是否存在"""
    print("\n检查文件是否存在:")
    
    files_to_check = [
        'src/__init__.py',
        'src/data/__init__.py',
        'src/data/data_loader.py',
        'src/models/__init__.py',
        'src/models/mean_model.py',
        'src/models/log_model.py',
        'src/training/__init__.py',
        'src/training/trainer.py',
        'src/evaluation/__init__.py',
        'src/evaluation/evaluator.py',
        'src/utils/__init__.py',
        'src/utils/helpers.py',
        'config/config.yaml',
        'main.py'
    ]
    
    missing_files = []
    for file_path in files_to_check:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n缺失 {len(missing_files)} 个文件:")
        for f in missing_files:
            print(f"  - {f}")
        
        # 创建缺失的__init__.py文件
        print("\n尝试创建缺失的 __init__.py 文件...")
        for f in missing_files:
            if f.endswith('__init__.py'):
                file_path = project_root / f
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.touch()
                print(f"  创建了: {f}")

if __name__ == "__main__":
    print("项目导入测试工具")
    print("=" * 50)
    print(f"Python路径: {sys.executable}")
    print(f"项目根目录: {project_root}")
    print("=" * 50)
    
    # 检查文件
    check_file_exists()
    
    # 测试导入
    test_imports()