#!/usr/bin/env python3
"""
测试批量训练的完整流程
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.append(str(Path(__file__).resolve().parent))

from src.data.data_preprocessor import DataPreprocessor
from src.models.model_trainer import ModelTrainer

# 模拟session_state
class MockSessionState:
    def __init__(self):
        self.data = {}
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def __getitem__(self, key):
        return self.data[key]

# 创建模拟数据
np.random.seed(42)

# 创建144列的测试数据（模拟用户说的144个特征）
columns = [f'col_{i}' for i in range(144)] + ['target']
data = np.random.randn(1000, 145)
df = pd.DataFrame(data, columns=columns)

# 初始化模拟session
st = MockSessionState()
st['df'] = df
st['feature_cols'] = ['col_10', 'col_20']  # 用户选择了2个特征
st['target_col'] = 'target'
st['scale_method'] = 'standard'

print("="*50)
print("测试批量训练完整流程")
print("="*50)

# 1. 模拟批量训练的数据获取逻辑
print("\n1. 数据获取逻辑:")
feature_cols_batch = st.get('feature_cols', [])
target_col_batch = st.get('target_col', '')

print(f"用户选择的特征列: {feature_cols_batch}")
print(f"用户选择的目标列: {target_col_batch}")
print(f"特征列数量: {len(feature_cols_batch)}")

# 2. 提取数据
df_original = st.get('df', None)
X = df_original[feature_cols_batch].copy()
y = df_original[target_col_batch].copy()

print(f"✓ 数据提取 - X shape: {X.shape}, y shape: {y.shape}")

# 3. 预处理和划分
preprocessor_batch = DataPreprocessor()
X_scaled = preprocessor_batch.scale_data(X, method=st.get('scale_method', 'standard'))
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor_batch.split_data(
    X_scaled, y, test_size=0.2, val_size=0.1, random_state=42, time_series=True
)

print(f"✓ 数据划分 - X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# 4. 模拟批量训练的模型选择和训练
print("\n2. 批量训练逻辑:")

# 模型名称映射
model_name_map = {
    "线性回归": "linear_regression",
    "Ridge": "ridge", 
    "Lasso": "lasso",
    "随机森林": "random_forest",
    "梯度提升": "gradient_boosting",
    "SVR": "svr",
    "KNN": "knn",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm"
}

# 要训练的模型
batch_models = ["线性回归", "随机森林", "梯度提升"]

batch_results = []

for i, model_display_name in enumerate(batch_models):
    print(f"\n  训练模型: {model_display_name}")
    
    model_type = model_name_map.get(model_display_name)
    trainer = ModelTrainer()
    trainer.select_model(model_type)
    
    # 训练模型
    try:
        trainer.train(X_train, y_train)
        print(f"    ✓ 模型训练成功")
        
        # 预测
        y_pred = trainer.predict(X_test)
        print(f"    ✓ 模型预测成功 - y_pred shape: {y_pred.shape}")
        
        # 评估
        metrics = trainer.evaluate(X_test, y_test)
        print(f"    ✓ 模型评估成功 - R²: {metrics['r2']:.4f}")
        
        batch_results.append({
            'model_name': model_display_name,
            'model_type': model_type,
            'trainer': trainer,
            'predictions': y_pred,
            'metrics': metrics
        })
    except Exception as e:
        print(f"    ✗ 模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise

print("\n" + "="*50)
print("✅ 批量训练流程测试成功！")
print("✅ 特征列数量正确：2个特征")
print("✅ 没有出现特征数量不匹配的错误")
print("="*50)
