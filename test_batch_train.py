#!/usr/bin/env python3
"""
测试批量训练的数据获取逻辑
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.append(str(Path(__file__).resolve().parent))

from src.data.data_preprocessor import DataPreprocessor

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
print("测试批量训练数据获取逻辑")
print("="*50)
print(f"数据总列数: {len(df.columns)}")
print(f"用户选择的特征列: {st['feature_cols']}")
print(f"用户选择的目标列: {st['target_col']}")
print(f"特征列数量: {len(st['feature_cols'])}")

# 1. 测试特征列提取
print("\n1. 测试特征列提取:")
try:
    X = df[st['feature_cols']].copy()
    y = df[st['target_col']].copy()
    print(f"✓ 成功提取 - X shape: {X.shape}, y shape: {y.shape}")
except Exception as e:
    print(f"✗ 提取失败: {e}")
    raise

# 2. 测试数据缩放
print("\n2. 测试数据缩放:")
try:
    preprocessor = DataPreprocessor()
    X_scaled = preprocessor.scale_data(X, method='standard')
    print(f"✓ 成功缩放 - X_scaled shape: {X_scaled.shape}")
except Exception as e:
    print(f"✗ 缩放失败: {e}")
    raise

# 3. 测试数据划分
print("\n3. 测试数据划分:")
try:
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X_scaled, y, test_size=0.2, val_size=0.1, random_state=42, time_series=True
    )
    print(f"✓ 成功划分 - X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
except Exception as e:
    print(f"✗ 划分失败: {e}")
    raise

# 4. 测试模型训练
print("\n4. 测试模型训练:")
try:
    from sklearn.ensemble import RandomForestRegressor
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    print(f"✓ 模型训练成功")
    
    # 测试预测
    y_pred = model.predict(X_test)
    print(f"✓ 模型预测成功 - y_pred shape: {y_pred.shape}")
except Exception as e:
    print(f"✗ 模型操作失败: {e}")
    raise

print("\n" + "="*50)
print("✅ 所有测试都通过了！")
print("="*50)
