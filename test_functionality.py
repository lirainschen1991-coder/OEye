"""
功能自测脚本 - 测试所有新增功能
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.data.data_loader import read_data_file
from src.data.data_preprocessor import DataPreprocessor
from src.models.model_trainer import ModelTrainer
from src.visualization.visualizer import DataVisualizer

print("="*60)
print("海洋工程结构物运动响应预测平台 - 功能自测")
print("="*60)

# 1. 测试数据加载
print("\n[1/8] 测试数据加载...")
try:
    df = read_data_file('primary.out')
    print(f"✓ 数据加载成功: {df.shape}")
    print(f"  列名: {list(df.columns[:5])}...")
except Exception as e:
    print(f"✗ 数据加载失败: {e}")

# 2. 测试数据预处理功能
print("\n[2/8] 测试数据预处理功能...")
try:
    preprocessor = DataPreprocessor()
    
    # 测试滞后特征
    df_lagged = preprocessor.create_lagged_features(df, ['PtfmSurge', 'PtfmSway'], lags=2)
    print(f"✓ 滞后特征创建成功: {df_lagged.shape}")
    
    # 测试滚动窗口
    df_rolling = preprocessor.rolling_window(df, ['PtfmSurge'], window_size=5)
    print(f"✓ 滚动窗口特征创建成功: {df_rolling.shape}")
    
    # 测试差分
    df_diff = preprocessor.differencing(df, ['PtfmSurge'], order=1)
    print(f"✓ 差分特征创建成功: {df_diff.shape}")
    
    # 测试异常检测
    outliers = preprocessor.detect_outliers(df, ['PtfmSurge', 'PtfmSway'], method='iqr')
    print(f"✓ 异常值检测成功: 发现 {sum(len(v) for v in outliers.values())} 个异常值")
    
    # 测试数据平滑
    df_smooth = preprocessor.smooth_data(df, ['PtfmSurge'], method='moving_average', window_length=5)
    print(f"✓ 数据平滑成功")
    
    # 测试趋势特征
    df_trend = preprocessor.add_trend_features(df, ['PtfmSurge'], window=10)
    print(f"✓ 趋势特征添加成功: {df_trend.shape}")
    
except Exception as e:
    print(f"✗ 数据预处理失败: {e}")
    import traceback
    traceback.print_exc()

# 3. 测试数据增强
print("\n[3/8] 测试数据增强功能...")
try:
    df_sample = df.head(100)
    df_aug = preprocessor.augment_data(df_sample, target_col='PtfmSurge', method='noise', n_augmentations=2)
    print(f"✓ 数据增强成功: {df_sample.shape} -> {df_aug.shape}")
except Exception as e:
    print(f"✗ 数据增强失败: {e}")

# 4. 测试模型训练
print("\n[4/8] 测试模型训练功能...")
try:
    trainer = ModelTrainer()
    
    # 准备数据
    feature_cols = ['PtfmSway', 'PtfmHeave', 'PtfmRoll', 'PtfmPitch']
    target_col = 'PtfmSurge'
    
    X = df[feature_cols].head(500)
    y = df[target_col].head(500)
    
    # 测试线性回归
    trainer.select_model('linear_regression')
    trainer.train(X, y)
    y_pred = trainer.predict(X)
    print(f"✓ 线性回归模型训练成功")
    
    # 测试随机森林
    trainer.select_model('random_forest', n_estimators=10, max_depth=5)
    trainer.train(X, y)
    y_pred = trainer.predict(X)
    print(f"✓ 随机森林模型训练成功")
    
except Exception as e:
    print(f"✗ 模型训练失败: {e}")
    import traceback
    traceback.print_exc()

# 5. 测试时间序列评估指标
print("\n[5/8] 测试时间序列评估指标...")
try:
    metrics = trainer.evaluate_time_series(y, y_pred)
    print(f"✓ 时间序列评估成功:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
except Exception as e:
    print(f"✗ 时间序列评估失败: {e}")

# 6. 测试集成学习
print("\n[6/8] 测试集成学习功能...")
try:
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    
    models_dict = {
        'linear': LinearRegression(),
        'ridge': Ridge(),
        'rf': RandomForestRegressor(n_estimators=10, random_state=42)
    }
    
    # 测试投票集成
    ensemble = trainer.create_voting_ensemble(models_dict)
    ensemble.fit(X, y)
    y_pred_ensemble = ensemble.predict(X)
    print(f"✓ 投票集成模型训练成功")
    
    # 测试加权平均
    weighted_ensemble = trainer.create_weighted_average(models_dict, weights=[0.2, 0.3, 0.5])
    weighted_ensemble.fit(X, y)
    y_pred_weighted = weighted_ensemble.predict(X)
    print(f"✓ 加权平均集成模型训练成功")
    
except Exception as e:
    print(f"✗ 集成学习失败: {e}")
    import traceback
    traceback.print_exc()

# 7. 测试可视化功能
print("\n[7/8] 测试可视化功能...")
try:
    visualizer = DataVisualizer()
    
    # 测试模型对比图
    models_metrics = {
        'Linear Regression': {'mse': 0.1, 'mae': 0.2, 'r2': 0.85},
        'Random Forest': {'mse': 0.05, 'mae': 0.15, 'r2': 0.92},
        'SVR': {'mse': 0.08, 'mae': 0.18, 'r2': 0.88}
    }
    fig = visualizer.plot_model_comparison(models_metrics, use_plotly=False)
    print(f"✓ 模型对比图生成成功")
    
    # 测试预测对比图
    predictions_dict = {
        'Linear': y_pred,
        'RF': y_pred_ensemble
    }
    fig2 = visualizer.plot_prediction_comparison(y, predictions_dict, use_plotly=False)
    print(f"✓ 预测对比图生成成功")
    
    # 测试误差热力图
    fig3 = visualizer.plot_error_heatmap(models_metrics, use_plotly=False)
    print(f"✓ 误差热力图生成成功")
    
except Exception as e:
    print(f"✗ 可视化功能失败: {e}")
    import traceback
    traceback.print_exc()

# 8. 测试交叉验证
print("\n[8/8] 测试交叉验证功能...")
try:
    trainer.select_model('random_forest', n_estimators=10, max_depth=5)
    cv_results = trainer.cross_validate(X, y, cv=3)
    print(f"✓ 交叉验证成功:")
    print(f"  Mean R²: {cv_results['mean']:.4f}")
    print(f"  Std R²: {cv_results['std']:.4f}")
except Exception as e:
    print(f"✗ 交叉验证失败: {e}")

print("\n" + "="*60)
print("功能自测完成!")
print("="*60)
