"""
自定义模型模板和示例
=====================

本文件提供自定义模型的编写模板，用户可以基于此模板创建自己的模型。
支持两种类型的自定义模型：
1. Sklearn风格模型（实现fit/predict接口）
2. Keras/TensorFlow风格模型（深度学习模型）

使用方式：
1. 在下方编写自定义模型代码
2. 在应用中选择"自定义模型"选项
3. 将编写好的模型代码粘贴到文本框中
4. 程序会自动加载并使用该模型
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 模板1：Sklearn风格回归模型（用于时序预测）
# ============================================================
class CustomRegressionTemplate(BaseEstimator, RegressorMixin):
    """
    自定义回归模型模板
    
    用户需要实现：
    - __init__: 模型初始化参数
    - fit: 训练模型的方法
    - predict: 预测的方法
    
    示例：简单移动平均模型
    """
    def __init__(self, window_size=5):
        self.window_size = window_size
    
    def fit(self, X, y):
        """
        训练模型
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            训练数据特征
        y : array-like, shape (n_samples,)
            训练数据标签
            
        Returns:
        --------
        self : object
            返回自身
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # 保存训练数据的统计信息
        self.mean_ = np.mean(y)
        self.std_ = np.std(y)
        
        # 如果有额外特征，使用简单线性回归
        if X.shape[1] > 0:
            from sklearn.linear_model import LinearRegression
            self.linear_model = LinearRegression()
            self.linear_model.fit(X, y)
        else:
            self.linear_model = None
        
        return self
    
    def predict(self, X):
        """
        预测
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            预测数据特征
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            预测结果
        """
        X = np.asarray(X)
        
        if self.linear_model is not None:
            return self.linear_model.predict(X)
        else:
            return np.full(X.shape[0], self.mean_)


# ============================================================
# 模板2：Sklearn风格分类模型（用于分类任务）
# ============================================================
class CustomClassificationTemplate(BaseEstimator, ClassifierMixin):
    """
    自定义分类模型模板
    
    用户需要实现：
    - __init__: 模型初始化参数
    - fit: 训练模型的方法
    - predict: 预测的方法
    - predict_proba: 预测概率的方法（可选，用于概率输出）
    
    示例：基于距离的简单分类器
    """
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y):
        """
        训练模型
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        
        return self
    
    def predict(self, X):
        """
        预测类别
        """
        X = np.asarray(X)
        predictions = []
        
        for sample in X:
            # 简单KNN逻辑：找最近的n个邻居
            distances = np.sqrt(np.sum((self.X_ - sample) ** 2, axis=1))
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_labels = self.y_[nearest_indices]
            
            # 多数投票
            from collections import Counter
            vote_result = Counter(nearest_labels)
            predictions.append(vote_result.most_common(1)[0][0])
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        预测各类别的概率
        """
        X = np.asarray(X)
        probas = []
        
        for sample in X:
            distances = np.sqrt(np.sum((self.X_ - sample) ** 2, axis=1))
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_labels = self.y_[nearest_indices]
            
            # 计算各类别的概率
            proba = np.zeros(len(self.classes_))
            for label in nearest_labels:
                idx = np.where(self.classes_ == label)[0][0]
                proba[idx] += 1
            
            proba = proba / self.n_neighbors
            probas.append(proba)
        
        return np.array(probas)


# ============================================================
# 模板3：指数加权移动平均模型（用于时序预测）
# ============================================================
class ExponentialSmoothingModel(BaseEstimator, RegressorMixin):
    """
    指数加权移动平均模型
    
    适用于时序数据的简单预测
    """
    def __init__(self, alpha=0.3):
        self.alpha = alpha
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim > 1 and X.shape[1] > 0:
            # 如果有特征，使用线性模型
            from sklearn.linear_model import Ridge
            self.linear_model = Ridge(alpha=1.0)
            self.linear_model.fit(X, y)
        else:
            self.linear_model = None
            # 保存最后一个值作为预测
            self.last_value_ = y[-1] if len(y) > 0 else 0
        
        return self
    
    def predict(self, X):
        X = np.asarray(X)
        
        if self.linear_model is not None:
            return self.linear_model.predict(X)
        else:
            return np.full(X.shape[0], self.last_value_)


# ============================================================
# 模板4：组合模型（集成多个模型）
# ============================================================
class WeightedEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    加权集成回归模型
    
    将多个模型组合在一起，根据性能自动分配权重
    """
    def __init__(self, models=None, weights=None):
        """
        Parameters:
        -----------
        models : list, 模型列表
            例如：[('rf', RandomForestRegressor()), ('gb', GradientBoostingRegressor())]
        weights : list, 各模型的权重
            如果为None，则使用交叉验证自动确定权重
        """
        self.models = models if models is not None else []
        self.weights = weights
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.fitted_models_ = []
        
        # 训练每个模型
        for name, model in self.models:
            fitted_model = model.fit(X, y)
            self.fitted_models_.append((name, fitted_model))
        
        # 如果没有指定权重，使用交叉验证确定权重
        if self.weights is None:
            scores = []
            for name, model in self.fitted_models_:
                try:
                    score = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
                    scores.append(-score.mean())
                except:
                    scores.append(1e10)
            
            # 权重与误差成反比
            min_score = min(scores)
            self.weights = [min_score / s for s in scores]
        
        # 归一化权重
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        
        return self
    
    def predict(self, X):
        X = np.asarray(X)
        
        # 加权平均预测
        predictions = []
        for i, (name, model) in enumerate(self.fitted_models_):
            pred = model.predict(X)
            predictions.append(pred * self.weights[i])
        
        return np.sum(predictions, axis=0)


# ============================================================
# 模板5：特征工程+基础模型组合
# ============================================================
class FeatureEngineeredModel(BaseEstimator, RegressorMixin):
    """
    带特征工程的模型
    
    在预测前自动进行特征工程处理
    """
    def __init__(self, base_model=None, add_lag_features=True, add_rolling_stats=True):
        """
        Parameters:
        -----------
        base_model : 模型对象
            基础预测模型，默认为Ridge回归
        add_lag_features : bool
            是否添加滞后特征
        add_rolling_stats : bool
            是否添加滚动统计特征
        """
        from sklearn.linear_model import Ridge
        self.base_model = base_model if base_model is not None else Ridge()
        self.add_lag_features = add_lag_features
        self.add_rolling_stats = add_rolling_stats
    
    def _create_features(self, X):
        """创建新特征"""
        df = pd.DataFrame(X)
        
        if self.add_lag_features and len(df) > 1:
            # 添加滞后特征
            for i in range(1, min(4, len(df.columns) + 1)):
                df[f'lag_{i}'] = df.iloc[:, i-1].shift(1)
        
        if self.add_rolling_stats and len(df) > 3:
            # 添加滚动统计
            for col in df.columns[:3]:  # 只对前几列添加
                df[f'{col}_roll_mean'] = df[col].rolling(window=3, min_periods=1).mean()
                df[f'{col}_roll_std'] = df[col].rolling(window=3, min_periods=1).std().fillna(0)
        
        return df.fillna(0).values
    
    def fit(self, X, y):
        X_new = self._create_features(X)
        self.base_model.fit(X_new, y)
        return self
    
    def predict(self, X):
        X_new = self._create_features(X)
        return self.base_model.predict(X_new)


# ============================================================
# 如何使用自定义模型
# ============================================================
"""
使用说明：
----------

1. 选择模板或创建新模型：
   - 复制上面的模板类
   - 修改类名和参数

2. 实现必需的方法：
   - __init__(self, ...): 初始化参数
   - fit(self, X, y): 训练模型
   - predict(self, X): 进行预测

3. 可选方法：
   - predict_proba(self, X): 预测概率（分类模型）
   - score(self, X, y): 评估模型（用于AutoML）

4. 在应用中使用：
   - 选择"自定义模型"选项
   - 粘贴完整的模型代码
   - 设置模型参数
   - 开始训练
"""


# ============================================================
# 快速示例：用户可以直接复制下面的代码到应用中使用
# ============================================================

# 示例1：自定义线性回归（带特征选择）
"""
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
import numpy as np

class MyLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, use_regularization=False, alpha=1.0):
        self.use_regularization = use_regularization
        self.alpha = alpha
    
    def fit(self, X, y):
        if self.use_regularization:
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=self.alpha)
        else:
            self.model = LinearRegression()
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
"""

# 示例2：自定义随机森林（带参数调优）
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class MyRandomForest(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    
    def fit(self, X, y):
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=42
        )
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
"""

# 示例3：自定义LSTM模型
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class MyLSTMModel(BaseEstimator, RegressorMixin):
    def __init__(self, units=50, dropout=0.2, epochs=50, batch_size=32):
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_ = None
    
    def _build_model(self, n_features):
        model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=(1, n_features)),
            Dropout(self.dropout),
            LSTM(self.units // 2),
            Dropout(self.dropout),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def fit(self, X, y):
        # 重塑数据为LSTM输入格式 [samples, timesteps, features]
        n_samples = X.shape[0]
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        X_reshaped = X.reshape(n_samples, 1, n_features)
        
        self.model_ = self._build_model(n_features)
        self.model_.fit(X_reshaped, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self
    
    def predict(self, X):
        n_samples = X.shape[0]
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        X_reshaped = X.reshape(n_samples, 1, n_features)
        return self.model_.predict(X_reshaped, verbose=0).flatten()
"""
