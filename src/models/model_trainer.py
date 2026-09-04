import pandas as pd
import numpy as np
import pickle
import os
import json
import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, Perceptron
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, 
                               StackingRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor)
from sklearn.svm import SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad, Adadelta
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.model_type = None
        self.is_deep_learning = False
    
    def select_model(self, model_type, **kwargs):
        """
        选择模型
        """
        self.model_type = model_type
        # 重置深度学习标志
        self.is_deep_learning = False
        
        # 机器学习模型
        if model_type == 'linear_regression':
            self.model = LinearRegression(**kwargs)
        elif model_type == 'ridge':
            self.model = Ridge(**kwargs)
        elif model_type == 'lasso':
            self.model = Lasso(**kwargs)
        elif model_type == 'elastic_net':
            self.model = ElasticNet(**kwargs)
        elif model_type == 'bayesian_ridge':
            self.model = BayesianRidge(**kwargs)
        elif model_type == 'perceptron':
            from sklearn.neural_network import MLPRegressor
            self.model = MLPRegressor(hidden_layer_sizes=(100,), **kwargs)
        # 集成学习模型
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(**kwargs)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(**kwargs)
        elif model_type == 'xgboost':
            try:
                from xgboost import XGBRegressor
                self.model = XGBRegressor(**kwargs)
            except ImportError:
                raise ImportError("XGBoost未安装，请运行: pip install xgboost")
        elif model_type == 'lightgbm':
            try:
                from lightgbm import LGBMRegressor
                self.model = LGBMRegressor(**kwargs)
            except ImportError:
                raise ImportError("LightGBM未安装，请运行: pip install lightgbm")
        elif model_type == 'catboost':
            try:
                from catboost import CatBoostRegressor
                self.model = CatBoostRegressor(verbose=False, **kwargs)
            except ImportError:
                raise ImportError("CatBoost未安装，请运行: pip install catboost")
        elif model_type == 'adaboost':
            self.model = AdaBoostRegressor(**kwargs)
        elif model_type == 'extra_trees':
            self.model = ExtraTreesRegressor(**kwargs)
        elif model_type == 'bagging':
            self.model = BaggingRegressor(**kwargs)
        elif model_type == 'voting':
            estimators = [
                ('lr', LinearRegression()),
                ('rf', RandomForestRegressor(n_estimators=50)),
                ('gb', GradientBoostingRegressor(n_estimators=50))
            ]
            self.model = VotingRegressor(estimators=estimators)
        # 支持向量机
        elif model_type == 'svr':
            self.model = SVR(**kwargs)
        elif model_type == 'nu_svr':
            self.model = NuSVR(**kwargs)
        # 近邻算法
        elif model_type == 'knn':
            self.model = KNeighborsRegressor(**kwargs)
        elif model_type == 'radius_neighbors':
            self.model = RadiusNeighborsRegressor(**kwargs)
        # 树模型
        elif model_type == 'decision_tree':
            self.model = DecisionTreeRegressor(**kwargs)
        elif model_type == 'extra_tree':
            self.model = ExtraTreeRegressor(**kwargs)
        # 高斯过程
        elif model_type == 'gaussian_process':
            # 默认使用RBF核
            kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            self.model = GaussianProcessRegressor(kernel=kernel, **kwargs)
        # 深度学习模型
        elif model_type == 'ann':
            self.is_deep_learning = True
            self.model = self._create_ann_model(**kwargs)
        elif model_type == 'lstm':
            self.is_deep_learning = True
            self.model = self._create_lstm_model(**kwargs)
        elif model_type == 'gru':
            self.is_deep_learning = True
            self.model = self._create_gru_model(**kwargs)
        elif model_type == 'cnn':
            self.is_deep_learning = True
            self.model = self._create_cnn_model(**kwargs)
        elif model_type == 'transformer':
            self.is_deep_learning = True
            self.model = self._create_transformer_model(**kwargs)
        elif model_type == 'mlp':
            self.is_deep_learning = True
            self.model = self._create_mlp_model(**kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        return self.model
    
    def _create_ann_model(self, input_dim, hidden_layers=[64, 32], activation='relu', 
                         output_activation='linear', learning_rate=0.001, optimizer='adam',
                         loss='mse', metrics=['mae', 'mse'], **kwargs):
        """
        创建人工神经网络模型
        """
        model = Sequential()
        
        # 输入层
        model.add(Dense(hidden_layers[0], activation=activation, input_dim=input_dim))
        if 'batch_norm' in kwargs and kwargs['batch_norm']:
            model.add(BatchNormalization())
        if 'dropout' in kwargs and kwargs['dropout'] > 0:
            model.add(Dropout(kwargs['dropout']))
        
        # 隐藏层
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation=activation))
            if 'batch_norm' in kwargs and kwargs['batch_norm']:
                model.add(BatchNormalization())
            if 'dropout' in kwargs and kwargs['dropout'] > 0:
                model.add(Dropout(kwargs['dropout']))
        
        # 输出层
        model.add(Dense(1, activation=output_activation))
        
        # 选择优化器
        opt = self._get_optimizer(optimizer, learning_rate)
        
        # 编译模型
        model.compile(optimizer=opt, loss=loss, metrics=metrics)
        
        return model
    
    def _get_optimizer(self, optimizer_name, learning_rate):
        """
        获取优化器
        """
        optimizer_name = optimizer_name.lower()
        
        if optimizer_name == 'adam':
            return Adam(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            return RMSprop(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            return SGD(learning_rate=learning_rate)
        elif optimizer_name == 'adagrad':
            return Adagrad(learning_rate=learning_rate)
        elif optimizer_name == 'adadelta':
            return Adadelta(learning_rate=learning_rate)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    def _create_lstm_model(self, input_shape, hidden_layers=[64, 32], activation='relu', 
                          output_activation='linear', learning_rate=0.001, optimizer='adam',
                          loss='mse', metrics=['mae', 'mse'], **kwargs):
        """
        创建LSTM模型
        """
        model = Sequential()
        
        # 输入层
        model.add(LSTM(hidden_layers[0], activation=activation, return_sequences=True, input_shape=input_shape))
        if 'dropout' in kwargs and kwargs['dropout'] > 0:
            model.add(Dropout(kwargs['dropout']))
        
        # 隐藏层
        for i, units in enumerate(hidden_layers[1:]):
            if i == len(hidden_layers[1:]) - 1:
                model.add(LSTM(units, activation=activation))
            else:
                model.add(LSTM(units, activation=activation, return_sequences=True))
            if 'dropout' in kwargs and kwargs['dropout'] > 0:
                model.add(Dropout(kwargs['dropout']))
        
        # 输出层
        model.add(Dense(1, activation=output_activation))
        
        # 编译模型
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        
        return model
    
    def _create_gru_model(self, input_shape, hidden_layers=[64, 32], activation='relu', 
                         output_activation='linear', learning_rate=0.001, optimizer='adam',
                         loss='mse', metrics=['mae', 'mse'], **kwargs):
        """
        创建GRU模型
        """
        model = Sequential()
        
        # 输入层
        model.add(GRU(hidden_layers[0], activation=activation, return_sequences=True, input_shape=input_shape))
        if 'batch_norm' in kwargs and kwargs['batch_norm']:
            model.add(BatchNormalization())
        if 'dropout' in kwargs and kwargs['dropout'] > 0:
            model.add(Dropout(kwargs['dropout']))
        
        # 隐藏层
        for i, units in enumerate(hidden_layers[1:]):
            if i == len(hidden_layers[1:]) - 1:
                model.add(GRU(units, activation=activation))
            else:
                model.add(GRU(units, activation=activation, return_sequences=True))
            if 'batch_norm' in kwargs and kwargs['batch_norm']:
                model.add(BatchNormalization())
            if 'dropout' in kwargs and kwargs['dropout'] > 0:
                model.add(Dropout(kwargs['dropout']))
        
        # 输出层
        model.add(Dense(1, activation=output_activation))
        
        # 选择优化器
        opt = self._get_optimizer(optimizer, learning_rate)
        
        # 编译模型
        model.compile(optimizer=opt, loss=loss, metrics=metrics)
        
        return model
    
    def _create_cnn_model(self, input_shape, hidden_layers=[64, 32], activation='relu',
                         output_activation='linear', learning_rate=0.001, optimizer='adam',
                         loss='mse', metrics=['mae', 'mse'], **kwargs):
        """
        创建CNN模型（用于回归任务）
        """
        from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D
        
        model = Sequential()
        
        # Conv层
        model.add(Conv1D(filters=hidden_layers[0], kernel_size=3, activation=activation, 
                        input_shape=input_shape, padding='same'))
        if 'batch_norm' in kwargs and kwargs['batch_norm']:
            model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        if 'dropout' in kwargs and kwargs['dropout'] > 0:
            model.add(Dropout(kwargs['dropout']))
        
        # 额外的Conv层
        for units in hidden_layers[1:]:
            model.add(Conv1D(filters=units, kernel_size=3, activation=activation, padding='same'))
            if 'batch_norm' in kwargs and kwargs['batch_norm']:
                model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))
            if 'dropout' in kwargs and kwargs['dropout'] > 0:
                model.add(Dropout(kwargs['dropout']))
        
        # 全局池化
        model.add(GlobalAveragePooling1D())
        
        # 全连接层
        model.add(Dense(32, activation=activation))
        if 'dropout' in kwargs and kwargs['dropout'] > 0:
            model.add(Dropout(kwargs['dropout'] / 2))
        
        # 输出层
        model.add(Dense(1, activation=output_activation))
        
        # 选择优化器
        opt = self._get_optimizer(optimizer, learning_rate)
        
        # 编译模型
        model.compile(optimizer=opt, loss=loss, metrics=metrics)
        
        return model
    
    def _create_transformer_model(self, input_shape, hidden_layers=[64, 32], activation='relu',
                                   output_activation='linear', learning_rate=0.001, optimizer='adam',
                                   loss='mse', metrics=['mae', 'mse'], num_heads=2, num_layers=2,
                                   **kwargs):
        """
        创建Transformer模型（简化版）
        """
        from tensorflow.keras.layers import (Input, MultiHeadAttention, LayerNormalization,
                                             GlobalAveragePooling1D)
        from tensorflow.keras.models import Model
        
        inputs = Input(shape=input_shape)
        x = inputs
        
        # Transformer编码器层
        for _ in range(num_layers):
            # 多头注意力
            attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_layers[0])(x, x)
            x = LayerNormalization(epsilon=1e-6)(x + attn_output)
            
            # 前馈网络
            ff_output = Dense(hidden_layers[0] * 2, activation=activation)(x)
            ff_output = Dense(int(x.shape[-1]))(ff_output)
            x = LayerNormalization(epsilon=1e-6)(x + ff_output)
        
        # 全局池化
        x = GlobalAveragePooling1D()(x)
        
        # 全连接层
        for units in hidden_layers:
            x = Dense(units, activation=activation)(x)
            if 'dropout' in kwargs and kwargs['dropout'] > 0:
                x = Dropout(kwargs['dropout'])(x)
        
        # 输出层
        outputs = Dense(1, activation=output_activation)(x)
        
        # 创建模型
        model = Model(inputs=inputs, outputs=outputs)
        
        # 选择优化器
        opt = self._get_optimizer(optimizer, learning_rate)
        
        # 编译模型
        model.compile(optimizer=opt, loss=loss, metrics=metrics)
        
        return model
    
    def _create_mlp_model(self, input_dim, hidden_layers=[64, 32], activation='relu',
                          output_activation='linear', learning_rate=0.001, optimizer='adam',
                          loss='mse', metrics=['mae', 'mse'], **kwargs):
        """
        创建MLP（多层感知机）模型
        """
        model = Sequential()
        
        # 输入层
        model.add(Dense(hidden_layers[0], activation=activation, input_dim=input_dim))
        if 'batch_norm' in kwargs and kwargs['batch_norm']:
            model.add(BatchNormalization())
        if 'dropout' in kwargs and kwargs['dropout'] > 0:
            model.add(Dropout(kwargs['dropout']))
        
        # 隐藏层
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation=activation))
            if 'batch_norm' in kwargs and kwargs['batch_norm']:
                model.add(BatchNormalization())
            if 'dropout' in kwargs and kwargs['dropout'] > 0:
                model.add(Dropout(kwargs['dropout']))
        
        # 输出层
        model.add(Dense(1, activation=output_activation))
        
        # 选择优化器
        opt = self._get_optimizer(optimizer, learning_rate)
        
        # 编译模型
        model.compile(optimizer=opt, loss=loss, metrics=metrics)
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, 
             patience=10, **kwargs):
        """
        训练模型
        """
        if self.is_deep_learning:
            # 深度学习模型训练
            callbacks = []
            if X_val is not None and y_val is not None:
                callbacks.append(EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True))
            
            history = self.model.fit(
                X_train, y_train, 
                validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            return history
        else:
            # 机器学习模型训练
            if hasattr(X_train, 'values'):
                X_train = X_train.values
            if hasattr(y_train, 'values'):
                y_train = y_train.values
            
            X_train = np.asarray(X_train, dtype=np.float64)
            y_train = np.asarray(y_train, dtype=np.float64)
            
            if y_train.ndim > 1:
                y_train = y_train.ravel()
            
            self.model.fit(X_train, y_train)
            return None
    
    def evaluate(self, X_test, y_test):
        """
        评估模型
        """
        y_pred = self.predict(X_test)
        
        if hasattr(y_test, 'values'):
            y_test = y_test.values
        y_test = np.asarray(y_test, dtype=np.float64)
        
        if y_test.ndim > 1:
            y_test = y_test.ravel()
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': rmse
        }
    
    def predict(self, X):
        """
        使用模型进行预测
        """
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(1, -1) if len(X) > 1 else X.reshape(-1, 1)
        
        y_pred = self.model.predict(X)
        if len(y_pred.shape) == 1:
            return y_pred
        elif len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred
    
    def save_model(self, file_path):
        """
        保存模型
        """
        if self.is_deep_learning:
            # 保存深度学习模型
            self.model.save(file_path)
        else:
            # 保存机器学习模型
            with open(file_path, 'wb') as f:
                pickle.dump(self.model, f)
    
    def hyperparameter_tuning(self, X_train, y_train, model_type, param_grid, tuning_type='grid', 
                             cv=5, n_iter=100, random_state=42, **kwargs):
        """
        超参数调优
        """
        # 选择基础模型
        if model_type == 'linear_regression':
            base_model = LinearRegression()
        elif model_type == 'ridge':
            base_model = Ridge()
        elif model_type == 'lasso':
            base_model = Lasso()
        elif model_type == 'random_forest':
            base_model = RandomForestRegressor(random_state=random_state)
        elif model_type == 'gradient_boosting':
            base_model = GradientBoostingRegressor(random_state=random_state)
        elif model_type == 'svr':
            base_model = SVR()
        elif model_type == 'knn':
            base_model = KNeighborsRegressor()
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 选择调优方法
        if tuning_type == 'grid':
            tuner = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=cv, scoring='r2', n_jobs=-1)
        elif tuning_type == 'random':
            tuner = RandomizedSearchCV(estimator=base_model, param_distributions=param_grid, 
                                      n_iter=n_iter, cv=cv, scoring='r2', random_state=random_state, n_jobs=-1)
        else:
            raise ValueError(f"不支持的调优方法: {tuning_type}")
        
        # 执行调优
        tuner.fit(X_train, y_train)
        
        # 更新模型
        self.model = tuner.best_estimator_
        self.model_type = model_type
        self.is_deep_learning = False
        
        return {
            'best_params': tuner.best_params_,
            'best_score': tuner.best_score_,
            'cv_results': tuner.cv_results_
        }
    
    def save_model(self, file_path, model_info=None):
        """
        保存模型（支持版本控制）
        """
        save_dir = os.path.dirname(file_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        if self.is_deep_learning:
            if not file_path.endswith('.keras'):
                file_path = file_path.rsplit('.', 1)[0] + '.keras'
            self.model.save(file_path)
        else:
            if not file_path.endswith('.pkl'):
                file_path = file_path.rsplit('.', 1)[0] + '.pkl'
            with open(file_path, 'wb') as f:
                pickle.dump(self.model, f)
        
        # 保存模型信息（版本控制）
        if model_info is None:
            model_info = {
                'model_type': self.model_type,
                'is_deep_learning': self.is_deep_learning,
                'save_time': datetime.datetime.now().isoformat(),
                'version': '1.0.0'
            }
        
        # 保存模型信息到JSON文件
        info_file_path = file_path + '.json'
        with open(info_file_path, 'w') as f:
            json.dump(model_info, f, indent=4, ensure_ascii=False)
        
        return model_info
    
    def load_model(self, file_path, model_type=None, custom_json_path=None):
        """
        加载模型（支持版本控制）
        兼容多种保存方式：
        1. 分开保存：model.pkl + model.pkl.json
        2. 一起保存：model.pkl (包含所有信息)
        3. 用户自定义提供JSON文件
        """
        model_info = None
        
        # 方式1：如果用户提供了自定义JSON路径，优先使用
        if custom_json_path and os.path.exists(custom_json_path):
            try:
                with open(custom_json_path, 'r') as f:
                    model_info = json.load(f)
            except:
                pass
        
        # 方式2：尝试读取同目录下的JSON文件（model.pkl.json）
        if model_info is None:
            info_file_path = file_path + '.json'
            if os.path.exists(info_file_path):
                try:
                    with open(info_file_path, 'r') as f:
                        model_info = json.load(f)
                except:
                    pass
        
        # 方式3：如果没有分离的JSON文件，尝试从pkl文件中读取
        if model_info is None and file_path.endswith('.pkl'):
            try:
                with open(file_path, 'rb') as f:
                    pkl_content = pickle.load(f)
                    # 检查是否是字典格式（包含model_info的格式）
                    if isinstance(pkl_content, dict):
                        model_info = pkl_content
            except:
                pass
        
        deep_learning_types = ['ann', 'lstm', 'gru', 'cnn', 'transformer', 'mlp']
        
        # 判断是否是字典格式的保存（包含模型和所有信息）
        is_dict_format = model_info is not None and isinstance(model_info, dict) and 'model' in model_info
        
        if model_type is None:
            if file_path.endswith('.h5') or file_path.endswith('.keras'):
                from tensorflow.keras.models import load_model
                self.model = load_model(file_path)
                self.is_deep_learning = True
            else:
                if is_dict_format:
                    # 从字典中提取模型
                    self.model = model_info['model']
                    self.is_deep_learning = model_info.get('is_deep_learning', False)
                else:
                    with open(file_path, 'rb') as f:
                        self.model = pickle.load(f)
                    self.is_deep_learning = False
        else:
            self.model_type = model_type
            if model_type in deep_learning_types:
                from tensorflow.keras.models import load_model
                self.model = load_model(file_path)
                self.is_deep_learning = True
            else:
                if is_dict_format:
                    self.model = model_info['model']
                    self.is_deep_learning = model_info.get('is_deep_learning', False)
                else:
                    with open(file_path, 'rb') as f:
                        self.model = pickle.load(f)
                    self.is_deep_learning = False
        
        return self.model, model_info
    
    # ==================== 集成学习方法 ====================
    
    def create_voting_ensemble(self, models_dict, voting='hard', weights=None):
        """
        创建投票集成模型
        
        Parameters:
        -----------
        models_dict : dict
            模型字典，格式为 {'model_name': model_instance}
        voting : str
            投票方式: 'hard' 或 'soft'
        weights : list, optional
            各模型的权重
            
        Returns:
        --------
        VotingRegressor : 投票集成模型
        """
        estimators = [(name, model) for name, model in models_dict.items()]
        self.model = VotingRegressor(estimators=estimators, weights=weights)
        self.model_type = 'voting_ensemble'
        self.is_deep_learning = False
        return self.model
    
    def create_stacking_ensemble(self, models_dict, meta_model=None, cv=5):
        """
        创建堆叠集成模型
        
        Parameters:
        -----------
        models_dict : dict
            基模型字典，格式为 {'model_name': model_instance}
        meta_model : estimator, optional
            元模型，默认为线性回归
        cv : int
            交叉验证折数
            
        Returns:
        --------
        StackingRegressor : 堆叠集成模型
        """
        if meta_model is None:
            meta_model = LinearRegression()
        
        estimators = [(name, model) for name, model in models_dict.items()]
        self.model = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_model,
            cv=cv,
            passthrough=False
        )
        self.model_type = 'stacking_ensemble'
        self.is_deep_learning = False
        return self.model
    
    def create_weighted_average(self, models_dict, weights=None):
        """
        创建加权平均集成模型
        
        Parameters:
        -----------
        models_dict : dict
            模型字典
        weights : list, optional
            各模型的权重，默认为等权重
            
        Returns:
        --------
        WeightedAverageEnsemble : 加权平均集成模型
        """
        self.model = WeightedAverageEnsemble(models_dict, weights)
        self.model_type = 'weighted_average'
        self.is_deep_learning = False
        return self.model
    
    # ==================== 时间序列特有评估指标 ====================
    
    def evaluate_time_series(self, y_true, y_pred, fast_mode=True):
        """
        时间序列模型评估（包含特有指标）
        
        Parameters:
        -----------
        y_true : array-like
            真实值
        y_pred : array-like
            预测值
        fast_mode : bool
            快速模式，只计算关键指标
            
        Returns:
        --------
        dict : 评估指标字典
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # 基本指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 额外基础指标
        # 解释方差分数
        explained_variance = 1 - np.var(y_true - y_pred) / np.var(y_true)
        # 最大误差
        max_error = np.max(np.abs(y_true - y_pred))
        # 中位数绝对误差
        median_ae = np.median(np.abs(y_true - y_pred))
        # 平均对数平方误差
        msle = np.mean((np.log1p(y_true) - np.log1p(y_pred))**2)
        
        # 时间序列特有指标
        # 1. 平均绝对百分比误差 (MAPE)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        # 2. 对称平均绝对百分比误差 (SMAPE)
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100
        
        # 3. 平均方向准确率 (MDA)
        direction_true = np.sign(np.diff(y_true))
        direction_pred = np.sign(np.diff(y_pred))
        mda = np.mean(direction_true == direction_pred) * 100
        
        # 4. 相对平方误差 (RRSE)
        rrse = np.sqrt(np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
        
        # 5. 相对绝对误差 (RAE)
        rae = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))
        
        # 6. 决定系数调整版
        n = len(y_true)
        p = 1  # 简化处理，假设一个特征
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
        
        if fast_mode:
            # 快速模式：简化计算
            trend_consistency = 0.0
            peak_accuracy = 0.0
            theil_u = 0.0
            correlation = 0.0
            covariance = 0.0
        else:
            # 7. 趋势一致性 (Trend Consistency)
            trend_consistency = self._calculate_trend_consistency(y_true, y_pred)
            
            # 8. 峰值检测准确率
            peak_accuracy = self._calculate_peak_accuracy(y_true, y_pred)
            
            # 9. Theil's U统计量
            theil_u = self._calculate_theil_u(y_true, y_pred)
            
            # 10. 相关系数
            correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
            
            # 11. 协方差
            covariance = np.cov(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'adjusted_r2': adjusted_r2,
            'mape': mape,
            'smape': smape,
            'mda': mda,
            'explained_variance': explained_variance,
            'max_error': max_error,
            'median_ae': median_ae,
            'msle': msle,
            'rrse': rrse,
            'rae': rae,
            'trend_consistency': trend_consistency,
            'peak_accuracy': peak_accuracy,
            'theil_u': theil_u,
            'correlation': correlation,
            'covariance': covariance
        }
    
    def _calculate_trend_consistency(self, y_true, y_pred, window=5):
        """
        计算趋势一致性 - 优化版本
        """
        if len(y_true) < window:
            return 0.0
        
        # 使用向量化操作替代循环
        n = len(y_true) - window + 1
        true_trends = np.zeros(n)
        pred_trends = np.zeros(n)
        
        for i in range(n):
            true_trends[i] = np.sign(y_true[i+window-1] - y_true[i])
            pred_trends[i] = np.sign(y_pred[i+window-1] - y_pred[i])
        
        return np.mean(true_trends == pred_trends) * 100
    
    def _calculate_peak_accuracy(self, y_true, y_pred, threshold_percentile=90):
        """
        计算峰值检测准确率
        """
        threshold = np.percentile(y_true, threshold_percentile)
        true_peaks = y_true > threshold
        pred_peaks = y_pred > threshold
        
        if np.sum(true_peaks) == 0:
            return 100.0
        
        return np.mean(true_peaks == pred_peaks) * 100
    
    def _calculate_theil_u(self, y_true, y_pred):
        """
        计算Theil's U统计量
        """
        numerator = np.sqrt(np.mean((y_pred - y_true) ** 2))
        denominator = np.sqrt(np.mean(y_pred ** 2)) + np.sqrt(np.mean(y_true ** 2))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def cross_validate(self, X, y, cv=5, scoring='r2'):
        """
        交叉验证评估
        
        Parameters:
        -----------
        X : array-like
            特征数据
        y : array-like
            目标数据
        cv : int
            交叉验证折数
        scoring : str
            评分指标
            
        Returns:
        --------
        dict : 交叉验证结果
        """
        if self.model is None:
            raise ValueError("请先选择或训练模型")
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        }


class WeightedAverageEnsemble:
    """
    加权平均集成模型
    """
    def __init__(self, models_dict, weights=None):
        self.models = models_dict
        self.weights = weights
        self.model_names = list(models_dict.keys())
        
        if weights is None:
            self.weights = [1.0 / len(models_dict)] * len(models_dict)
        else:
            # 归一化权重
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def fit(self, X, y):
        """
        训练所有基模型
        """
        for name, model in self.models.items():
            model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        加权平均预测
        """
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
        
        # 加权平均
        weighted_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            weighted_pred += self.weights[i] * pred
        
        return weighted_pred
