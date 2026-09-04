"""
迁移学习模块
支持模型微调和知识迁移
"""
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
import warnings
warnings.filterwarnings('ignore')


class TransferLearningModel:
    """
    迁移学习模型基类
    支持从预训练模型迁移知识到新任务
    """
    
    def __init__(self, base_model=None, freeze_layers=None):
        """
        初始化迁移学习模型
        
        Parameters:
        -----------
        base_model : trained model
            预训练的基础模型
        freeze_layers : list
            要冻结的层（对于深度学习模型）
        """
        self.base_model = base_model
        self.freeze_layers = freeze_layers
        self.target_model = None
        self.transfer_history = []
        
    def load_pretrained_model(self, model_path):
        """
        加载预训练模型
        
        Parameters:
        -----------
        model_path : str
            预训练模型文件路径
        """
        try:
            with open(model_path, 'rb') as f:
                self.base_model = pickle.load(f)
            print(f"成功加载预训练模型: {model_path}")
            return True
        except Exception as e:
            print(f"加载预训练模型失败: {str(e)}")
            return False
    
    def save_model(self, model_path):
        """
        保存迁移后的模型
        """
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(self.target_model, f)
            print(f"模型已保存到: {model_path}")
            return True
        except Exception as e:
            print(f"保存模型失败: {str(e)}")
            return False


class SklearnTransferLearning(TransferLearningModel):
    """
    基于Scikit-learn的迁移学习
    适用于随机森林、梯度提升等模型
    """
    
    def __init__(self, base_model=None, transfer_strategy='fine_tune'):
        """
        初始化
        
        Parameters:
        -----------
        transfer_strategy : str
            迁移策略: 'fine_tune', 'feature_extractor', 'ensemble'
        """
        super().__init__(base_model)
        self.transfer_strategy = transfer_strategy
        
    def transfer(self, X_source, y_source, X_target, y_target, 
                 sample_weight_source=0.3, sample_weight_target=0.7):
        """
        执行知识迁移
        
        Parameters:
        -----------
        X_source : array-like
            源域特征数据
        y_source : array-like
            源域目标数据
        X_target : array-like
            目标域特征数据
        y_target : array-like
            目标域目标数据
        sample_weight_source : float
            源域样本权重
        sample_weight_target : float
            目标域样本权重
        """
        if self.transfer_strategy == 'fine_tune':
            return self._fine_tune(X_source, y_source, X_target, y_target, 
                                 sample_weight_source, sample_weight_target)
        elif self.transfer_strategy == 'feature_extractor':
            return self._feature_extractor_transfer(X_source, y_source, X_target, y_target)
        elif self.transfer_strategy == 'ensemble':
            return self._ensemble_transfer(X_source, y_source, X_target, y_target)
        else:
            raise ValueError(f"未知的迁移策略: {self.transfer_strategy}")
    
    def _fine_tune(self, X_source, y_source, X_target, y_target, 
                   sample_weight_source, sample_weight_target):
        """
        微调策略：使用源域和目标域数据联合训练
        """
        # 合并数据
        X_combined = np.vstack([X_source, X_target])
        y_combined = np.concatenate([y_source, y_target])
        
        # 创建样本权重
        weights_source = np.ones(len(y_source)) * sample_weight_source
        weights_target = np.ones(len(y_target)) * sample_weight_target
        sample_weights = np.concatenate([weights_source, weights_target])
        
        # 创建新模型（基于源模型的类型）
        if isinstance(self.base_model, RandomForestRegressor):
            self.target_model = RandomForestRegressor(
                n_estimators=self.base_model.n_estimators,
                max_depth=self.base_model.max_depth,
                random_state=42
            )
        elif isinstance(self.base_model, GradientBoostingRegressor):
            self.target_model = GradientBoostingRegressor(
                n_estimators=self.base_model.n_estimators,
                learning_rate=self.base_model.learning_rate,
                max_depth=self.base_model.max_depth,
                random_state=42
            )
        else:
            # 默认使用相同的模型类型
            self.target_model = type(self.base_model)()
        
        # 训练模型
        self.target_model.fit(X_combined, y_combined, sample_weight=sample_weights)
        
        # 记录迁移历史
        self.transfer_history.append({
            'strategy': 'fine_tune',
            'source_samples': len(X_source),
            'target_samples': len(X_target),
            'source_weight': sample_weight_source,
            'target_weight': sample_weight_target
        })
        
        print(f"微调完成: 使用 {len(X_source)} 个源域样本和 {len(X_target)} 个目标域样本")
        return self.target_model
    
    def _feature_extractor_transfer(self, X_source, y_source, X_target, y_target):
        """
        特征提取器策略：使用源模型提取特征，训练新模型
        """
        # 使用源模型生成伪标签或特征重要性
        if hasattr(self.base_model, 'feature_importances_'):
            # 基于特征重要性选择重要特征
            importances = self.base_model.feature_importances_
            top_features_idx = np.argsort(importances)[-int(len(importances)*0.7):]
            
            X_target_selected = X_target[:, top_features_idx]
            
            # 在选中的特征上训练新模型
            self.target_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.target_model.fit(X_target_selected, y_target)
            
            # 保存特征选择信息
            self.selected_features = top_features_idx
            
            self.transfer_history.append({
                'strategy': 'feature_extractor',
                'selected_features': len(top_features_idx),
                'total_features': len(importances)
            })
            
            print(f"特征提取器迁移完成: 选择了 {len(top_features_idx)} 个重要特征")
        else:
            # 如果不支持特征重要性，使用全部特征
            self.target_model = type(self.base_model)()
            self.target_model.fit(X_target, y_target)
            
            self.transfer_history.append({
                'strategy': 'feature_extractor',
                'selected_features': X_target.shape[1],
                'total_features': X_target.shape[1]
            })
        
        return self.target_model
    
    def _ensemble_transfer(self, X_source, y_source, X_target, y_target):
        """
        集成策略：组合源模型和目标域模型
        """
        # 在目标域数据上训练新模型
        target_model = type(self.base_model)()
        target_model.fit(X_target, y_target)
        
        # 保存两个模型
        self.source_model = self.base_model
        self.target_model = target_model
        
        self.transfer_history.append({
            'strategy': 'ensemble',
            'source_samples': len(X_source),
            'target_samples': len(X_target)
        })
        
        print(f"集成迁移完成: 源模型 + 目标域模型")
        return self.target_model
    
    def predict(self, X):
        """
        使用迁移后的模型进行预测
        """
        if self.transfer_strategy == 'ensemble':
            # 集成预测：平均两个模型的预测结果
            pred_source = self.source_model.predict(X)
            pred_target = self.target_model.predict(X)
            return (pred_source + pred_target) / 2
        elif self.transfer_strategy == 'feature_extractor' and hasattr(self, 'selected_features'):
            # 使用选中的特征
            X_selected = X[:, self.selected_features]
            return self.target_model.predict(X_selected)
        else:
            return self.target_model.predict(X)


class DeepTransferLearning(TransferLearningModel):
    """
    深度学习迁移学习
    适用于神经网络模型
    """
    
    def __init__(self, base_model=None, freeze_layers_ratio=0.5):
        """
        初始化
        
        Parameters:
        -----------
        freeze_layers_ratio : float
            冻结层比例（0-1之间）
        """
        super().__init__(base_model)
        self.freeze_layers_ratio = freeze_layers_ratio
        self.transfer_model = None
        
    def build_transfer_model(self, input_shape, output_units=1, 
                            new_layers=[64, 32], dropout_rate=0.2):
        """
        构建迁移模型架构
        
        Parameters:
        -----------
        input_shape : tuple
            输入形状
        output_units : int
            输出单元数
        new_layers : list
            新添加的全连接层大小
        dropout_rate : float
            Dropout比率
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            # 创建基础模型（如果提供了预训练模型）
            if self.base_model is not None:
                # 复制基础模型架构
                base_config = self.base_model.get_config()
                base_model_copy = type(self.base_model).from_config(base_config)
                
                # 冻结部分层
                if hasattr(base_model_copy, 'layers'):
                    n_layers = len(base_model_copy.layers)
                    freeze_until = int(n_layers * self.freeze_layers_ratio)
                    
                    for i, layer in enumerate(base_model_copy.layers):
                        if i < freeze_until:
                            layer.trainable = False
                        else:
                            layer.trainable = True
                
                # 构建迁移模型
                inputs = keras.Input(shape=input_shape)
                x = base_model_copy(inputs)
                
                # 添加新的全连接层
                for units in new_layers:
                    x = layers.Dense(units, activation='relu')(x)
                    x = layers.Dropout(dropout_rate)(x)
                
                outputs = layers.Dense(output_units)(x)
                
                self.transfer_model = keras.Model(inputs=inputs, outputs=outputs)
                
            else:
                # 从头创建新模型
                inputs = keras.Input(shape=input_shape)
                x = inputs
                
                for units in new_layers:
                    x = layers.Dense(units, activation='relu')(x)
                    x = layers.Dropout(dropout_rate)(x)
                
                outputs = layers.Dense(output_units)(x)
                
                self.transfer_model = keras.Model(inputs=inputs, outputs=outputs)
            
            # 编译模型
            self.transfer_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            print("迁移模型构建完成")
            self.transfer_model.summary()
            
            return self.transfer_model
            
        except ImportError:
            print("TensorFlow未安装，无法构建深度学习迁移模型")
            return None
    
    def fit(self, X, y, validation_split=0.2, epochs=50, batch_size=32):
        """
        训练迁移模型
        """
        if self.transfer_model is None:
            raise ValueError("请先调用build_transfer_model()构建模型")
        
        try:
            import tensorflow as tf
            
            history = self.transfer_model.fit(
                X, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            self.transfer_history.append({
                'epochs': epochs,
                'batch_size': batch_size,
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1]
            })
            
            return history
            
        except Exception as e:
            print(f"训练失败: {str(e)}")
            return None
    
    def predict(self, X):
        """
        使用迁移模型进行预测
        """
        if self.transfer_model is None:
            raise ValueError("模型尚未训练")
        
        return self.transfer_model.predict(X)


class DomainAdaptation:
    """
    域适应技术
    处理源域和目标域分布不一致的问题
    """
    
    def __init__(self, method='correlation_alignment'):
        """
        初始化
        
        Parameters:
        -----------
        method : str
            域适应方法: 'correlation_alignment', 'subspace_alignment', 'instance_weighting'
        """
        self.method = method
        self.transformation_matrix = None
        self.source_mean = None
        self.target_mean = None
        
    def fit(self, X_source, X_target):
        """
        学习域适应变换
        
        Parameters:
        -----------
        X_source : array-like
            源域数据
        X_target : array-like
            目标域数据
        """
        if self.method == 'correlation_alignment':
            self._fit_correlation_alignment(X_source, X_target)
        elif self.method == 'subspace_alignment':
            self._fit_subspace_alignment(X_source, X_target)
        elif self.method == 'instance_weighting':
            self._fit_instance_weighting(X_source, X_target)
        else:
            raise ValueError(f"未知的域适应方法: {self.method}")
    
    def _fit_correlation_alignment(self, X_source, X_target):
        """
        相关对齐（CORAL）
        """
        # 计算协方差矩阵
        cov_source = np.atleast_2d(np.cov(X_source.T)) + np.eye(X_source.shape[1]) * 1e-6
        cov_target = np.atleast_2d(np.cov(X_target.T)) + np.eye(X_target.shape[1]) * 1e-6
        
        # 计算变换矩阵
        cov_source_sqrt = self._matrix_sqrt(cov_source)
        cov_target_sqrt = self._matrix_sqrt(cov_target)
        
        self.transformation_matrix = np.linalg.pinv(cov_source_sqrt) @ cov_target_sqrt
        
        print("相关对齐完成")
    
    def _matrix_sqrt(self, matrix):
        """
        计算矩阵的平方根
        """
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, 0)  # 确保非负
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    
    def _fit_subspace_alignment(self, X_source, X_target, n_components=10):
        """
        子空间对齐
        """
        from sklearn.decomposition import PCA
        n_components = max(1, min(n_components, X_source.shape[0], X_target.shape[0], X_source.shape[1], X_target.shape[1]))
        
        # 对源域和目标域分别进行PCA
        pca_source = PCA(n_components=n_components)
        pca_target = PCA(n_components=n_components)
        
        pca_source.fit(X_source)
        pca_target.fit(X_target)
        
        # 保存PCA模型
        self.pca_source = pca_source
        self.pca_target = pca_target
        
        # 计算对齐矩阵
        source_components = pca_source.components_.T
        target_components = pca_target.components_.T
        
        self.alignment_matrix = source_components.T @ target_components
        
        print(f"子空间对齐完成: {n_components} 个主成分")
    
    def _fit_instance_weighting(self, X_source, X_target):
        """
        实例加权
        """
        # 计算源域和目标域的均值
        self.source_mean = np.mean(X_source, axis=0)
        self.target_mean = np.mean(X_target, axis=0)
        
        # 计算协方差
        self.source_cov = np.cov(X_source.T)
        self.target_cov = np.cov(X_target.T)
        
        print("实例加权准备完成")
    
    def transform(self, X, domain='source'):
        """
        应用域适应变换
        
        Parameters:
        -----------
        X : array-like
            要变换的数据
        domain : str
            数据域: 'source' 或 'target'
        """
        if self.method == 'correlation_alignment':
            if domain == 'source':
                return X @ self.transformation_matrix
            else:
                return X
        elif self.method == 'subspace_alignment':
            if domain == 'source':
                X_pca = self.pca_source.transform(X)
                return X_pca @ self.alignment_matrix
            else:
                return self.pca_target.transform(X)
        elif self.method == 'instance_weighting':
            # 计算实例权重
            if domain == 'source':
                weights = self._compute_instance_weights(X)
                return X, weights
            else:
                return X, np.ones(len(X))
        else:
            return X
    
    def _compute_instance_weights(self, X):
        """
        计算实例权重
        """
        # 简单的基于距离的权重计算
        distances = np.linalg.norm(X - self.target_mean, axis=1)
        weights = np.exp(-distances / np.mean(distances))
        return weights / np.sum(weights) * len(weights)


class TransferLearningManager:
    """
    迁移学习管理器
    管理多个迁移学习任务和预训练模型库
    """
    
    def __init__(self, model_library_path='transfer_models'):
        """
        初始化
        
        Parameters:
        -----------
        model_library_path : str
            预训练模型库路径
        """
        self.model_library_path = model_library_path
        self.pretrained_models = {}
        self.transfer_tasks = []
        
        # 创建模型库目录
        os.makedirs(model_library_path, exist_ok=True)
        
    def register_pretrained_model(self, name, model, metadata=None):
        """
        注册预训练模型到模型库
        
        Parameters:
        -----------
        name : str
            模型名称
        model : trained model
            预训练模型
        metadata : dict
            模型元数据
        """
        self.pretrained_models[name] = {
            'model': model,
            'metadata': metadata or {},
            'registered_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存到文件
        model_path = os.path.join(self.model_library_path, f"{name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # 保存元数据
        if metadata:
            metadata_path = os.path.join(self.model_library_path, f"{name}_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
        
        print(f"预训练模型 '{name}' 已注册并保存")
    
    def load_pretrained_model(self, name):
        """
        从模型库加载预训练模型
        """
        if name in self.pretrained_models:
            return self.pretrained_models[name]['model']
        
        model_path = os.path.join(self.model_library_path, f"{name}.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # 加载元数据
            metadata_path = os.path.join(self.model_library_path, f"{name}_metadata.pkl")
            metadata = None
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
            
            self.pretrained_models[name] = {
                'model': model,
                'metadata': metadata,
                'loaded_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return model
        else:
            print(f"模型 '{name}' 不存在")
            return None
    
    def list_pretrained_models(self):
        """
        列出所有可用的预训练模型
        """
        models = []
        for filename in os.listdir(self.model_library_path):
            if filename.endswith('.pkl') and not filename.endswith('_metadata.pkl'):
                name = filename[:-4]
                metadata_file = f"{name}_metadata.pkl"
                metadata = None
                
                if metadata_file in os.listdir(self.model_library_path):
                    with open(os.path.join(self.model_library_path, metadata_file), 'rb') as f:
                        metadata = pickle.load(f)
                
                models.append({
                    'name': name,
                    'metadata': metadata
                })
        
        return models
    
    def create_transfer_task(self, source_model_name, target_data_info, 
                           transfer_strategy='fine_tune'):
        """
        创建迁移学习任务
        """
        task = {
            'task_id': len(self.transfer_tasks) + 1,
            'source_model': source_model_name,
            'target_data': target_data_info,
            'strategy': transfer_strategy,
            'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'pending'
        }
        
        self.transfer_tasks.append(task)
        print(f"迁移任务 {task['task_id']} 已创建")
        
        return task
    
    def execute_transfer_task(self, task_id, X_source, y_source, X_target, y_target):
        """
        执行迁移学习任务
        """
        task = next((t for t in self.transfer_tasks if t['task_id'] == task_id), None)
        if task is None:
            print(f"任务 {task_id} 不存在")
            return None
        
        # 加载源模型
        source_model = self.load_pretrained_model(task['source_model'])
        if source_model is None:
            task['status'] = 'failed'
            return None
        
        # 创建迁移学习器
        transfer_learner = SklearnTransferLearning(
            base_model=source_model,
            transfer_strategy=task['strategy']
        )
        
        # 执行迁移
        target_model = transfer_learner.transfer(X_source, y_source, X_target, y_target)
        
        task['status'] = 'completed'
        task['completed_at'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return target_model
