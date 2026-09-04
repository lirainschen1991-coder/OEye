"""
自动机器学习(AutoML)模块
自动选择最优模型和超参数
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings('ignore')


class AutoMLTrainer:
    """
    自动机器学习训练器
    自动搜索最优模型和超参数组合
    """
    
    def __init__(self, task='regression', time_limit=300, cv_folds=5, cv_strategy='auto', random_state=42):
        """
        初始化AutoML训练器
        
        Parameters:
        -----------
        task : str
            任务类型，'regression' 或 'classification'
        time_limit : int
            自动搜索的时间限制（秒）
        cv_folds : int
            交叉验证折数
        """
        self.task = task
        self.time_limit = time_limit
        self.cv_folds = cv_folds
        self.cv_strategy = cv_strategy
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.results_df = None
        self.search_history = []
        
        # 定义模型搜索空间
        self.model_space = self._define_model_space()

    def _build_cv(self, y=None):
        """Build a validation splitter appropriate for the task/data type."""
        if self.cv_strategy == 'time_series':
            return TimeSeriesSplit(n_splits=self.cv_folds)
        if self.task == 'classification' and self.cv_strategy in ('auto', 'stratified'):
            return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        if self.cv_strategy == 'kfold':
            return KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        return self.cv_folds
        
    def _define_model_space(self):
        """
        定义模型和超参数搜索空间
        """
        if self.task == 'regression':
            return {
                'linear_regression': {
                    'model': LinearRegression(),
                    'params': {}
                },
                'ridge': {
                    'model': Ridge(),
                    'params': {
                        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr']
                    }
                },
                'lasso': {
                    'model': Lasso(),
                    'params': {
                        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                        'max_iter': [1000, 2000, 5000]
                    }
                },
                'random_forest': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7, 10, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'gradient_boosting': {
                    'model': GradientBoostingRegressor(random_state=42),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.05, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                },
                'svr': {
                    'model': SVR(),
                    'params': {
                        'C': [0.1, 1.0, 10.0, 100.0],
                        'kernel': ['rbf', 'linear', 'poly'],
                        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                    }
                },
                'knn': {
                    'model': KNeighborsRegressor(),
                    'params': {
                        'n_neighbors': [3, 5, 7, 10, 15],
                        'weights': ['uniform', 'distance'],
                        'metric': ['euclidean', 'manhattan', 'minkowski']
                    }
                }
            }
        else:
            # 分类任务的搜索空间
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            from sklearn.neighbors import KNeighborsClassifier
            
            return {
                'logistic_regression': {
                    'model': LogisticRegression(random_state=42, max_iter=1000),
                    'params': {
                        'C': [0.1, 1.0, 10.0, 100.0],
                        'solver': ['lbfgs', 'liblinear']
                    }
                },
                'random_forest': {
                    'model': RandomForestClassifier(random_state=42),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7, 10, None],
                        'min_samples_split': [2, 5, 10]
                    }
                },
                'gradient_boosting': {
                    'model': GradientBoostingClassifier(random_state=42),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth': [3, 5, 7]
                    }
                },
                'svc': {
                    'model': SVC(random_state=42),
                    'params': {
                        'C': [0.1, 1.0, 10.0],
                        'kernel': ['rbf', 'linear'],
                        'gamma': ['scale', 'auto']
                    }
                },
                'knn': {
                    'model': KNeighborsClassifier(),
                    'params': {
                        'n_neighbors': [3, 5, 7, 10],
                        'weights': ['uniform', 'distance']
                    }
                }
            }
    
    def search(self, X, y, scoring=None, search_method='random', n_iter=20, verbose=1):
        """
        执行自动模型搜索
        
        Parameters:
        -----------
        X : array-like
            训练特征
        y : array-like
            训练目标
        scoring : str or callable
            评估指标
        search_method : str
            搜索方法，'grid' 或 'random'
        n_iter : int
            随机搜索的迭代次数
        verbose : int
            输出详细程度
            
        Returns:
        --------
        dict
            搜索结果，包含最佳模型、参数和分数
        """
        if scoring is None:
            scoring = 'neg_mean_squared_error' if self.task == 'regression' else 'accuracy'
        
        results = []
        
        print(f"开始AutoML搜索，任务类型: {self.task}")
        print(f"搜索方法: {search_method}, 迭代次数: {n_iter}")
        print(f"交叉验证折数: {self.cv_folds}")
        print("-" * 60)
        
        for model_name, config in self.model_space.items():
            if verbose > 0:
                print(f"\n正在搜索模型: {model_name}...")
            
            model = config['model']
            param_grid = config['params']
            
            try:
                if search_method == 'grid' and param_grid:
                    # 网格搜索
                    search = GridSearchCV(
                        model, param_grid,
                        cv=self._build_cv(y),
                        scoring=scoring,
                        n_jobs=-1,
                        verbose=0
                    )
                elif param_grid:
                    # 随机搜索
                    search = RandomizedSearchCV(
                        model, param_grid,
                        n_iter=min(n_iter, self._count_combinations(param_grid)),
                        cv=self._build_cv(y),
                        scoring=scoring,
                        n_jobs=-1,
                        random_state=42,
                        verbose=0
                    )
                else:
                    # 无参数需要调优
                    scores = cross_val_score(model, X, y, cv=self._build_cv(y), scoring=scoring)
                    best_score = scores.mean()
                    best_params = {}
                    
                    results.append({
                        'model_name': model_name,
                        'best_score': best_score,
                        'best_params': best_params,
                        'cv_scores': scores,
                        'std': scores.std()
                    })
                    
                    self.search_history.append({
                        'model': model_name,
                        'params': best_params,
                        'score': best_score
                    })
                    
                    if verbose > 0:
                        print(f"  分数: {best_score:.4f} (+/- {scores.std()*2:.4f})")
                    continue
                
                # 执行搜索
                search.fit(X, y)
                
                results.append({
                    'model_name': model_name,
                    'best_score': search.best_score_,
                    'best_params': search.best_params_,
                    'cv_results': search.cv_results_,
                    'best_estimator': search.best_estimator_
                })
                
                self.search_history.append({
                    'model': model_name,
                    'params': search.best_params_,
                    'score': search.best_score_
                })
                
                if verbose > 0:
                    print(f"  最佳分数: {search.best_score_:.4f}")
                    print(f"  最佳参数: {search.best_params_}")
                    
            except Exception as e:
                print(f"  模型 {model_name} 搜索失败: {str(e)}")
                continue
        
        # 创建结果DataFrame
        self.results_df = pd.DataFrame([
            {
                'Model': r['model_name'],
                'Score': r['best_score'],
                'Parameters': str(r['best_params'])
            }
            for r in results
        ])
        
        # 按分数排序（注意：对于neg_mean_squared_error，越大越好）
        self.results_df = self.results_df.sort_values('Score', ascending=False)
        
        # 选择最佳模型
        if results:
            best_result = max(results, key=lambda x: x['best_score'])
            self.best_model = best_result.get('best_estimator')
            self.best_params = best_result['best_params']
            self.best_score = best_result['best_score']
            
            print("\n" + "=" * 60)
            print(f"最佳模型: {best_result['model_name']}")
            print(f"最佳分数: {self.best_score:.4f}")
            print(f"最佳参数: {self.best_params}")
            print("=" * 60)
        
        return {
            'best_model': self.best_model,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': results,
            'results_df': self.results_df
        }
    
    def _count_combinations(self, param_grid):
        """计算参数组合数量"""
        count = 1
        for values in param_grid.values():
            count *= len(values)
        return count
    
    def get_search_summary(self):
        """
        获取搜索摘要
        """
        if self.results_df is None:
            return "尚未执行搜索"
        
        summary = {
            'total_models_tested': len(self.results_df),
            'best_model': self.results_df.iloc[0]['Model'] if len(self.results_df) > 0 else None,
            'best_score': self.results_df.iloc[0]['Score'] if len(self.results_df) > 0 else None,
            'all_models': self.results_df.to_dict('records')
        }
        
        return summary
    
    def predict(self, X):
        """
        使用最佳模型进行预测
        """
        if self.best_model is None:
            raise ValueError("尚未训练模型，请先调用search()方法")
        
        return self.best_model.predict(X)
    
    def get_feature_importance(self, feature_names=None):
        """
        获取特征重要性（如果最佳模型支持）
        """
        if self.best_model is None:
            return None
        
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importances = np.abs(self.best_model.coef_)
            if importances.ndim > 1:
                importances = importances.flatten()
        else:
            return None
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df


class AutoMLClassifier(AutoMLTrainer):
    """
    自动机器学习分类器
    """
    
    def __init__(self, time_limit=300, cv_folds=5, cv_strategy='auto', random_state=42):
        super().__init__(
            task='classification',
            time_limit=time_limit,
            cv_folds=cv_folds,
            cv_strategy=cv_strategy,
            random_state=random_state
        )
    
    def predict_proba(self, X):
        """
        预测概率（分类任务）
        """
        if self.best_model is None:
            raise ValueError("尚未训练模型")
        
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X)
        else:
            raise ValueError("当前模型不支持概率预测")


class AutoMLRegressor(AutoMLTrainer):
    """
    自动机器学习回归器
    """
    
    def __init__(self, time_limit=300, cv_folds=5, cv_strategy='auto', random_state=42):
        super().__init__(
            task='regression',
            time_limit=time_limit,
            cv_folds=cv_folds,
            cv_strategy=cv_strategy,
            random_state=random_state
        )
