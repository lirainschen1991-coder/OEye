"""
模型可解释性模块
提供模型决策过程的可视化和解释
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.inspection import permutation_importance, partial_dependence
import warnings
warnings.filterwarnings('ignore')

# 可选导入 - 如果未安装则提供备用实现
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("警告: shap模块未安装，SHAP功能将不可用。请运行: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("警告: lime模块未安装，LIME功能将不可用。请运行: pip install lime")


class ModelExplainer:
    """
    模型解释器
    提供多种模型可解释性方法
    """
    
    def __init__(self, model, feature_names=None, task='regression'):
        """
        初始化模型解释器
        
        Parameters:
        -----------
        model : trained model
            已训练的模型
        feature_names : list
            特征名称列表
        task : str
            任务类型，'regression' 或 'classification'
        """
        self.model = model
        self.feature_names = feature_names
        self.task = task
        self.explainer = None
        self.shap_values = None
        self.lime_explainer = None
        
    def explain_with_shap(self, X, sample_size=100):
        """
        使用SHAP解释模型预测
        
        Parameters:
        -----------
        X : array-like
            特征数据
        sample_size : int
            用于解释的样本数量
        """
        if not SHAP_AVAILABLE:
            print("SHAP模块未安装，无法使用SHAP解释功能")
            return None
            
        try:
            # 采样以减少计算时间
            if len(X) > sample_size:
                X_sample = shap.sample(X, sample_size)
            else:
                X_sample = X
            
            # 创建SHAP解释器
            if hasattr(self.model, 'predict_proba'):
                self.explainer = shap.TreeExplainer(self.model) if hasattr(self.model, 'tree_') or hasattr(self.model, 'estimators_') else shap.KernelExplainer(self.model.predict_proba, shap.sample(X, 50))
            else:
                self.explainer = shap.TreeExplainer(self.model) if hasattr(self.model, 'tree_') or hasattr(self.model, 'estimators_') else shap.KernelExplainer(self.model.predict, shap.sample(X, 50))
            
            # 计算SHAP值
            self.shap_values = self.explainer.shap_values(X_sample)
            
            return {
                'shap_values': self.shap_values,
                'expected_value': self.explainer.expected_value,
                'X_sample': X_sample
            }
        except Exception as e:
            print(f"SHAP解释失败: {str(e)}")
            return None
    
    def plot_shap_summary(self, use_plotly=False):
        """
        绘制SHAP摘要图
        """
        if self.shap_values is None:
            raise ValueError("请先调用explain_with_shap()方法")
        
        if use_plotly:
            # 使用Plotly绘制交互式SHAP图
            if isinstance(self.shap_values, list):
                shap_vals = self.shap_values[0]
            else:
                shap_vals = self.shap_values
            
            # 计算特征重要性
            feature_importance = np.abs(shap_vals).mean(axis=0)
            if self.feature_names is None:
                self.feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
            
            # 创建DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=True)
            
            fig = go.Figure(go.Bar(
                x=importance_df['importance'],
                y=importance_df['feature'],
                orientation='h',
                marker_color='skyblue'
            ))
            
            fig.update_layout(
                title='SHAP Feature Importance',
                xaxis_title='Mean |SHAP Value|',
                yaxis_title='Feature',
                height=500
            )
            
            return fig
        else:
            # 使用matplotlib
            plt.figure(figsize=(10, 6))
            shap.summary_plot(self.shap_values, feature_names=self.feature_names, show=False)
            plt.title('SHAP Feature Importance')
            plt.tight_layout()
            return plt
    
    def plot_shap_waterfall(self, instance_idx=0):
        """
        绘制SHAP瀑布图（单样本解释）
        """
        if self.shap_values is None:
            raise ValueError("请先调用explain_with_shap()方法")
        
        plt.figure(figsize=(10, 6))
        
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0][instance_idx]
        else:
            shap_vals = self.shap_values[instance_idx]
        
        # 创建瀑布图
        shap.waterfall_plot(shap.Explanation(
            values=shap_vals,
            base_values=self.explainer.expected_value if not isinstance(self.explainer.expected_value, list) else self.explainer.expected_value[0],
            feature_names=self.feature_names
        ), show=False)
        
        plt.title(f'SHAP Waterfall Plot - Instance {instance_idx}')
        plt.tight_layout()
        return plt
    
    def explain_with_lime(self, X_train, X_instance, num_features=10):
        """
        使用LIME解释单条预测
        
        Parameters:
        -----------
        X_train : array-like
            训练数据（用于生成解释器）
        X_instance : array-like
            要解释的样本
        num_features : int
            显示的特征数量
        """
        if not LIME_AVAILABLE:
            print("LIME模块未安装，无法使用LIME解释功能")
            return None
            
        try:
            # 创建LIME解释器
            if self.lime_explainer is None:
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train,
                    feature_names=self.feature_names,
                    class_names=['prediction'],
                    mode='regression' if self.task == 'regression' else 'classification'
                )
            
            # 生成解释
            if self.task == 'classification' and hasattr(self.model, 'predict_proba'):
                explanation = self.lime_explainer.explain_instance(
                    X_instance, 
                    self.model.predict_proba,
                    num_features=num_features
                )
            else:
                explanation = self.lime_explainer.explain_instance(
                    X_instance,
                    self.model.predict,
                    num_features=num_features
                )
            
            return explanation
        except Exception as e:
            print(f"LIME解释失败: {str(e)}")
            return None
    
    def plot_lime_explanation(self, explanation, use_plotly=False):
        """
        绘制LIME解释图
        """
        if explanation is None:
            return None
        
        if use_plotly:
            # 转换为Plotly图表
            features = explanation.as_list()
            
            # 解析特征和权重
            feature_names = [f[0] for f in features]
            weights = [f[1] for f in features]
            colors = ['red' if w < 0 else 'green' for w in weights]
            
            fig = go.Figure(go.Bar(
                x=weights,
                y=feature_names,
                orientation='h',
                marker_color=colors
            ))
            
            fig.update_layout(
                title='LIME Explanation',
                xaxis_title='Weight',
                yaxis_title='Feature',
                height=400
            )
            
            return fig
        else:
            # 使用LIME内置绘图
            return explanation.as_pyplot_figure()
    
    def get_permutation_importance(self, X, y, n_repeats=10):
        """
        计算排列重要性
        """
        try:
            result = permutation_importance(
                self.model, X, y,
                n_repeats=n_repeats,
                random_state=42,
                n_jobs=-1
            )
            
            if self.feature_names is None:
                self.feature_names = [f'Feature_{i}' for i in range(len(result.importances_mean))]
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': result.importances_mean,
                'std': result.importances_std
            }).sort_values('importance', ascending=False)
            
            return importance_df
        except Exception as e:
            print(f"排列重要性计算失败: {str(e)}")
            return None
    
    def plot_permutation_importance(self, importance_df, use_plotly=False):
        """
        绘制排列重要性图
        """
        if importance_df is None:
            return None
        
        if use_plotly:
            fig = go.Figure(go.Bar(
                x=importance_df['importance'],
                y=importance_df['feature'],
                orientation='h',
                error_x=dict(
                    type='data',
                    array=importance_df['std'],
                    visible=True
                ),
                marker_color='lightcoral'
            ))
            
            fig.update_layout(
                title='Permutation Feature Importance',
                xaxis_title='Importance',
                yaxis_title='Feature',
                height=500
            )
            
            return fig
        else:
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['feature'], importance_df['importance'], 
                    xerr=importance_df['std'], color='lightcoral')
            plt.xlabel('Importance')
            plt.title('Permutation Feature Importance')
            plt.tight_layout()
            return plt
    
    def get_partial_dependence(self, X, features, grid_resolution=20):
        """
        计算部分依赖图数据
        """
        try:
            pd_results = partial_dependence(
                self.model, X, features,
                grid_resolution=grid_resolution,
                kind='average'
            )
            
            return pd_results
        except Exception as e:
            print(f"部分依赖图计算失败: {str(e)}")
            return None
    
    def plot_partial_dependence(self, pd_results, feature_names=None, use_plotly=False):
        """
        绘制部分依赖图
        """
        if pd_results is None:
            return None
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(pd_results['grid_values']))]
        
        n_features = len(pd_results['grid_values'])
        
        if use_plotly:
            fig = make_subplots(
                rows=(n_features + 1) // 2,
                cols=2,
                subplot_titles=feature_names
            )
            
            for i, (grid_values, pd_values) in enumerate(zip(pd_results['grid_values'], pd_results['average'])):
                row = i // 2 + 1
                col = i % 2 + 1
                
                fig.add_trace(
                    go.Scatter(
                        x=grid_values,
                        y=pd_values[0] if pd_values.ndim > 1 else pd_values,
                        mode='lines',
                        name=feature_names[i]
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title='Partial Dependence Plots',
                height=300 * ((n_features + 1) // 2),
                showlegend=False
            )
            
            return fig
        else:
            # 使用matplotlib
            fig, axes = plt.subplots((n_features + 1) // 2, 2, figsize=(12, 4 * ((n_features + 1) // 2)))
            if n_features == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, (grid_values, pd_values) in enumerate(zip(pd_results['grid_values'], pd_results['average'])):
                axes[i].plot(grid_values, pd_values[0] if pd_values.ndim > 1 else pd_values)
                axes[i].set_xlabel(feature_names[i])
                axes[i].set_ylabel('Partial Dependence')
                axes[i].set_title(f'PDP: {feature_names[i]}')
                axes[i].grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            return plt
    
    def get_feature_importance_summary(self, X=None, y=None):
        """
        获取特征重要性综合摘要
        """
        summary = {}
        
        # 1. 模型内置特征重要性
        if hasattr(self.model, 'feature_importances_'):
            summary['built_in'] = pd.DataFrame({
                'feature': self.feature_names or [f'Feature_{i}' for i in range(len(self.model.feature_importances_))],
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # 2. 系数（线性模型）
        elif hasattr(self.model, 'coef_'):
            coefs = np.abs(self.model.coef_)
            if coefs.ndim > 1:
                coefs = coefs.flatten()
            summary['coefficients'] = pd.DataFrame({
                'feature': self.feature_names or [f'Feature_{i}' for i in range(len(coefs))],
                'importance': coefs
            }).sort_values('importance', ascending=False)
        
        # 3. SHAP重要性
        if self.shap_values is not None:
            if isinstance(self.shap_values, list):
                shap_vals = self.shap_values[0]
            else:
                shap_vals = self.shap_values
            
            shap_importance = np.abs(shap_vals).mean(axis=0)
            summary['shap'] = pd.DataFrame({
                'feature': self.feature_names or [f'Feature_{i}' for i in range(len(shap_importance))],
                'importance': shap_importance
            }).sort_values('importance', ascending=False)
        
        # 4. 排列重要性
        if X is not None and y is not None:
            perm_importance = self.get_permutation_importance(X, y)
            if perm_importance is not None:
                summary['permutation'] = perm_importance
        
        return summary


class DiagnosisSystem:
    """
    系统状态识别和诊断系统
    基于机器学习模型识别系统状态并进行故障诊断
    """
    
    def __init__(self, model=None):
        """
        初始化诊断系统
        
        Parameters:
        -----------
        model : trained model
            已训练的状态识别模型
        """
        self.model = model
        self.state_labels = None
        self.thresholds = {}
        self.diagnosis_rules = {}
        
    def set_state_labels(self, labels):
        """
        设置状态标签
        
        Parameters:
        -----------
        labels : dict
            状态编码到状态名称的映射，如 {0: '正常', 1: '故障1', 2: '故障2'}
        """
        self.state_labels = labels
    
    def set_thresholds(self, thresholds):
        """
        设置诊断阈值
        
        Parameters:
        -----------
        thresholds : dict
            各物理量的阈值，如 {'surge': (-5, 5), 'heave': (-3, 3)}
        """
        self.thresholds = thresholds
    
    def set_diagnosis_rules(self, rules):
        """
        设置诊断规则
        
        Parameters:
        -----------
        rules : dict
            诊断规则字典
        """
        self.diagnosis_rules = rules
    
    def predict_state(self, X):
        """
        预测系统状态
        
        Parameters:
        -----------
        X : array-like
            输入特征
            
        Returns:
        --------
        dict
            包含状态预测结果的字典
        """
        if self.model is None:
            raise ValueError("尚未设置诊断模型")
        
        predictions = self.model.predict(X)
        
        results = {
            'state_code': predictions,
            'state_name': [self.state_labels.get(p, f'未知状态_{p}') if self.state_labels else str(p) for p in predictions]
        }
        
        # 如果模型支持概率预测
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            results['state_probability'] = probabilities.max(axis=1)
            results['all_probabilities'] = probabilities
        
        return results
    
    def diagnose(self, X, feature_names=None):
        """
        执行故障诊断
        
        Parameters:
        -----------
        X : array-like
            输入数据
        feature_names : list
            特征名称列表
            
        Returns:
        --------
        dict
            诊断结果
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        diagnosis_results = []
        
        for i, sample in enumerate(X):
            sample_diagnosis = {
                'sample_id': i,
                'anomalies': [],
                'warnings': [],
                'recommendations': []
            }
            
            # 1. 检查阈值违规
            for j, (feature, value) in enumerate(zip(feature_names, sample)):
                if feature in self.thresholds:
                    low, high = self.thresholds[feature]
                    if value < low or value > high:
                        sample_diagnosis['anomalies'].append({
                            'feature': feature,
                            'value': value,
                            'threshold': (low, high),
                            'severity': 'high' if abs(value) > max(abs(low), abs(high)) * 1.5 else 'medium'
                        })
            
            # 2. 应用诊断规则
            for rule_name, rule_func in self.diagnosis_rules.items():
                if rule_func(sample, feature_names):
                    sample_diagnosis['warnings'].append(rule_name)
            
            # 3. 生成建议
            if sample_diagnosis['anomalies']:
                sample_diagnosis['recommendations'].append("检测到异常参数，建议检查传感器数据")
            
            if len(sample_diagnosis['anomalies']) > 2:
                sample_diagnosis['recommendations'].append("多个参数异常，可能存在系统性故障")
            
            diagnosis_results.append(sample_diagnosis)
        
        return diagnosis_results
    
    def get_state_statistics(self, X):
        """
        获取状态统计信息
        
        Parameters:
        -----------
        X : array-like
            输入数据
            
        Returns:
        --------
        dict
            状态统计信息
        """
        predictions = self.model.predict(X)
        unique, counts = np.unique(predictions, return_counts=True)
        
        stats = {
            'total_samples': len(predictions),
            'state_distribution': {}
        }
        
        for state, count in zip(unique, counts):
            state_name = self.state_labels.get(state, f'状态_{state}') if self.state_labels else f'状态_{state}'
            stats['state_distribution'][state_name] = {
                'count': count,
                'percentage': count / len(predictions) * 100
            }
        
        return stats
    
    def plot_state_distribution(self, X, use_plotly=False):
        """
        绘制状态分布图
        """
        predictions = self.model.predict(X)
        unique, counts = np.unique(predictions, return_counts=True)
        
        labels = [self.state_labels.get(u, f'状态_{u}') if self.state_labels else f'状态_{u}' for u in unique]
        
        if use_plotly:
            fig = go.Figure(data=[go.Pie(labels=labels, values=counts, hole=.3)])
            fig.update_layout(title='系统状态分布')
            return fig
        else:
            plt.figure(figsize=(8, 6))
            plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.title('系统状态分布')
            plt.axis('equal')
            return plt
    
    def generate_report(self, X, feature_names=None):
        """
        生成诊断报告
        
        Parameters:
        -----------
        X : array-like
            输入数据
        feature_names : list
            特征名称
            
        Returns:
        --------
        dict
            完整的诊断报告
        """
        report = {
            'summary': {},
            'state_prediction': None,
            'diagnosis': None,
            'statistics': None,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 1. 状态预测
        report['state_prediction'] = self.predict_state(X)
        
        # 2. 故障诊断
        report['diagnosis'] = self.diagnose(X, feature_names)
        
        # 3. 统计信息
        report['statistics'] = self.get_state_statistics(X)
        
        # 4. 摘要
        total_anomalies = sum(len(d['anomalies']) for d in report['diagnosis'])
        total_warnings = sum(len(d['warnings']) for d in report['diagnosis'])
        
        report['summary'] = {
            'total_samples': len(X),
            'anomalies_detected': total_anomalies,
            'warnings_generated': total_warnings,
            'overall_status': '正常' if total_anomalies == 0 else '需要关注' if total_anomalies < 5 else '异常'
        }
        
        return report
