import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools

class DataVisualizer:
    def __init__(self):
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def plot_time_series(self, df, columns=None, title='时间序列图', xlabel='时间', ylabel='数值', 
                        figsize=(12, 6), use_plotly=False):
        """
        绘制时间序列图
        """
        if columns is None:
            columns = df.columns
        
        if use_plotly:
            # 使用Plotly绘制交互式图表
            fig = go.Figure()
            
            for col in columns:
                if col in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))
            
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                legend_title='变量',
                hovermode='x unified'
            )
            
            return fig
        else:
            # 使用Matplotlib绘制静态图表
            plt.figure(figsize=figsize)
            
            for col in columns:
                if col in df.columns:
                    plt.plot(df.index, df[col], label=col)
            
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt
    
    def plot_correlation_matrix(self, df, title='相关系数矩阵', figsize=(12, 10), use_plotly=False):
        """
        绘制相关系数矩阵
        """
        # 复制数据并处理NaN值
        df_clean = df.copy()
        
        # 只选择数值列
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean = df_clean[numeric_cols]
        
        # 删除全为NaN的列
        df_clean = df_clean.dropna(axis=1, how='all')
        
        # 用列均值填充NaN值
        df_clean = df_clean.fillna(df_clean.mean())
        
        # 计算相关系数矩阵
        corr_matrix = df_clean.corr()
        
        # 将相关系数矩阵中的NaN替换为0（如果还有的话）
        corr_matrix = corr_matrix.fillna(0)
        
        if use_plotly:
            # 使用Plotly绘制交互式热力图
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu_r',
                zmin=-1,
                zmax=1,
                text=corr_matrix.values.round(2),
                texttemplate='%{text}'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='变量',
                yaxis_title='变量',
                xaxis_tickangle=-45
            )
            
            return fig
        else:
            # 使用Seaborn绘制静态热力图
            plt.figure(figsize=figsize)
            
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap='RdBu_r',
                vmin=-1,
                vmax=1,
                fmt='.2f',
                linewidths=0.5,
                cbar_kws={'label': '相关系数'}
            )
            
            plt.title(title)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            return plt
    
    def plot_feature_importance(self, feature_importance, feature_names, title='特征重要性', 
                               figsize=(12, 8), use_plotly=False):
        """
        绘制特征重要性
        """
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        })
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        if use_plotly:
            # 使用Plotly绘制交互式柱状图
            fig = go.Figure(data=go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Feature'],
                orientation='h'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='重要性',
                yaxis_title='特征',
                yaxis={'categoryorder': 'total ascending'}
            )
            
            return fig
        else:
            # 使用Matplotlib绘制静态柱状图
            plt.figure(figsize=figsize)
            
            plt.barh(range(len(importance_df)), importance_df['Importance'], tick_label=importance_df['Feature'])
            plt.title(title)
            plt.xlabel('重要性')
            plt.ylabel('特征')
            plt.tight_layout()
            
            return plt
    
    def plot_prediction_results(self, y_true, y_pred, title='预测结果对比', 
                              xlabel='样本', ylabel='数值', figsize=(12, 6), use_plotly=False):
        """
        绘制预测结果对比
        """
        # 创建结果对比DataFrame
        results_df = pd.DataFrame({
            '真实值': y_true.flatten() if hasattr(y_true, 'flatten') else y_true,
            '预测值': y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred
        })
        
        if use_plotly:
            # 使用Plotly绘制交互式对比图
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=results_df.index, y=results_df['真实值'], mode='lines', name='真实值', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=results_df.index, y=results_df['预测值'], mode='lines', name='预测值', line=dict(color='red', dash='dash')))
            
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                legend_title='类型',
                hovermode='x unified'
            )
            
            return fig
        else:
            # 使用Matplotlib绘制静态对比图
            plt.figure(figsize=figsize)
            
            plt.plot(results_df.index, results_df['真实值'], label='真实值', color='blue')
            plt.plot(results_df.index, results_df['预测值'], label='预测值', color='red', linestyle='--')
            
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt
    
    def plot_residuals(self, y_true, y_pred, title='残差图', figsize=(12, 6), use_plotly=False):
        """
        绘制残差图
        """
        # 计算残差
        residuals = (y_true.flatten() if hasattr(y_true, 'flatten') else y_true) - \
                    (y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred)
        
        if use_plotly:
            # 使用Plotly绘制交互式残差图
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=y_pred.flatten(), y=residuals, mode='markers', name='残差'))
            fig.add_hline(y=0, line_dash='dash', line_color='red', name='零残差线')
            
            fig.update_layout(
                title=title,
                xaxis_title='预测值',
                yaxis_title='残差',
                hovermode='closest'
            )
            
            return fig
        else:
            # 使用Matplotlib绘制静态残差图
            plt.figure(figsize=figsize)
            
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            
            plt.title(title)
            plt.xlabel('预测值')
            plt.ylabel('残差')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt
    
    def plot_error_distribution(self, y_true, y_pred, title='预测误差分布', bins=50, 
                              figsize=(12, 6), use_plotly=False):
        """
        绘制预测误差分布
        """
        # 计算预测误差
        errors = (y_true.flatten() if hasattr(y_true, 'flatten') else y_true) - \
                 (y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred)
        
        if use_plotly:
            # 使用Plotly绘制交互式直方图
            fig = go.Figure(data=go.Histogram(
                x=errors,
                nbinsx=bins,
                histnorm='probability density',
                opacity=0.75
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='预测误差',
                yaxis_title='概率密度',
                bargap=0.1
            )
            
            return fig
        else:
            # 使用Matplotlib绘制静态直方图
            plt.figure(figsize=figsize)
            
            plt.hist(errors, bins=bins, density=True, alpha=0.75, edgecolor='black')
            plt.title(title)
            plt.xlabel('预测误差')
            plt.ylabel('概率密度')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt
    
    def plot_training_history(self, history, metrics=['loss', 'val_loss'], title='模型训练历史', 
                             figsize=(12, 8), use_plotly=False):
        """
        绘制模型训练历史
        """
        if use_plotly:
            # 使用Plotly绘制交互式训练历史图
            fig = go.Figure()
            
            for metric in metrics:
                if metric in history.history:
                    fig.add_trace(go.Scatter(x=list(range(1, len(history.history[metric]) + 1)), 
                                           y=history.history[metric], mode='lines', name=metric))
            
            fig.update_layout(
                title=title,
                xaxis_title='epoch',
                yaxis_title='数值',
                legend_title='指标',
                hovermode='x unified'
            )
            
            return fig
        else:
            # 使用Matplotlib绘制静态训练历史图
            plt.figure(figsize=figsize)
            
            for metric in metrics:
                if metric in history.history:
                    plt.plot(history.history[metric], label=metric)
            
            plt.title(title)
            plt.xlabel('epoch')
            plt.ylabel('数值')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt
    
    def plot_boxplot(self, df, columns=None, title='箱线图', figsize=(12, 8), use_plotly=False):
        """
        绘制箱线图
        """
        if columns is None:
            columns = df.columns
        
        if use_plotly:
            # 使用Plotly绘制交互式箱线图
            fig = go.Figure()
            
            for col in columns:
                if col in df.columns:
                    fig.add_trace(go.Box(y=df[col], name=col))
            
            fig.update_layout(
                title=title,
                yaxis_title='数值',
                boxmode='group'
            )
            
            return fig
        else:
            # 使用Matplotlib绘制静态箱线图
            plt.figure(figsize=figsize)
            
            df[columns].boxplot()
            plt.title(title)
            plt.ylabel('数值')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            return plt

    def plot_violin(self, df, columns=None, title='小提琴图', figsize=(12, 8), use_plotly=False, color='blue'):
        """
        绘制小提琴图
        """
        if columns is None:
            columns = df.columns
        
        if use_plotly:
            # 使用Plotly绘制交互式小提琴图
            fig = go.Figure()
            
            for col in columns:
                if col in df.columns:
                    fig.add_trace(go.Violin(y=df[col], name=col, marker_color=color))
            
            fig.update_layout(
                title=title,
                yaxis_title='数值',
                boxmode='group'
            )
            
            return fig
        else:
            # 使用Matplotlib绘制静态小提琴图
            plt.figure(figsize=figsize)
            
            df[columns].plot(kind='violin')
            plt.title(title)
            plt.ylabel('数值')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            return plt

    def plot_histogram(self, df, columns=None, title='直方图', xlabel='数值', ylabel='频率', 
                     figsize=(12, 8), use_plotly=False, bins=30, color='blue'):
        """
        绘制直方图
        """
        if columns is None:
            columns = df.columns
        
        if use_plotly:
            # 使用Plotly绘制交互式直方图
            n_cols = min(2, len(columns))
            n_rows = (len(columns) + n_cols - 1) // n_cols
            
            fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=columns)
            
            for i, col in enumerate(columns, start=1):
                if col in df.columns:
                    row = (i - 1) // n_cols + 1
                    col_pos = (i - 1) % n_cols + 1
                    fig.add_trace(go.Histogram(x=df[col], name=col, nbinsx=bins, 
                                             marker_color=color), row=row, col=col_pos)
            
            fig.update_layout(
                height=400 * n_rows,
                width=500 * n_cols,
                title_text=title,
                showlegend=False
            )
            
            fig.update_xaxes(title_text=xlabel)
            fig.update_yaxes(title_text=ylabel)
            
            return fig
        else:
            # 使用Matplotlib绘制静态直方图
            plot_data = df[[col for col in columns if col in df.columns]]
            plot_data.hist(bins=bins, figsize=figsize, alpha=0.6, color=color, edgecolor='black')
            
            plt.suptitle(title)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            return plt

    def plot_scatter(self, df, x_col, y_col, title='散点图', xlabel=None, ylabel=None, 
                   figsize=(10, 8), use_plotly=False, color='blue', marker_size=8, 
                   trendline=False):
        """
        绘制散点图
        """
        if xlabel is None:
            xlabel = x_col
        if ylabel is None:
            ylabel = y_col
        
        if use_plotly:
            # 使用Plotly绘制交互式散点图
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode='markers', 
                                   marker=dict(color=color, size=marker_size), name=y_col))
            
            # 添加趋势线
            if trendline:
                # 计算趋势线
                z = np.polyfit(df[x_col], df[y_col], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(df[x_col].min(), df[x_col].max(), 100)
                y_trend = p(x_trend)
                
                fig.add_trace(go.Scatter(x=x_trend, y=y_trend, mode='lines', 
                                       name='趋势线', line=dict(color='red', width=2)))
            
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                hovermode='closest'
            )
            
            return fig
        else:
            # 使用Matplotlib绘制静态散点图
            plt.figure(figsize=figsize)
            
            plt.scatter(df[x_col], df[y_col], alpha=0.6, color=color, s=marker_size*2)
            
            # 添加趋势线
            if trendline:
                z = np.polyfit(df[x_col], df[y_col], 1)
                p = np.poly1d(z)
                plt.plot(df[x_col], p(df[x_col]), 'r-', linewidth=2, label='趋势线')
                plt.legend()
            
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt

    def plot_pairplot(self, df, columns=None, title='变量关系图', figsize=(12, 12), hue=None):
        """
        绘制变量关系图
        """
        if columns is None:
            columns = df.columns[:5]  # 默认只显示前5列
        
        # 使用Seaborn绘制变量关系图
        plt.figure(figsize=figsize)
        
        sns.pairplot(df[columns], diag_kind='kde', hue=hue)
        
        plt.suptitle(title, y=1.02)
        plt.tight_layout()
        
        return plt
    
    # ==================== 模型对比和评估可视化 ====================
    
    def plot_model_comparison(self, models_metrics, metrics=['mse', 'mae', 'r2'], 
                             title='模型性能对比', figsize=(14, 8), use_plotly=False):
        """
        绘制模型性能对比图
        
        Parameters:
        -----------
        models_metrics : dict
            模型评估指标字典，格式为 {'model_name': {'mse': 0.1, 'mae': 0.2, ...}}
        metrics : list
            要对比的指标列表
        title : str
            图表标题
        figsize : tuple
            图表大小
        use_plotly : bool
            是否使用Plotly
            
        Returns:
        --------
        figure : 图表对象
        """
        # 准备数据
        model_names = list(models_metrics.keys())
        n_metrics = len(metrics)
        
        if use_plotly:
            # 使用Plotly绘制交互式对比图
            fig = make_subplots(
                rows=1, cols=n_metrics,
                subplot_titles=metrics,
                horizontal_spacing=0.1
            )
            
            colors = px.colors.qualitative.Set3[:len(model_names)]
            
            for i, metric in enumerate(metrics):
                values = [models_metrics[model].get(metric, 0) for model in model_names]
                
                fig.add_trace(
                    go.Bar(x=model_names, y=values, name=metric, 
                          marker_color=colors, showlegend=(i==0)),
                    row=1, col=i+1
                )
            
            fig.update_layout(
                title=title,
                height=500,
                width=400 * n_metrics
            )
            
            return fig
        else:
            # 使用Matplotlib绘制静态对比图
            fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
            if n_metrics == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics):
                values = [models_metrics[model].get(metric, 0) for model in model_names]
                
                axes[i].bar(model_names, values, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'{metric.upper()}')
                axes[i].set_ylabel('数值')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3, axis='y')
            
            plt.suptitle(title)
            plt.tight_layout()
            
            return plt
    
    def plot_prediction_comparison(self, y_true, predictions_dict, title='多模型预测结果对比',
                                   figsize=(14, 8), use_plotly=False):
        """
        绘制多模型预测结果对比图
        
        Parameters:
        -----------
        y_true : array-like
            真实值
        predictions_dict : dict
            预测结果字典，格式为 {'model_name': y_pred}
        title : str
            图表标题
        figsize : tuple
            图表大小
        use_plotly : bool
            是否使用Plotly
            
        Returns:
        --------
        figure : 图表对象
        """
        y_true = np.array(y_true).flatten()
        
        if use_plotly:
            fig = go.Figure()
            
            # 添加真实值
            fig.add_trace(go.Scatter(
                x=list(range(len(y_true))), y=y_true,
                mode='lines', name='真实值',
                line=dict(color='black', width=2)
            ))
            
            # 添加各模型预测值
            colors = px.colors.qualitative.Set3[:len(predictions_dict)]
            for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
                y_pred = np.array(y_pred).flatten()
                fig.add_trace(go.Scatter(
                    x=list(range(len(y_pred))), y=y_pred,
                    mode='lines', name=model_name,
                    line=dict(color=colors[i], width=1.5)
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title='样本索引',
                yaxis_title='数值',
                hovermode='x unified',
                height=500
            )
            
            return fig
        else:
            plt.figure(figsize=figsize)
            
            # 绘制真实值
            plt.plot(y_true, label='真实值', color='black', linewidth=2)
            
            # 绘制各模型预测值
            for model_name, y_pred in predictions_dict.items():
                y_pred = np.array(y_pred).flatten()
                plt.plot(y_pred, label=model_name, linewidth=1.5, alpha=0.7)
            
            plt.title(title)
            plt.xlabel('样本索引')
            plt.ylabel('数值')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt
    
    def plot_residual_comparison(self, y_true, predictions_dict, title='多模型残差对比',
                                 figsize=(14, 8), use_plotly=False):
        """
        绘制多模型残差对比图
        
        Parameters:
        -----------
        y_true : array-like
            真实值
        predictions_dict : dict
            预测结果字典
        title : str
            图表标题
        figsize : tuple
            图表大小
        use_plotly : bool
            是否使用Plotly
            
        Returns:
        --------
        figure : 图表对象
        """
        y_true = np.array(y_true).flatten()
        
        if use_plotly:
            n_models = len(predictions_dict)
            fig = make_subplots(
                rows=1, cols=n_models,
                subplot_titles=list(predictions_dict.keys()),
                horizontal_spacing=0.05
            )
            
            for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
                y_pred = np.array(y_pred).flatten()
                residuals = y_true - y_pred
                
                fig.add_trace(
                    go.Scatter(x=y_pred, y=residuals, mode='markers',
                              marker=dict(size=6, opacity=0.6), showlegend=False),
                    row=1, col=i+1
                )
                
                # 添加零线
                fig.add_hline(y=0, line_dash='dash', line_color='red', row=1, col=i+1)
            
            fig.update_layout(title=title, height=400)
            
            return fig
        else:
            n_models = len(predictions_dict)
            fig, axes = plt.subplots(1, n_models, figsize=figsize)
            if n_models == 1:
                axes = [axes]
            
            for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
                y_pred = np.array(y_pred).flatten()
                residuals = y_true - y_pred
                
                axes[i].scatter(y_pred, residuals, alpha=0.5)
                axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.8)
                axes[i].set_title(model_name)
                axes[i].set_xlabel('预测值')
                axes[i].set_ylabel('残差')
                axes[i].grid(True, alpha=0.3)
            
            plt.suptitle(title)
            plt.tight_layout()
            
            return plt
    
    def plot_error_heatmap(self, models_metrics, metrics=None, title='模型误差热力图',
                          figsize=(12, 8), use_plotly=False):
        """
        绘制模型误差热力图
        
        Parameters:
        -----------
        models_metrics : dict
            模型评估指标字典
        metrics : list, optional
            要显示的指标列表
        title : str
            图表标题
        figsize : tuple
            图表大小
        use_plotly : bool
            是否使用Plotly
            
        Returns:
        --------
        figure : 图表对象
        """
        if metrics is None:
            # 获取所有可用的指标
            all_metrics = set()
            for model_metrics in models_metrics.values():
                all_metrics.update(model_metrics.keys())
            metrics = sorted(list(all_metrics))
        
        # 创建数据矩阵
        model_names = list(models_metrics.keys())
        data_matrix = []
        
        for model in model_names:
            row = [models_metrics[model].get(metric, 0) for metric in metrics]
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        if use_plotly:
            fig = go.Figure(data=go.Heatmap(
                z=data_matrix,
                x=metrics,
                y=model_names,
                colorscale='RdYlGn_r',
                text=data_matrix.round(4),
                texttemplate='%{text}'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='评估指标',
                yaxis_title='模型',
                height=400 + 50 * len(model_names)
            )
            
            return fig
        else:
            plt.figure(figsize=figsize)
            
            sns.heatmap(data_matrix, annot=True, fmt='.4f', cmap='RdYlGn_r',
                       xticklabels=metrics, yticklabels=model_names,
                       linewidths=0.5, cbar_kws={'label': '数值'})
            
            plt.title(title)
            plt.xlabel('评估指标')
            plt.ylabel('模型')
            plt.tight_layout()
            
            return plt
    
    def plot_roc_pr_curve(self, y_true, predictions_dict, title='模型性能曲线',
                          figsize=(14, 6), use_plotly=False):
        """
        绘制ROC和PR曲线对比（适用于分类问题）
        
        Parameters:
        -----------
        y_true : array-like
            真实标签
        predictions_dict : dict
            预测概率字典
        title : str
            图表标题
        figsize : tuple
            图表大小
        use_plotly : bool
            是否使用Plotly
            
        Returns:
        --------
        figure : 图表对象
        """
        from sklearn.metrics import roc_curve, auc, precision_recall_curve
        
        if use_plotly:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['ROC曲线', 'PR曲线'],
                horizontal_spacing=0.15
            )
            
            colors = px.colors.qualitative.Set3[:len(predictions_dict)]
            
            for i, (model_name, y_pred_proba) in enumerate(predictions_dict.items()):
                # ROC曲线
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                fig.add_trace(
                    go.Scatter(x=fpr, y=tpr, mode='lines',
                              name=f'{model_name} (AUC={roc_auc:.3f})',
                              line=dict(color=colors[i])),
                    row=1, col=1
                )
                
                # PR曲线
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
                pr_auc = auc(recall, precision)
                
                fig.add_trace(
                    go.Scatter(x=recall, y=precision, mode='lines',
                              name=f'{model_name} (AP={pr_auc:.3f})',
                              line=dict(color=colors[i]), showlegend=False),
                    row=1, col=2
                )
            
            # 添加对角线
            fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                          line=dict(dash='dash', color='gray'), showlegend=False),
                row=1, col=1
            )
            
            fig.update_layout(title=title, height=500, width=1000)
            
            return fig
        else:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            for model_name, y_pred_proba in predictions_dict.items():
                # ROC曲线
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                axes[0].plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.3f})')
                
                # PR曲线
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
                pr_auc = auc(recall, precision)
                axes[1].plot(recall, precision, label=f'{model_name} (AP={pr_auc:.3f})')
            
            # ROC图设置
            axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[0].set_xlabel('假阳性率')
            axes[0].set_ylabel('真阳性率')
            axes[0].set_title('ROC曲线')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # PR图设置
            axes[1].set_xlabel('召回率')
            axes[1].set_ylabel('精确率')
            axes[1].set_title('PR曲线')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.suptitle(title)
            plt.tight_layout()
            
            return plt
    
    def plot_learning_curves(self, train_sizes, train_scores, val_scores, 
                            title='学习曲线', figsize=(12, 6), use_plotly=False):
        """
        绘制学习曲线
        
        Parameters:
        -----------
        train_sizes : array-like
            训练样本数量
        train_scores : array-like
            训练集得分
        val_scores : array-like
            验证集得分
        title : str
            图表标题
        figsize : tuple
            图表大小
        use_plotly : bool
            是否使用Plotly
            
        Returns:
        --------
        figure : 图表对象
        """
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        if use_plotly:
            fig = go.Figure()
            
            # 训练集曲线
            fig.add_trace(go.Scatter(
                x=train_sizes, y=train_mean, mode='lines+markers',
                name='训练集得分', line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=np.concatenate([train_sizes, train_sizes[::-1]]),
                y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
                fill='toself', fillcolor='rgba(0,0,255,0.1)',
                line=dict(color='rgba(255,255,255,0)'), showlegend=False
            ))
            
            # 验证集曲线
            fig.add_trace(go.Scatter(
                x=train_sizes, y=val_mean, mode='lines+markers',
                name='验证集得分', line=dict(color='red')
            ))
            fig.add_trace(go.Scatter(
                x=np.concatenate([train_sizes, train_sizes[::-1]]),
                y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
                fill='toself', fillcolor='rgba(255,0,0,0.1)',
                line=dict(color='rgba(255,255,255,0)'), showlegend=False
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='训练样本数',
                yaxis_title='得分',
                hovermode='x unified',
                height=500
            )
            
            return fig
        else:
            plt.figure(figsize=figsize)
            
            plt.plot(train_sizes, train_mean, 'o-', color='blue', label='训练集得分')
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            
            plt.plot(train_sizes, val_mean, 'o-', color='red', label='验证集得分')
            plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
            
            plt.title(title)
            plt.xlabel('训练样本数')
            plt.ylabel('得分')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt

    def interactive_data_explorer(self, df, title='交互式数据探索'):
        """
        交互式数据探索
        """
        import streamlit as st
        
        st.title(title)
        
        # 数据概览
        st.subheader('数据概览')
        st.write(df.describe())
        
        # 选择变量
        selected_columns = st.multiselect('选择变量', df.columns, default=df.columns[:3])
        
        if selected_columns:
            # 时间序列图
            st.subheader('时间序列图')
            fig = self.plot_time_series(df[selected_columns], use_plotly=True)
            st.plotly_chart(fig)
            
            # 相关系数矩阵
            st.subheader('相关系数矩阵')
            fig = self.plot_correlation_matrix(df[selected_columns], use_plotly=True)
            st.plotly_chart(fig)
            
            # 箱线图
            st.subheader('箱线图')
            fig = self.plot_boxplot(df[selected_columns], use_plotly=True)
            st.plotly_chart(fig)
            
            # 直方图
            st.subheader('直方图')
            fig = self.plot_histogram(df[selected_columns], use_plotly=True)
            st.plotly_chart(fig)
            
            # 散点图（选择两个变量）
            if len(selected_columns) >= 2:
                st.subheader('散点图')
                x_var = st.selectbox('X轴变量', selected_columns, index=0)
                y_var = st.selectbox('Y轴变量', selected_columns, index=1)
                show_trendline = st.checkbox('显示趋势线')
                
                fig = self.plot_scatter(df, x_var, y_var, use_plotly=True, trendline=show_trendline)
                st.plotly_chart(fig)

    def plot_feature_importance_advanced(self, feature_importance, feature_names, title='特征重要性', 
                                      figsize=(12, 10), threshold=0.01, color='blue'):
        """
        绘制高级特征重要性图
        """
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        })
        
        # 按重要性排序
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        # 过滤掉重要性低于阈值的特征
        importance_df = importance_df[importance_df['Importance'] >= threshold]
        
        # 计算累计重要性
        importance_df['Cumulative Importance'] = importance_df['Importance'].cumsum()
        importance_df['Cumulative Importance'] /= importance_df['Cumulative Importance'].max()
        
        # 创建双轴图
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # 柱状图 - 特征重要性
        bars = ax1.bar(range(len(importance_df)), importance_df['Importance'], 
                      color=color, alpha=0.6, label='特征重要性')
        ax1.set_xlabel('特征')
        ax1.set_ylabel('重要性', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xticks(range(len(importance_df)))
        ax1.set_xticklabels(importance_df['Feature'], rotation=45, ha='right')
        
        # 折线图 - 累计重要性
        ax2 = ax1.twinx()
        ax2.plot(range(len(importance_df)), importance_df['Cumulative Importance'], 
                color='red', linewidth=2, marker='o', label='累计重要性')
        ax2.set_ylabel('累计重要性', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # 添加标题和网格
        plt.title(title)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        plt.tight_layout()
        
        return plt
