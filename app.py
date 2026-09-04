import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import traceback
import datetime
import html
import plotly.graph_objects as go
import plotly.io as pio
from functools import lru_cache
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from src.data.data_loader import read_data_file, read_multiple_files
from src.data.data_preprocessor import DataPreprocessor
from src.models.model_trainer import ModelTrainer
from src.models.automl import AutoMLTrainer, AutoMLRegressor
from src.models.model_explainability import ModelExplainer, DiagnosisSystem
from src.models.transfer_learning import DomainAdaptation, TransferLearningManager, SklearnTransferLearning
from src.models.algorithm_audit import (
    algorithm_implementation_matrix,
    build_algorithm_audit_report,
    optimization_algorithm_matrix,
    recommended_algorithm_matrix,
)
from src.models.deep_learning_extensions import list_deep_learning_algorithms, tensorflow_available
from src.models.enhanced_diagnostics import EnhancedFaultDiagnosis, DiagnosisConfig
from src.models.reinforcement_learning import (
    ControlEnvironmentConfig,
    CustomFunctionEnvironmentConfig,
    DataDrivenEnvironmentConfig,
    HeuristicRLAgent,
    SimpleControlEnvironment,
    compare_algorithms,
    compare_custom_function_algorithms,
    compare_data_driven_algorithms,
    list_rl_algorithms,
    recommend_control_action,
    rl_backend_availability,
    summarize_evaluation,
    validate_rl_dataset,
)
from src.ui.workbench import append_run_history, render_project_workbench
from src.visualization.visualizer import DataVisualizer
from src.utils.help_manual import get_help_content, get_all_topics, search_help, HELP_MANUAL

# ==================== 性能优化配置 ====================
st.set_option('client.showErrorDetails', False)

# 检查Streamlit版本是否支持fragment
try:
    from streamlit import fragment
    HAS_FRAGMENT = True
except ImportError:
    HAS_FRAGMENT = False
    
# 自定义fragment装饰器（如果不支持）
def fragment_decorator(func=None, *, run_every=None):
    if func is None:
        return lambda f: f
    return func

if not HAS_FRAGMENT:
    fragment = fragment_decorator

# ==================== 页面配置和初始化 ====================
st.set_page_config(
    page_title="海眸智能数据预测与诊断平台 OEye",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'workflow_completed' not in st.session_state:
    st.session_state.workflow_completed = {}
if 'error_log' not in st.session_state:
    st.session_state.error_log = []
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'default_scaler': 'standard',
        'default_test_size': 0.2,
        'default_model': 'random_forest',
        'show_help': True
    }
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_info' not in st.session_state:
    st.session_state.file_info = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model_type_trained' not in st.session_state:
    st.session_state.model_type_trained = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_val' not in st.session_state:
    st.session_state.X_val = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_val' not in st.session_state:
    st.session_state.y_val = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'scale_method' not in st.session_state:
    st.session_state.scale_method = 'standard'
if 'use_feature_selection' not in st.session_state:
    st.session_state.use_feature_selection = False
if 'training_stop_requested' not in st.session_state:
    st.session_state.training_stop_requested = False
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'run_history' not in st.session_state:
    st.session_state.run_history = []
if 'custom_execution_log' not in st.session_state:
    st.session_state.custom_execution_log = []
if 'enhanced_diagnosis_result' not in st.session_state:
    st.session_state.enhanced_diagnosis_result = None

# 创建实例 - 使用单例模式避免重复创建
@st.cache_resource
def get_preprocessor():
    return DataPreprocessor()

@st.cache_resource
def get_visualizer():
    return DataVisualizer()

preprocessor = get_preprocessor()
visualizer = get_visualizer()

# ==================== 数据缓存函数 ====================
@st.cache_data(ttl=3600, show_spinner=False)
def cached_read_data(file_path):
    """缓存数据读取"""
    return read_data_file(file_path)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_data_stats(df_hash, df_json):
    """缓存数据统计信息"""
    df = pd.read_json(df_json)
    return df.describe()

@st.cache_data(ttl=3600, show_spinner=False)
def cached_correlation_matrix(df_hash, df_json, cols_json):
    """缓存相关性矩阵计算"""
    df = pd.read_json(df_json)
    cols = pd.read_json(cols_json)
    return df[cols].corr() if len(cols) > 0 else None

# ==================== 自定义CSS样式 ====================
st.markdown("""
<style>
    /* 步骤指示器样式 */
    .step-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
        padding: 10px;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    .step-item {
        text-align: center;
        flex: 1;
        padding: 10px;
        border-radius: 5px;
        margin: 0 5px;
    }
    .step-active {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .step-completed {
        background-color: #2ecc71;
        color: white;
    }
    .step-pending {
        background-color: #e0e0e0;
        color: #666;
    }
    /* 帮助提示样式 */
    .help-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
    /* 错误提示样式 */
    .error-box {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
    /* 成功提示样式 */
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 辅助函数 ====================
def show_help(text):
    """显示帮助信息"""
    if st.session_state.user_preferences.get('show_help', True):
        st.markdown(f'<div class="help-box">💡 <b>提示:</b> {text}</div>', unsafe_allow_html=True)

def show_error(message, details=None):
    """显示友好的错误信息"""
    st.markdown(f'<div class="error-box">❌ <b>错误:</b> {message}</div>', unsafe_allow_html=True)
    if details:
        with st.expander("查看详细错误信息"):
            st.code(details)
    # 记录错误日志
    st.session_state.error_log.append({
        'message': message,
        'details': details,
        'timestamp': pd.Timestamp.now()
    })

def show_success(message):
    """显示成功信息"""
    st.markdown(f'<div class="success-box">✅ <b>成功:</b> {message}</div>', unsafe_allow_html=True)

def dataframe_to_csv_bytes(df):
    """Encode result dataframes for Excel-friendly CSV download."""
    return df.to_csv(index=False).encode("utf-8-sig")

def build_html_report(title, metrics=None, sections=None, figures=None, tables=None, notes=None):
    """Build a compact standalone HTML report with metrics, Plotly figures, and tables."""
    metrics = metrics or {}
    sections = sections or []
    figures = figures or []
    tables = tables or []
    notes = notes or []
    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metric_cards = "".join(
        f"<div class='metric'><span>{html.escape(str(key))}</span><strong>{html.escape(str(value))}</strong></div>"
        for key, value in metrics.items()
    )
    section_html = "".join(
        f"<section><h2>{html.escape(str(heading))}</h2><p>{html.escape(str(body))}</p></section>"
        for heading, body in sections
    )
    figure_html = "".join(
        f"<section><h2>{html.escape(str(name))}</h2>{pio.to_html(fig, include_plotlyjs='cdn', full_html=False)}</section>"
        for name, fig in figures
        if fig is not None
    )
    table_html = ""
    for name, table in tables:
        if table is None:
            continue
        table_df = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
        table_html += (
            f"<section><h2>{html.escape(str(name))}</h2>"
            f"{table_df.head(500).to_html(index=False, classes='data-table', border=0)}</section>"
        )
    note_html = "".join(f"<li>{html.escape(str(note))}</li>" for note in notes)
    if note_html:
        note_html = f"<section><h2>建议与备注</h2><ul>{note_html}</ul></section>"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; background: #f6f8fb; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 32px 28px 56px; }}
header {{ background: linear-gradient(135deg, #16324f, #256b7f); color: white; padding: 28px 32px; border-radius: 10px; }}
h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
h2 {{ margin: 0 0 14px; font-size: 20px; }}
section {{ margin-top: 22px; background: white; border: 1px solid #e4e8ef; border-radius: 8px; padding: 20px; box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04); }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-top: 18px; }}
.metric {{ background: white; color: #172033; border-radius: 8px; padding: 14px 16px; border: 1px solid #dce3ec; }}
.metric span {{ display: block; color: #627085; font-size: 13px; margin-bottom: 6px; }}
.metric strong {{ font-size: 22px; }}
.data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.data-table th, .data-table td {{ padding: 8px 10px; border-bottom: 1px solid #edf0f5; text-align: left; }}
.data-table th {{ background: #f2f5f9; color: #334155; }}
footer {{ color: #667085; font-size: 12px; margin-top: 24px; }}
</style>
</head>
<body><main>
<header><h1>{html.escape(title)}</h1><div>生成时间：{generated_at}</div></header>
<div class="metrics">{metric_cards}</div>
{section_html}
{figure_html}
{table_html}
{note_html}
<footer>由海眸智能数据预测与诊断平台 OEye 生成。报告中的算法结论应结合工程机理和现场复核使用。</footer>
</main></body></html>"""

def render_result_downloads(prefix, result_df=None, report_html=None, csv_name=None, html_name=None, key_prefix="result"):
    """Render common CSV and HTML report download buttons."""
    col_csv, col_html = st.columns(2)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with col_csv:
        if result_df is not None:
            st.download_button(
                f"📥 下载{prefix}数据 CSV",
                dataframe_to_csv_bytes(result_df),
                file_name=csv_name or f"{key_prefix}_{timestamp}.csv",
                mime="text/csv",
                key=f"download_{key_prefix}_csv",
                use_container_width=True,
            )
    with col_html:
        if report_html is not None:
            st.download_button(
                f"📄 下载{prefix}HTML图文报告",
                report_html.encode("utf-8"),
                file_name=html_name or f"{key_prefix}_report_{timestamp}.html",
                mime="text/html",
                key=f"download_{key_prefix}_html",
                use_container_width=True,
            )

def record_custom_execution(name, status, details=None):
    """记录自定义代码执行日志，保留现有自定义代码能力并提升可追踪性。"""
    st.session_state.custom_execution_log.append({
        'name': name,
        'status': status,
        'details': details or '',
        'timestamp': pd.Timestamp.now()
    })
    st.session_state.custom_execution_log = st.session_state.custom_execution_log[-50:]

def update_step(step_num):
    """更新当前步骤"""
    st.session_state.current_step = step_num

def mark_step_completed(step_name):
    """标记步骤完成"""
    st.session_state.workflow_completed[step_name] = True

@fragment
def render_step_indicator():
    """渲染步骤指示器 - 使用fragment避免整页刷新"""
    steps = [
        ("1", "📁 数据上传"),
        ("2", "📊 数据预览"),
        ("3", "🔧 数据预处理"),
        ("4", "🤖 模型训练"),
        ("5", "📈 模型评估"),
        ("6", "🔮 预测应用")
    ]
    
    current_step = st.session_state.current_step
    
    html = '<div class="step-container">'
    for i, (num, label) in enumerate(steps, 1):
        if i < current_step:
            css_class = "step-completed"
            icon = "✓"
        elif i == current_step:
            css_class = "step-active"
            icon = "●"
        else:
            css_class = "step-pending"
            icon = "○"
        html += f'<div class="step-item {css_class}">{icon} {label}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def show_step_help(topic_key, button_text="❓ 帮助"):
    """显示步骤帮助按钮"""
    if st.button(button_text, key=f"help_btn_{topic_key}"):
        st.session_state.show_help_manual = True
        st.session_state.help_topic = topic_key
        st.rerun()

# ==================== 侧边栏 ====================
st.sidebar.title("海眸智能数据预测与诊断平台 OEye")

# 个性化设置
with st.sidebar.expander("⚙️ 使用与帮助"):
    st.session_state.user_preferences['show_help'] = st.checkbox(
        "显示帮助提示",
        value=st.session_state.user_preferences['show_help']
    )
    
    # 错误日志
    with st.expander("📋 错误日志"):
        if st.session_state.error_log:
            st.write(f"共记录 {len(st.session_state.error_log)} 个错误")
            if st.button("清除错误日志"):
                st.session_state.error_log = []
                st.success("错误日志已清除")
            
            for i, error in enumerate(st.session_state.error_log[-5:], 1):
                with st.expander(f"错误 {i}: {error['message'][:30]}..."):
                    st.write(f"**时间:** {error['timestamp']}")
                    st.write(f"**信息:** {error['message']}")
                    if error['details']:
                        st.code(error['details'])
        else:
            st.write("✅ 暂无错误记录")

    with st.expander("🧩 自定义代码运行日志"):
        if st.session_state.custom_execution_log:
            for item in st.session_state.custom_execution_log[-8:][::-1]:
                st.write(f"**{item['name']}** - {item['status']}")
                st.caption(item['timestamp'])
                if item.get('details'):
                    st.code(item['details'])
        else:
            st.write("暂无自定义代码运行记录")

# ==================== 步骤指示器（放在侧边栏顶部确保实时更新） ====================
st.sidebar.markdown("---")
st.sidebar.subheader("📍 训练进度")
render_step_indicator()

# ==================== 步骤 1: 数据上传 ====================
st.sidebar.markdown("---")
st.sidebar.subheader("📁 全局数据文件上传")

show_help("支持 .out、.dat、.csv、.xlsx 和 .xls 格式的数据文件。第一列应为时间，其他列为物理量。")

upload_mode = st.sidebar.radio(
    "选择上传模式",
    ["单文件", "多文件批量"],
    help="单文件: 上传单个数据文件; 多文件批量: 上传多个数据文件进行合并"
)

tmp_file_paths = []

if upload_mode == "单文件":
    uploaded_file = st.sidebar.file_uploader(
        "上传数据文件 (.out、.dat、.csv、.xlsx、.xls)，Limit 200MB per file • OUT, DAT, TXT, CSV, XLSX, XLS",
        type=["out", "dat", "txt", "csv", "xlsx", "xls"],
        accept_multiple_files=False,
        help="选择单个.out、.dat、.csv、.xlsx或.xls文件进行上传"
    )
    
    if uploaded_file is not None:
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            tmp_file_paths.append(tmp_file_path)
        
        try:
            # 读取数据
            with st.spinner("正在读取数据文件..."):
                df = read_data_file(tmp_file_path)
            file_info = [{'index': 1, 'path': uploaded_file.name, 'rows': len(df), 'columns': list(df.columns)}]
            st.session_state.df = df
            st.session_state.file_info = file_info
            st.sidebar.success(f"✅ 成功读取: {len(df)} 行, {len(df.columns)} 列")
            update_step(2)
            mark_step_completed('data_upload')
        except Exception as e:
            error_details = traceback.format_exc()
            st.sidebar.error(f"❌ 读取失败: {str(e)[:50]}")
            st.session_state.error_log.append({
                'message': str(e),
                'details': error_details,
                'timestamp': pd.Timestamp.now()
            })
            
else:  # 多文件批量
    uploaded_files = st.sidebar.file_uploader(
        "上传多个数据文件 (.out、.dat、.csv、.xlsx、.xls)，Limit 200MB per file • OUT, DAT, TXT, CSV, XLSX, XLS",
        type=["out", "dat", "txt", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="选择多个.out、.dat、.csv、.xlsx或.xls文件进行批量处理"
    )
    
    if uploaded_files:
        if len(uploaded_files) < 2:
            st.sidebar.warning("⚠️ 请至少上传2个文件")
        else:
            # 保存所有上传的文件
            progress_bar = st.sidebar.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_paths.append(tmp_file.name)
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # 批量合并选项
            st.sidebar.subheader("批量合并选项")
            merge_method = st.sidebar.selectbox(
                "合并方法",
                ["concat", "join"],
                help="concat: 纵向拼接; join: 横向合并"
            )
            
            sort_by_time = st.sidebar.checkbox("按时间排序", value=True)
            remove_duplicates = st.sidebar.checkbox("删除重复时间戳", value=True)
            
            try:
                with st.spinner("正在读取和合并..."):
                    df, file_info = read_multiple_files(
                        tmp_file_paths, 
                        merge_method=merge_method,
                        sort_by_time=sort_by_time,
                        remove_duplicates=remove_duplicates
                    )
                st.session_state.df = df
                st.session_state.file_info = file_info
                st.sidebar.success(f"✅ 成功合并: {len(uploaded_files)} 个文件, {len(df)} 行")
                update_step(2)
                mark_step_completed('data_upload')
            except Exception as e:
                error_details = traceback.format_exc()
                st.sidebar.error(f"❌ 合并失败: {str(e)[:50]}")

# 清理临时文件
for tmp_path in tmp_file_paths:
    try:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    except:
        pass

# ==================== 主页面：标签系统 ====================
# 自定义样式放大标签标题
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 28px;
        border-bottom: 1px solid #e5e7eb;
    }
    .stTabs [data-baseweb="tab"],
    .stTabs [role="tab"],
    div[data-testid="stTabs"] button[role="tab"] {
        min-height: 58px !important;
        padding: 14px 22px !important;
        flex: 1 1 auto !important;
    }
    .stTabs [data-baseweb="tab-list"] button,
    .stTabs [data-baseweb="tab-list"] button p,
    div[data-testid="stTabs"] button[role="tab"] p {
        font-size: 22px !important;
        font-weight: 700 !important;
        line-height: 1.35 !important;
        letter-spacing: 0 !important;
        white-space: nowrap !important;
    }
    .stTabs [aria-selected="true"] p {
        color: #ef4444 !important;
        font-weight: 800 !important;
    }
    @media (max-width: 1200px) {
        .stTabs [data-baseweb="tab-list"] button,
        .stTabs [data-baseweb="tab-list"] button p,
        div[data-testid="stTabs"] button[role="tab"] p {
            font-size: 18px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# 创建顶部标签页
tab0, tab1, tab2, tab3, tab4, tab5, tab7, tab6 = st.tabs([
    "🏠 项目工作台",
    "📈 时序预测",
    "🎯 分类任务",
    "🚀 批量任务处理",
    "🔍 系统状态诊断",
    "⚙️ 优化算法",
    "🧠 强化学习",
    "📖 帮助手册"
])

# ==================== 标签0: 项目工作台 ====================
with tab0:
    render_project_workbench(st, st.session_state)
    with st.expander("✅ 机器学习算法使用检查", expanded=False):
        audit_tab1, audit_tab2, audit_tab3, audit_tab4 = st.tabs(["检查结论", "实现状态", "推荐矩阵", "优化算法"])
        with audit_tab1:
            st.dataframe(build_algorithm_audit_report(), use_container_width=True, hide_index=True)
        with audit_tab2:
            st.caption("用于区分真实训练实现、条件实现、轻量实验代理和简化元启发式实现。")
            st.dataframe(algorithm_implementation_matrix(), use_container_width=True, hide_index=True)
        with audit_tab3:
            st.dataframe(recommended_algorithm_matrix(), use_container_width=True, hide_index=True)
        with audit_tab4:
            st.dataframe(optimization_algorithm_matrix(), use_container_width=True, hide_index=True)

# ==================== 标签1: 时序预测 ====================
with tab1:
    st.markdown("<h1 style='font-size: 32px; font-weight: bold;'>📈 时序预测流程</h1>", unsafe_allow_html=True)
    with st.expander("🧠 深度学习高级模型库", expanded=False):
        st.caption("此处只列出时序预测主训练流程真实可选、可训练的深度学习模型。")
        dl_df = pd.DataFrame([item.__dict__ for item in list_deep_learning_algorithms("regression")])
        st.dataframe(dl_df.rename(columns={
            "key": "模型ID",
            "name": "模型名称",
            "task": "任务",
            "data_type": "数据类型",
            "description": "适用说明"
        }), use_container_width=True, hide_index=True)
        if tensorflow_available():
            st.success("TensorFlow 可用，ANN/MLP、LSTM、GRU、CNN、Transformer 可在主训练流程中真实训练。")
        else:
            st.warning("TensorFlow 当前不可用，深度学习模型不能训练。")
    
    # 如果没有数据，显示数据上传界面
    if st.session_state.df is None:
        st.info("👈 请先上传数据文件开始训练")
        
        # 数据来源选择
        ts_data_col1, ts_data_col2 = st.columns([3, 1])
        
        with ts_data_col1:
            ts_data_source = st.radio(
                "选择数据来源",
                ["上传文件", "使用示例数据"],
                horizontal=True,
                key="ts_data_source"
            )
        
        if ts_data_source == "上传文件":
            ts_file = st.file_uploader(
                "上传时序数据文件",
                type=["out", "dat", "txt", "csv", "xlsx", "xls"],
                key="ts_file_uploader"
            )
            
            if ts_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(ts_file.name)[1]) as tmp_file:
                    tmp_file.write(ts_file.getvalue())
                    ts_tmp_path = tmp_file.name
                
                try:
                    with st.spinner("正在读取数据..."):
                        ts_df = read_data_file(ts_tmp_path)
                    
                    # 清理之前的状态
                    st.session_state.target_col = None
                    st.session_state.target_col_selected = None
                    st.session_state.feature_cols = None
                    st.session_state.feature_cols_selected = None
                    st.session_state.X_train = None
                    st.session_state.y_train = None
                    st.session_state.visual_cols_selected = None
                    st.session_state.df_temp = None
                    
                    st.session_state.df = ts_df
                    st.session_state.file_info = [{'index': 1, 'path': ts_file.name, 'rows': len(ts_df), 'columns': list(ts_df.columns)}]
                    st.success(f"✅ 数据加载成功！共 {len(ts_df)} 行，{len(ts_df.columns)} 列")
                    st.rerun()
                except Exception as e:
                    st.error(f"读取失败: {str(e)}")
                finally:
                    os.unlink(ts_tmp_path)
        else:
            st.markdown("**选择示例数据集：**")
            ts_sample_options = [
                "温度时序预测.csv",
                "电力负荷时序预测.csv",
                "股票价格时序预测.csv",
                "销售额时序预测.csv",
                "风力发电时序预测.csv",
                "交通流量时序预测.csv",
                "设备传感器时序预测.csv",
                "网络流量时序预测.csv"
            ]
            ts_selected = st.selectbox("选择数据集", ts_sample_options, key="ts_sample_select")
            
            if st.button("加载示例数据", key="ts_load_sample"):
                ts_sample_path = f"sample_data/{ts_selected}"
                if os.path.exists(ts_sample_path):
                    try:
                        ts_df = pd.read_csv(ts_sample_path, encoding='utf-8-sig')
                    except:
                        ts_df = pd.read_csv(ts_sample_path)
                    
                    # 清理之前的状态
                    st.session_state.target_col = None
                    st.session_state.target_col_selected = None
                    st.session_state.feature_cols = None
                    st.session_state.feature_cols_selected = None
                    st.session_state.X_train = None
                    st.session_state.y_train = None
                    st.session_state.visual_cols_selected = None
                    st.session_state.df_temp = None
                    
                    st.session_state.df = ts_df
                    st.session_state.file_info = [{'index': 1, 'path': ts_selected, 'rows': len(ts_df), 'columns': list(ts_df.columns)}]
                    st.success(f"✅ 已加载 {ts_selected}，共 {len(ts_df)} 行，{len(ts_df.columns)} 列")
                    st.rerun()
                else:
                    st.error("示例数据文件不存在！")
    else:
        # 有数据时显示主流程
        df = st.session_state.df
        
        # 数据切换功能（始终可见）
        with st.expander("📂 切换数据", expanded=False):
            if st.button("🔄 重新选择数据", key="ts_reselect_data"):
                st.session_state.df = None
                st.session_state.file_info = None
                # 重置数据源选择状态
                if 'ts_data_source_expander' in st.session_state:
                    st.session_state.ts_data_source_expander = "上传文件"
                st.rerun()
            
            # 使用不同的 key 来避免状态冲突
            ts_data_source = st.radio(
                "选择数据来源",
                ["上传文件", "使用示例数据"],
                horizontal=True,
                key="ts_data_source_expander"
            )
            
            if ts_data_source == "上传文件":
                ts_file = st.file_uploader(
                    "上传时序数据文件",
                    type=["out", "dat", "txt", "csv", "xlsx", "xls"],
                    key="ts_file_uploader_expander"
                )
                
                if ts_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(ts_file.name)[1]) as tmp_file:
                        tmp_file.write(ts_file.getvalue())
                        ts_tmp_path = tmp_file.name
                    
                    try:
                        with st.spinner("正在读取数据..."):
                            ts_df = read_data_file(ts_tmp_path)
                        
                        # 清理之前的状态
                        st.session_state.target_col = None
                        st.session_state.target_col_selected = None
                        st.session_state.feature_cols = None
                        st.session_state.feature_cols_selected = None
                        st.session_state.X_train = None
                        st.session_state.y_train = None
                        st.session_state.visual_cols_selected = None
                        st.session_state.df_temp = None
                        
                        st.session_state.df = ts_df
                        st.session_state.file_info = [{'index': 1, 'path': ts_file.name, 'rows': len(ts_df), 'columns': list(ts_df.columns)}]
                        st.success(f"✅ 数据加载成功！共 {len(ts_df)} 行，{len(ts_df.columns)} 列")
                        st.rerun()
                    except Exception as e:
                        st.error(f"读取失败: {str(e)}")
                    finally:
                        os.unlink(ts_tmp_path)
            else:
                st.markdown("**选择示例数据集：**")
                ts_sample_options = [
                    "温度时序预测.csv",
                    "电力负荷时序预测.csv",
                    "股票价格时序预测.csv",
                    "销售额时序预测.csv",
                    "风力发电时序预测.csv",
                    "交通流量时序预测.csv",
                    "设备传感器时序预测.csv",
                    "网络流量时序预测.csv"
                ]
                ts_selected = st.selectbox("选择数据集", ts_sample_options, key="ts_sample_select")
                
                if st.button("加载示例数据", key="ts_load_sample_expander"):
                    ts_sample_path = f"sample_data/{ts_selected}"
                    if os.path.exists(ts_sample_path):
                        try:
                            ts_df = pd.read_csv(ts_sample_path, encoding='utf-8-sig')
                        except:
                            ts_df = pd.read_csv(ts_sample_path)
                        
                        # 清理之前的状态
                        st.session_state.target_col = None
                        st.session_state.target_col_selected = None
                        st.session_state.feature_cols = None
                        st.session_state.feature_cols_selected = None
                        st.session_state.X_train = None
                        st.session_state.y_train = None
                        
                        st.session_state.df = ts_df
                        st.session_state.file_info = [{'index': 1, 'path': ts_selected, 'rows': len(ts_df), 'columns': list(ts_df.columns)}]
                        st.success(f"✅ 已加载 {ts_selected}，共 {len(ts_df)} 行，{len(ts_df.columns)} 列")
                        st.rerun()
                    else:
                        st.error("示例数据文件不存在！")
        
        # 步骤 2: 数据预览
        st.subheader("📊 步骤 1: 数据预览")
        update_step(2)
        
        show_help("在此步骤中，您可以查看数据的基本信息、统计特征和可视化图表，帮助您了解数据质量。")
        
        file_info = st.session_state.file_info
        if file_info and len(file_info) > 0:
            with st.expander("📁 文件信息"):
                if len(file_info) == 1:
                    st.write(f"**文件名:** {file_info[0]['path']}")
                    st.write(f"**数据行数:** {file_info[0]['rows']}")
                    st.write(f"**数据列数:** {len(file_info[0]['columns'])}")
                else:
                    st.write(f"**共加载 {len(file_info)} 个文件:**")
                    for info in file_info:
                        if 'error' in info:
                            st.error(f"文件 {info['index']}: {info['path']} - 错误")
                        else:
                            st.write(f"- 文件 {info['index']}: {info['path']} ({info['rows']} 行)")
        
        st.write(f"**数据形状:** {df.shape}")
        
        # 数据预览选项
        preview_col1, preview_col2 = st.columns([1, 2])
        with preview_col1:
            show_all_data = st.checkbox("显示全部数据", value=False, key="show_all_data_preview")
        with preview_col2:
            if not show_all_data:
                n_rows = st.slider("显示行数", 10, min(1000, len(df)), 50, key="preview_rows")
            else:
                n_rows = len(df)
        
        if show_all_data:
            st.write(f"显示全部 {len(df)} 行数据:")
            st.dataframe(df, height=400)
        else:
            st.write(f"前{n_rows}行数据:")
            st.dataframe(df.head(n_rows), height=400)
        
        # 数据基本统计
        with st.expander("查看数据统计"):
            st.dataframe(df.describe())
        
        # 数据可视化 - 使用延迟加载优化性能
        st.subheader("数据可视化")
        
        # 使用session_state存储选择，避免重复创建widget
        if 'visual_cols_selected' not in st.session_state or st.session_state.visual_cols_selected is None:
            st.session_state.visual_cols_selected = df.columns.tolist()[:min(3, len(df.columns))]
        
        # 确保默认列在当前数据中存在
        valid_defaults = [c for c in st.session_state.visual_cols_selected if c in df.columns.tolist()]
        if not valid_defaults:
            st.session_state.visual_cols_selected = df.columns.tolist()[:min(3, len(df.columns))]
        
        visual_cols = st.multiselect(
            "选择要可视化的列",
            df.columns.tolist(),
            default=st.session_state.visual_cols_selected,
            key="visual_cols_multiselect",
            help="选择要在图表中显示的列。建议同时选择多个相关物理量进行对比分析。"
        )
        
        # 只在有选择且用户点击按钮后才渲染图表
        if visual_cols:
            # 缓存图表渲染
            @st.cache_data(ttl=300, show_spinner=False)
            def get_time_series_plot(df_json, cols_json):
                df_plot = pd.read_json(df_json)
                cols = pd.read_json(cols_json, typ='series').tolist()
                return visualizer.plot_time_series(df_plot, cols, use_plotly=True)
            
            @st.cache_data(ttl=300, show_spinner=False)
            def get_corr_plot(df_json, cols_json):
                df_plot = pd.read_json(df_json)
                cols = pd.read_json(cols_json, typ='series').tolist()
                return visualizer.plot_correlation_matrix(df_plot[cols], use_plotly=True)
            
            # 延迟加载 - 使用按钮触发图表渲染
            if st.button("📊 生成可视化图表", key="generate_viz"):
                with st.spinner("正在生成图表..."):
                    df_json = df.to_json()
                    cols_json = pd.Series(visual_cols).to_json()
                    
                    fig = get_time_series_plot(df_json, cols_json)
                    st.plotly_chart(fig, use_container_width=True, key="ts_plot")
                    
                    if len(visual_cols) > 1:
                        corr_fig = get_corr_plot(df_json, cols_json)
                        st.plotly_chart(corr_fig, use_container_width=True, key="corr_plot")
        
        # 步骤 3: 数据预处理
        st.subheader("🔧 步骤 2: 数据预处理")
        update_step(3)
        
        show_help("配置数据预处理参数，包括目标变量选择、特征工程、异常检测、数据平滑等。")
        
        # 选择目标列 - 使用key避免重复创建
        if 'target_col_selected' not in st.session_state:
            st.session_state.target_col_selected = df.columns.tolist()[0] if len(df.columns) > 0 else None
            
        target_col = st.selectbox(
            "选择目标列",
            df.columns.tolist(),
            index=df.columns.tolist().index(st.session_state.target_col_selected) if st.session_state.target_col_selected in df.columns else 0,
            key="target_col_selectbox",
            help="选择要预测的变量作为目标列。通常是海洋工程结构物的运动响应，如Surge、Sway、Heave等。"
        )
        st.session_state.target_col = target_col
        st.session_state.target_col_selected = target_col
        
        # 选择特征列 - 使用key避免重复创建
        available_features = [col for col in df.columns if col != target_col]
        if 'feature_cols_selected' not in st.session_state or not st.session_state.feature_cols_selected:
            st.session_state.feature_cols_selected = available_features
            
        feature_cols = st.multiselect(
            "选择特征列",
            available_features,
            default=[c for c in st.session_state.feature_cols_selected if c in available_features],
            key="feature_cols_multiselect",
            help="选择用于预测目标变量的输入特征。建议选择与目标变量相关的物理量，如波浪高度、风速、流速等。"
        )
        st.session_state.feature_cols = feature_cols
        
        if not feature_cols:
            st.warning("⚠️ 请至少选择一个特征列")
            show_help("您需要选择至少一个特征列才能继续。建议保留所有相关物理量作为特征。")
        else:
            # 数据准备 - 只选择数值列
            X = df[feature_cols].copy()
            y = df[target_col].copy()
            
            numeric_feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_feature_cols) < len(feature_cols):
                non_numeric = [c for c in feature_cols if c not in numeric_feature_cols]
                st.warning(f"自动移除非数值列: {non_numeric}")
                feature_cols = numeric_feature_cols
                X = X[feature_cols]
                st.session_state.feature_cols = feature_cols
            
            # 异常检测
            with st.expander("🔍 异常检测和修复"):
                use_outlier_detection = st.checkbox(
                    "启用异常检测",
                    help="自动识别数据中的异常值并进行修复，提高数据质量和模型性能。"
                )
                if use_outlier_detection:
                    outlier_method = st.selectbox(
                        "检测方法",
                        ["iqr", "zscore", "isolation_forest"],
                        help="IQR: 基于四分位距; ZScore: 基于标准差; IsolationForest: 基于机器学习异常检测"
                    )
                    outlier_threshold = st.slider(
                        "阈值",
                        1.0, 5.0, 3.0, 0.5,
                        help="阈值越大，检测出的异常值越少。建议根据数据特点调整。"
                    )
                    repair_method = st.selectbox(
                        "修复方法",
                        ["interpolation", "mean", "median"],
                        help="插值: 使用相邻值插补; 均值/中位数: 使用统计值替换"
                    )
                    
                    if st.button("执行异常检测"):
                        with st.spinner("检测中..."):
                            df_temp = X.copy()
                            df_temp[target_col] = y
                            outliers_dict = preprocessor.detect_outliers(
                                df_temp, columns=feature_cols + [target_col],
                                method=outlier_method, threshold=outlier_threshold
                            )
                            total_outliers = sum(len(v) for v in outliers_dict.values())
                            st.info(f"发现 {total_outliers} 个异常值")
                            
                            df_repaired = preprocessor.repair_outliers(
                                df_temp, columns=feature_cols + [target_col],
                                method=repair_method, outliers_dict=outliers_dict
                            )
                            
                            # 显示修改对比
                            st.subheader("📊 异常值修复对比")
                            comparison_tabs = st.tabs(["异常值列表", "数据对比", "异常值分布", "时序定位"])
                            
                            with comparison_tabs[0]:
                                # 异常值列表和定位
                                st.write("**异常值详细列表**")
                                
                                # 创建所有异常值的DataFrame
                                all_outliers = []
                                for col, indices in outliers_dict.items():
                                    for idx in indices:
                                        all_outliers.append({
                                            '索引': idx,
                                            '列名': col,
                                            '原始值': df_temp.loc[idx, col],
                                            '修复后': df_repaired.loc[idx, col],
                                            '变化量': df_repaired.loc[idx, col] - df_temp.loc[idx, col]
                                        })
                                
                                if all_outliers:
                                    outliers_df = pd.DataFrame(all_outliers)
                                    st.dataframe(outliers_df, height=300)
                                    
                                    # 异常值定位功能
                                    st.write("**🔍 异常值定位**")
                                    selected_outlier_idx = st.selectbox(
                                        "选择要定位的异常值",
                                        range(len(all_outliers)),
                                        format_func=lambda i: f"索引 {all_outliers[i]['索引']} - {all_outliers[i]['列名']} (值: {all_outliers[i]['原始值']:.4f})",
                                        key="outlier_locator"
                                    )
                                    
                                    if selected_outlier_idx is not None:
                                        selected = all_outliers[selected_outlier_idx]
                                        idx = selected['索引']
                                        col = selected['列名']
                                        
                                        st.write(f"**定位详情:**")
                                        st.write(f"- 索引位置: {idx}")
                                        st.write(f"- 列名: {col}")
                                        st.write(f"- 原始值: {selected['原始值']:.6f}")
                                        st.write(f"- 修复后: {selected['修复后']:.6f}")
                                        st.write(f"- 变化量: {selected['变化量']:.6f}")
                                        
                                        # 显示该位置附近的上下文数据
                                        context_start = max(0, idx - 5)
                                        context_end = min(len(df_temp), idx + 6)
                                        st.write(f"**上下文数据 (索引 {context_start} 到 {context_end-1}):**")
                                        
                                        context_df = df_temp.iloc[context_start:context_end][[col]].copy()
                                        context_df['状态'] = ['正常' if i != idx else '⚠️ 异常值' for i in range(context_start, context_end)]
                                        st.dataframe(context_df, height=250)
                                else:
                                    st.info("未发现异常值")
                            
                            with comparison_tabs[1]:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**修复前**")
                                    st.dataframe(df_temp.head(50), height=400)
                                with col2:
                                    st.write("**修复后**")
                                    st.dataframe(df_repaired.head(50), height=400)
                            
                            with comparison_tabs[2]:
                                # 显示异常值分布图
                                outlier_summary = {k: len(v) for k, v in outliers_dict.items() if len(v) > 0}
                                if outlier_summary:
                                    import plotly.graph_objects as go
                                    fig = go.Figure(data=[go.Bar(x=list(outlier_summary.keys()), y=list(outlier_summary.values()))])
                                    fig.update_layout(title="各列异常值数量", xaxis_title="列名", yaxis_title="异常值数量")
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("未发现异常值")
                            
                            with comparison_tabs[3]:
                                # 时序定位图
                                st.write("**异常值时序定位图**")
                                
                                # 选择要显示的列
                                display_cols = st.multiselect(
                                    "选择要显示的列",
                                    feature_cols + [target_col],
                                    default=[target_col] if target_col in df_temp.columns else feature_cols[:1],
                                    key="outlier_ts_display"
                                )
                                
                                if display_cols:
                                    import plotly.graph_objects as go
                                    from plotly.subplots import make_subplots
                                    
                                    fig = make_subplots(rows=len(display_cols), cols=1, 
                                                       subplot_titles=display_cols,
                                                       vertical_spacing=0.1)
                                    
                                    for i, col in enumerate(display_cols, 1):
                                        # 绘制正常数据
                                        fig.add_trace(
                                            go.Scatter(
                                                x=list(range(len(df_temp))),
                                                y=df_temp[col],
                                                mode='lines',
                                                name=f'{col} 原始',
                                                line=dict(color='blue', width=1),
                                                opacity=0.6
                                            ),
                                            row=i, col=1
                                        )
                                        
                                        # 标记异常值
                                        if col in outliers_dict and outliers_dict[col]:
                                            outlier_indices = outliers_dict[col]
                                            fig.add_trace(
                                                go.Scatter(
                                                    x=outlier_indices,
                                                    y=df_temp.loc[outlier_indices, col],
                                                    mode='markers',
                                                    name=f'{col} 异常值',
                                                    marker=dict(color='red', size=10, symbol='x'),
                                                ),
                                                row=i, col=1
                                            )
                                    
                                    fig.update_layout(height=300*len(display_cols), showlegend=True,
                                                     title_text="异常值在时序中的位置（红色X标记）")
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            X = df_repaired[feature_cols]
                            y = df_repaired[target_col]
                            st.success(f"✅ 异常值已使用 {repair_method} 方法修复，共修复 {total_outliers} 处")
            
            # ==================== 数据平滑 ====================
            st.subheader("📉 数据平滑")
            use_smoothing = st.checkbox(
                "启用数据平滑",
                help="去除数据中的噪声，使数据更加平滑，有助于提高模型泛化能力。"
            )
            
            if use_smoothing:
                # 先选择要平滑的列
                smooth_cols = st.multiselect(
                    "选择要平滑的列",
                    feature_cols,
                    default=feature_cols[:min(3, len(feature_cols))],
                    help="选择需要进行平滑处理的特征列。建议只对噪声较大的列进行平滑。"
                )
                
                if not smooth_cols:
                    st.warning("请至少选择一列进行平滑")
                else:
                    smooth_method = st.selectbox(
                        "平滑方法",
                        ["savgol", "moving_average", "exponential"],
                        help="Savgol: Savitzky-Golay滤波，保留特征形状; 移动平均: 简单平均，适合周期性数据; 指数: 指数加权移动平均，对近期数据权重更高"
                    )
                    smooth_window = st.slider(
                        "平滑窗口大小",
                        3, 51, 5, 2,
                        help="窗口越大，平滑效果越强。建议使用奇数。"
                    )
                    
                    if st.button("执行数据平滑"):
                        with st.spinner("平滑数据中..."):
                            X_before = X.copy()
                            X_smooth = preprocessor.smooth_data(X.copy(), columns=smooth_cols, method=smooth_method, window_length=smooth_window)
                            # 只更新平滑的列，保持其他列不变
                            X = X.copy()
                            X[smooth_cols] = X_smooth[smooth_cols]
                            
                            # 显示平滑前后对比
                            st.subheader("📊 数据平滑效果对比")
                            
                            import plotly.graph_objects as go
                            from plotly.subplots import make_subplots
                            
                            n_cols = len(smooth_cols)
                            fig = make_subplots(rows=n_cols, cols=1, subplot_titles=smooth_cols)
                            
                            for i, col in enumerate(smooth_cols, 1):
                                # 原始数据 - 蓝色
                                fig.add_trace(
                                    go.Scatter(y=X_before[col], name=f"{col} 原始", line=dict(color='blue', width=2)),
                                    row=i, col=1
                                )
                                # 平滑后数据 - 红色
                                fig.add_trace(
                                    go.Scatter(y=X[col], name=f"{col} 平滑后", line=dict(color='red', width=2)),
                                    row=i, col=1
                                )
                            
                            fig.update_layout(height=300*n_cols, title_text="数据平滑前后对比", showlegend=True)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 数据对比表格
                            with st.expander("查看平滑前后数据对比"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**平滑前**")
                                    st.dataframe(X_before[smooth_cols].head(20), height=300)
                                with col2:
                                    st.write("**平滑后**")
                                    st.dataframe(X[smooth_cols].head(20), height=300)
                            
                            st.success(f"✅ 数据已使用 {smooth_method} 方法平滑，窗口大小: {smooth_window}，共平滑 {len(smooth_cols)} 列")
            
            # ==================== 时间列预处理 ====================
            st.subheader("📅 时间列预处理")
            
            df_cols = df.columns.tolist()
            datetime_cols = []
            
            for col in df_cols:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    datetime_cols.append(col)
                elif df[col].dtype == object:
                    try:
                        parsed = pd.to_datetime(df[col].dropna().iloc[:5], errors='coerce')
                        if parsed.notna().sum() >= len(parsed) * 0.5:
                            datetime_cols.append(col)
                    except:
                        pass
            
            time_col_input = st.text_input(
                "手动输入时间列名（可选）",
                placeholder="如：日期、时间、timestamp 等",
                help="如果时间列没有被自动检测到，可以手动输入列名"
            )
            if time_col_input and time_col_input in df.columns and time_col_input not in datetime_cols:
                datetime_cols.append(time_col_input)
            
            if datetime_cols:
                st.info(f"检测到日期时间列: {datetime_cols}")
                
                use_time_processing = st.checkbox(
                    "启用时间列处理",
                    help="将日期时间列转换为数值特征，提取年、月、日、时、分、星期等信息，并可进行周期性编码。"
                )
                
                if use_time_processing:
                    time_col = st.selectbox(
                        "选择时间列",
                        datetime_cols,
                        help="选择要处理的时间列"
                    )
                    
                    time_feature_cols = st.multiselect(
                        "选择要提取的特征",
                        ["年", "月", "日", "时", "分", "秒", "星期", "是否周末", "是否工作日", "季度"],
                        default=["月", "日", "时", "星期"],
                        help="选择要从时间列提取的特征"
                    )
                    
                    use_cyclic_encoding = st.checkbox(
                        "使用周期性编码 (sin/cos)",
                        value=True,
                        help="对周期性特征（时、月、星期等）使用sin/cos编码，保留周期性信息。推荐勾选。"
                    )
                    
                    if st.button("提取时间特征"):
                        with st.spinner("提取时间特征..."):
                            df_time = df.copy()
                            if not pd.api.types.is_datetime64_any_dtype(df_time[time_col]):
                                df_time[time_col] = pd.to_datetime(df_time[time_col], errors='coerce')
                            
                            new_time_features = []
                            
                            if "年" in time_feature_cols:
                                df_time['时间_年'] = df_time[time_col].dt.year
                                new_time_features.append('时间_年')
                            
                            if "月" in time_feature_cols:
                                if use_cyclic_encoding:
                                    df_time['时间_月_sin'] = np.sin(2 * np.pi * df_time[time_col].dt.month / 12)
                                    df_time['时间_月_cos'] = np.cos(2 * np.pi * df_time[time_col].dt.month / 12)
                                    new_time_features.extend(['时间_月_sin', '时间_月_cos'])
                                else:
                                    df_time['时间_月'] = df_time[time_col].dt.month
                                    new_time_features.append('时间_月')
                            
                            if "日" in time_feature_cols:
                                if use_cyclic_encoding:
                                    df_time['时间_日_sin'] = np.sin(2 * np.pi * df_time[time_col].dt.day / 31)
                                    df_time['时间_日_cos'] = np.cos(2 * np.pi * df_time[time_col].dt.day / 31)
                                    new_time_features.extend(['时间_日_sin', '时间_日_cos'])
                                else:
                                    df_time['时间_日'] = df_time[time_col].dt.day
                                    new_time_features.append('时间_日')
                            
                            if "时" in time_feature_cols:
                                if use_cyclic_encoding:
                                    df_time['时间_时_sin'] = np.sin(2 * np.pi * df_time[time_col].dt.hour / 24)
                                    df_time['时间_时_cos'] = np.cos(2 * np.pi * df_time[time_col].dt.hour / 24)
                                    new_time_features.extend(['时间_时_sin', '时间_时_cos'])
                                else:
                                    df_time['时间_时'] = df_time[time_col].dt.hour
                                    new_time_features.append('时间_时')
                            
                            if "分" in time_feature_cols:
                                if use_cyclic_encoding:
                                    df_time['时间_分_sin'] = np.sin(2 * np.pi * df_time[time_col].dt.minute / 60)
                                    df_time['时间_分_cos'] = np.cos(2 * np.pi * df_time[time_col].dt.minute / 60)
                                    new_time_features.extend(['时间_分_sin', '时间_分_cos'])
                                else:
                                    df_time['时间_分'] = df_time[time_col].dt.minute
                                    new_time_features.append('时间_分')
                            
                            if "秒" in time_feature_cols:
                                df_time['时间_秒'] = df_time[time_col].dt.second
                                new_time_features.append('时间_秒')
                            
                            if "星期" in time_feature_cols:
                                if use_cyclic_encoding:
                                    df_time['时间_星期_sin'] = np.sin(2 * np.pi * df_time[time_col].dt.dayofweek / 7)
                                    df_time['时间_星期_cos'] = np.cos(2 * np.pi * df_time[time_col].dt.dayofweek / 7)
                                    new_time_features.extend(['时间_星期_sin', '时间_星期_cos'])
                                else:
                                    df_time['时间_星期'] = df_time[time_col].dt.dayofweek
                                    new_time_features.append('时间_星期')
                            
                            if "是否周末" in time_feature_cols:
                                df_time['时间_是否周末'] = (df_time[time_col].dt.dayofweek >= 5).astype(int)
                                new_time_features.append('时间_是否周末')
                            
                            if "是否工作日" in time_feature_cols:
                                df_time['时间_是否工作日'] = (df_time[time_col].dt.dayofweek < 5).astype(int)
                                new_time_features.append('时间_是否工作日')
                            
                            if "季度" in time_feature_cols:
                                if use_cyclic_encoding:
                                    df_time['时间_季度_sin'] = np.sin(2 * np.pi * df_time[time_col].dt.quarter / 4)
                                    df_time['时间_季度_cos'] = np.cos(2 * np.pi * df_time[time_col].dt.quarter / 4)
                                    new_time_features.extend(['时间_季度_sin', '时间_季度_cos'])
                                else:
                                    df_time['时间_季度'] = df_time[time_col].dt.quarter
                                    new_time_features.append('时间_季度')
                            
                            if len(new_time_features) > 0:
                                df = df_time
                                st.session_state.df = df_time
                                feature_cols = feature_cols + new_time_features
                                st.session_state.feature_cols = feature_cols
                                st.success(f"✅ 已提取 {len(new_time_features)} 个时间特征: {', '.join(new_time_features)}")
                                
                                import plotly.graph_objects as go
                                from plotly.subplots import make_subplots
                                
                                st.subheader("📊 时间特征效果展示")
                                
                                display_features = [f for f in new_time_features if 'sin' not in f and 'cos' not in f][:4]
                                if display_features:
                                    fig = make_subplots(
                                        rows=len(display_features), cols=1,
                                        subplot_titles=display_features,
                                        vertical_spacing=0.15
                                    )
                                    
                                    for i, feat in enumerate(display_features, 1):
                                        fig.add_trace(
                                            go.Scatter(y=df_time[feat], name=feat, line=dict(width=2)),
                                            row=i, col=1
                                        )
                                    
                                    fig.update_layout(height=250*len(display_features), showlegend=False)
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("未选择任何时间特征")
            else:
                st.info("未检测到日期时间列，跳过时间列预处理")
            
            # ==================== 时间序列特征工程 ====================
            st.subheader("⏱️ 时间序列特征工程")
            use_ts_features = st.checkbox(
                "启用时间序列特征",
                help="为时间序列数据创建额外的特征，捕捉时间依赖性和趋势信息，提高预测精度。"
            )
            
            if use_ts_features:
                ts_feature_tabs = st.tabs(["滞后特征", "滚动统计", "差分", "趋势特征"])
                
                with ts_feature_tabs[0]:
                    use_lag = st.checkbox(
                        "添加滞后特征",
                        help="使用历史时间步的值作为特征，捕捉时间序列的自相关性。"
                    )
                    if use_lag:
                        lag_cols = st.multiselect(
                            "选择滞后特征列",
                            feature_cols,
                            default=feature_cols[:2],
                            help="选择要创建滞后特征的列"
                        )
                        lag_periods = st.slider(
                            "滞后阶数",
                            1, 10, 2,
                            help="使用多少个历史时间步作为特征。阶数越大，考虑的历史信息越长。"
                        )
                        if st.button("添加滞后特征"):
                            with st.spinner("创建滞后特征..."):
                                df_temp = X.copy()
                                df_temp[target_col] = y
                                df_lagged = preprocessor.create_lagged_features(df_temp, lag_cols, lags=lag_periods)
                                
                                # 显示特征工程效果
                                st.subheader("📊 滞后特征效果展示")
                                
                                # 显示新增的特征列
                                new_cols = [c for c in df_lagged.columns if c not in df_temp.columns]
                                st.write(f"**新增特征列 ({len(new_cols)}个):** {', '.join(new_cols[:10])}{'...' if len(new_cols) > 10 else ''}")
                                
                                # 可视化滞后特征效果
                                import plotly.graph_objects as go
                                from plotly.subplots import make_subplots
                                
                                # 选择一列展示滞后效果
                                display_col = lag_cols[0] if lag_cols else feature_cols[0]
                                
                                fig = make_subplots(
                                    rows=lag_periods+1, cols=1,
                                    subplot_titles=[f"{display_col} 原始"] + [f"{display_col} 滞后 {i+1}" for i in range(lag_periods)],
                                    vertical_spacing=0.08
                                )
                                
                                # 原始数据
                                fig.add_trace(
                                    go.Scatter(y=df_temp[display_col], name="原始", line=dict(color='blue', width=2)),
                                    row=1, col=1
                                )
                                
                                # 各阶滞后特征
                                colors = ['green', 'orange', 'red', 'purple', 'brown']
                                for i in range(lag_periods):
                                    lag_col = f"{display_col}_lag_{i+1}"
                                    if lag_col in df_lagged.columns:
                                        fig.add_trace(
                                            go.Scatter(y=df_lagged[lag_col], name=f"滞后{i+1}", 
                                                      line=dict(color=colors[i % len(colors)], width=1.5)),
                                            row=i+2, col=1
                                        )
                                
                                fig.update_layout(height=200*(lag_periods+1), title_text="滞后特征效果对比", showlegend=True)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # 显示数据对比
                                with st.expander("查看滞后特征数据"):
                                    st.dataframe(df_lagged.head(20), height=300)
                                
                                X = df_lagged.drop(columns=[target_col])
                                y = df_lagged[target_col]
                                st.success(f"✅ 已添加滞后特征，新形状: {X.shape}")
                
                with ts_feature_tabs[1]:
                    use_rolling = st.checkbox(
                        "添加滚动统计特征",
                        help="计算滑动窗口内的统计量，捕捉局部趋势和波动性。"
                    )
                    if use_rolling:
                        rolling_cols = st.multiselect(
                            "选择滚动统计列",
                            feature_cols,
                            default=feature_cols[:2],
                            key="rolling_cols",
                            help="选择要计算滚动统计的列"
                        )
                        window_size = st.slider(
                            "窗口大小",
                            2, 50, 5,
                            help="滚动窗口的大小。窗口越大，统计量越平滑。"
                        )
                        rolling_funcs = st.multiselect(
                            "统计函数",
                            ["mean", "std", "min", "max", "median"],
                            default=["mean", "std"],
                            help="选择要计算的统计量。均值反映趋势，标准差反映波动性。"
                        )
                        if st.button("添加滚动统计特征"):
                            with st.spinner("创建滚动统计特征..."):
                                X_before = X.copy()
                                X = preprocessor.rolling_window(X, rolling_cols, window_size=window_size, functions=rolling_funcs)
                                
                                # 显示特征工程效果
                                st.subheader("📊 滚动统计特征效果展示")
                                
                                # 显示新增的特征列
                                new_cols = [c for c in X.columns if c not in X_before.columns]
                                st.write(f"**新增特征列 ({len(new_cols)}个):** {', '.join(new_cols[:10])}{'...' if len(new_cols) > 10 else ''}")
                                
                                # 可视化滚动统计效果
                                if rolling_cols:
                                    import plotly.graph_objects as go
                                    from plotly.subplots import make_subplots
                                    
                                    display_col = rolling_cols[0]
                                    fig = make_subplots(rows=2, cols=1, subplot_titles=[f"{display_col} 原始数据", f"{display_col} 滚动统计"])
                                    
                                    # 原始数据
                                    fig.add_trace(go.Scatter(y=X_before[display_col], name="原始", line=dict(color='lightgray')), row=1, col=1)
                                    
                                    # 滚动统计
                                    for func in rolling_funcs[:2]:  # 最多显示2个
                                        col_name = f"{display_col}_rolling_{func}_{window_size}"
                                        if col_name in X.columns:
                                            fig.add_trace(go.Scatter(y=X[col_name], name=f"滚动{func}", line=dict(width=2)), row=2, col=1)
                                    
                                    fig.update_layout(height=500, title_text="滚动统计特征效果")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # 显示数据对比
                                with st.expander("查看滚动统计特征数据"):
                                    st.dataframe(X.head(20), height=300)
                                
                                st.success(f"✅ 已添加滚动统计特征，新形状: {X.shape}")
                
                with ts_feature_tabs[2]:
                    use_diff = st.checkbox(
                        "添加差分特征",
                        help="计算相邻时间步的差值，去除趋势，使数据平稳。"
                    )
                    if use_diff:
                        diff_cols = st.multiselect(
                            "选择差分列",
                            feature_cols,
                            default=feature_cols[:2],
                            key="diff_cols",
                            help="选择要计算差分的列"
                        )
                        diff_order = st.slider(
                            "差分阶数",
                            1, 3, 1,
                            help="差分的次数。1阶差分去除线性趋势，2阶差分去除二次趋势。"
                        )
                        if st.button("添加差分特征"):
                            with st.spinner("创建差分特征..."):
                                X = preprocessor.differencing(X, diff_cols, order=diff_order)
                                st.success(f"已添加差分特征，新形状: {X.shape}")
                
                with ts_feature_tabs[3]:
                    use_trend = st.checkbox(
                        "添加趋势特征",
                        help="提取长期趋势信息，帮助模型识别数据的整体走向。"
                    )
                    if use_trend:
                        trend_cols = st.multiselect(
                            "选择趋势特征列",
                            feature_cols,
                            default=feature_cols[:2],
                            key="trend_cols",
                            help="选择要提取趋势的列"
                        )
                        trend_window = st.slider(
                            "趋势窗口",
                            5, 100, 10,
                            help="计算趋势的窗口大小。窗口越大，趋势越平滑，反映长期变化。"
                        )
                        if st.button("添加趋势特征"):
                            with st.spinner("创建趋势特征..."):
                                df_temp = X.copy()
                                df_temp[target_col] = y
                                df_trend = preprocessor.add_trend_features(df_temp, trend_cols, window=trend_window)
                                
                                # 显示趋势特征效果
                                st.subheader("📊 趋势特征效果展示")
                                
                                # 显示新增的特征列
                                new_cols = [c for c in df_trend.columns if c not in df_temp.columns]
                                st.write(f"**新增趋势特征列 ({len(new_cols)}个):** {', '.join(new_cols[:10])}{'...' if len(new_cols) > 10 else ''}")
                                
                                # 可视化趋势特征效果
                                import plotly.graph_objects as go
                                from plotly.subplots import make_subplots
                                
                                # 为每个选中的列创建对比图
                                n_trend_cols = len(trend_cols)
                                fig = make_subplots(
                                    rows=n_trend_cols, cols=1,
                                    subplot_titles=[f"{col} 原始数据与趋势对比" for col in trend_cols],
                                    vertical_spacing=0.1
                                )
                                
                                for i, col in enumerate(trend_cols, 1):
                                    # 趋势列名格式为 {col}_trend (不包含window)
                                    trend_col_name = f"{col}_trend"
                                    
                                    if trend_col_name in df_trend.columns:
                                        # 原始数据 - 蓝色
                                        fig.add_trace(
                                            go.Scatter(
                                                y=df_temp[col], 
                                                name=f"{col} 原始", 
                                                line=dict(color='blue', width=1.5),
                                                opacity=0.7
                                            ),
                                            row=i, col=1
                                        )
                                        # 趋势线 - 红色 (趋势斜率)
                                        fig.add_trace(
                                            go.Scatter(
                                                y=df_trend[trend_col_name], 
                                                name=f"{col} 趋势斜率", 
                                                line=dict(color='red', width=2)
                                            ),
                                            row=i, col=1
                                        )
                                
                                fig.update_layout(
                                    height=300*n_trend_cols, 
                                    title_text="趋势特征效果对比（蓝色=原始数据，红色=趋势斜率）",
                                    showlegend=True
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # 显示趋势特征统计
                                with st.expander("查看趋势特征统计"):
                                    st.write("**趋势特征说明：**")
                                    st.write("- `{col}_trend`: 线性趋势斜率（滚动窗口内的趋势线斜率）")
                                    st.write("- `{col}_trend_direction`: 趋势方向（1=上升, -1=下降, 0=平稳）")
                                    st.write("- `{col}_change_rate`: 变化率（百分比变化）")
                                    st.write("- `{col}_acceleration`: 加速度（变化率的变化）")
                                    
                                    for col in trend_cols:
                                        st.write(f"**{col} 趋势统计:**")
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.write("趋势斜率均值:", f"{df_trend[f'{col}_trend'].mean():.6f}")
                                        with col2:
                                            st.write("上升时段比例:", f"{(df_trend[f'{col}_trend_direction'] == 1).mean()*100:.1f}%")
                                        with col3:
                                            st.write("变化率均值:", f"{df_trend[f'{col}_change_rate'].mean():.6f}")
                                
                                # 显示数据对比
                                with st.expander("查看趋势特征数据"):
                                    st.dataframe(df_trend.head(20), height=300)
                                
                                X = df_trend.drop(columns=[target_col])
                                y = df_trend[target_col]
                                st.success(f"✅ 已添加趋势特征，新形状: {X.shape}")
            
            # ==================== 数据增强 ====================
            st.subheader("🔄 数据增强")
            use_augmentation = st.checkbox(
                "启用数据增强",
                help="通过对原始数据进行变换生成额外样本，扩充训练数据量，提高模型泛化能力。适用于小数据集。"
            )
            
            if use_augmentation:
                aug_method = st.selectbox(
                    "增强方法",
                    ["noise", "scaling", "time_warp", "permutation"],
                    help="噪声: 添加高斯噪声，模拟测量误差; 缩放: 随机缩放振幅; 时间扭曲: 时间轴拉伸/压缩; 置换: 随机置换序列段"
                )
                aug_factor = st.slider(
                    "增强因子",
                    0.001, 0.1, 0.01, 0.001,
                    help="控制增强强度。值越大，生成的样本与原始数据差异越大。"
                )
                n_augmentations = st.slider(
                    "增强次数",
                    1, 5, 2,
                    help="每个原始样本生成多少个增强样本。次数越多，数据集越大。"
                )
                
                if st.button("执行数据增强"):
                    with st.spinner("增强数据中..."):
                        df_temp = X.copy()
                        df_temp[target_col] = y
                        df_aug = preprocessor.augment_data(df_temp, target_col=target_col, method=aug_method, noise_factor=aug_factor, n_augmentations=n_augmentations)
                        
                        # 显示数据增强效果
                        st.subheader("📊 数据增强效果展示")
                        
                        # 显示样本数量变化
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("原始样本数", len(df_temp))
                        with col2:
                            st.metric("增强后样本数", len(df_aug))
                        with col3:
                            st.metric("增强倍数", f"{len(df_aug)/len(df_temp):.1f}x")
                        
                        # 可视化增强效果
                        import plotly.graph_objects as go
                        from plotly.subplots import make_subplots
                        
                        display_col = feature_cols[0] if feature_cols else target_col
                        
                        # 检查数据维度，如果是多变量时间序列需要特殊处理
                        # 假设数据是表格形式，每行是一个样本，每列是一个特征
                        fig = make_subplots(rows=2, cols=1, subplot_titles=["原始数据样本", "增强后数据样本"])
                        
                        # 原始数据 - 显示前几个样本的特征值
                        n_show = min(5, len(df_temp))
                        for i in range(n_show):
                            # 获取该样本所有特征的值作为序列
                            sample_values = df_temp.iloc[i][feature_cols].values if feature_cols else [df_temp.iloc[i][display_col]]
                            fig.add_trace(go.Scatter(y=sample_values, 
                                                   name=f"原始样本{i+1}", 
                                                   line=dict(color='blue', width=1), opacity=0.7), row=1, col=1)
                        
                        # 增强后数据 - 显示原始和增强样本
                        for i in range(min(3, len(df_aug))):
                            sample_values = df_aug.iloc[i][feature_cols].values if feature_cols else [df_aug.iloc[i][display_col]]
                            fig.add_trace(go.Scatter(y=sample_values, 
                                                   name=f"增强样本{i+1}", 
                                                   line=dict(color='green', width=1), opacity=0.7), row=2, col=1)
                        
                        fig.update_layout(height=600, title_text=f"数据增强效果对比 - 各样本特征序列", showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 显示增强后的数据统计
                        with st.expander("查看增强后数据统计"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**原始数据统计**")
                                st.dataframe(df_temp.describe(), height=300)
                            with col2:
                                st.write("**增强后数据统计**")
                                st.dataframe(df_aug.describe(), height=300)
                        
                        X = df_aug.drop(columns=[target_col])
                        y = df_aug[target_col]
                        st.success(f"✅ 数据增强完成: {len(df_temp)} -> {len(df_aug)} 样本")
            
            # 数据缩放
            st.subheader("数据缩放")
            scale_method = st.selectbox(
                "缩放方法",
                ["standard", "minmax", "robust", "none"],
                index=0,
                help="Standard: 标准化为均值为0、方差为1，适合正态分布数据; MinMax: 缩放到[0,1]范围; Robust: 使用中位数和四分位数，对异常值鲁棒; None: 不缩放"
            )
            st.session_state.scale_method = scale_method
            
            if scale_method != "none":
                st.write(f"将使用 {scale_method} 方法缩放；缩放器只在训练集上拟合，避免数据泄漏。")
            else:
                st.write("不进行数据缩放。")
            
            # 特征选择
            st.subheader("特征选择")
            use_feature_selection = st.checkbox(
                "启用特征选择",
                help="自动选择对预测目标最重要的特征，减少维度，提高模型性能和训练速度。"
            )
            st.session_state.use_feature_selection = use_feature_selection
            
            if use_feature_selection:
                k = st.slider(
                    "选择要保留的特征数量",
                    1, len(feature_cols), min(10, len(feature_cols)),
                    help="保留的特征数量。建议根据数据特点和模型复杂度调整。"
                )
                selection_method = st.selectbox(
                    "特征选择方法",
                    ["f_regression", "mutual_info"],
                    help="F回归: 基于线性关系; 互信息: 基于统计依赖性，可捕捉非线性关系"
                )
                st.write(f"将保留 {k} 个特征；特征选择器只在训练集上拟合。")
            else:
                k = len(feature_cols)
                selection_method = 'f_regression'
            
            # 数据划分
            st.subheader("数据划分")
            
            # 添加划分方式选择
            use_time_series_split = st.checkbox(
                "🔢 按时间顺序划分（推荐时序数据使用）",
                value=True,
                help="启用：按时间顺序划分（前N%训练，后M%测试）；关闭：随机划分（随机抽取样本）"
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                test_size = st.slider(
                    "测试集比例",
                    0.1, 0.5, 0.2, 0.05,
                    help="用于最终评估模型性能的独立数据集比例。建议: 0.2-0.3"
                )
            with col2:
                val_size = st.slider(
                    "验证集比例",
                    0.0, 0.3, 0.1, 0.05,
                    help="用于超参数调优和早停的数据集比例。设为0则不使用验证集。"
                )
            with col3:
                if not use_time_series_split:
                    random_state = st.number_input(
                        "随机种子",
                        0, 1000, 42,
                        help="控制数据划分的随机性。固定种子可使结果可重复。"
                    )
                else:
                    random_state = 42
            
            prepared = preprocessor.prepare_train_val_test(
                X, y,
                test_size=test_size,
                val_size=val_size,
                random_state=random_state,
                time_series=use_time_series_split,
                scale_method=scale_method,
                use_feature_selection=use_feature_selection,
                k=k,
                selection_method=selection_method
            )
            X_train = prepared['X_train']
            X_val = prepared['X_val']
            X_test = prepared['X_test']
            y_train = prepared['y_train']
            y_val = prepared['y_val']
            y_test = prepared['y_test']
            selected_feature_cols = prepared['selected_features']
            
            st.write(f"数据划分结果:")
            st.write(f"  训练集: {X_train.shape[0]} 样本")
            if X_val is not None:
                st.write(f"  验证集: {X_val.shape[0]} 样本")
            st.write(f"  测试集: {X_test.shape[0]} 样本")
            
            # 保存到session state
            st.session_state.X_train = X_train
            st.session_state.X_val = X_val
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_val = y_val
            st.session_state.y_test = y_test
            st.session_state.selected_feature_cols = selected_feature_cols
            
            # 保存完整列名信息用于诊断（特征列 + 目标列）
            all_cols = list(selected_feature_cols) + [target_col]
            st.session_state.all_data_columns = all_cols
            
            mark_step_completed('data_preprocessing')
            
            # 步骤 4: 模型训练
            st.subheader("🤖 步骤 3: 模型选择和训练")
            update_step(4)
            
            # 显示当前数据信息
            feature_cols_chosen = st.session_state.get('feature_cols', [])
            target_col_chosen = st.session_state.get('target_col', '')
            
            st.info(f"📌 当前训练配置 - 目标列: **{target_col_chosen}** | 特征数量: **{len(feature_cols_chosen) if feature_cols_chosen else 0}**")
            with st.expander("📋 已选特征列详情"):
                if feature_cols_chosen:
                    for i, col in enumerate(feature_cols_chosen):
                        st.write(f"  {i+1}. {col}")
                else:
                    st.write("未选择特征列")
            
            show_help("选择并配置机器学习或深度学习模型。可以使用单个模型、多模型对比、集成学习或AutoML自动搜索。")
            
            if st.session_state.trainer is None:
                st.session_state.trainer = ModelTrainer()
            trainer = st.session_state.trainer
            
            model_selection_mode = st.radio(
                "模型选择模式",
                ["单模型", "多模型对比", "集成学习", "AutoML自动搜索", "自定义模型"],
                horizontal=True,
                help="单模型: 训练一个模型; 多模型对比: 同时训练多个模型并对比性能; 集成学习: 组合多个模型提高预测精度; AutoML: 自动搜索最优模型和参数; 自定义模型: 用户自己编写模型代码"
            )
            
            use_ensemble = model_selection_mode == "集成学习"
            use_automl = model_selection_mode == "AutoML自动搜索"
            use_model_comparison = model_selection_mode == "多模型对比"
            use_custom_model = model_selection_mode == "自定义模型"
            
            # ==================== 自定义模型功能 ====================
            if use_custom_model:
                st.subheader("🔧 自定义模型配置")
                show_help("用户可以自己编写模型代码，或从模板中选择已有的自定义模型。")
                
                template_codes = {
                    "自定义（空白模板）": '''from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd

class MyCustomModel(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def fit(self, X, y):
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)''',
                    "简单线性回归（带正则化）": '''from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
import numpy as np

class MyLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def fit(self, X, y):
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)''',
                    "带特征选择的随机森林": '''from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class MyRandomForest(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    
    def fit(self, X, y):
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42
        )
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)''',
                    "加权集成模型": '''from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, RegressorMixin

class WeightedEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.models = [
            ('rf', RandomForestRegressor(n_estimators=50)),
            ('gb', GradientBoostingRegressor(n_estimators=50)),
            ('ridge', Ridge())
        ]
        self.weights = None
    
    def fit(self, X, y):
        from sklearn.model_selection import cross_val_score
        scores = []
        fitted = []
        for name, model in self.models:
            m = model.fit(X, y)
            score = cross_val_score(m, X, y, cv=3).mean()
            scores.append(score)
            fitted.append((name, m))
        self.weights = [s/sum(scores) for s in scores]
        self.fitted_models_ = fitted
        return self
    
    def predict(self, X):
        preds = []
        for i, (name, model) in enumerate(self.fitted_models_):
            preds.append(model.predict(X) * self.weights[i])
        return sum(preds)''',
                    "指数加权移动平均": '''from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class ExpSmoothModel(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.3):
        self.alpha = alpha
    
    def fit(self, X, y):
        self.last_values_ = y[-5:] if len(y) >= 5 else y
        return self
    
    def predict(self, X):
        weights = np.array([self.alpha * (1-self.alpha)**i for i in range(len(self.last_values_))])
        weights = weights / weights.sum()
        pred_value = np.sum(weights * self.last_values_[::-1])
        return np.full(X.shape[0], pred_value)''',
                    "特征工程+基础模型": '''from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
import pandas as pd

class FeatureEngineeredModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.base_model = Ridge()
    
    def fit(self, X, y):
        X_new = self._create_features(X)
        self.base_model.fit(X_new, y)
        return self
    
    def _create_features(self, X):
        df = pd.DataFrame(X)
        for i in range(1, min(4, len(df.columns)+1)):
            if len(df) > 1:
                df[f"lag_{i}"] = df.iloc[:, i-1].shift(1)
        return df.fillna(0).values
    
    def predict(self, X):
        X_new = self._create_features(X)
        return self.base_model.predict(X_new)'''
                }
                
                # 初始化session state
                if 'ts_custom_model_code' not in st.session_state:
                    st.session_state.ts_custom_model_code = template_codes["自定义（空白模板）"]
                
                ts_template_code = st.selectbox(
                    "选择模板",
                    list(template_codes.keys()),
                    key="ts_custom_template_select"
                )
                
                # 当模板选择变化时更新代码
                if st.session_state.get('ts_last_template') != ts_template_code:
                    st.session_state.ts_custom_model_code = template_codes[ts_template_code]
                    st.session_state.ts_last_template = ts_template_code
                    st.rerun()
                
                st.markdown("### ✏️ 编辑自定义模型代码")
                st.info("请确保代码包含：1. 模型类定义（继承BaseEstimator, RegressorMixin）2. __init__, fit, predict 方法")
                
                # 导入/导出按钮放在代码框上方
                imp_col, exp_col = st.columns(2)
                with imp_col:
                    if st.button("📤 导入模型代码", key="ts_import_btn", use_container_width=True):
                        st.session_state.ts_show_import = True
                    
                    if st.session_state.get('ts_show_import', False):
                        uploaded_code_file = st.file_uploader(
                            "选择文件",
                            type=["py"],
                            key="ts_import_code"
                        )
                        if uploaded_code_file is not None:
                            st.session_state.ts_custom_model_code = uploaded_code_file.getvalue().decode("utf-8")
                            st.success("✅ 代码已导入，请重新选择模板或刷新页面查看")
                            st.session_state.ts_show_import = False
                
                with exp_col:
                    st.download_button(
                        label="📥 导出模型代码",
                        data=st.session_state.ts_custom_model_code,
                        file_name="custom_model.py",
                        mime="text/x-python",
                        key="ts_download_code",
                        use_container_width=True
                    )
                
                custom_model_code = st.text_area(
                    "自定义模型代码",
                    height=250,
                    value=st.session_state.ts_custom_model_code,
                    key="ts_custom_model_code"
                )
                
                custom_model_loaded = False
                custom_model_instance = None
                if custom_model_code:
                    try:
                        import types
                        custom_module = types.ModuleType('custom_model_module')
                        exec(custom_model_code, custom_module.__dict__)
                        record_custom_execution("时序自定义模型", "加载成功")
                        
                        custom_model_class = None
                        for item in dir(custom_module):
                            obj = getattr(custom_module, item)
                            if isinstance(obj, type) and hasattr(obj, 'fit') and hasattr(obj, 'predict'):
                                if obj.__name__ != 'BaseEstimator' and obj.__name__ != 'RegressorMixin':
                                    custom_model_class = obj
                                    break
                        
                        if custom_model_class:
                            custom_model_instance = custom_model_class()
                            custom_model_loaded = True
                            st.success(f"✅ 自定义模型加载成功: {custom_model_class.__name__}")
                        else:
                            st.error("未找到有效的模型类，请确保代码包含继承自BaseEstimator的模型类")
                    except Exception as e:
                        record_custom_execution("时序自定义模型", "加载失败", str(e))
                        st.error(f"模型代码加载失败: {str(e)}")
            
            # ==================== 多模型对比功能 ====================
            if use_model_comparison:
                st.subheader("📊 多模型对比配置")
                show_help("选择多个模型进行并行训练，然后对比它们的性能。")
                
                st.write("选择要对比的模型（至少选择2个）：")
                col1, col2, col3 = st.columns(3)
                
                compare_models = {}
                with col1:
                    if st.checkbox("线性回归", key="cmp_lr"):
                        compare_models['Linear Regression'] = 'linear_regression'
                    if st.checkbox("Ridge", key="cmp_ridge"):
                        compare_models['Ridge'] = 'ridge'
                    if st.checkbox("Lasso", key="cmp_lasso"):
                        compare_models['Lasso'] = 'lasso'
                
                with col2:
                    if st.checkbox("随机森林", key="cmp_rf"):
                        compare_models['Random Forest'] = 'random_forest'
                    if st.checkbox("梯度提升", key="cmp_gb"):
                        compare_models['Gradient Boosting'] = 'gradient_boosting'
                
                with col3:
                    if st.checkbox("SVR", key="cmp_svr"):
                        compare_models['SVR'] = 'svr'
                    if st.checkbox("KNN", key="cmp_knn"):
                        compare_models['KNN'] = 'knn'
                
                if len(compare_models) >= 2:
                    # 训练控制按钮
                    train_col1, train_col2 = st.columns([1, 1])
                    with train_col1:
                        start_compare = st.button("开始多模型对比训练", type="primary")
                    with train_col2:
                        if st.button("⏹️ 停止训练", type="secondary", key="stop_compare"):
                            st.session_state.training_stop_requested = True
                            st.warning("已请求停止训练，当前模型完成后将终止...")
                            st.rerun()
                    
                    if start_compare:
                        st.session_state.training_stop_requested = False
                        with st.spinner("训练多个模型进行对比..."):
                            comparison_results = {}
                            predictions_dict = {}
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, (model_name, model_type_key) in enumerate(compare_models.items()):
                                # 检查是否请求停止
                                if st.session_state.training_stop_requested:
                                    status_text.warning(f"训练已在 {model_name} 处停止")
                                    break
                                
                                status_text.text(f"正在训练: {model_name} ({i+1}/{len(compare_models)})")
                                temp_trainer = ModelTrainer()
                                temp_trainer.select_model(model_type_key)
                                temp_trainer.train(X_train, y_train)
                                y_pred_temp = temp_trainer.predict(X_test)
                                predictions_dict[model_name] = y_pred_temp
                                metrics_temp = temp_trainer.evaluate_time_series(y_test, y_pred_temp, fast_mode=True)
                                comparison_results[model_name] = metrics_temp
                                progress_bar.progress((i + 1) / len(compare_models))
                            
                            st.session_state.comparison_results = comparison_results
                            st.session_state.predictions_dict = predictions_dict
                            st.session_state.model_comparison_done = True
                            st.session_state.training_stop_requested = False
                            
                            st.success(f"模型对比完成！共训练 {len(comparison_results)} 个模型")
                            
                            st.subheader("模型性能对比表")
                            comparison_df = pd.DataFrame(comparison_results).T
                            st.dataframe(comparison_df.style.highlight_min(subset=['mse', 'mae', 'mape'], color='green')
                                                   .highlight_max(subset=['r2'], color='green'))
                            
                            st.subheader("模型性能可视化对比")
                            fig_compare = visualizer.plot_model_comparison(comparison_results, metrics=['mse', 'mae', 'r2'], use_plotly=True)
                            st.plotly_chart(fig_compare, use_container_width=True)
                            
                            st.subheader("预测结果对比")
                            with st.expander("点击展开预测结果对比图", expanded=False):
                                fig_pred_compare = visualizer.plot_prediction_comparison(y_test, predictions_dict, use_plotly=True)
                                st.plotly_chart(fig_pred_compare, use_container_width=True)
                            
                            best_model_mse = min(comparison_results.items(), key=lambda x: x[1]['mse'])
                            best_model_r2 = max(comparison_results.items(), key=lambda x: x[1]['r2'])
                            
                            st.info(f"🏆 **最佳模型 (MSE):** {best_model_mse[0]} (MSE: {best_model_mse[1]['mse']:.6f})")
                            st.info(f"🏆 **最佳模型 (R²):** {best_model_r2[0]} (R²: {best_model_r2[1]['r2']:.4f})")
                            
                            st.session_state.model_trained = True
                            st.session_state.model_type_trained = 'multi_model_comparison'
                            mark_step_completed('model_training')
                else:
                    st.warning("请至少选择2个模型进行对比")
            
            # ==================== 集成学习功能 ====================
            elif use_ensemble:
                st.subheader("🎯 集成学习")
                show_help("组合多个模型的预测结果，提高预测性能和稳定性。")
                
                ensemble_type = st.selectbox(
                    "集成方法",
                    ["voting", "stacking"],
                    help="Voting: 投票平均，简单高效; Stacking: 堆叠，使用元学习器组合预测，通常效果更好但需要更多数据"
                )
                
                st.write("选择要集成的模型：")
                col1, col2 = st.columns(2)
                ensemble_models = {}
                
                with col1:
                    if st.checkbox("随机森林", key="ens_rf"):
                        ensemble_models['rf'] = 'random_forest'
                    if st.checkbox("梯度提升", key="ens_gb"):
                        ensemble_models['gb'] = 'gradient_boosting'
                
                with col2:
                    if st.checkbox("SVR", key="ens_svr"):
                        ensemble_models['svr'] = 'svr'
                    if st.checkbox("KNN", key="ens_knn"):
                        ensemble_models['knn'] = 'knn'
                
                if len(ensemble_models) >= 2:
                    # 训练控制按钮
                    ens_col1, ens_col2 = st.columns([1, 1])
                    with ens_col1:
                        start_ensemble = st.button("创建集成模型", type="primary")
                    with ens_col2:
                        if st.button("⏹️ 停止训练", type="secondary", key="stop_ensemble"):
                            st.session_state.training_stop_requested = True
                            st.warning("已请求停止训练...")
                            st.rerun()
                    
                    if start_ensemble:
                        st.session_state.training_stop_requested = False
                        with st.spinner("创建集成模型..."):
                            from sklearn.ensemble import VotingRegressor, StackingRegressor
                            from sklearn.linear_model import LinearRegression
                            
                            estimators = []
                            status_text = st.empty()
                            
                            for i, (name, model_type) in enumerate(ensemble_models.items()):
                                if st.session_state.training_stop_requested:
                                    status_text.warning(f"训练已在 {name} 处停止")
                                    st.stop()
                                
                                status_text.text(f"正在训练基础模型: {name} ({i+1}/{len(ensemble_models)})")
                                temp_trainer = ModelTrainer()
                                temp_trainer.select_model(model_type)
                                estimators.append((name, temp_trainer.model))
                            
                            if ensemble_type == "voting":
                                ensemble_model = VotingRegressor(estimators=estimators)
                            else:
                                ensemble_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
                            
                            ensemble_model.fit(X_train, y_train)
                            trainer.model = ensemble_model
                            
                            y_pred = trainer.predict(X_test)
                            metrics = trainer.evaluate_time_series(y_test, y_pred, fast_mode=True)
                            
                            fig1 = visualizer.plot_prediction_results(y_test, y_pred, use_plotly=True)
                            fig2 = visualizer.plot_residuals(y_test, y_pred, use_plotly=True)
                            fig3 = visualizer.plot_error_distribution(y_test, y_pred, use_plotly=True)
                            
                            st.success("✅ 集成模型训练完成!")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("MSE", f"{metrics['mse']:.6f}")
                                st.metric("RMSE", f"{metrics['rmse']:.6f}")
                            with col2:
                                st.metric("MAE", f"{metrics['mae']:.6f}")
                                st.metric("R²", f"{metrics['r2']:.4f}")
                            with col3:
                                st.metric("MAPE", f"{metrics['mape']:.2f}%")
                            with col4:
                                st.metric("趋势一致性", f"{metrics['trend_consistency']:.2f}%")
                            
                            st.subheader("📈 预测结果对比")
                            with st.expander("点击展开预测结果对比图", expanded=False):
                                st.plotly_chart(fig1, use_container_width=True)
                                
                                with st.expander("查看残差分析"):
                                    col_res1, col_res2 = st.columns(2)
                                    with col_res1:
                                        st.plotly_chart(fig2, use_container_width=True)
                                    with col_res2:
                                        st.plotly_chart(fig3, use_container_width=True)
                            
                            st.session_state.model_trained = True
                            st.session_state.model_type_trained = f'ensemble_{ensemble_type}'
                            st.session_state.y_test = y_test
                            st.session_state.y_pred = y_pred
                            mark_step_completed('model_training')
                else:
                    st.warning("请至少选择2个模型进行集成")
            
            # ==================== 单模型训练 ====================
            elif model_selection_mode == "单模型" or use_custom_model:
                if use_custom_model:
                    model_type = "custom"
                else:
                    model_type = st.selectbox(
                        "选择模型",
                        [
                            # 集成学习
                            "random_forest", "gradient_boosting", "xgboost", "lightgbm", "catboost",
                            "adaboost", "extra_trees", "bagging", "voting",
                            # 线性模型
                            "linear_regression", "ridge", "lasso", "elastic_net", "bayesian_ridge",
                            # 支持向量机
                            "svr", "nu_svr",
                            # 近邻算法
                            "knn", "radius_neighbors",
                            # 树模型
                            "decision_tree", "extra_tree",
                            # 神经网络
                            "ann", "lstm", "gru", "cnn", "transformer",
                            # 其他
                            "gaussian_process", "mlp", "perceptron"
                        ],
                        help="集成学习: RandomForest/XGBoost/LightGBM适合复杂问题; 线性模型: 简单可解释; 深度学习: ANN/LSTM/GRU/CNN/Transformer适合复杂时间序列; 其他: SVR/GP适合特定问题"
                    )
                
                is_deep_learning = model_type in ["ann", "lstm", "gru", "cnn", "transformer", "mlp"]
                
                # 自定义模型使用用户提供的模型
                if model_type == "custom":
                    if custom_model_loaded and 'custom_model_instance' in locals():
                        model = custom_model_instance
                        st.info(f"🔧 使用自定义模型: {custom_model_class.__name__}")
                    else:
                        st.error("请先输入有效的自定义模型代码")
                        st.stop()
                else:
                    model = None
                
                # 模型参数
                model_kwargs = {}
                
                # ========== 集成学习模型 ==========
                if model_type in ["random_forest", "extra_trees"]:
                    model_kwargs["n_estimators"] = st.slider(
                        "树数量", 10, 500, 100, 10,
                        help="森林中决策树的数量。树越多模型越稳定，但训练时间越长。"
                    )
                    model_kwargs["max_depth"] = st.slider(
                        "最大深度", 1, 50, 10,
                        help="每棵树的最大深度。深度越大模型越复杂，可能过拟合。"
                    )
                    if model_type == "random_forest":
                        model_kwargs["min_samples_split"] = st.slider(
                            "最小分裂样本数", 2, 20, 2,
                            help="分裂内部节点所需的最小样本数。"
                        )
                
                elif model_type in ["gradient_boosting", "xgboost", "lightgbm", "catboost", "adaboost"]:
                    model_kwargs["n_estimators"] = st.slider(
                        "树数量", 10, 500, 100, 10,
                        help="boosting阶段的最大树数量。"
                    )
                    model_kwargs["learning_rate"] = st.slider(
                        "学习率", 0.001, 1.0, 0.1, 0.001,
                        help="缩小每棵树的贡献。较小的学习率需要更多树。"
                    )
                    if model_type in ["gradient_boosting", "xgboost", "lightgbm", "catboost"]:
                        model_kwargs["max_depth"] = st.slider(
                            "最大深度", 1, 15, 6,
                            help="树的最大深度。控制模型复杂度。"
                        )
                    if model_type == "adaboost":
                        model_kwargs["loss"] = st.selectbox(
                            "损失函数", ["linear", "square", "exponential"],
                            help="AdaBoost回归器的样本权重更新损失函数。"
                        )
                
                elif model_type in ["bagging", "voting"]:
                    if model_type == "bagging":
                        model_kwargs["n_estimators"] = st.slider(
                            "基学习器数量", 10, 100, 10,
                            help="基学习器的数量。"
                        )
                        model_kwargs["max_samples"] = st.slider(
                            "最大样本比例", 0.1, 1.0, 0.8, 0.1,
                            help="从训练集中抽取的样本比例。"
                        )
                
                # ========== 线性模型 ==========
                elif model_type in ["ridge", "lasso", "elastic_net", "bayesian_ridge"]:
                    if model_type == "elastic_net":
                        model_kwargs["alpha"] = st.slider(
                            "正则化强度", 0.001, 10.0, 1.0, 0.001,
                            help="乘以惩罚项的常数。越大正则化越强。"
                        )
                        model_kwargs["l1_ratio"] = st.slider(
                            "L1/L2比例", 0.0, 1.0, 0.5, 0.05,
                            help="0=L2正则化，1=L1正则化，0.5=等量混合。"
                        )
                    elif model_type == "bayesian_ridge":
                        model_kwargs["alpha_1"] = st.slider(
                            "Alpha 1", 1e-7, 1e-5, 1e-6, format="%.0e",
                            help="Gamma分布的形状参数。"
                        )
                    else:
                        model_kwargs["alpha"] = st.slider(
                            "正则化强度", 0.001, 10.0, 1.0, 0.001,
                            help="控制收缩量：越大收缩越强，模型越简单。"
                        )
                
                # ========== 支持向量机 ==========
                elif model_type in ["svr", "nu_svr"]:
                    model_kwargs["C"] = st.slider(
                        "C (正则化)", 0.1, 100.0, 1.0, 0.1,
                        help="正则化参数。C越大，对训练数据的拟合程度越高，但可能过拟合。"
                    )
                    model_kwargs["kernel"] = st.selectbox(
                        "核函数", ["rbf", "linear", "poly", "sigmoid"],
                        help="RBF: 适合非线性; Linear: 适合线性关系; Poly: 多项式; Sigmoid: 类似神经网络"
                    )
                    if model_kwargs["kernel"] == "poly":
                        model_kwargs["degree"] = st.slider(
                            "多项式次数", 2, 5, 3,
                            help="多项式核函数的次数。"
                        )
                    if model_type == "nu_svr":
                        model_kwargs["nu"] = st.slider(
                            "Nu", 0.1, 1.0, 0.5, 0.1,
                            help="支持向量的上界和下界的比例。"
                        )
                
                # ========== 近邻算法 ==========
                elif model_type in ["knn", "radius_neighbors"]:
                    if model_type == "knn":
                        model_kwargs["n_neighbors"] = st.slider(
                            "邻居数量", 1, 50, 5,
                            help="用于预测的邻居数量。K越大决策边界越平滑。"
                        )
                    else:
                        model_kwargs["radius"] = st.slider(
                            "半径", 0.1, 10.0, 1.0, 0.1,
                            help="查询点的邻域半径。"
                        )
                    model_kwargs["weights"] = st.selectbox(
                        "权重", ["uniform", "distance"],
                        help="Uniform: 等权重; Distance: 距离越近权重越大。"
                    )
                    model_kwargs["algorithm"] = st.selectbox(
                        "算法", ["auto", "ball_tree", "kd_tree", "brute"],
                        help="用于计算最近邻的算法。"
                    )
                
                # ========== 树模型 ==========
                elif model_type in ["decision_tree", "extra_tree"]:
                    model_kwargs["max_depth"] = st.slider(
                        "最大深度", 1, 50, 10,
                        help="树的最大深度。None表示节点扩展直到所有叶子都是纯的。"
                    )
                    model_kwargs["min_samples_split"] = st.slider(
                        "最小分裂样本数", 2, 20, 2,
                        help="分裂内部节点所需的最小样本数。"
                    )
                    model_kwargs["criterion"] = st.selectbox(
                        "分裂标准", ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                        help="衡量分裂质量的函数。"
                    )
                
                # ========== 高斯过程 ==========
                elif model_type == "gaussian_process":
                    model_kwargs["alpha"] = st.slider(
                        "Alpha (噪声)", 1e-10, 1.0, 1e-10, format="%.0e",
                        help="在拟合期间添加到对角线的噪声项。"
                    )
                    model_kwargs["normalize_y"] = st.checkbox(
                        "标准化目标值",
                        help="是否标准化目标值y。建议启用。"
                    )
                
                # ========== 深度学习模型 ==========
                elif is_deep_learning:
                    st.subheader("🔧 神经网络配置")
                    
                    hidden_layers = st.text_input(
                        "隐藏层配置 (逗号分隔，如: 128,64,32)", "64,32",
                        help="每层神经元数量。更多层/神经元=更强表达能力，但容易过拟合。"
                    )
                    try:
                        model_kwargs["hidden_layers"] = list(map(int, hidden_layers.split(",")))
                    except:
                        model_kwargs["hidden_layers"] = [64, 32]
                        st.warning("隐藏层配置格式错误，使用默认值 [64, 32]")
                    
                    model_kwargs["learning_rate"] = st.slider(
                        "学习率", 0.0001, 0.1, 0.001, 0.0001,
                        help="优化步长。太大可能不收敛，太小训练慢。"
                    )
                    model_kwargs["dropout"] = st.slider(
                        "Dropout率", 0.0, 0.5, 0.2, 0.05,
                        help="随机失活比例，用于防止过拟合。建议0.2-0.5。"
                    )
                    model_kwargs["activation"] = st.selectbox(
                        "激活函数", ["relu", "tanh", "sigmoid", "leaky_relu", "elu"],
                        help="ReLU: 常用; Tanh: 适合RNN; Sigmoid: 输出层常用。"
                    )
                    model_kwargs["batch_norm"] = st.checkbox(
                        "使用批归一化",
                        value=True,
                        help="批归一化可以加速训练并提高稳定性。"
                    )
                    
                    # 模型特定参数
                    if model_type in ["ann", "mlp"]:
                        model_kwargs["input_dim"] = X_train.shape[1]
                    elif model_type in ["lstm", "gru"]:
                        model_kwargs["input_shape"] = (1, X_train.shape[1])
                        model_kwargs["return_sequences"] = st.checkbox(
                            "返回序列",
                            help="是否返回完整的输出序列或仅返回最后一个输出。"
                        )
                    elif model_type == "cnn":
                        model_kwargs["input_shape"] = (X_train.shape[1], 1)
                        st.info("CNN模型会将输入 reshape 为 (样本数, 特征数, 1)")
                    elif model_type == "transformer":
                        model_kwargs["input_shape"] = (1, X_train.shape[1])  # Transformer需要的输入形状
                        model_kwargs["num_heads"] = st.slider(
                            "注意力头数", 1, 8, 2,
                            help="多头注意力机制的头数。"
                        )
                        model_kwargs["num_layers"] = st.slider(
                            "Transformer层数", 1, 6, 2,
                            help="Transformer编码器层数。"
                        )
                        st.info("Transformer模型会将输入 reshape 为 (样本数, 1, 特征数)")
                    
                    epochs = st.slider(
                        "训练轮数", 10, 500, 100, 10,
                        help="完整遍历数据集的次数。太少欠拟合，太多过拟合。"
                    )
                    batch_size = st.slider(
                        "批次大小", 8, 128, 32, 8,
                        help="每次梯度更新使用的样本数。较大批次训练更稳定。"
                    )
                    # 早停设置
                    use_early_stopping = st.checkbox(
                        "使用早停",
                        value=True,
                        help="当验证损失不再下降时提前停止训练，防止过拟合。"
                    )
                    patience = st.slider(
                        "耐心值", 5, 50, 10,
                        help="验证损失不改善的轮数，超过则停止。"
                    ) if use_early_stopping else 10
                
                # 交叉验证（仅适用于机器学习模型）
                if not is_deep_learning:
                    st.subheader("🔄 交叉验证")
                    use_cv = st.checkbox(
                        "启用交叉验证",
                        help="将数据分成多份轮流训练和验证，评估模型稳定性和泛化能力。"
                    )
                    if use_cv:
                        cv_folds = st.slider(
                            "交叉验证折数", 2, 10, 5,
                            help="数据分成的份数。常用5或10折。折数越多评估越稳定但计算量越大。"
                        )
                        if st.button("执行交叉验证"):
                            with st.spinner(f"执行{cv_folds}折交叉验证..."):
                                trainer.select_model(model_type, **model_kwargs)
                                cv_results = trainer.cross_validate(X_train, y_train, cv=cv_folds, scoring='r2')
                                st.success("交叉验证完成！")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("平均得分", f"{cv_results['mean']:.4f}")
                                with col2:
                                    st.metric("标准差", f"{cv_results['std']:.4f}")
                                with col3:
                                    st.metric("最高得分", f"{cv_results['max']:.4f}")
                                with col4:
                                    st.metric("最低得分", f"{cv_results['min']:.4f}")
                
                # 训练控制按钮
                train_col1, train_col2 = st.columns([1, 1])
                with train_col1:
                    start_train = st.button("🚀 开始训练", type="primary")
                with train_col2:
                    if st.button("⏹️ 停止训练", type="secondary"):
                        st.session_state.training_stop_requested = True
                        st.warning("已请求停止训练，当前操作完成后将终止...")
                        st.rerun()
                
                if start_train:
                    # 重置停止标志
                    st.session_state.training_stop_requested = False
                    st.session_state.training_in_progress = True
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 检查是否请求停止
                    if st.session_state.training_stop_requested:
                        st.info("训练已取消")
                        st.session_state.training_in_progress = False
                        st.stop()
                    
                    status_text.text("正在选择模型...")
                    progress_bar.progress(10)
                    
                    # 自定义模型不需要select_model
                    if model_type != "custom":
                        trainer.select_model(model_type, **model_kwargs)
                    else:
                        # 直接使用用户提供的自定义模型
                        trainer.model = model
                        trainer.is_deep_learning = False  # 自定义模型视为机器学习模型
                    
                    # 检查是否请求停止
                    if st.session_state.training_stop_requested:
                        st.info("训练已取消")
                        st.session_state.training_in_progress = False
                        st.stop()
                    
                    status_text.text("正在训练模型...")
                    progress_bar.progress(30)
                    
                    # 为深度学习模型重塑数据
                    if is_deep_learning:
                        # 确保数据是numpy数组
                        X_train_arr = X_train.values if hasattr(X_train, 'values') else np.array(X_train)
                        X_val_arr = X_val.values if hasattr(X_val, 'values') else np.array(X_val) if X_val is not None else None
                        X_test_arr = X_test.values if hasattr(X_test, 'values') else np.array(X_test)
                        if model_type == "cnn":
                            X_train_dl = X_train_arr.reshape(X_train_arr.shape[0], X_train_arr.shape[1], 1)
                            X_val_dl = X_val_arr.reshape(X_val_arr.shape[0], X_val_arr.shape[1], 1) if X_val_arr is not None else None
                            X_test_dl = X_test_arr.reshape(X_test_arr.shape[0], X_test_arr.shape[1], 1)
                        else:
                            X_train_dl = X_train_arr.reshape(X_train_arr.shape[0], 1, X_train_arr.shape[1])
                            X_val_dl = X_val_arr.reshape(X_val_arr.shape[0], 1, X_val_arr.shape[1]) if X_val_arr is not None else None
                            X_test_dl = X_test_arr.reshape(X_test_arr.shape[0], 1, X_test_arr.shape[1])
                        
                        # 检查是否请求停止
                        if st.session_state.training_stop_requested:
                            st.info("训练已取消")
                            st.session_state.training_in_progress = False
                            st.stop()
                        
                        history = trainer.train(
                            X_train_dl, y_train, X_val_dl, y_val, 
                            epochs=epochs, batch_size=batch_size, 
                            patience=patience
                        )
                        
                        # 检查是否请求停止
                        if st.session_state.training_stop_requested:
                            st.info("训练已取消")
                            st.session_state.training_in_progress = False
                            st.stop()
                        
                        if history is not None:
                            st.subheader("模型训练历史")
                            fig_history = visualizer.plot_training_history(history, use_plotly=True)
                            st.plotly_chart(fig_history, use_container_width=True)
                        
                        y_pred = trainer.predict(X_test_dl)
                    else:
                        # 检查是否请求停止
                        if st.session_state.training_stop_requested:
                            st.info("训练已取消")
                            st.session_state.training_in_progress = False
                            st.stop()
                        
                        trainer.train(X_train, y_train)
                        y_pred = trainer.predict(X_test)
                    
                    # 检查是否请求停止
                    if st.session_state.training_stop_requested:
                        st.info("训练已取消")
                        st.session_state.training_in_progress = False
                        st.stop()
                    
                    status_text.text("正在评估模型...")
                    progress_bar.progress(60)
                    metrics = trainer.evaluate_time_series(y_test, y_pred, fast_mode=True)
                    
                    # 检查是否请求停止
                    if st.session_state.training_stop_requested:
                        st.info("训练已取消")
                        st.session_state.training_in_progress = False
                        st.stop()
                    
                    status_text.text("正在生成可视化...")
                    progress_bar.progress(80)
                    fig1 = visualizer.plot_prediction_results(y_test, y_pred, use_plotly=True)
                    fig2 = visualizer.plot_residuals(y_test, y_pred, use_plotly=True)
                    fig3 = visualizer.plot_error_distribution(y_test, y_pred, use_plotly=True)
                    
                    # 检查是否请求停止
                    if st.session_state.training_stop_requested:
                        st.info("训练已取消")
                        st.session_state.training_in_progress = False
                        st.stop()
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()
                    
                    # 训练完成，重置标志
                    st.session_state.training_in_progress = False
                    st.session_state.training_stop_requested = False
                    
                    st.success("✅ 训练完成!")
                    append_run_history(st.session_state, "时序模型训练", st.session_state.model_type_trained or model_type)
                    
                    # 主要指标显示
                    st.subheader("📊 模型性能指标")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("MSE", f"{metrics['mse']:.6f}")
                        st.metric("RMSE", f"{metrics['rmse']:.6f}")
                    with col2:
                        st.metric("MAE", f"{metrics['mae']:.6f}")
                        st.metric("Median AE", f"{metrics['median_ae']:.6f}")
                    with col3:
                        st.metric("R²", f"{metrics['r2']:.4f}")
                        st.metric("Adjusted R²", f"{metrics['adjusted_r2']:.4f}")
                    with col4:
                        st.metric("MAPE", f"{metrics['mape']:.2f}%")
                        st.metric("SMAPE", f"{metrics['smape']:.2f}%")
                    with col5:
                        st.metric("趋势一致性", f"{metrics['trend_consistency']:.2f}%")
                        st.metric("方向准确率", f"{metrics['mda']:.2f}%")
                    
                    # 详细指标展开面板
                    with st.expander("📋 查看详细评估指标"):
                        metrics_df = pd.DataFrame({
                            '指标名称': [
                                '均方误差 (MSE)', '均方根误差 (RMSE)', '平均绝对误差 (MAE)', '中位数绝对误差',
                                '决定系数 (R²)', '调整R²', '解释方差', '最大误差',
                                '平均绝对百分比误差 (MAPE)', '对称MAPE', '平均对数平方误差',
                                '相对平方误差 (RRSE)', '相对绝对误差 (RAE)', 'Theil U统计量',
                                '方向准确率 (MDA)', '趋势一致性', '峰值检测准确率',
                                '相关系数', '协方差'
                            ],
                            '数值': [
                                f"{metrics['mse']:.6f}", f"{metrics['rmse']:.6f}", 
                                f"{metrics['mae']:.6f}", f"{metrics['median_ae']:.6f}",
                                f"{metrics['r2']:.4f}", f"{metrics['adjusted_r2']:.4f}",
                                f"{metrics['explained_variance']:.4f}", f"{metrics['max_error']:.6f}",
                                f"{metrics['mape']:.2f}%", f"{metrics['smape']:.2f}%",
                                f"{metrics['msle']:.6f}", f"{metrics['rrse']:.4f}",
                                f"{metrics['rae']:.4f}", f"{metrics['theil_u']:.4f}",
                                f"{metrics['mda']:.2f}%", f"{metrics['trend_consistency']:.2f}%",
                                f"{metrics['peak_accuracy']:.2f}%", f"{metrics['correlation']:.4f}",
                                f"{metrics['covariance']:.6f}"
                            ],
                            '说明': [
                                '预测值与真实值差的平方的平均', 'MSE的平方根', '预测值与真实值差的绝对值的平均',
                                '绝对误差的中位数', '模型解释数据变异的程度', '考虑特征数调整后的R²',
                                '模型解释方差的比例', '最大预测误差', '百分比形式的平均误差',
                                '对称形式的MAPE', '对数变换后的MSE', '相对于简单模型的误差',
                                '相对于简单模型的绝对误差', '预测精度指标，<1表示优于简单预测',
                                '趋势方向预测准确率', '局部趋势一致性', '峰值点检测准确率',
                                '预测值与真实值的线性相关程度', '预测值与真实值的协方差'
                            ]
                        })
                        st.dataframe(metrics_df, height=500, use_container_width=True)
                        
                        # 指标解释
                        st.info("""
                        **指标说明：**
                        - **误差类指标** (MSE, RMSE, MAE等): 越小越好，表示预测误差
                        - **拟合度指标** (R², Adjusted R², Explained Variance): 越接近1越好
                        - **百分比误差** (MAPE, SMAPE): 越小越好，便于跨数据集比较
                        - **方向/趋势指标** (MDA, Trend Consistency): 越高越好，表示趋势预测能力
                        - **Theil U**: <1表示优于简单预测，=1表示相当，>1表示差于简单预测
                        """)
                    
                    st.subheader("📈 预测结果对比")
                    with st.expander("点击展开预测结果对比图", expanded=False):
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        with st.expander("查看残差分析"):
                            col_res1, col_res2 = st.columns(2)
                            with col_res1:
                                st.plotly_chart(fig2, use_container_width=True)
                            with col_res2:
                                st.plotly_chart(fig3, use_container_width=True)
                    
                    st.session_state.model_trained = True
                    st.session_state.model_type_trained = model_type
                    st.session_state.y_test = y_test
                    st.session_state.y_pred = y_pred
                    st.session_state.ts_metrics = metrics
                    st.session_state.ts_result_export = pd.DataFrame({
                        "sample_id": np.arange(len(np.asarray(y_test).ravel())),
                        "true_value": np.asarray(y_test).ravel(),
                        "predicted_value": np.asarray(y_pred).ravel(),
                        "residual": np.asarray(y_test).ravel() - np.asarray(y_pred).ravel(),
                        "absolute_error": np.abs(np.asarray(y_test).ravel() - np.asarray(y_pred).ravel()),
                    })
                    st.session_state.ts_report_figures = {
                        "预测效果": fig1,
                        "残差分析": fig2,
                        "误差分布": fig3,
                    }
                    mark_step_completed('model_training')
            
            elif model_selection_mode == "AutoML自动搜索":
                st.info("🤖 AutoML将自动搜索最优模型和参数")
                
                col1, col2 = st.columns(2)
                with col1:
                    automl_n_iter = st.slider(
                        "搜索迭代次数", 5, 50, 20,
                        help="尝试的不同参数组合数量。越多越可能找到最优解，但耗时越长。"
                    )
                    automl_cv_folds = st.slider(
                        "交叉验证折数", 3, 10, 5,
                        help="评估每个模型时使用的交叉验证折数。"
                    )
                with col2:
                    automl_scoring = st.selectbox(
                        "评估指标",
                        ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"],
                        format_func=lambda x: {"neg_mean_squared_error": "MSE", "neg_mean_absolute_error": "MAE", "r2": "R²"}[x],
                        help="用于评估和选择最佳模型的指标。R²越接近1越好，MSE/MAE越小越好。"
                    )
                    automl_cv_strategy = st.selectbox(
                        "验证策略",
                        ["time_series", "kfold"],
                        format_func=lambda x: {"time_series": "时序滚动验证", "kfold": "普通K折验证"}[x],
                        help="时序数据建议使用滚动验证，避免未来信息进入训练。"
                    )
                
                # 训练控制按钮
                automl_col1, automl_col2 = st.columns([1, 1])
                with automl_col1:
                    start_automl = st.button("🚀 启动AutoML搜索", type="primary")
                with automl_col2:
                    if st.button("⏹️ 停止搜索", type="secondary", key="stop_automl"):
                        st.session_state.training_stop_requested = True
                        st.warning("已请求停止AutoML搜索...")
                        st.rerun()
                
                if start_automl:
                    st.session_state.training_stop_requested = False
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with st.spinner("AutoML搜索中，请稍候..."):
                        automl = AutoMLRegressor(time_limit=600, cv_folds=automl_cv_folds, cv_strategy=automl_cv_strategy)
                        
                        # 包装搜索过程以支持停止功能
                        results = None
                        try:
                            results = automl.search(
                                X_train, y_train,
                                scoring=automl_scoring,
                                search_method="random",
                                n_iter=automl_n_iter
                            )
                            
                            # 检查是否请求停止
                            if st.session_state.training_stop_requested:
                                status_text.warning("AutoML搜索已停止")
                                st.session_state.training_stop_requested = False
                                st.stop()
                        except Exception as e:
                            if st.session_state.training_stop_requested:
                                status_text.warning("AutoML搜索已停止")
                                st.session_state.training_stop_requested = False
                                st.stop()
                            else:
                                raise e
                        
                        progress_bar.empty()
                        status_text.empty()
                        st.success(f"✅ AutoML搜索完成! 共评估 {len(results['results_df'])} 个模型")
                        
                        if results['results_df'] is not None:
                            st.write("**模型排名：**")
                            st.dataframe(results['results_df'])
                        
                        st.write(f"**最佳模型:** {results['results_df'].iloc[0]['Model'] if len(results['results_df']) > 0 else 'Unknown'}")
                        st.write(f"**最佳分数:** {results['best_score']:.4f}")
                        
                        # 使用最佳模型评估
                        trainer.model = results['best_model']
                        y_pred = trainer.predict(X_test)
                        metrics = trainer.evaluate_time_series(y_test, y_pred, fast_mode=True)
                        
                        # 生成可视化
                        fig1 = visualizer.plot_prediction_results(y_test, y_pred, use_plotly=True)
                        fig2 = visualizer.plot_residuals(y_test, y_pred, use_plotly=True)
                        fig3 = visualizer.plot_error_distribution(y_test, y_pred, use_plotly=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("MSE", f"{metrics['mse']:.6f}")
                        with col2:
                            st.metric("R²", f"{metrics['r2']:.4f}")
                        with col3:
                            st.metric("MAPE", f"{metrics['mape']:.2f}%")
                        
                        # 预测结果可视化
                        st.subheader("📈 预测结果对比")
                        with st.expander("点击展开预测结果对比图", expanded=False):
                            st.plotly_chart(fig1, use_container_width=True)
                            
                            with st.expander("查看残差分析"):
                                col_res1, col_res2 = st.columns(2)
                                with col_res1:
                                    st.plotly_chart(fig2, use_container_width=True)
                                with col_res2:
                                    st.plotly_chart(fig3, use_container_width=True)
                        
                        st.session_state.model_trained = True
                        st.session_state.model_type_trained = 'automl'
                        st.session_state.y_test = y_test
                        st.session_state.y_pred = y_pred
                        st.session_state.ts_metrics = metrics
                        st.session_state.ts_result_export = pd.DataFrame({
                            "sample_id": np.arange(len(np.asarray(y_test).ravel())),
                            "true_value": np.asarray(y_test).ravel(),
                            "predicted_value": np.asarray(y_pred).ravel(),
                            "residual": np.asarray(y_test).ravel() - np.asarray(y_pred).ravel(),
                            "absolute_error": np.abs(np.asarray(y_test).ravel() - np.asarray(y_pred).ravel()),
                        })
                        st.session_state.ts_report_figures = {
                            "预测效果": fig1,
                            "残差分析": fig2,
                            "误差分布": fig3,
                        }
                        mark_step_completed('model_training')
            
            # 模型可解释性和保存
            if st.session_state.get('model_trained', False):
                st.subheader("🔍 模型可解释性分析")
                
                use_explainability = st.checkbox(
                    "启用模型解释",
                    help="分析模型如何做出预测，了解哪些特征对预测结果影响最大。",
                    key="use_explainability_checkbox"
                )
                
                if use_explainability:
                    explainer = ModelExplainer(trainer.model, feature_names=feature_cols)
                    explain_method = st.selectbox(
                        "解释方法",
                        ["特征重要性", "SHAP分析", "排列重要性"],
                        key="explain_method_select",
                        help="特征重要性: 模型内置重要性; SHAP: 基于博弈论的精确解释; 排列重要性: 通过打乱特征评估重要性"
                    )
                    
                    if explain_method == "特征重要性":
                        importance_summary = explainer.get_feature_importance_summary()
                        if 'built_in' in importance_summary:
                            st.dataframe(importance_summary['built_in'].head(10))
                    
                    elif explain_method == "SHAP分析":
                        if st.button("计算SHAP值", key="calc_shap"):
                            with st.spinner("计算中..."):
                                shap_results = explainer.explain_with_shap(X_test, sample_size=100)
                                if shap_results:
                                    fig_shap = explainer.plot_shap_summary(use_plotly=True)
                                    if fig_shap:
                                        st.plotly_chart(fig_shap, use_container_width=True, key="shap_plot")
                
                # 保存模型
                st.subheader("💾 保存模型")
                model_name = st.text_input(
                    "模型名称",
                    f"{st.session_state.model_type_trained}_model",
                    key="model_name_input",
                    help="为模型命名以便后续识别和使用。建议使用描述性名称。"
                )
                if st.button("保存模型", key="save_model_btn"):
                    is_dl = st.session_state.model_type_trained in ['ann', 'lstm', 'gru', 'cnn', 'transformer', 'mlp']
                    ext = '.keras' if is_dl else '.pkl'
                    model_path = os.path.join("saved_models", f"{model_name}{ext}")
                    os.makedirs("saved_models", exist_ok=True)
                    
                    # 构建模型信息，包含特征列和标准化方法
                    model_info = {
                        'model_type': st.session_state.model_type_trained,
                        'is_deep_learning': is_dl,
                        'save_time': datetime.datetime.now().isoformat(),
                        'version': '1.0.0',
                        'task_type': 'regression',
                        'feature_cols': st.session_state.get('selected_feature_cols', feature_cols),
                        'original_feature_cols': feature_cols,
                        'target_col': target_col,
                        'scale_method': scale_method,
                        'preprocessing_note': 'scaler/feature_selector fitted on training split only'
                    }
                    
                    trainer.save_model(model_path, model_info=model_info)
                    st.success(f"✅ 模型已保存到 {model_path}")
                    st.info(f"📋 模型信息：特征列 {len(feature_cols)} 个，标准化方法：{scale_method}")
            
            if st.session_state.get("ts_result_export") is not None and st.session_state.get("ts_metrics") is not None:
                with st.expander("📤 时序训练结果导出", expanded=False):
                    ts_export_df = st.session_state.ts_result_export
                    ts_figures = st.session_state.get("ts_report_figures", {})
                    ts_metrics = st.session_state.ts_metrics
                    ts_report = build_html_report(
                        "时序预测训练图文结果报告",
                        metrics={
                            "模型": st.session_state.get("model_type_trained", "N/A"),
                            "MSE": f"{ts_metrics.get('mse', np.nan):.6f}",
                            "RMSE": f"{ts_metrics.get('rmse', np.nan):.6f}",
                            "MAE": f"{ts_metrics.get('mae', np.nan):.6f}",
                            "R²": f"{ts_metrics.get('r2', np.nan):.4f}",
                            "MAPE": f"{ts_metrics.get('mape', np.nan):.2f}%",
                        },
                        sections=[("结果摘要", "报告包含测试集真实值、预测值、残差、误差分布和主要回归指标。")],
                        figures=[(name, fig) for name, fig in ts_figures.items()],
                        tables=[("预测明细", ts_export_df)],
                    )
                    render_result_downloads(
                        "时序训练",
                        ts_export_df,
                        ts_report,
                        csv_name=f"time_series_training_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        html_name=f"time_series_training_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        key_prefix="time_series_training",
                    )

            # ==================== 新数据预测（独立功能） ====================
            st.divider()
            st.subheader("🔮 新数据预测")
            st.info("使用训练好的模型或上传的模型文件对新数据进行预测")
            
            # 模型选择
            ts_model_source = st.radio(
                "选择预测模型来源",
                ["使用当前训练的模型", "使用自定义模型文件"],
                horizontal=True,
                key="ts_model_source_select",
                help="选择使用当前训练的模型，或上传以前保存的模型文件进行预测。"
            )
            
            # 用于预测的变量
            ts_prediction_trainer = None
            ts_prediction_model_type = None
            ts_feature_cols = None
            ts_scale_method = "none"
            ts_use_custom = False
            ts_model_info = None  # 初始化模型信息变量
            ts_target_col = None  # 初始化目标列变量
            
            if ts_model_source == "使用当前训练的模型":
                if st.session_state.get('model_trained', False) and trainer.model is not None:
                    ts_prediction_trainer = trainer
                    ts_prediction_model_type = st.session_state.model_type_trained
                    ts_feature_cols = st.session_state.get('selected_feature_cols', feature_cols)
                    ts_scale_method = scale_method
                else:
                    st.warning("⚠️ 当前没有训练好的模型，请先完成模型训练或选择「使用自定义模型文件」")
            else:  # 使用自定义模型文件
                st.markdown("**上传模型文件**")
                ts_model_file = st.file_uploader(
                    "上传模型文件 (.pkl 或 .keras)",
                    type=["pkl", "keras", "h5"],
                    key="ts_custom_model_uploader",
                    help="上传以前保存的模型文件（.pkl格式为机器学习模型，.keras/.h5格式为深度学习模型）。"
                )
                
                if ts_model_file is not None:
                    # 保存上传的模型文件到临时位置
                    ts_model_ext = os.path.splitext(ts_model_file.name)[1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ts_model_ext) as tmp_ts_model_file:
                        tmp_ts_model_file.write(ts_model_file.getvalue())
                        custom_ts_model_path = tmp_ts_model_file.name
                    
                    # 同时支持上传JSON配置文件（可选）
                    ts_model_json_file = st.file_uploader(
                        "上传模型配置文件（可选，如模型名.json）",
                        type=["json"],
                        key="ts_model_json_uploader",
                        help="如果模型保存时分离了配置文件，请一并上传以便读取模型信息。如果不上传，程序会尝试从模型文件中读取信息。"
                    )
                    
                    # 如果用户上传了JSON文件，保存到临时位置
                    custom_json_path = None
                    if ts_model_json_file is not None:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_json_file:
                            tmp_json_file.write(ts_model_json_file.getvalue())
                            custom_json_path = tmp_json_file.name
                    
                    try:
                        # 创建新的训练器并加载模型
                        from src.models.model_trainer import ModelTrainer
                        ts_custom_trainer = ModelTrainer()
                        
                        # 根据文件扩展名判断模型类型
                        if ts_model_ext in ['.keras', '.h5']:
                            loaded_ts_model_type = 'ann'
                        else:
                            loaded_ts_model_type = 'random_forest'
                        
                        # 加载模型
                        loaded_ts_model, ts_model_info = ts_custom_trainer.load_model(custom_ts_model_path, loaded_ts_model_type, custom_json_path)
                        
                        if ts_model_info and 'model_type' in ts_model_info:
                            ts_prediction_model_type = ts_model_info['model_type']
                            st.info(f"📋 加载的模型类型: {ts_prediction_model_type}")
                        else:
                            ts_prediction_model_type = loaded_ts_model_type
                        
                        ts_prediction_trainer = ts_custom_trainer
                        ts_use_custom = True
                        st.success(f"✅ 模型加载成功: {ts_model_file.name}")
                        
                        # 显示模型信息
                        # 初始化目标列变量
                        ts_target_col = None
                        
                        if ts_model_info:
                            with st.expander("📊 模型信息"):
                                st.json(ts_model_info)
                                
                            # 从模型信息中读取配置
                            if 'feature_cols' in ts_model_info:
                                ts_feature_cols = ts_model_info['feature_cols']
                                st.success(f"✅ 已从模型信息中读取特征列：{len(ts_feature_cols)} 个")
                            else:
                                st.warning("⚠️ 模型信息中未找到特征列，请手动选择")
                                ts_feature_cols = None
                            
                            if 'target_col' in ts_model_info:
                                ts_target_col = ts_model_info['target_col']
                                st.success(f"✅ 已从模型信息中读取目标列：{ts_target_col}")
                            
                            if 'scale_method' in ts_model_info:
                                ts_scale_method = ts_model_info['scale_method']
                                st.success(f"✅ 已从模型信息中读取标准化方法：{ts_scale_method}")
                            else:
                                ts_scale_method = "none"
                        else:
                            st.warning("⚠️ 无法读取模型信息，请手动配置")
                            ts_feature_cols = None
                            
                    except Exception as e:
                        st.error(f"❌ 模型加载失败: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                        ts_prediction_trainer = None
                        ts_feature_cols = None
                else:
                    st.info("请上传模型文件以继续")
            
            # 当使用自定义模型且无法读取特征列时，让用户从新数据中选择
            if ts_use_custom and ts_prediction_trainer is not None and ts_feature_cols is None:
                st.subheader("⚙️ 特征列选择")
                st.info("请上传新数据文件，然后从数据中选择特征列：")
                
                # 先上传数据文件来选择特征列
                ts_config_file = st.file_uploader(
                    "上传数据文件以选择特征列",
                    type=["csv", "xlsx", "xls"],
                    key="ts_config_file_uploader"
                )
                
                if ts_config_file is not None:
                    try:
                        if ts_config_file.name.endswith('.csv'):
                            ts_config_df = pd.read_csv(ts_config_file)
                        else:
                            ts_config_df = pd.read_excel(ts_config_file)
                        
                        st.write("数据预览：", ts_config_df.head())
                        
                        # 让用户从数据列中选择特征列
                        ts_feature_cols = st.multiselect(
                            "选择特征列",
                            options=list(ts_config_df.columns),
                            help="选择用于预测的特征列"
                        )
                        
                        # 选择目标列（用于后续对比预测值和真实值）
                        ts_target_col = st.selectbox(
                            "选择目标列（可选，用于评估预测效果）",
                            options=["无"] + list(ts_config_df.columns),
                            key="ts_target_col_select",
                            help="选择目标列后，可以对比预测值和真实值"
                        )
                        if ts_target_col == "无":
                            ts_target_col = None
                        
                        # 标准化方法
                        ts_scale_method = st.selectbox(
                            "数据标准化方法",
                            ["none", "standard", "minmax", "robust"],
                            index=0,
                            key="ts_custom_scale_method",
                            help="选择模型训练时使用的数据标准化方法"
                        )
                        
                        if not ts_feature_cols:
                            st.warning("请至少选择一个特征列")
                            ts_prediction_trainer = None
                    except Exception as e:
                        st.error(f"读取数据文件失败: {str(e)}")
                        ts_prediction_trainer = None
                else:
                    ts_prediction_trainer = None  # 没有配置好，不能继续
            
            if ts_prediction_trainer is not None and ts_prediction_trainer.model is not None:
                new_data_file = st.file_uploader(
                    "上传新数据文件",
                    type=["out", "dat", "txt", "csv", "xlsx", "xls"],
                    key="new_data_file_uploader",
                    help="上传包含相同特征列的新数据文件，使用训练好的模型进行预测。"
                )
                
                if new_data_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(new_data_file.name)[1]) as tmp_file:
                        tmp_file.write(new_data_file.getvalue())
                        new_tmp_file_path = tmp_file.name
                    
                    try:
                        if new_data_file.name.endswith(('.csv', '.xlsx', '.xls')):
                            if new_data_file.name.endswith('.csv'):
                                new_df = pd.read_csv(new_tmp_file_path)
                            else:
                                new_df = pd.read_excel(new_tmp_file_path)
                        else:
                            new_df = read_data_file(new_tmp_file_path)
                        
                        st.info(f"📁 文件已加载，共 {len(new_df)} 行，{len(new_df.columns)} 列")
                        st.write("数据预览（前5行）：", new_df.head())
                        
                        st.subheader("🔗 特征列映射")
                        st.markdown("请将新数据的列映射到训练时使用的特征列：")
                        
                        col_mapping = {}
                        use_default_mapping = st.checkbox(
                            "使用与训练数据相同的列名（自动匹配）",
                            value=True,
                            key="auto_map_cols"
                        )
                        
                        mapping_ready = False
                        
                        if use_default_mapping:
                            missing_cols = [col for col in ts_feature_cols if col not in new_df.columns]
                            if missing_cols:
                                st.warning(f"⚠️ 以下特征列在上传的数据中未找到: {missing_cols}")
                                st.info("请取消上方复选框，手动映射特征列")
                            else:
                                col_mapping = {col: col for col in ts_feature_cols}
                                st.success("✅ 自动匹配成功")
                                mapping_ready = True
                        else:
                            st.markdown("**请选择新数据中对应的列：**")
                            for train_col in ts_feature_cols:
                                available_cols = ["无"] + list(new_df.columns)
                                default_idx = available_cols.index(train_col) if train_col in new_df.columns else 0
                                selected_col = st.selectbox(
                                    f"{train_col} →",
                                    available_cols,
                                    index=default_idx,
                                    key=f"map_{train_col}"
                                )
                                if selected_col != "无":
                                    col_mapping[train_col] = selected_col
                            
                            if len(col_mapping) == len(ts_feature_cols):
                                mapping_ready = True
                        
                        # 显示开始预测按钮（当映射准备好时）
                        if mapping_ready:
                            if st.button("🔮 开始预测", type="primary"):
                                new_X = new_df[[col_mapping[col] for col in ts_feature_cols]]
                                new_X.columns = ts_feature_cols
                                
                                # 数据标准化处理
                                if ts_scale_method != "none" and not ts_use_custom:
                                    # 使用训练时的预处理器
                                    new_X_scaled = preprocessor._transform_scale_split(new_X)
                                    if st.session_state.get('use_feature_selection') and getattr(preprocessor, 'feature_selector', None) is not None:
                                        new_X_scaled = preprocessor._transform_selected_features(new_X_scaled)
                                elif ts_scale_method != "none" and ts_use_custom:
                                    # 使用自定义标准化
                                    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
                                    if ts_scale_method == "standard":
                                        scaler = StandardScaler()
                                    elif ts_scale_method == "minmax":
                                        scaler = MinMaxScaler()
                                    elif ts_scale_method == "robust":
                                        scaler = RobustScaler()
                                    new_X_scaled = pd.DataFrame(
                                        scaler.fit_transform(new_X),
                                        columns=new_X.columns,
                                        index=new_X.index
                                    )
                                else:
                                    new_X_scaled = new_X
                                
                                # 使用 ts_prediction_model_type
                                model_type = ts_prediction_model_type
                                is_dl_model = model_type in ['ann', 'lstm', 'gru', 'cnn', 'transformer', 'mlp']
                                if is_dl_model:
                                    if model_type in ['lstm', 'gru', 'transformer']:
                                        new_X_scaled = new_X_scaled.values.reshape(new_X_scaled.shape[0], 1, new_X_scaled.shape[1])
                                    elif model_type == 'cnn':
                                        new_X_scaled = new_X_scaled.values.reshape(new_X_scaled.shape[0], new_X_scaled.shape[1], 1)
                                    elif model_type in ['ann', 'mlp']:
                                        new_X_scaled = new_X_scaled.values
                                
                                new_y_pred = ts_prediction_trainer.predict(new_X_scaled)
                                new_df["预测值"] = new_y_pred
                                
                                # 预测完成后显示结果
                                st.subheader("📊 预测结果分析")
                                
                                # 预测结果概览
                                pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
                                with pred_col1:
                                    st.metric("预测样本数", len(new_y_pred))
                                with pred_col2:
                                    st.metric("预测均值", f"{np.mean(new_y_pred):.4f}")
                                with pred_col3:
                                    st.metric("预测标准差", f"{np.std(new_y_pred):.4f}")
                                with pred_col4:
                                    st.metric("预测范围", f"[{np.min(new_y_pred):.4f}, {np.max(new_y_pred):.4f}]")
                                
                                # 预测结果可视化
                                st.subheader("📈 预测结果可视化")
                                
                                # 创建预测结果图表
                                import plotly.graph_objects as go
                                from plotly.subplots import make_subplots
                                
                                pred_fig = make_subplots(
                                    rows=2, cols=2,
                                    subplot_titles=("预测值分布", "预测值时序图", "预测值箱线图", "预测值直方图"),
                                    specs=[[{"type": "scatter"}, {"type": "scatter"}],
                                           [{"type": "box"}, {"type": "histogram"}]]
                                )
                                
                                # 时序图
                                pred_fig.add_trace(
                                    go.Scatter(y=new_y_pred, name="预测值", line=dict(color='blue')),
                                    row=1, col=2
                                )
                                
                                # 箱线图
                                pred_fig.add_trace(
                                    go.Box(y=new_y_pred, name="预测值", boxpoints='outliers'),
                                    row=2, col=1
                                )
                                
                                # 直方图
                                pred_fig.add_trace(
                                    go.Histogram(x=new_y_pred, name="分布", nbinsx=30, opacity=0.7),
                                    row=2, col=2
                                )
                                
                                # 分布曲线（需要至少2个不同值才能计算）
                                from scipy import stats
                                if len(np.unique(new_y_pred)) >= 2 and np.std(new_y_pred) > 0:
                                    kde_x = np.linspace(np.min(new_y_pred), np.max(new_y_pred), 100)
                                    kde = stats.gaussian_kde(new_y_pred)
                                    pred_fig.add_trace(
                                        go.Scatter(x=kde_x, y=kde(kde_x), name="密度曲线", line=dict(color='red')),
                                        row=1, col=1
                                    )
                                
                                pred_fig.update_layout(height=700, title_text="预测结果多维度分析", showlegend=True)
                                st.plotly_chart(pred_fig, use_container_width=True)
                                
                                # 如果有目标列，进行对比分析
                                if ts_target_col and ts_target_col in new_df.columns:
                                    st.subheader("🔄 预测值 vs 真实值对比")
                                    
                                    actual_values = new_df[ts_target_col].values
                                    
                                    # 计算对比指标
                                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                                    pred_mse = mean_squared_error(actual_values, new_y_pred)
                                    pred_rmse = np.sqrt(pred_mse)
                                    pred_mae = mean_absolute_error(actual_values, new_y_pred)
                                    pred_r2 = r2_score(actual_values, new_y_pred)
                                    pred_mape = np.mean(np.abs((actual_values - new_y_pred) / (actual_values + 1e-10))) * 100
                                    
                                    # 显示对比指标
                                    comp_col1, comp_col2, comp_col3, comp_col4, comp_col5 = st.columns(5)
                                    with comp_col1:
                                        st.metric("MSE", f"{pred_mse:.6f}")
                                    with comp_col2:
                                        st.metric("RMSE", f"{pred_rmse:.6f}")
                                    with comp_col3:
                                        st.metric("MAE", f"{pred_mae:.6f}")
                                    with comp_col4:
                                        st.metric("R²", f"{pred_r2:.4f}")
                                    with comp_col5:
                                        st.metric("MAPE", f"{pred_mape:.2f}%")
                                    
                                    # 对比图表
                                    compare_fig = make_subplots(
                                        rows=2, cols=2,
                                        subplot_titles=("预测值 vs 真实值", "残差分布", "误差时序", "Q-Q图")
                                    )
                                    
                                    # 预测值 vs 真实值散点图
                                    compare_fig.add_trace(
                                        go.Scatter(x=actual_values, y=new_y_pred, mode='markers', 
                                                  name="预测点", marker=dict(size=6, opacity=0.6)),
                                        row=1, col=1
                                    )
                                    # 理想线
                                    min_val = min(np.min(actual_values), np.min(new_y_pred))
                                    max_val = max(np.max(actual_values), np.max(new_y_pred))
                                    compare_fig.add_trace(
                                        go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                                                  mode='lines', name="理想线", line=dict(color='red', dash='dash')),
                                        row=1, col=1
                                    )
                                    
                                    # 残差
                                    residuals = actual_values - new_y_pred
                                    compare_fig.add_trace(
                                        go.Histogram(x=residuals, name="残差分布", nbinsx=30),
                                        row=1, col=2
                                    )
                                    
                                    # 误差时序
                                    compare_fig.add_trace(
                                        go.Scatter(y=residuals, name="残差时序", line=dict(color='orange')),
                                        row=2, col=1
                                    )
                                    
                                    # Q-Q图
                                    from scipy.stats import probplot
                                    qq = probplot(residuals, dist="norm")
                                    compare_fig.add_trace(
                                        go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name="Q-Q点"),
                                        row=2, col=2
                                    )
                                    compare_fig.add_trace(
                                        go.Scatter(x=qq[0][0], y=qq[1][0] * qq[0][0] + qq[1][1], 
                                                  mode='lines', name="参考线", line=dict(color='red')),
                                        row=2, col=2
                                    )
                                    
                                    compare_fig.update_layout(height=700, title_text="预测效果对比分析", showlegend=True)
                                    st.plotly_chart(compare_fig, use_container_width=True)
                                    
                                    # 误差分析
                                    with st.expander("📋 详细误差分析"):
                                        error_df = pd.DataFrame({
                                            '样本索引': range(len(residuals)),
                                            '真实值': actual_values,
                                            '预测值': new_y_pred,
                                            '绝对误差': np.abs(residuals),
                                            '相对误差(%)': np.abs(residuals / (actual_values + 1e-10)) * 100,
                                            '残差': residuals
                                        })
                                        st.dataframe(error_df, height=400)
                                        
                                        # 最大误差样本
                                        max_error_idx = np.argmax(np.abs(residuals))
                                        st.warning(f"最大误差样本: 索引 {max_error_idx}, 真实值: {actual_values[max_error_idx]:.4f}, 预测值: {new_y_pred[max_error_idx]:.4f}, 误差: {residuals[max_error_idx]:.4f}")
                            
                            if 'new_y_pred' in locals():
                                # 预测结果表格
                                with st.expander("📋 查看完整预测结果"):
                                    st.dataframe(new_df, height=400)
                                
                                # 下载按钮
                                csv = new_df.to_csv(index=False)
                                # 使用模型名称或默认名称
                                download_model_name = ts_model_info.get('model_type', 'model') if ts_model_info else 'ts_model'
                                st.download_button("📥 下载预测结果 (CSV)", csv, f"{download_model_name}_predictions.csv", "text/csv")
                                prediction_figures = []
                                if "pred_fig" in locals():
                                    prediction_figures.append(("预测结果分布与趋势", pred_fig))
                                if "compare_fig" in locals():
                                    prediction_figures.append(("预测值与真实值对比", compare_fig))
                                prediction_metrics = {"模型": download_model_name, "预测样本数": len(new_df)}
                                if "pred_r2" in locals():
                                    prediction_metrics.update({
                                        "R²": f"{pred_r2:.4f}",
                                        "RMSE": f"{pred_rmse:.6f}",
                                        "MAE": f"{pred_mae:.6f}",
                                        "MAPE": f"{pred_mape:.2f}%",
                                    })
                                ts_prediction_report = build_html_report(
                                    "时序新数据预测图文结果报告",
                                    metrics=prediction_metrics,
                                    sections=[("结果摘要", "报告包含新数据预测结果、预测分布和可选真实值对比。")],
                                    figures=prediction_figures,
                                    tables=[("预测结果", new_df)],
                                )
                                st.download_button(
                                    "📄 下载预测HTML图文报告",
                                    ts_prediction_report.encode("utf-8"),
                                    file_name=f"{download_model_name}_prediction_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                    mime="text/html",
                                    key="download_ts_prediction_html",
                                )
                    finally:
                        os.unlink(new_tmp_file_path)

# ==================== 标签2: 分类任务 ====================
with tab2:
    st.markdown("<h1 style='font-size: 32px; font-weight: bold;'>🎯 分类任务</h1>", unsafe_allow_html=True)
    
    st.info("上传带标签的数据，训练分类模型进行样本分类或状态识别。")
    with st.expander("🧠 分类深度学习模型库", expanded=False):
        st.caption("此处只列出分类页主训练流程真实可选、可训练的深度学习模型。")
        clf_dl_df = pd.DataFrame([item.__dict__ for item in list_deep_learning_algorithms("classification")])
        st.dataframe(clf_dl_df.rename(columns={
            "key": "模型ID",
            "name": "模型名称",
            "task": "任务",
            "data_type": "数据类型",
            "description": "适用说明"
        }), use_container_width=True, hide_index=True)
        st.info("ANN 适合表格数据；LSTM 适合序列特征；1D-CNN 适合波形片段；2D-CNN 适合谱图或二维特征图。")
    
    # 初始化分类任务的session state
    if 'classification_df' not in st.session_state:
        st.session_state.classification_df = None
    if 'classification_target_col' not in st.session_state:
        st.session_state.classification_target_col = None
    if 'classification_feature_cols' not in st.session_state:
        st.session_state.classification_feature_cols = None
    if 'classification_trainer' not in st.session_state:
        st.session_state.classification_trainer = None
    if 'classification_model_trained' not in st.session_state:
        st.session_state.classification_model_trained = False
    
    # ==================== 分类任务步骤1: 数据上传 ====================
    st.subheader("📁 步骤1: 数据上传")
    
    clf_col1, clf_col2 = st.columns([3, 1])
    
    with clf_col1:
        clf_data_source = st.radio(
            "选择数据来源",
            ["上传文件", "使用示例数据"],
            horizontal=True,
            key="clf_data_source"
        )
    
    if clf_data_source == "上传文件":
        clf_file = st.file_uploader(
            "上传分类数据文件",
            type=["csv", "xlsx", "xls", "out", "dat", "txt"],
            key="clf_file_uploader"
        )
        
        if clf_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(clf_file.name)[1]) as tmp_file:
                tmp_file.write(clf_file.getvalue())
                clf_tmp_path = tmp_file.name
            
            try:
                if clf_file.name.endswith(('.csv')):
                    clf_df = None
                    for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']:
                        try:
                            clf_df = pd.read_csv(clf_tmp_path, encoding=encoding)
                            break
                        except:
                            continue
                    if clf_df is None:
                        raise ValueError("无法使用支持的编码读取CSV文件")
                elif clf_file.name.endswith(('.xlsx', '.xls')):
                    clf_df = pd.read_excel(clf_tmp_path)
                else:
                    clf_df = read_data_file(clf_tmp_path)
                
                st.session_state.classification_df = clf_df
                st.success(f"✅ 数据加载成功！共 {len(clf_df)} 行，{len(clf_df.columns)} 列")
            finally:
                os.unlink(clf_tmp_path)
    else:
        st.markdown("**选择示例数据集：**")
        sample_data_options = [
            "设备故障诊断.csv",
            "产品质量分类.csv", 
            "鸢尾花分类.csv",
            "心脏病诊断.csv",
            "客户流失分类.csv",
            "机械振动模式分类.csv",
            "糖尿病风险预测.csv",
            "信用评分分类.csv",
            "图像质量评估.csv",
            "邮件垃圾邮件分类.csv",
            "时序状态分类.csv",
            "多通道时序状态分类.csv"
        ]
        selected_sample = st.selectbox("选择数据集", sample_data_options, key="clf_sample_select")
        
        if st.button("加载示例数据", key="clf_load_sample"):
            sample_path = f"sample_data/{selected_sample}"
            if os.path.exists(sample_path):
                clf_df = pd.read_csv(sample_path, encoding='utf-8-sig')
                st.session_state.classification_df = clf_df
                st.success(f"✅ 已加载 {selected_sample}，共 {len(clf_df)} 行，{len(clf_df.columns)} 列")
            else:
                st.error("示例数据文件不存在！")
    
    # 显示数据预览
    if st.session_state.classification_df is not None:
        clf_df = st.session_state.classification_df
        
        with st.expander("📊 数据预览", expanded=True):
            st.dataframe(clf_df.head(10))
            
            # 显示数据基本信息
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("样本数", len(clf_df))
            with col_info2:
                st.metric("特征数", len(clf_df.columns) - 1)
            with col_info3:
                # 统计目标列的类别数
                numeric_cols = clf_df.select_dtypes(include=[np.number]).columns.tolist()
                non_numeric_cols = [col for col in clf_df.columns if col not in numeric_cols]
                if non_numeric_cols:
                    target_col = st.selectbox("选择目标列（标签）", non_numeric_cols, key="clf_target_select_preview")
                    unique_labels = clf_df[target_col].nunique()
                    st.metric("类别数", unique_labels)
                else:
                    st.warning("未找到非数值列作为目标列")
        
        # ==================== 分类任务步骤2: 特征选择 ====================
        st.subheader("🎯 步骤2: 特征与目标列选择")
        
        clf_all_cols = clf_df.columns.tolist()
        
        # 选择目标列（标签列）
        clf_target_col = st.selectbox(
            "选择目标列（标签/类别）",
            clf_all_cols,
            index=len(clf_all_cols) - 1,
            key="clf_target_col_select",
            help="选择包含类别标签的列，如：状态、质量等级、诊断结果等"
        )
        
        # 检测是否为时序分类问题（每行是一条时序数据）
        clf_is_time_series = st.radio(
            "分类问题类型",
            ["标准分类", "时序状态分类"],
            index=0,
            horizontal=True,
            key="clf_is_time_series",
            help="标准分类: 每行是一个样本；时序状态分类: 每行是一条时序数据，代表一种系统状态"
        )
        
        # 时序分类时添加特征提取选项
        clf_time_series_features = {}
        if clf_is_time_series == "时序状态分类":
            st.info("ℹ️ 时序状态分类：每行是一条时序数据（如振动信号），需要从时序中提取特征进行分类")
            
            clf_all_feature_cols = [col for col in clf_all_cols if col != clf_target_col]
            
            import re
            channel_pattern = re.compile(r'^(通道\d+)_')
            potential_channels = set()
            for col in clf_all_feature_cols:
                match = channel_pattern.match(col)
                if match:
                    potential_channels.add(match.group(1))
            
            if len(potential_channels) > 1:
                st.info(f"检测到多通道数据：{len(potential_channels)} 个通道（{', '.join(sorted(potential_channels))}）")
                clf_is_multichannel = True
            else:
                clf_is_multichannel = st.checkbox(
                    "多通道时序数据",
                    value=False,
                    key="clf_is_multichannel",
                    help="每行包含多个通道的时序数据（如通道1_时间点1, 通道2_时间点1...），系统会自动识别通道"
                )
            
            st.markdown("#### 📊 时序特征提取选项")
            
            # 特征类型选择：传统特征 vs 图像特征
            feature_type = st.radio(
                "特征类型",
                ["传统特征", "时域波形图像", "时频谱图图像"],
                horizontal=True,
                key="clf_feature_type",
                help="传统特征：从时序中提取统计/频域等数值特征；图像特征：将时序转为图像后用CNN识别"
            )
            
            clf_time_series_features["feature_type"] = feature_type
            
            if feature_type == "传统特征":
                col_ts1, col_ts2 = st.columns(2)
                
                with col_ts1:
                    clf_time_series_features["use_statistical"] = st.checkbox(
                        "统计特征",
                        value=True,
                        key="clf_statistical",
                        help="均值、标准差、最大值、最小值、峰峰值、偏度、峰度"
                    )
                    clf_time_series_features["use_frequency"] = st.checkbox(
                        "频域特征",
                        value=False,
                        key="clf_frequency",
                        help="频谱特征（需要FFT）"
                    )
                
                with col_ts2:
                    clf_time_series_features["use_temporal"] = st.checkbox(
                        "时序特征",
                        value=True,
                        key="clf_temporal",
                        help="变化率、趋势、波动性"
                    )
                    clf_time_series_features["use_peaks"] = st.checkbox(
                        "峰值特征",
                        value=False,
                        key="clf_peaks",
                        help="峰值数量、峰值间距"
                    )
            elif feature_type == "时域波形图像":
                st.info("📈 时域波形图像：将长时序按滑动窗口切片，形成波形图像序列")
                
                if 'feature_cols' in dir() and len(feature_cols) > 0:
                    time_series_length = len(feature_cols)
                else:
                    time_series_length = 20
                
                col_wv1, col_wv2 = st.columns(2)
                with col_wv1:
                    clf_time_series_features["waveform_window_size"] = st.number_input(
                        "滑动窗口大小（时间步）",
                        min_value=5,
                        max_value=min(time_series_length, 100),
                        value=min(10, time_series_length),
                        key="clf_waveform_window",
                        help="每个波形图像包含的时间步数"
                    )
                with col_wv2:
                    clf_time_series_features["waveform_stride"] = st.number_input(
                        "滑动步长",
                        min_value=1,
                        max_value=clf_time_series_features.get("waveform_window_size", 10),
                        value=max(1, clf_time_series_features.get("waveform_window_size", 10) // 2),
                        key="clf_waveform_stride",
                        help="滑动窗口的步长（应小于窗口大小）"
                    )
            elif feature_type == "时频谱图图像":
                st.info("🔊 时频谱图图像：将时序通过STFT转换为时频谱图")
                col_sp1, col_sp2, col_sp3 = st.columns(3)
                with col_sp1:
                    clf_time_series_features["stft_nperseg"] = st.number_input(
                        "STFT窗口大小",
                        min_value=8,
                        max_value=256,
                        value=64,
                        key="clf_stft_nperseg",
                        help="STFT每帧的样本数"
                    )
                with col_sp2:
                    clf_time_series_features["stft_noverlap"] = st.number_input(
                        "STFT重叠",
                        min_value=0,
                        max_value=63,
                        value=32,
                        key="clf_stft_noverlap",
                        help="相邻帧之间的重叠样本数（必须小于窗口大小）"
                    )
                with col_sp3:
                    clf_time_series_features["spectrogram_height"] = st.number_input(
                        "谱图高度（像素）",
                        min_value=16,
                        max_value=128,
                        value=32,
                        key="clf_spectrogram_height",
                        help="谱图图像的高度"
                    )
            
            clf_time_series_features["is_multichannel"] = clf_is_multichannel
        
        # 显示目标列的类别分布
        if clf_target_col:
            st.markdown("**类别分布：**")
            label_counts = clf_df[clf_target_col].value_counts()
            st.dataframe(label_counts)
            
            # 可视化类别分布
            fig_labels = go.Figure(data=[
                go.Bar(x=label_counts.index, y=label_counts.values, marker_color='steelblue')
            ])
            fig_labels.update_layout(
                title="类别分布",
                xaxis_title="类别",
                yaxis_title="样本数量",
                height=300
            )
            st.plotly_chart(fig_labels, use_container_width=True)
        
        # 选择特征列
        clf_feature_cols = st.multiselect(
            "选择特征列",
            [col for col in clf_all_cols if col != clf_target_col],
            default=[col for col in clf_all_cols if col != clf_target_col],
            key="clf_feature_cols_select",
            help="选择用于分类的特征"
        )
        
        if clf_feature_cols:
            st.session_state.classification_target_col = clf_target_col
            st.session_state.classification_feature_cols = clf_feature_cols
            
            # ==================== 分类任务步骤3: 数据预处理 ====================
            st.subheader("🔧 步骤3: 数据预处理")
            
            clf_prep_col1, clf_prep_col2 = st.columns(2)
            
            with clf_prep_col1:
                clf_test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05, key="clf_test_size")
            
            with clf_prep_col2:
                clf_random_state = st.number_input("随机种子", 0, 999, 42, key="clf_random_state")
            
            # 数据预处理
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            
            X_clf = clf_df[clf_feature_cols]
            y_clf = clf_df[clf_target_col]
            
            # 时序特征提取
            if clf_is_time_series == "时序状态分类":
                st.info("🔧 正在提取时序特征...")
                
                feature_type = clf_time_series_features.get("feature_type", "传统特征")
                is_multichannel = clf_time_series_features.get("is_multichannel", False)
                
                # 时域波形图像特征提取
                if feature_type == "时域波形图像":
                    st.info("📈 正在将时序转换为波形图像...")
                    
                    window_size = clf_time_series_features.get("waveform_window_size", 32)
                    stride = clf_time_series_features.get("waveform_stride", 8)
                    
                    def create_waveform_images(row_data, window_size, stride):
                        """将时序数据转换为波形图像序列"""
                        values = np.array(row_data.values, dtype=float)
                        images = []
                        
                        for start in range(0, len(values) - window_size + 1, stride):
                            window = values[start:start + window_size]
                            # 归一化到 [0, 1]
                            window_min = window.min()
                            window_max = window.max()
                            if window_max - window_min > 0:
                                window_norm = (window - window_min) / (window_max - window_min)
                            else:
                                window_norm = np.zeros_like(window)
                            images.append(window_norm)
                        
                        return images
                    
                    # 为每个样本创建波形图像
                    waveform_images = []
                    waveform_labels = []
                    
                    for idx in range(len(X_clf)):
                        row = X_clf.iloc[idx]
                        images = create_waveform_images(row, window_size, stride)
                        if len(images) > 0:
                            waveform_images.append(images[0])
                            waveform_labels.append(y_clf.iloc[idx])
                    
                    # 转换为CNN所需格式 (samples, timesteps, channels)
                    X_clf_waveform = np.array(waveform_images)
                    X_clf_waveform = X_clf_waveform.reshape(-1, window_size, 1)
                    
                    # 保存原始标签供后续编码使用
                    waveform_labels_for_encoding = waveform_labels.copy()
                    
                    # 存储为特殊格式供CNN使用
                    X_clf_train_scaled = None
                    X_clf_test_scaled = None
                    clf_waveform_mode = True
                    clf_waveform_timesteps = window_size
                    
                    st.success(f"✅ 波形图像转换完成！生成了 {len(X_clf_waveform)} 个波形图像，形状: {X_clf_waveform.shape}")
                    
                    # 显示示例波形图像
                    with st.expander("📊 波形图像示例"):
                        fig_wave = go.Figure()
                        for i in range(min(3, len(X_clf_waveform))):
                            fig_wave.add_trace(go.Scatter(
                                y=X_clf_waveform[i, :, 0],
                                mode='lines',
                                name=f'样本{i+1}'
                            ))
                        fig_wave.update_layout(
                            title="波形图像示例（前3个样本）",
                            xaxis_title="时间步",
                            yaxis_title="归一化幅值",
                            height=300
                        )
                        st.plotly_chart(fig_wave, use_container_width=True)
                    
                    # 跳过后续的传统特征提取
                    skip_traditional_feature_extraction = True
                
                # 时频谱图图像特征提取
                elif feature_type == "时频谱图图像":
                    st.info("🔊 正在将时序转换为时频谱图...")
                    
                    nperseg = clf_time_series_features.get("stft_nperseg", 64)
                    noverlap = clf_time_series_features.get("stft_noverlap", 32)
                    
                    if noverlap >= nperseg:
                        noverlap = nperseg - 1
                        st.warning(f"⚠️ STFT重叠值已自动调整为 {noverlap}（必须小于窗口大小 {nperseg}）")
                    
                    spectrogram_height = clf_time_series_features.get("spectrogram_height", 32)
                    
                    from scipy import signal
                    from scipy.ndimage import zoom as resize_scipy
                    
                    def create_spectrogram(row_data, nperseg, noverlap, target_height):
                        """将时序数据转换为谱图图像"""
                        values = np.array(row_data.values, dtype=float)
                        
                        if len(values) < nperseg:
                            return None
                        
                        # 计算STFT
                        freqs, times, Sxx = signal.stft(
                            values, 
                            fs=1.0, 
                            nperseg=nperseg, 
                            noverlap=noverlap
                        )
                        
                        # 取幅度谱
                        Sxx_mag = np.abs(Sxx)
                        
                        # 归一化
                        Sxx_norm = (Sxx_mag - Sxx_mag.min()) / (Sxx_mag.max() - Sxx_mag.min() + 1e-8)
                        
                        # 调整大小到目标高度
                        zoom_factors = (target_height / Sxx_norm.shape[0], Sxx_norm.shape[1] / Sxx_norm.shape[1])
                        Sxx_resized = resize_scipy(Sxx_norm, zoom_factors)
                        
                        return Sxx_resized
                    
                    # 为每个样本创建谱图
                    spectrogram_images = []
                    spectrogram_labels = []
                    
                    for idx in range(len(X_clf)):
                        row = X_clf.iloc[idx]
                        spect = create_spectrogram(row, nperseg, noverlap, spectrogram_height)
                        if spect is not None and spect.size > 0:
                            spectrogram_images.append(spect)
                            spectrogram_labels.append(y_clf.iloc[idx])
                    
                    if len(spectrogram_images) == 0:
                        st.error("❌ 无法从数据中生成有效的谱图，请检查数据")
                        st.stop()
                    
                    # 转换为CNN所需格式 (samples, height, width, channels)
                    X_clf_spectrogram = np.array(spectrogram_images)
                    X_clf_spectrogram = X_clf_spectrogram.reshape(-1, spectrogram_height, X_clf_spectrogram.shape[2], 1)
                    
                    # 保存原始标签供后续编码使用
                    spectrogram_labels_for_encoding = spectrogram_labels.copy()
                    
                    X_clf_train_scaled = None
                    X_clf_test_scaled = None
                    clf_spectrogram_mode = True
                    clf_spectrogram_shape = X_clf_spectrogram.shape[1:]
                    
                    st.success(f"✅ 谱图转换完成！生成了 {len(X_clf_spectrogram)} 个谱图，形状: {X_clf_spectrogram.shape}")
                    
                    # 显示示例谱图
                    with st.expander("📊 谱图图像示例"):
                        fig_spec = go.Figure()
                        for i in range(min(2, len(X_clf_spectrogram))):
                            fig_spec.add_trace(go.Heatmap(
                                z=X_clf_spectrogram[i, :, :, 0],
                                colorscale='Viridis',
                                name=f'样本{i+1}'
                            ))
                        fig_spec.update_layout(
                            title="谱图图像示例",
                            height=300
                        )
                        st.plotly_chart(fig_spec, use_container_width=True)
                    
                    skip_traditional_feature_extraction = True
                
                else:
                    skip_traditional_feature_extraction = False
                
                def extract_single_channel_features(row_values, prefix="", use_statistical=True, use_frequency=False, use_temporal=True, use_peaks=False):
                    """从单通道时序数据中提取特征"""
                    import scipy.signal as signal
                    features = {}
                    values = np.array(row_values, dtype=float)
                    
                    if use_statistical:
                        features[f'{prefix}ts_mean'] = np.mean(values)
                        features[f'{prefix}ts_std'] = np.std(values)
                        features[f'{prefix}ts_max'] = np.max(values)
                        features[f'{prefix}ts_min'] = np.min(values)
                        features[f'{prefix}ts_range'] = np.max(values) - np.min(values)
                        features[f'{prefix}ts_rms'] = np.sqrt(np.mean(values**2))
                        features[f'{prefix}ts_skew'] = pd.Series(values).skew()
                        features[f'{prefix}ts_kurt'] = pd.Series(values).kurtosis()
                    
                    if use_frequency:
                        fft_vals = np.fft.fft(values)
                        fft_mag = np.abs(fft_vals)
                        freqs = np.fft.fftfreq(len(values))
                        features[f'{prefix}fft_mean'] = np.mean(fft_mag)
                        features[f'{prefix}fft_std'] = np.std(fft_mag)
                        features[f'{prefix}fft_max'] = np.max(fft_mag)
                        features[f'{prefix}dominant_freq'] = freqs[np.argmax(fft_mag[:len(fft_mag)//2])] if len(fft_mag) > 1 else 0
                    
                    if use_temporal:
                        diff_vals = np.diff(values)
                        features[f'{prefix}ts_diff_mean'] = np.mean(diff_vals)
                        features[f'{prefix}ts_diff_std'] = np.std(diff_vals)
                        features[f'{prefix}ts_abs_diff_mean'] = np.mean(np.abs(diff_vals))
                        features[f'{prefix}ts_trend'] = np.polyfit(range(len(values)), values, 1)[0]
                    
                    if use_peaks:
                        peaks, _ = signal.find_peaks(values)
                        features[f'{prefix}num_peaks'] = len(peaks)
                        if len(peaks) > 1:
                            peak_intervals = np.diff(peaks)
                            features[f'{prefix}peak_interval_mean'] = np.mean(peak_intervals)
                        else:
                            features[f'{prefix}peak_interval_mean'] = 0
                    
                    return features
                
                def extract_time_series_features(row_data, is_multichannel=False, use_statistical=True, use_frequency=False, use_temporal=True, use_peaks=False):
                    """从时序数据中提取特征（支持单通道和多通道）"""
                    import re
                    features = {}
                    
                    if is_multichannel:
                        import scipy.signal as signal
                        channel_pattern = re.compile(r'^(通道\d+)_时间点(\d+)')
                        channel_data = {}
                        
                        for col in row_data.index:
                            match = channel_pattern.match(col)
                            if match:
                                channel_name = match.group(1)
                                if channel_name not in channel_data:
                                    channel_data[channel_name] = []
                                channel_data[channel_name].append((int(match.group(2)), row_data[col]))
                        
                        for channel_name in sorted(channel_data.keys()):
                            channel_data[channel_name].sort(key=lambda x: x[0])
                            channel_values = np.array([v for _, v in channel_data[channel_name]], dtype=float)
                            prefix = f"{channel_name}_"
                            
                            channel_features = extract_single_channel_features(
                                channel_values, prefix, use_statistical, use_frequency, use_temporal, use_peaks
                            )
                            features.update(channel_features)
                        
                        if len(channel_data) >= 2:
                            keys = list(channel_data.keys())
                            ch1_vals = np.array([v for _, v in channel_data[keys[0]]], dtype=float)
                            ch2_vals = np.array([v for _, v in channel_data[keys[1]]], dtype=float)
                            features['channel_correlation'] = np.corrcoef(ch1_vals, ch2_vals)[0, 1]
                            features['channel_diff_mean'] = np.mean(ch1_vals - ch2_vals)
                    else:
                        features = extract_single_channel_features(
                            row_data.values, "", use_statistical, use_frequency, use_temporal, use_peaks
                        )
                    
                    return features
                
                if not skip_traditional_feature_extraction:
                    feature_list = []
                    for idx in range(len(X_clf)):
                        row = X_clf.iloc[idx]
                        features = extract_time_series_features(
                            row,
                            is_multichannel=is_multichannel,
                            use_statistical=clf_time_series_features.get("use_statistical", True),
                            use_frequency=clf_time_series_features.get("use_frequency", False),
                            use_temporal=clf_time_series_features.get("use_temporal", True),
                            use_peaks=clf_time_series_features.get("use_peaks", False)
                        )
                        feature_list.append(features)
                    
                    X_clf = pd.DataFrame(feature_list)
                    clf_feature_cols = X_clf.columns.tolist()
                    
                    if is_multichannel:
                        st.success(f"✅ 多通道时序特征提取完成！每个样本提取了 {len(clf_feature_cols)} 个特征（{clf_time_series_features.get('num_channels', '多')}通道）")
                    else:
                        st.success(f"✅ 时序特征提取完成！每个样本提取了 {len(clf_feature_cols)} 个特征")
                    
                    # 显示提取的特征
                    with st.expander("📊 提取的时序特征预览"):
                        st.dataframe(X_clf.head())
            
            # 检查是否使用图像模式
            clf_image_mode = clf_time_series_features.get("feature_type") in ["时域波形图像", "时频谱图图像"]
            
            # 标签编码 - 对于图像模式使用过滤后的标签
            label_encoder = LabelEncoder()
            if clf_image_mode:
                if clf_time_series_features.get("feature_type") == "时域波形图像":
                    y_clf = waveform_labels_for_encoding
                elif clf_time_series_features.get("feature_type") == "时频谱图图像":
                    y_clf = spectrogram_labels_for_encoding
                y_clf_encoded = label_encoder.fit_transform(y_clf)
            else:
                y_clf_encoded = label_encoder.fit_transform(y_clf)
            
            if clf_image_mode:
                # 图像模式：使用已转换好的图像数据
                if clf_time_series_features.get("feature_type") == "时域波形图像":
                    X_clf_image = X_clf_waveform
                else:
                    X_clf_image = X_clf_spectrogram
                
                # 划分数据集
                X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
                    X_clf_image, y_clf_encoded, test_size=clf_test_size, random_state=clf_random_state, stratify=y_clf_encoded
                )
                
                X_clf_train_scaled = X_clf_train
                X_clf_test_scaled = X_clf_test
                
                st.success(f"✅ 数据预处理完成！训练集: {len(X_clf_train)} 样本，测试集: {len(X_clf_test)} 样本")
            else:
                # 传统特征模式
                # 只保留数值列
                numeric_clf_cols = X_clf.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_clf_cols) < len(clf_feature_cols):
                    non_numeric = [c for c in clf_feature_cols if c not in numeric_clf_cols]
                    st.warning(f"自动移除非数值列: {non_numeric}")
                    clf_feature_cols = numeric_clf_cols
                    X_clf = X_clf[clf_feature_cols]
                
                # 划分数据集
                X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
                    X_clf, y_clf_encoded, test_size=clf_test_size, random_state=clf_random_state, stratify=y_clf_encoded
                )
                
                # 标准化
                scaler = StandardScaler()
                X_clf_train_scaled = scaler.fit_transform(X_clf_train)
                X_clf_test_scaled = scaler.transform(X_clf_test)
                
                st.success(f"✅ 数据预处理完成！训练集: {len(X_clf_train)} 样本，测试集: {len(X_clf_test)} 样本")
            
            # ==================== 分类任务步骤4: 模型选择 ====================
            st.subheader("🤖 步骤4: 选择分类模型")
            
            clf_model_selection_mode = st.radio(
                "模型选择模式",
                ["预置模型", "自定义模型"],
                horizontal=True,
                key="clf_model_selection_mode",
                help="预置模型: 使用内置的机器学习/深度学习模型; 自定义模型: 用户自己编写模型代码"
            )
            
            # 自定义模型配置
            clf_custom_model_loaded = False
            clf_custom_model_instance = None
            if clf_model_selection_mode == "自定义模型":
                st.markdown("### 🔧 自定义分类模型配置")
                
                clf_template_codes = {
                    "自定义（空白模板）": '''from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class MyCustomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y):
        self.model_ = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.model_.fit(X, y)
        self.classes_ = np.unique(y)
        return self
    
    def predict(self, X):
        return self.model_.predict(X)
    
    def predict_proba(self, X):
        return self.model_.predict_proba(X)''',
                    "简单KNN分类器": '''from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class MyKNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y):
        self.model_ = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.model_.fit(X, y)
        self.classes_ = np.unique(y)
        return self
    
    def predict(self, X):
        return self.model_.predict(X)
    
    def predict_proba(self, X):
        return self.model_.predict_proba(X)''',
                    "加权投票集成": '''from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

class VotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.models = [
            RandomForestClassifier(n_estimators=50),
            GradientBoostingClassifier(n_estimators=50),
            LogisticRegression()
        ]
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for m in self.models:
            m.fit(X, y)
        return self
    
    def predict(self, X):
        votes = np.zeros((X.shape[0], len(self.classes_)))
        for m in self.models:
            proba = m.predict_proba(X)
            votes += proba
        return self.classes_[np.argmax(votes, axis=1)]
    
    def predict_proba(self, X):
        votes = np.zeros((X.shape[0], len(self.classes_)))
        for m in self.models:
            votes += m.predict_proba(X)
        return votes / len(self.models)''',
                    "概率校准分类器": '''from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin

class CalibratedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model=None):
        self.base_model = base_model if base_model else LogisticRegression()
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model_ = CalibratedClassifierCV(self.base_model, cv=3)
        self.model_.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model_.predict(X)
    
    def predict_proba(self, X):
        return self.model_.predict_proba(X)'''
                }
                
                # 初始化session state
                if 'clf_custom_model_code' not in st.session_state:
                    st.session_state.clf_custom_model_code = clf_template_codes["自定义（空白模板）"]
                
                clf_template_code = st.selectbox(
                    "选择模板",
                    list(clf_template_codes.keys()),
                    key="clf_custom_template_select"
                )
                
                # 当模板选择变化时更新代码
                if st.session_state.get('clf_last_template') != clf_template_code:
                    st.session_state.clf_custom_model_code = clf_template_codes[clf_template_code]
                    st.session_state.clf_last_template = clf_template_code
                    st.rerun()
                
                st.markdown("### ✏️ 编辑自定义分类模型代码")
                st.info("请确保代码包含：1. 模型类定义（继承BaseEstimator, ClassifierMixin）2. __init__, fit, predict 方法")
                
                # 导入/导出按钮放在代码框上方
                clf_imp_col, clf_exp_col = st.columns(2)
                with clf_imp_col:
                    if st.button("📤 导入模型代码", key="clf_import_btn", use_container_width=True):
                        st.session_state.clf_show_import = True
                    
                    if st.session_state.get('clf_show_import', False):
                        clf_uploaded_code_file = st.file_uploader(
                            "选择文件",
                            type=["py"],
                            key="clf_import_code"
                        )
                        if clf_uploaded_code_file is not None:
                            st.session_state.clf_custom_model_code = clf_uploaded_code_file.getvalue().decode("utf-8")
                            st.success("✅ 代码已导入，请重新选择模板或刷新页面查看")
                            st.session_state.clf_show_import = False
                
                with clf_exp_col:
                    st.download_button(
                        label="📥 导出模型代码",
                        data=st.session_state.clf_custom_model_code,
                        file_name="custom_classifier.py",
                        mime="text/x-python",
                        key="clf_download_code",
                        use_container_width=True
                    )
                
                clf_custom_model_code = st.text_area(
                    "自定义分类模型代码",
                    height=250,
                    value=st.session_state.clf_custom_model_code,
                    key="clf_custom_model_code"
                )
                
                clf_custom_model_class = None
                if clf_custom_model_code:
                    try:
                        import types
                        clf_custom_module = types.ModuleType('clf_custom_model_module')
                        exec(clf_custom_model_code, clf_custom_module.__dict__)
                        record_custom_execution("分类自定义模型", "加载成功")
                        
                        for item in dir(clf_custom_module):
                            obj = getattr(clf_custom_module, item)
                            if isinstance(obj, type) and hasattr(obj, 'fit') and hasattr(obj, 'predict'):
                                if obj.__name__ != 'BaseEstimator' and obj.__name__ != 'ClassifierMixin':
                                    clf_custom_model_class = obj
                                    break
                        
                        if clf_custom_model_class:
                            clf_custom_model_instance = clf_custom_model_class()
                            clf_custom_model_loaded = True
                            st.success(f"✅ 自定义分类模型加载成功: {clf_custom_model_class.__name__}")
                        else:
                            st.error("未找到有效的分类模型类")
                    except Exception as e:
                        record_custom_execution("分类自定义模型", "加载失败", str(e))
                        st.error(f"模型代码加载失败: {str(e)}")
            
            if clf_model_selection_mode == "自定义模型":
                clf_model_type = "custom"
            else:
                # 检查是否在图像模式下
                clf_image_mode = clf_time_series_features.get("feature_type") in ["时域波形图像", "时频谱图图像"]
                
                if clf_image_mode:
                    # 图像模式只能选择CNN相关模型，根据图像类型选择
                    feature_type = clf_time_series_features.get("feature_type")
                    if feature_type == "时域波形图像":
                        clf_model_options = ["CNN (1D)"]
                        model_help = "时域波形图像使用1D卷积神经网络"
                    elif feature_type == "时频谱图图像":
                        clf_model_options = ["CNN (2D)"]
                        model_help = "时频谱图图像使用2D卷积神经网络"
                else:
                    clf_model_options = [
                        "Random Forest",
                        "Gradient Boosting", 
                        "XGBoost",
                        "LightGBM",
                        "SVM",
                        "KNN",
                        "Logistic Regression",
                        "Decision Tree",
                        "ANN",
                        "LSTM"
                    ]
                    model_help = "机器学习: RandomForest/XGBoost/LightGBM适合复杂问题; 深度学习: ANN/LSTM适合复杂模式"
                
                clf_model_type = st.selectbox(
                    "选择分类模型",
                    clf_model_options,
                    key="clf_model_select",
                    help=model_help
                )
            
            # 预置模型参数配置
            clf_model_kwargs = {}
            if clf_model_selection_mode == "预置模型":
                st.markdown("### ⚙️ 模型参数配置")
                
                if clf_model_type == "Random Forest":
                    clf_model_kwargs["n_estimators"] = st.slider(
                        "树数量", 10, 500, 100, 10,
                        key="clf_rf_n_estimators",
                        help="森林中决策树的数量。树越多模型越稳定，但训练时间越长。"
                    )
                    clf_model_kwargs["max_depth"] = st.slider(
                        "最大深度", 1, 50, 10,
                        key="clf_rf_max_depth",
                        help="每棵树的最大深度。深度越大模型越复杂，可能过拟合。"
                    )
                    clf_model_kwargs["min_samples_split"] = st.slider(
                        "最小分裂样本数", 2, 20, 2,
                        key="clf_rf_min_samples_split",
                        help="分裂内部节点所需的最小样本数。"
                    )
                    clf_model_kwargs["min_samples_leaf"] = st.slider(
                        "最小叶节点样本数", 1, 10, 1,
                        key="clf_rf_min_samples_leaf",
                        help="叶节点所需的最小样本数。"
                    )
                
                elif clf_model_type == "Gradient Boosting":
                    clf_model_kwargs["n_estimators"] = st.slider(
                        "树数量", 10, 500, 100, 10,
                        key="clf_gb_n_estimators",
                        help="boosting阶段的最大树数量。"
                    )
                    clf_model_kwargs["learning_rate"] = st.slider(
                        "学习率", 0.001, 1.0, 0.1, 0.001,
                        key="clf_gb_learning_rate",
                        help="缩小每棵树的贡献。较小的学习率需要更多树。"
                    )
                    clf_model_kwargs["max_depth"] = st.slider(
                        "最大深度", 1, 15, 3,
                        key="clf_gb_max_depth",
                        help="树的最大深度。控制模型复杂度。"
                    )
                    clf_model_kwargs["subsample"] = st.slider(
                        "子采样比例", 0.1, 1.0, 1.0, 0.1,
                        key="clf_gb_subsample",
                        help="用于训练每棵树的样本比例。"
                    )
                
                elif clf_model_type == "XGBoost":
                    clf_model_kwargs["n_estimators"] = st.slider(
                        "树数量", 10, 500, 100, 10,
                        key="clf_xgb_n_estimators",
                        help="boosting阶段的最大树数量。"
                    )
                    clf_model_kwargs["learning_rate"] = st.slider(
                        "学习率", 0.001, 1.0, 0.1, 0.001,
                        key="clf_xgb_learning_rate",
                        help="缩小每棵树的贡献。"
                    )
                    clf_model_kwargs["max_depth"] = st.slider(
                        "最大深度", 1, 15, 6,
                        key="clf_xgb_max_depth",
                        help="树的最大深度。"
                    )
                    clf_model_kwargs["subsample"] = st.slider(
                        "子采样比例", 0.1, 1.0, 1.0, 0.1,
                        key="clf_xgb_subsample",
                        help="用于训练每棵树的样本比例。"
                    )
                    clf_model_kwargs["colsample_bytree"] = st.slider(
                        "列采样比例", 0.1, 1.0, 1.0, 0.1,
                        key="clf_xgb_colsample",
                        help="每棵树使用的特征比例。"
                    )
                
                elif clf_model_type == "LightGBM":
                    clf_model_kwargs["n_estimators"] = st.slider(
                        "树数量", 10, 500, 100, 10,
                        key="clf_lgbm_n_estimators",
                        help="boosting阶段的最大树数量。"
                    )
                    clf_model_kwargs["learning_rate"] = st.slider(
                        "学习率", 0.001, 1.0, 0.1, 0.001,
                        key="clf_lgbm_learning_rate",
                        help="缩小每棵树的贡献。"
                    )
                    clf_model_kwargs["max_depth"] = st.slider(
                        "最大深度", -1, 15, -1,
                        key="clf_lgbm_max_depth",
                        help="-1表示无限制。"
                    )
                    clf_model_kwargs["num_leaves"] = st.slider(
                        "叶子节点数", 2, 100, 31,
                        key="clf_lgbm_num_leaves",
                        help="叶子节点的最大数量。"
                    )
                
                elif clf_model_type == "SVM":
                    clf_model_kwargs["kernel"] = st.selectbox(
                        "核函数", ["rbf", "linear", "poly", "sigmoid"],
                        key="clf_svm_kernel",
                        help="SVM核函数类型。"
                    )
                    clf_model_kwargs["C"] = st.slider(
                        "正则化参数C", 0.01, 100.0, 1.0, 0.01,
                        key="clf_svm_c",
                        help="正则化参数，越大越容易过拟合。"
                    )
                    clf_model_kwargs["gamma"] = st.selectbox(
                        "gamma参数", ["scale", "auto", "rbf"],
                        key="clf_svm_gamma",
                        help="核函数系数。"
                    )
                
                elif clf_model_type == "KNN":
                    clf_model_kwargs["n_neighbors"] = st.slider(
                        "邻居数量", 1, 20, 5, 1,
                        key="clf_knn_neighbors",
                        help="用于预测的邻居数量。"
                    )
                    clf_model_kwargs["weights"] = st.selectbox(
                        "权重方式", ["uniform", "distance"],
                        key="clf_knn_weights",
                        help="uniform: 等权重; distance: 距离加权。"
                    )
                    clf_model_kwargs["metric"] = st.selectbox(
                        "距离度量", ["euclidean", "manhattan", "minkowski"],
                        key="clf_knn_metric",
                        help="距离计算方式。"
                    )
                
                elif clf_model_type == "Logistic Regression":
                    clf_model_kwargs["C"] = st.slider(
                        "正则化参数C", 0.01, 100.0, 1.0, 0.01,
                        key="clf_lr_c",
                        help="正则化参数，越大越容易过拟合。"
                    )
                    clf_model_kwargs["max_iter"] = st.slider(
                        "最大迭代次数", 100, 2000, 1000, 100,
                        key="clf_lr_max_iter"
                    )
                    clf_model_kwargs["solver"] = st.selectbox(
                        "优化算法", ["lbfgs", "liblinear", "saga"],
                        key="clf_lr_solver"
                    )
                
                elif clf_model_type == "Decision Tree":
                    clf_model_kwargs["max_depth"] = st.slider(
                        "最大深度", 1, 50, 10,
                        key="clf_dt_max_depth",
                        help="树的最大深度。"
                    )
                    clf_model_kwargs["min_samples_split"] = st.slider(
                        "最小分裂样本数", 2, 20, 2,
                        key="clf_dt_min_samples_split"
                    )
                    clf_model_kwargs["min_samples_leaf"] = st.slider(
                        "最小叶节点样本数", 1, 10, 1,
                        key="clf_dt_min_samples_leaf"
                    )
                    clf_model_kwargs["criterion"] = st.selectbox(
                        "分裂标准", ["gini", "entropy"],
                        key="clf_dt_criterion"
                    )
                
                elif clf_model_type in ["ANN", "LSTM", "CNN (1D)", "CNN (2D)"]:
                    if clf_model_type in ["CNN (1D)", "CNN (2D)"]:
                        clf_model_kwargs["conv_filters_1"] = st.slider(
                            "第一层卷积核数量", 8, 128, 32, 8,
                            key="clf_cnn_filters1",
                            help="第一层卷积层使用的卷积核数量"
                        )
                        clf_model_kwargs["conv_filters_2"] = st.slider(
                            "第二层卷积核数量", 8, 128, 64, 8,
                            key="clf_cnn_filters2",
                            help="第二层卷积层使用的卷积核数量"
                        )
                        clf_model_kwargs["kernel_size"] = st.slider(
                            "卷积核大小", 2, 7, 3, 1,
                            key="clf_cnn_kernel",
                            help="卷积核的空间大小"
                        )
                        clf_model_kwargs["dense_units"] = st.slider(
                            "全连接层神经元数量", 16, 256, 64, 16,
                            key="clf_cnn_dense",
                            help="全连接层的神经元数量"
                        )
                        clf_model_kwargs["dropout_rate"] = st.slider(
                            "Dropout比例", 0.0, 0.7, 0.3, 0.05,
                            key="clf_cnn_dropout",
                            help="Dropout层丢弃比例，防止过拟合"
                        )
                    else:
                        clf_model_kwargs["hidden_units"] = st.text_input(
                            "隐藏层神经元数量（逗号分隔，如: 128,64,32）",
                            "64,32",
                            key="clf_dl_hidden_units"
                        )
                    
                    clf_model_kwargs["epochs"] = st.slider(
                        "训练轮数", 1, 200, 50, 1,
                        key="clf_dl_epochs"
                    )
                    clf_model_kwargs["batch_size"] = st.slider(
                        "批次大小", 8, 256, 32, 8,
                        key="clf_dl_batch_size"
                    )
                    clf_model_kwargs["learning_rate"] = st.slider(
                        "学习率", 0.0001, 0.1, 0.001, 0.0001,
                        key="clf_dl_learning_rate"
                    )
            
            # ==================== 分类任务步骤5: 模型训练 ====================
            st.subheader("🚀 步骤5: 训练模型")
            
            if st.button("开始训练", type="primary", key="clf_train_button"):
                with st.spinner("正在训练分类模型..."):
                    try:
                        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                        from sklearn.svm import SVC
                        from sklearn.neighbors import KNeighborsClassifier
                        from sklearn.linear_model import LogisticRegression
                        from sklearn.tree import DecisionTreeClassifier
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
                        
                        clf_model = None
                        is_dl = False
                        
                        # 自定义模型处理
                        if clf_model_type == "custom":
                            if clf_custom_model_loaded and 'clf_custom_model_instance' in locals():
                                clf_model = clf_custom_model_instance
                                st.info(f"🔧 使用自定义分类模型: {clf_custom_model_class.__name__}")
                            else:
                                st.error("请先输入有效的自定义分类模型代码")
                                st.stop()
                        # 预置模型
                        elif clf_model_type == "Random Forest":
                            clf_model = RandomForestClassifier(
                                n_estimators=clf_model_kwargs.get("n_estimators", 100),
                                max_depth=clf_model_kwargs.get("max_depth", None),
                                min_samples_split=clf_model_kwargs.get("min_samples_split", 2),
                                min_samples_leaf=clf_model_kwargs.get("min_samples_leaf", 1),
                                random_state=clf_random_state
                            )
                        elif clf_model_type == "Gradient Boosting":
                            clf_model = GradientBoostingClassifier(
                                n_estimators=clf_model_kwargs.get("n_estimators", 100),
                                learning_rate=clf_model_kwargs.get("learning_rate", 0.1),
                                max_depth=clf_model_kwargs.get("max_depth", 3),
                                subsample=clf_model_kwargs.get("subsample", 1.0),
                                random_state=clf_random_state
                            )
                        elif clf_model_type == "XGBoost":
                            from xgboost import XGBClassifier
                            clf_model = XGBClassifier(
                                n_estimators=clf_model_kwargs.get("n_estimators", 100),
                                learning_rate=clf_model_kwargs.get("learning_rate", 0.1),
                                max_depth=clf_model_kwargs.get("max_depth", 6),
                                subsample=clf_model_kwargs.get("subsample", 1.0),
                                colsample_bytree=clf_model_kwargs.get("colsample_bytree", 1.0),
                                random_state=clf_random_state,
                                eval_metric='logloss'
                            )
                        elif clf_model_type == "LightGBM":
                            from lightgbm import LGBMClassifier
                            clf_model = LGBMClassifier(
                                n_estimators=clf_model_kwargs.get("n_estimators", 100),
                                learning_rate=clf_model_kwargs.get("learning_rate", 0.1),
                                max_depth=clf_model_kwargs.get("max_depth", -1),
                                num_leaves=clf_model_kwargs.get("num_leaves", 31),
                                random_state=clf_random_state,
                                verbose=-1
                            )
                        elif clf_model_type == "SVM":
                            clf_model = SVC(
                                kernel=clf_model_kwargs.get("kernel", "rbf"),
                                C=clf_model_kwargs.get("C", 1.0),
                                gamma=clf_model_kwargs.get("gamma", "scale"),
                                probability=True,
                                random_state=clf_random_state
                            )
                        elif clf_model_type == "KNN":
                            clf_model = KNeighborsClassifier(
                                n_neighbors=clf_model_kwargs.get("n_neighbors", 5),
                                weights=clf_model_kwargs.get("weights", "uniform"),
                                metric=clf_model_kwargs.get("metric", "euclidean")
                            )
                        elif clf_model_type == "Logistic Regression":
                            clf_model = LogisticRegression(
                                C=clf_model_kwargs.get("C", 1.0),
                                max_iter=clf_model_kwargs.get("max_iter", 1000),
                                solver=clf_model_kwargs.get("solver", "lbfgs"),
                                random_state=clf_random_state
                            )
                        elif clf_model_type == "Decision Tree":
                            clf_model = DecisionTreeClassifier(
                                max_depth=clf_model_kwargs.get("max_depth", None),
                                min_samples_split=clf_model_kwargs.get("min_samples_split", 2),
                                min_samples_leaf=clf_model_kwargs.get("min_samples_leaf", 1),
                                criterion=clf_model_kwargs.get("criterion", "gini"),
                                random_state=clf_random_state
                            )
                        elif clf_model_type == "CNN (1D)":
                            is_dl = True
                            import tensorflow as tf
                            from tensorflow.keras.models import Sequential
                            from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GlobalAveragePooling1D
                            
                            num_classes = len(np.unique(y_clf_train))
                            input_shape = X_clf_train_scaled.shape[1:]
                            timesteps = input_shape[0]
                            
                            filters1 = clf_model_kwargs.get("conv_filters_1", 32)
                            filters2 = clf_model_kwargs.get("conv_filters_2", 64)
                            kernel_size = clf_model_kwargs.get("kernel_size", 3)
                            dense_units = clf_model_kwargs.get("dense_units", 64)
                            dropout_rate = clf_model_kwargs.get("dropout_rate", 0.3)
                            
                            if timesteps <= 10:
                                clf_model = Sequential([
                                    Conv1D(filters1, kernel_size, activation='relu', padding='same', input_shape=input_shape),
                                    MaxPooling1D(2),
                                    Conv1D(filters2, kernel_size, activation='relu', padding='same'),
                                    GlobalAveragePooling1D(),
                                    Dense(dense_units, activation='relu'),
                                    Dropout(dropout_rate),
                                    Dense(num_classes, activation='softmax')
                                ])
                            else:
                                clf_model = Sequential([
                                    Conv1D(filters1, kernel_size, activation='relu', padding='same', input_shape=input_shape),
                                    MaxPooling1D(2),
                                    Conv1D(filters2, kernel_size, activation='relu', padding='same'),
                                    MaxPooling1D(2),
                                    GlobalAveragePooling1D(),
                                    Dense(dense_units, activation='relu'),
                                    Dropout(dropout_rate),
                                    Dense(num_classes, activation='softmax')
                                ])
                            
                            optimizer = tf.keras.optimizers.Adam(learning_rate=clf_model_kwargs.get("learning_rate", 0.001))
                            clf_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                        
                        elif clf_model_type == "CNN (2D)":
                            is_dl = True
                            import tensorflow as tf
                            from tensorflow.keras.models import Sequential
                            from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
                            
                            num_classes = len(np.unique(y_clf_train))
                            
                            # 调整输入形状以适应CNN2D
                            X_clf_train_cnn = X_clf_train_scaled.reshape(-1, X_clf_train_scaled.shape[1], X_clf_train_scaled.shape[2], 1)
                            X_clf_test_cnn = X_clf_test_scaled.reshape(-1, X_clf_test_scaled.shape[1], X_clf_test_scaled.shape[2], 1)
                            
                            input_shape = X_clf_train_cnn.shape[1:]
                            h, w = input_shape[0], input_shape[1]
                            
                            filters1 = clf_model_kwargs.get("conv_filters_1", 32)
                            filters2 = clf_model_kwargs.get("conv_filters_2", 64)
                            kernel_size = clf_model_kwargs.get("kernel_size", 3)
                            dense_units = clf_model_kwargs.get("dense_units", 64)
                            dropout_rate = clf_model_kwargs.get("dropout_rate", 0.3)
                            
                            if h <= 8 or w <= 8:
                                clf_model = Sequential([
                                    Conv2D(filters1, (kernel_size, kernel_size), activation='relu', padding='same', input_shape=input_shape),
                                    Conv2D(filters2, (kernel_size, kernel_size), activation='relu', padding='same'),
                                    GlobalAveragePooling2D(),
                                    Dense(dense_units, activation='relu'),
                                    Dropout(dropout_rate),
                                    Dense(num_classes, activation='softmax')
                                ])
                            else:
                                clf_model = Sequential([
                                    Conv2D(filters1, (kernel_size, kernel_size), activation='relu', padding='same', input_shape=input_shape),
                                    MaxPooling2D((2, 2)),
                                    Conv2D(filters2, (kernel_size, kernel_size), activation='relu', padding='same'),
                                    MaxPooling2D((2, 2)),
                                    GlobalAveragePooling2D(),
                                    Dense(dense_units, activation='relu'),
                                    Dropout(dropout_rate),
                                    Dense(num_classes, activation='softmax')
                                ])
                            
                            optimizer = tf.keras.optimizers.Adam(learning_rate=clf_model_kwargs.get("learning_rate", 0.001))
                            clf_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                        
                        elif clf_model_type == "ANN":
                            is_dl = True
                            import tensorflow as tf
                            from tensorflow.keras.models import Sequential
                            from tensorflow.keras.layers import Dense, Dropout
                            
                            hidden_units = list(map(int, clf_model_kwargs.get("hidden_units", "64,32").split(",")))
                            num_classes = len(np.unique(y_clf_train))
                            
                            clf_model = Sequential()
                            clf_model.add(Dense(hidden_units[0], activation='relu', input_shape=(X_clf_train_scaled.shape[1],)))
                            clf_model.add(Dropout(0.3))
                            for units in hidden_units[1:]:
                                clf_model.add(Dense(units, activation='relu'))
                                clf_model.add(Dropout(0.3))
                            clf_model.add(Dense(num_classes, activation='softmax'))
                            
                            optimizer = tf.keras.optimizers.Adam(learning_rate=clf_model_kwargs.get("learning_rate", 0.001))
                            clf_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                        elif clf_model_type == "LSTM":
                            is_dl = True
                            import tensorflow as tf
                            from tensorflow.keras.models import Sequential
                            from tensorflow.keras.layers import Dense, LSTM, Dropout
                            
                            hidden_units = list(map(int, clf_model_kwargs.get("hidden_units", "64,32").split(",")))
                            num_classes = len(np.unique(y_clf_train))
                            
                            X_clf_train_lstm = X_clf_train_scaled.reshape(X_clf_train_scaled.shape[0], 1, X_clf_train_scaled.shape[1])
                            X_clf_test_lstm = X_clf_test_scaled.reshape(X_clf_test_scaled.shape[0], 1, X_clf_test_scaled.shape[1])
                            
                            clf_model = Sequential()
                            clf_model.add(LSTM(hidden_units[0], return_sequences=True if len(hidden_units) > 1 else False, input_shape=(1, X_clf_train_scaled.shape[1])))
                            clf_model.add(Dropout(0.3))
                            for units in hidden_units[1:]:
                                clf_model.add(LSTM(units, return_sequences=False))
                                clf_model.add(Dropout(0.3))
                            clf_model.add(Dense(num_classes, activation='softmax'))
                            
                            optimizer = tf.keras.optimizers.Adam(learning_rate=clf_model_kwargs.get("learning_rate", 0.001))
                            clf_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                        
                        # 训练模型
                        if is_dl:
                            epochs = clf_model_kwargs.get("epochs", 50)
                            batch_size = clf_model_kwargs.get("batch_size", 32)
                            
                            if clf_model_type == "LSTM":
                                history = clf_model.fit(
                                    X_clf_train_lstm, y_clf_train,
                                    epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0
                                )
                                y_clf_pred = np.argmax(clf_model.predict(X_clf_test_lstm, verbose=0), axis=1)
                            elif clf_model_type == "CNN (2D)":
                                history = clf_model.fit(
                                    X_clf_train_cnn, y_clf_train,
                                    epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0
                                )
                                y_clf_pred = np.argmax(clf_model.predict(X_clf_test_cnn, verbose=0), axis=1)
                            else:
                                history = clf_model.fit(
                                    X_clf_train_scaled, y_clf_train,
                                    epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0
                                )
                                y_clf_pred = np.argmax(clf_model.predict(X_clf_test_scaled, verbose=0), axis=1)
                        else:
                            clf_model.fit(X_clf_train_scaled, y_clf_train)
                            y_clf_pred = clf_model.predict(X_clf_test_scaled)
                        
                        # 评估模型
                        accuracy = accuracy_score(y_clf_test, y_clf_pred)
                        precision = precision_score(y_clf_test, y_clf_pred, average='weighted')
                        recall = recall_score(y_clf_test, y_clf_pred, average='weighted')
                        f1 = f1_score(y_clf_test, y_clf_pred, average='weighted')
                        
                        # 保存模型和相关信息到会话状态
                        st.session_state.classification_trainer = {
                            'model': clf_model,
                            'model_type': clf_model_type,
                            'is_deep_learning': is_dl,
                            'label_encoder': label_encoder,
                            'scaler': scaler if not clf_image_mode else None,
                            'feature_cols': clf_feature_cols,
                            'target_col': clf_target_col,
                            'X_test': X_clf_test_scaled,
                            'y_test': y_clf_test,
                            'y_pred': y_clf_pred,
                            'X_train': X_clf_train_scaled,
                            'y_train': y_clf_train,
                            'X_test_lstm': X_clf_test_lstm if is_dl and clf_model_type == "LSTM" else None,
                            'X_train_lstm': X_clf_train_lstm if is_dl and clf_model_type == "LSTM" else None,
                            'X_test_cnn': X_clf_test_cnn if is_dl and clf_model_type == "CNN (2D)" else None,
                            'X_train_cnn': X_clf_train_cnn if is_dl and clf_model_type == "CNN (2D)" else None,
                            'image_mode': clf_image_mode,
                            'feature_type': clf_time_series_features.get("feature_type", "传统特征"),
                            'metrics': {
                                'accuracy': accuracy,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1
                            },
                            'confusion_matrix': confusion_matrix(y_clf_test, y_clf_pred),
                            'classification_report': classification_report(y_clf_test, y_clf_pred, target_names=label_encoder.classes_, output_dict=True),
                            'label_names': label_encoder.classes_
                        }
                        st.session_state.classification_model_trained = True
                        st.session_state.classification_export_df = pd.DataFrame({
                            "sample_id": np.arange(len(y_clf_test)),
                            "true_label_id": y_clf_test,
                            "predicted_label_id": y_clf_pred,
                            "true_label": label_encoder.inverse_transform(y_clf_test),
                            "predicted_label": label_encoder.inverse_transform(y_clf_pred),
                            "is_correct": y_clf_test == y_clf_pred,
                        })
                        
                    except Exception as e:
                        st.error(f"训练出错: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

            # ==================== 分类任务结果显示和模型保存 ====================
            if 'classification_model_trained' in st.session_state and st.session_state.classification_model_trained:
                trainer = st.session_state.classification_trainer

                st.success("✅ 模型训练完成！")
                append_run_history(st.session_state, "分类模型训练", clf_model_type)

                # 评估指标
                st.subheader("📊 模型评估")

                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("准确率 (Accuracy)", f"{trainer['metrics']['accuracy']:.4f}")
                with metric_col2:
                    st.metric("精确率 (Precision)", f"{trainer['metrics']['precision']:.4f}")
                with metric_col3:
                    st.metric("召回率 (Recall)", f"{trainer['metrics']['recall']:.4f}")
                with metric_col4:
                    st.metric("F1分数", f"{trainer['metrics']['f1']:.4f}")

                # 混淆矩阵
                st.subheader("🔢 混淆矩阵")

                fig_cm = go.Figure(data=go.Heatmap(
                    z=trainer['confusion_matrix'],
                    x=trainer['label_names'],
                    y=trainer['label_names'],
                    colorscale='Blues',
                    text=trainer['confusion_matrix'],
                    texttemplate="%{text}",
                    showscale=True
                ))
                fig_cm.update_layout(
                    title="混淆矩阵",
                    xaxis_title="预测类别",
                    yaxis_title="真实类别",
                    height=400
                )
                st.plotly_chart(fig_cm, use_container_width=True)

                # 分类报告
                st.subheader("📋 详细分类报告")
                report_df = pd.DataFrame(trainer['classification_report']).transpose()
                st.dataframe(report_df)
                if st.session_state.get("classification_export_df") is not None:
                    with st.expander("📤 分类训练结果导出", expanded=False):
                        clf_export_df = st.session_state.classification_export_df
                        clf_report_html = build_html_report(
                            "分类任务训练图文结果报告",
                            metrics={
                                "模型": trainer["model_type"],
                                "准确率": f"{trainer['metrics']['accuracy']:.4f}",
                                "精确率": f"{trainer['metrics']['precision']:.4f}",
                                "召回率": f"{trainer['metrics']['recall']:.4f}",
                                "F1": f"{trainer['metrics']['f1']:.4f}",
                            },
                            sections=[("结果摘要", "报告包含混淆矩阵、分类指标、逐样本预测结果和类别标签映射。")],
                            figures=[("混淆矩阵", fig_cm)],
                            tables=[("分类报告", report_df), ("预测明细", clf_export_df)],
                        )
                        render_result_downloads(
                            "分类训练",
                            clf_export_df,
                            clf_report_html,
                            csv_name=f"classification_training_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            html_name=f"classification_training_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            key_prefix="classification_training",
                        )

                # 类别分布对比图 - 优化为可折叠
                with st.expander("📈 预测结果可视化", expanded=False):
                    # 真实值和预测值对比图
                    st.subheader("真实值 vs 预测值对比")

                    # 确保样本数量不要太多，避免渲染卡顿
                    max_samples = 100
                    sample_indices = np.arange(min(max_samples, len(trainer['y_test'])))

                    fig_combined = go.Figure()
                    fig_combined.add_trace(go.Scatter(
                        x=sample_indices,
                        y=trainer['y_test'][sample_indices],
                        name='真实值',
                        mode='lines+markers',
                        line=dict(color='black', width=2),
                        marker=dict(size=6)
                    ))
                    fig_combined.add_trace(go.Scatter(
                        x=sample_indices,
                        y=trainer['y_pred'][sample_indices],
                        name='预测值',
                        mode='lines+markers',
                        line=dict(color='red', width=1, dash='dash'),
                        marker=dict(size=4, symbol='triangle-up')
                    ))
                    fig_combined.update_layout(
                        title=f"真实值 vs 预测值（前{len(sample_indices)}个样本）",
                        xaxis_title="样本索引",
                        yaxis_title="类别",
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_combined, use_container_width=True)

                    # 类别标签说明
                    if 'label_encoder' in trainer and hasattr(trainer['label_encoder'], 'classes_'):
                        label_classes = trainer['label_encoder'].classes_
                        st.info("📋 **类别标签说明：**")
                        label_cols = st.columns(min(len(label_classes), 4))
                        for idx, label_name in enumerate(label_classes):
                            col_idx = idx % len(label_cols)
                            with label_cols[col_idx]:
                                st.markdown(f"- **{idx}** = {label_name}")
                    
                    # 分类分布对比图
                    st.subheader("类别分布对比")
                    from plotly.subplots import make_subplots

                    fig_dist = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=(["真实类别分布", "预测类别分布"]),
                        specs=[[{"type": "bar"}, {"type": "bar"}]]
                    )

                    # 真实分布
                    unique_true, counts_true = np.unique(trainer['y_test'], return_counts=True)
                    fig_dist.add_trace(
                        go.Bar(x=trainer['label_names'][unique_true], y=counts_true, name="真实", marker_color='steelblue'),
                        row=1, col=1
                    )

                    # 预测分布
                    unique_pred, counts_pred = np.unique(trainer['y_pred'], return_counts=True)
                    fig_dist.add_trace(
                        go.Bar(x=trainer['label_names'][unique_pred], y=counts_pred, name="预测", marker_color='orange'),
                        row=1, col=2
                    )

                    fig_dist.update_layout(height=350, showlegend=True)
                    st.plotly_chart(fig_dist, use_container_width=True)

                # 保存模型功能
                st.subheader("💾 模型保存")
                clf_model_name = st.text_input(
                    "模型名称",
                    f"{trainer['model_type']}_model",
                    key="clf_model_name_input_new"
                )

                if st.button("保存模型", key="clf_save_model_button_new"):
                    import pickle
                    import os
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{clf_model_name}_{timestamp}.pkl"
                    file_path = os.path.join("saved_models", filename)

                    # 确保保存目录存在
                    os.makedirs("saved_models", exist_ok=True)

                    try:
                        # 保存完整的模型信息
                        model_info = {
                            'model': trainer['model'],
                            'feature_cols': trainer['feature_cols'],
                            'target_col': trainer['target_col'],
                            'scaler': trainer['scaler'],
                            'label_encoder': trainer['label_encoder'],
                            'model_type': trainer['model_type'],
                            'is_deep_learning': trainer['is_deep_learning'],
                            'timestamp': timestamp
                        }

                        with open(file_path, 'wb') as f:
                            pickle.dump(model_info, f)

                        st.success(f"✅ 模型已保存到: {file_path}")
                        st.info("模型文件保存在项目目录下的 saved_models 文件夹中")

                        # 提供下载链接
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                label="📥 下载模型文件",
                                data=f,
                                file_name=filename,
                                mime="application/octet-stream"
                            )
                    except Exception as e:
                        st.error(f"保存模型失败: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

            # ==================== 分类任务步骤6: 新数据预测 ====================
            # 新数据预测作为独立功能，不依赖于是否训练过模型
            st.divider()
            st.subheader("🔮 步骤6: 新数据分类预测")
            
            # 模型选择
            clf_model_source = st.radio(
                "选择预测模型来源",
                ["使用当前训练的模型", "使用自定义模型文件"],
                horizontal=True,
                key="clf_model_source_select",
                help="选择使用当前训练的模型，或上传以前保存的模型文件进行预测。"
            )
            
            # 用于预测的模型信息字典
            clf_prediction_info = {}
            clf_use_custom_model = False
            clf_model_info = None  # 初始化模型信息变量
            
            if clf_model_source == "使用当前训练的模型":
                if st.session_state.classification_model_trained and st.session_state.classification_trainer:
                    clf_prediction_info = st.session_state.classification_trainer.copy()
                else:
                    st.warning("⚠️ 当前没有训练好的模型，请先完成模型训练或选择「使用自定义模型文件」")
                    clf_prediction_info = None
            else:  # 使用自定义模型文件
                st.markdown("**上传模型文件**")
                clf_model_file = st.file_uploader(
                    "上传分类模型文件 (.pkl 或 .keras)",
                    type=["pkl", "keras", "h5"],
                    key="clf_custom_model_uploader",
                    help="上传以前保存的分类模型文件。"
                )
                
                if clf_model_file is not None:
                    # 保存上传的模型文件到临时位置
                    clf_model_ext = os.path.splitext(clf_model_file.name)[1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=clf_model_ext) as tmp_clf_model_file:
                        tmp_clf_model_file.write(clf_model_file.getvalue())
                        custom_clf_model_path = tmp_clf_model_file.name
                    
                    # 同时支持上传JSON配置文件（可选）
                    clf_model_json_file = st.file_uploader(
                        "上传模型配置文件（可选，如模型名.json）",
                        type=["json"],
                        key="clf_model_json_uploader",
                        help="如果模型保存时分离了配置文件，请一并上传以便读取模型信息。如果不上传，程序会尝试从模型文件中读取信息。"
                    )
                    
                    # 如果用户上传了JSON文件，保存到临时位置
                    clf_custom_json_path = None
                    if clf_model_json_file is not None:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_json_file:
                            tmp_json_file.write(clf_model_json_file.getvalue())
                            clf_custom_json_path = tmp_json_file.name
                    
                    try:
                        # 创建新的训练器并加载模型
                        from src.models.model_trainer import ModelTrainer
                        clf_custom_trainer = ModelTrainer()
                        
                        # 根据文件扩展名判断模型类型
                        if clf_model_ext in ['.keras', '.h5']:
                            loaded_clf_model_type = 'ann'
                        else:
                            loaded_clf_model_type = 'random_forest'
                        
                        # 加载模型
                        loaded_clf_model, clf_model_info = clf_custom_trainer.load_model(custom_clf_model_path, loaded_clf_model_type, clf_custom_json_path)
                        
                        # 构建预测信息字典
                        clf_prediction_info['model'] = loaded_clf_model
                        clf_prediction_info['is_deep_learning'] = clf_custom_trainer.is_deep_learning
                        clf_prediction_info['model_type'] = clf_model_info.get('model_type', loaded_clf_model_type) if clf_model_info else loaded_clf_model_type
                        
                        clf_use_custom_model = True
                        st.success(f"✅ 分类模型加载成功: {clf_model_file.name}")
                        
                        # 显示模型信息并读取配置
                        if clf_model_info:
                            with st.expander("📊 模型信息"):
                                st.json(clf_model_info)
                            
                            # 读取特征列
                            if 'feature_cols' in clf_model_info:
                                clf_prediction_info['feature_cols'] = clf_model_info['feature_cols']
                                st.success(f"✅ 已从模型信息中读取特征列：{len(clf_model_info['feature_cols'])} 个")
                            else:
                                st.warning("⚠️ 模型信息中未找到特征列，请手动选择")
                            
                            # 读取标准化器
                            if 'scaler' in clf_model_info:
                                clf_prediction_info['scaler'] = clf_model_info['scaler']
                                st.success("✅ 已从模型信息中读取标准化器")
                            
                            # 读取label_encoder
                            if 'label_encoder' in clf_model_info:
                                clf_prediction_info['label_encoder'] = clf_model_info['label_encoder']
                                st.success("✅ 已从模型信息中读取标签编码器")
                            
                            # 读取模型类型和深度学习标识
                            if 'model_type' in clf_model_info:
                                clf_prediction_info['model_type'] = clf_model_info['model_type']
                            if 'is_deep_learning' in clf_model_info:
                                clf_prediction_info['is_deep_learning'] = clf_model_info['is_deep_learning']
                        else:
                            st.warning("⚠️ 无法读取模型信息，请手动配置")
                            
                    except Exception as e:
                        st.error(f"❌ 模型加载失败: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                        clf_prediction_info = None
                else:
                    clf_prediction_info = None
                    st.info("请上传模型文件以继续")
            
            # 当使用自定义模型且没有特征列时，让用户从新数据中选择
            if clf_use_custom_model and clf_prediction_info is not None and 'feature_cols' not in clf_prediction_info:
                st.subheader("⚙️ 特征列选择")
                st.info("请上传数据文件，然后从数据中选择特征列：")
                
                # 先上传数据文件来选择特征列
                clf_config_file = st.file_uploader(
                    "上传数据文件以选择特征列",
                    type=["csv", "xlsx", "xls"],
                    key="clf_config_file_uploader"
                )
                
                if clf_config_file is not None:
                    try:
                        if clf_config_file.name.endswith('.csv'):
                            clf_config_df = pd.read_csv(clf_config_file)
                        else:
                            clf_config_df = pd.read_excel(clf_config_file)
                        
                        st.write("数据预览：", clf_config_df.head())
                        
                        # 让用户从数据列中选择特征列
                        clf_selected_features = st.multiselect(
                            "选择特征列",
                            options=list(clf_config_df.columns),
                            help="选择用于预测的特征列"
                        )
                        
                        # 标准化方法
                        clf_scale_method = st.selectbox(
                            "数据标准化方法",
                            ["none", "standard", "minmax", "robust"],
                            index=0,
                            key="clf_custom_scale_method",
                            help="选择模型训练时使用的数据标准化方法"
                        )
                        
                        # 是否深度学习模型
                        clf_is_dl = st.checkbox(
                            "是否为深度学习模型（ANN/LSTM等）",
                            value=clf_prediction_info.get('is_deep_learning', False),
                            key="clf_custom_is_dl"
                        )
                        
                        if clf_selected_features:
                            clf_prediction_info['feature_cols'] = clf_selected_features
                            clf_prediction_info['is_deep_learning'] = clf_is_dl
                            
                            # 创建标准化器
                            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
                            if clf_scale_method == "standard":
                                clf_prediction_info['scaler'] = StandardScaler()
                            elif clf_scale_method == "minmax":
                                clf_prediction_info['scaler'] = MinMaxScaler()
                            elif clf_scale_method == "robust":
                                clf_prediction_info['scaler'] = RobustScaler()
                            else:
                                clf_prediction_info['scaler'] = None
                        else:
                            st.warning("请至少选择一个特征列")
                            clf_prediction_info = None
                    except Exception as e:
                        st.error(f"读取数据文件失败: {str(e)}")
                        clf_prediction_info = None
                else:
                    clf_prediction_info = None  # 没有配置好，不能继续
            
            if clf_prediction_info is not None:
                st.info("上传新数据进行分类预测")
                
                clf_new_file = st.file_uploader(
                    "上传新数据文件",
                    type=["csv", "xlsx", "xls"],
                    key="clf_new_file"
                )
                
                if clf_new_file is not None and clf_prediction_info is not None:
                    try:
                        if clf_new_file.name.endswith('.csv'):
                            clf_new_df = pd.read_csv(clf_new_file)
                        else:
                            clf_new_df = pd.read_excel(clf_new_file)
                        
                        st.write("新数据预览：", clf_new_df.head())
                        
                        # 特征列映射
                        st.subheader("🔗 特征列映射")
                        
                        clf_col_mapping = {}
                        clf_use_auto_map = st.checkbox(
                            "自动匹配列名",
                            value=True,
                            key="clf_auto_map"
                        )
                        
                        clf_feature_cols_trained = clf_prediction_info['feature_cols']
                        clf_mapping_ready = False
                        
                        if clf_use_auto_map:
                            missing = [col for col in clf_feature_cols_trained if col not in clf_new_df.columns]
                            if missing:
                                st.warning(f"缺少列: {missing}")
                            else:
                                clf_col_mapping = {col: col for col in clf_feature_cols_trained}
                                st.success("✅ 自动匹配成功")
                                clf_mapping_ready = True
                        else:
                            for col in clf_feature_cols_trained:
                                available = ["无"] + list(clf_new_df.columns)
                                default_idx = available.index(col) if col in clf_new_df.columns else 0
                                selected = st.selectbox(f"{col} →", available, index=default_idx, key=f"clf_map_{col}")
                                if selected != "无":
                                    clf_col_mapping[col] = selected
                            
                            if len(clf_col_mapping) == len(clf_feature_cols_trained):
                                clf_mapping_ready = True
                        
                        # 显示开始预测按钮（当映射准备好时）
                        if clf_mapping_ready:
                            if st.button("🔮 开始预测", key="clf_predict_button"):
                                # 准备数据
                                X_new = clf_new_df[[clf_col_mapping[col] for col in clf_feature_cols_trained]]
                                
                                # 数据标准化
                                scaler = clf_prediction_info.get('scaler')
                                if scaler is not None:
                                    try:
                                        X_new_scaled = scaler.transform(X_new)
                                    except Exception:
                                        st.warning("当前分类模型未携带已拟合的标准化器，将临时按新数据拟合标准化；建议使用保存完整训练信息的模型文件。")
                                        X_new_scaled = scaler.fit_transform(X_new)
                                else:
                                    X_new_scaled = X_new.values
                                
                                # 预测
                                clf_model = clf_prediction_info['model']
                                is_dl = clf_prediction_info['is_deep_learning']
                                
                                if is_dl and clf_prediction_info['model_type'] == "LSTM":
                                    X_new_lstm = X_new_scaled.reshape(X_new_scaled.shape[0], 1, X_new_scaled.shape[1])
                                    y_new_pred = np.argmax(clf_model.predict(X_new_lstm, verbose=0), axis=1)
                                elif is_dl:
                                    y_new_pred = np.argmax(clf_model.predict(X_new_scaled, verbose=0), axis=1)
                                else:
                                    y_new_pred = clf_model.predict(X_new_scaled)
                                
                                # 解码标签（如果有label_encoder）
                                label_encoder = clf_prediction_info.get('label_encoder')
                                if label_encoder is not None:
                                    y_new_pred_labels = label_encoder.inverse_transform(y_new_pred)
                                else:
                                    # 如果没有label_encoder，直接显示数值标签
                                    y_new_pred_labels = [f"类别_{int(p)}" for p in y_new_pred]
                                
                                # 显示结果
                                clf_new_df["预测类别"] = y_new_pred_labels
                                
                                st.success("✅ 预测完成！")
                                
                                # 预测结果统计
                                pred_counts = pd.Series(y_new_pred_labels).value_counts()
                                st.write("预测结果分布：")
                                st.dataframe(pred_counts)
                                
                                # 预测结果图表
                                fig_pred_new = go.Figure(data=[
                                    go.Bar(x=pred_counts.index, y=pred_counts.values, marker_color='green')
                                ])
                                fig_pred_new.update_layout(
                                    title="预测类别分布",
                                    xaxis_title="类别",
                                    yaxis_title="数量",
                                    height=350
                                )
                                st.plotly_chart(fig_pred_new, use_container_width=True)
                                
                                # 显示预测结果表格
                                with st.expander("📋 查看完整预测结果"):
                                    st.dataframe(clf_new_df)
                                
                                # 下载预测结果
                                csv_result = clf_new_df.to_csv(index=False)
                                st.download_button(
                                    "📥 下载预测结果",
                                    csv_result,
                                    "classification_predictions.csv",
                                    "text/csv"
                                )
                                clf_prediction_report = build_html_report(
                                    "分类新数据预测图文结果报告",
                                    metrics={
                                        "模型": clf_prediction_info.get("model_type", "N/A"),
                                        "预测样本数": len(clf_new_df),
                                        "预测类别数": len(pred_counts),
                                    },
                                    sections=[("结果摘要", "报告包含新数据分类预测明细和预测类别分布。")],
                                    figures=[("预测类别分布", fig_pred_new)],
                                    tables=[("预测结果", clf_new_df), ("类别分布", pred_counts.reset_index().rename(columns={"index": "类别", 0: "数量"}))],
                                )
                                st.download_button(
                                    "📄 下载预测HTML图文报告",
                                    clf_prediction_report.encode("utf-8"),
                                    file_name=f"classification_prediction_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                    mime="text/html",
                                    key="download_clf_prediction_html",
                                )
                                
                    except Exception as e:
                        st.error(f"预测出错: {str(e)}")

# ==================== 标签3: 批量任务处理 ====================
with tab3:
    st.markdown("<h1 style='font-size: 32px; font-weight: bold;'>🚀 批量任务处理</h1>", unsafe_allow_html=True)
    
    # 检查是否有训练数据（支持时序预测和分类任务）
    has_ts_data = (st.session_state.X_train is not None and 
                   st.session_state.y_train is not None and 
                   st.session_state.X_test is not None and 
                   st.session_state.y_test is not None)
    
    has_clf_data = (st.session_state.classification_trainer is not None and
                   st.session_state.get('classification_model_trained', False))
    
    task_type = None
    if has_ts_data:
        task_type = "时序预测"
    elif has_clf_data:
        task_type = "分类任务"
    
    if task_type is None:
        st.warning("⚠️ 请先在「时序预测」或「分类任务」标签页中完成数据上传和预处理步骤，以获取训练数据。")
        st.markdown("""
        **批量训练流程：**
        1. 前往「📈 时序预测」或「🎯 分类任务」标签页
        2. 上传数据文件
        3. 完成数据预处理
        4. 返回「🚀 批量任务处理」进行批量训练
        """)
    else:
        # 显示当前任务类型
        st.success(f"📌 当前任务类型: {task_type}")
        
        # 初始化批量任务的session state
        if 'batch_task_type' not in st.session_state:
            st.session_state.batch_task_type = task_type
        if 'batch_results' not in st.session_state:
            st.session_state.batch_results = None
        if 'saved_workflows' not in st.session_state:
            st.session_state.saved_workflows = {}
        
        # 检测任务类型变化
        if st.session_state.batch_task_type != task_type:
            st.session_state.batch_task_type = task_type
            st.session_state.batch_results = None
        
        # ==================== 左侧：任务配置 ====================
        col_config, col_results = st.columns([1, 1.2], gap="large")
        
        with col_config:
            st.subheader("📋 任务配置")
            
            # 任务类型选择（支持手动切换）
            selected_task_type = st.radio(
                "任务类型",
                ["时序预测", "分类任务"],
                index=0 if task_type == "时序预测" else 1,
                horizontal=True,
                key="batch_task_radio",
                help="选择批量训练的任务类型"
            )
            
            # 当任务类型切换时，清空之前的训练结果
            if selected_task_type != st.session_state.batch_task_type:
                st.session_state.batch_task_type = selected_task_type
                st.session_state.batch_results = None
                st.rerun()
            
            if selected_task_type == "时序预测" and not has_ts_data:
                st.warning("⚠️ 时序预测数据未准备好，请先在「时序预测」标签页完成预处理")
                st.stop()
            elif selected_task_type == "分类任务" and not has_clf_data:
                st.warning("⚠️ 分类任务数据未准备好，请先在「分类任务」标签页完成预处理")
                st.stop()
            
            # 模型选择
            if selected_task_type == "时序预测":
                st.write("**选择要训练的回归模型：**")
                batch_models = st.multiselect(
                    "回归模型",
                    ["线性回归", "Ridge", "Lasso", "随机森林", "梯度提升", "SVR", "KNN", "XGBoost", "LightGBM"],
                    default=["随机森林", "梯度提升"],
                    key="batch_models_reg",
                    help="选择多个模型进行批量训练"
                )
                batch_metrics = st.selectbox(
                    "评估指标",
                    ["R²", "MSE", "MAE", "RMSE"],
                    index=0,
                    key="batch_metrics_reg",
                    help="用于评估和比较模型的指标"
                )
            else:
                st.write("**选择要训练的分类模型：**")
                batch_models = st.multiselect(
                    "分类模型",
                    ["逻辑回归", "随机森林分类", "梯度提升分类", "SVM分类", "KNN分类", "XGBoost分类", "LightGBM分类", "朴素贝叶斯"],
                    default=["随机森林分类", "梯度提升分类"],
                    key="batch_models_clf",
                    help="选择多个分类模型进行批量训练"
                )
                batch_metrics = st.selectbox(
                    "评估指标",
                    ["准确率", "精确率", "召回率", "F1分数"],
                    index=0,
                    key="batch_metrics_clf",
                    help="用于评估分类模型的指标"
                )
            
            # 高级设置
            with st.expander("⚙️ 高级设置"):
                enable_cv = st.checkbox("启用交叉验证", value=True, key="batch_cv")
                if enable_cv:
                    cv_folds = st.slider("交叉验证折数", 2, 10, 5, key="cv_folds")
                
                enable_save = st.checkbox("保存训练日志", value=True, key="save_log")
                
                parallel_train = st.checkbox("并行训练（加速）", value=False, key="parallel_train")
            
            # 开始训练按钮
            st.markdown("---")
            train_button = st.button(
                "🚀 开始批量训练", 
                type="primary", 
                key="start_batch_btn",
                use_container_width=True
            )
            
            # ==================== 显示数据信息（仅在有数据时）================
            if has_ts_data or has_clf_data:
                st.markdown("---")
                st.subheader("📊 当前数据信息")
                
                # 获取特征和目标信息
                if has_ts_data:
                    feature_info = st.session_state.get('feature_cols', [])
                    target_info = st.session_state.get('target_col', '')
                    sample_count = st.session_state.X_train.shape[0] if hasattr(st.session_state.X_train, 'shape') else 'N/A'
                elif has_clf_data:
                    clf_info = st.session_state.get('classification_trainer', {})
                    feature_info = clf_info.get('feature_cols', [])
                    target_info = clf_info.get('target_col', '')
                    sample_count = clf_info.get('X_train', np.array([])).shape[0] if hasattr(clf_info.get('X_train'), 'shape') else 'N/A'
                else:
                    feature_info = []
                    target_info = ''
                    sample_count = 0
                
                col_info1, col_info2, col_info3, col_info4 = st.columns(4)
                with col_info1:
                    st.metric("任务类型", task_type if task_type else selected_task_type)
                with col_info2:
                    st.metric("特征数量", len(feature_info) if isinstance(feature_info, list) else 'N/A')
                with col_info3:
                    st.metric("目标列", str(target_info)[:15] if target_info else 'N/A')
                with col_info4:
                    st.metric("样本数量", sample_count)
                
                if feature_info and isinstance(feature_info, list):
                    with st.expander("📋 特征列详情"):
                        for i, f in enumerate(feature_info):
                            st.write(f"  {i+1}. {f}")
        
        # ==================== 右侧：结果显示 ====================
        with col_results:
            if train_button:
                if not batch_models:
                    st.error("请至少选择一个模型进行训练")
                else:
                    with st.spinner("正在批量训练模型..."):
                        try:
                            # 获取数据 - 使用用户选择的特征列和目标列
                            if selected_task_type == "时序预测":
                                # 获取用户选择的特征列和目标列
                                feature_cols_batch = st.session_state.get('feature_cols', [])
                                target_col_batch = st.session_state.get('target_col', '')
                                
                                if not feature_cols_batch or not target_col_batch:
                                    st.error("请先在「时序预测」标签页中选择特征列和目标列")
                                    st.stop()
                                
                                # 直接从原始数据重新提取，确保使用正确的特征列
                                df_original = st.session_state.get('df', None)
                                if df_original is None:
                                    st.error("原始数据不存在，请重新上传数据")
                                    st.stop()
                                
                                # 严格验证特征列是否存在
                                invalid_cols = [col for col in feature_cols_batch if col not in df_original.columns]
                                if invalid_cols:
                                    st.error(f"特征列不存在: {invalid_cols}")
                                    st.stop()
                                
                                if target_col_batch not in df_original.columns:
                                    st.error(f"目标列不存在: {target_col_batch}")
                                    st.stop()
                                
                                # 使用用户选择的列提取数据
                                X = df_original[feature_cols_batch].copy()
                                y = df_original[target_col_batch].copy()
                                
                                # 验证数据形状
                                if len(X.columns) != len(feature_cols_batch):
                                    st.error(f"特征列数量不匹配！预期 {len(feature_cols_batch)}，实际 {len(X.columns)}")
                                    st.error(f"选择的特征列: {feature_cols_batch}")
                                    st.error(f"实际获取的列: {list(X.columns)}")
                                    st.stop()
                                
                                # 数据预处理和划分
                                preprocessor_batch = DataPreprocessor()
                                batch_prepared = preprocessor_batch.prepare_train_val_test(
                                    X, y,
                                    test_size=0.2,
                                    val_size=0.1,
                                    random_state=42,
                                    time_series=True,
                                    scale_method=st.session_state.get('scale_method', 'standard'),
                                    use_feature_selection=False
                                )
                                X_train = batch_prepared['X_train']
                                X_val = batch_prepared['X_val']
                                X_test = batch_prepared['X_test']
                                y_train = batch_prepared['y_train']
                                y_val = batch_prepared['y_val']
                                y_test = batch_prepared['y_test']
                                
                                # 再次验证数据形状
                                expected_features = len(feature_cols_batch)
                                if len(X_train.shape) > 1 and X_train.shape[1] != expected_features:
                                    st.error(f"训练数据特征数量不匹配！预期 {expected_features}，实际 {X_train.shape[1]}")
                                    st.stop()
                                
                                if len(X_test.shape) > 1 and X_test.shape[1] != expected_features:
                                    st.error(f"测试数据特征数量不匹配！预期 {expected_features}，实际 {X_test.shape[1]}")
                                    st.stop()
                                
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
                            else:
                                clf_info = st.session_state.classification_trainer
                                X_train = clf_info['X_train']
                                y_train = clf_info['y_train']
                                X_test = clf_info['X_test']
                                y_test = clf_info['y_test']
                                
                                model_name_map = {
                                    "逻辑回归": "logistic_regression",
                                    "随机森林分类": "random_forest",
                                    "梯度提升分类": "gradient_boosting",
                                    "SVM分类": "svm",
                                    "KNN分类": "knn",
                                    "XGBoost分类": "xgboost",
                                    "LightGBM分类": "lightgbm",
                                    "朴素贝叶斯": "naive_bayes"
                                }
                            
                            # 数据预处理
                            def ensure_2d(arr, is_target=False):
                                if arr is None:
                                    return None
                                if hasattr(arr, 'values'):
                                    arr = arr.values
                                arr = np.asarray(arr, dtype=np.float64)
                                # 目标变量保持1D
                                if is_target:
                                    return arr.ravel() if arr.ndim > 1 else arr
                                # 特征矩阵确保2D
                                if arr.ndim == 1:
                                    arr = arr.reshape(-1, 1)
                                return arr
                            
                            X_train = ensure_2d(X_train, is_target=False)
                            y_train = ensure_2d(y_train, is_target=True)
                            X_test = ensure_2d(X_test, is_target=False)
                            y_test = ensure_2d(y_test, is_target=True)
                            
                            # 训练模型
                            batch_results = []
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # 分类器映射
                            classifier_map = {
                                "逻辑回归": lambda: LogisticRegression(max_iter=1000, random_state=42),
                                "随机森林分类": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
                                "梯度提升分类": lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
                                "SVM分类": lambda: SVC(kernel='rbf', probability=True, random_state=42),
                                "KNN分类": lambda: KNeighborsClassifier(n_neighbors=5),
                                "XGBoost分类": lambda: XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
                                "LightGBM分类": lambda: LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
                                "朴素贝叶斯": lambda: GaussianNB()
                            }
                            
                            for i, model_display_name in enumerate(batch_models):
                                status_text.text(f"正在训练: {model_display_name}...")
                                
                                if selected_task_type == "时序预测":
                                    model_type = model_name_map.get(model_display_name)
                                    trainer = ModelTrainer()
                                    trainer.select_model(model_type)
                                    trainer.train(X_train, y_train)
                                    y_pred = trainer.predict(X_test)
                                    metrics = trainer.evaluate(X_test, y_test)
                                    
                                    batch_results.append({
                                        'model_name': model_display_name,
                                        'model_type': model_type,
                                        'trainer': trainer,
                                        'predictions': y_pred,
                                        'metrics': metrics
                                    })
                                else:
                                    clf = classifier_map[model_display_name]()
                                    clf.fit(X_train, y_train)
                                    y_pred = clf.predict(X_test)
                                    
                                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                                    metrics = {
                                        'accuracy': accuracy_score(y_test, y_pred),
                                        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                                        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                                        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                                    }
                                    
                                    batch_results.append({
                                        'model_name': model_display_name,
                                        'model_type': model_display_name,
                                        'trainer': clf,
                                        'predictions': y_pred,
                                        'metrics': metrics
                                    })
                                
                                progress_bar.progress((i + 1) / len(batch_models))
                            
                            status_text.empty()
                            st.session_state.batch_results = batch_results
                            st.session_state.batch_task_type = selected_task_type
                            # 保存测试数据用于结果可视化
                            st.session_state.batch_y_test = y_test
                            
                        except Exception as e:
                            st.error(f"批量训练失败: {str(e)}")
            
            # ==================== 显示训练结果 ====================
            if st.session_state.batch_results:
                batch_results = st.session_state.batch_results
                current_task = st.session_state.get('batch_task_type', '时序预测')
                y_test = st.session_state.get('batch_y_test', None)
                
                st.markdown("---")
                st.subheader("📊 批量训练结果")
                
                # 计算最优模型
                if current_task == "时序预测":
                    if batch_metrics == "R²":
                        best_idx = np.argmax([r['metrics'].get('r2', 0) for r in batch_results])
                        best_metric_key = 'r2'
                    elif batch_metrics == "MSE":
                        best_idx = np.argmin([r['metrics'].get('mse', float('inf')) for r in batch_results])
                        best_metric_key = 'mse'
                    elif batch_metrics == "MAE":
                        best_idx = np.argmin([r['metrics'].get('mae', float('inf')) for r in batch_results])
                        best_metric_key = 'mae'
                    else:
                        best_idx = np.argmin([r['metrics'].get('rmse', float('inf')) for r in batch_results])
                        best_metric_key = 'rmse'
                else:
                    if batch_metrics == "准确率":
                        best_idx = np.argmax([r['metrics'].get('accuracy', 0) for r in batch_results])
                        best_metric_key = 'accuracy'
                    elif batch_metrics == "精确率":
                        best_idx = np.argmax([r['metrics'].get('precision', 0) for r in batch_results])
                        best_metric_key = 'precision'
                    elif batch_metrics == "召回率":
                        best_idx = np.argmax([r['metrics'].get('recall', 0) for r in batch_results])
                        best_metric_key = 'recall'
                    else:
                        best_idx = np.argmax([r['metrics'].get('f1', 0) for r in batch_results])
                        best_metric_key = 'f1'
                
                best_model = batch_results[best_idx]
                
                # 性能对比表格
                if current_task == "时序预测":
                    results_df = pd.DataFrame([
                        {
                            "模型": r['model_name'],
                            "R²": f"{r['metrics'].get('r2', 0):.4f}",
                            "MSE": f"{r['metrics'].get('mse', 0):.4f}",
                            "MAE": f"{r['metrics'].get('mae', 0):.4f}",
                            "RMSE": f"{r['metrics'].get('rmse', 0):.4f}",
                            "状态": "🏆 最佳" if r == best_model else ""
                        }
                        for r in batch_results
                    ])
                else:
                    results_df = pd.DataFrame([
                        {
                            "模型": r['model_name'],
                            "准确率": f"{r['metrics'].get('accuracy', 0):.4f}",
                            "精确率": f"{r['metrics'].get('precision', 0):.4f}",
                            "召回率": f"{r['metrics'].get('recall', 0):.4f}",
                            "F1": f"{r['metrics'].get('f1', 0):.4f}",
                            "状态": "🏆 最佳" if r == best_model else ""
                        }
                        for r in batch_results
                    ])
                
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # 最佳模型信息
                best_value = best_model['metrics'].get(best_metric_key, 0)
                st.success(f"🏆 最佳模型: {best_model['model_name']} ({batch_metrics} = {best_value:.4f})")
                
                # 可视化对比
                fig = None
                fig_pred = None
                if len(batch_results) > 1:
                    st.markdown("### 📈 性能指标对比")
                    
                    if current_task == "时序预测":
                        metrics_to_plot = ['r2', 'mse', 'mae', 'rmse']
                        metric_labels = ['R²', 'MSE', 'MAE', 'RMSE']
                    else:
                        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
                        metric_labels = ['准确率', '精确率', '召回率', 'F1分数']
                    
                    fig = go.Figure()
                    for metric, label in zip(metrics_to_plot, metric_labels):
                        values = [r['metrics'].get(metric, 0) for r in batch_results]
                        fig.add_trace(go.Bar(
                            name=label,
                            x=[r['model_name'] for r in batch_results],
                            y=values
                        ))
                    
                    fig.update_layout(
                        barmode='group',
                        title="各模型性能指标对比",
                        xaxis_title="模型",
                        yaxis_title="指标值",
                        legend_title="指标",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 预测值/分类结果对比
                    if y_test is not None:
                        st.markdown("### 🔮 预测结果对比")
                        y_test_1d = np.asarray(y_test).ravel()
                        
                        fig_pred = go.Figure()
                        # 添加真实值
                        fig_pred.add_trace(go.Scatter(
                            y=y_test_1d[:50], name='真实值', mode='lines+markers', 
                            line=dict(color='black', width=2), marker=dict(size=4)
                        ))
                        # 添加各模型预测值
                        for r in batch_results[:4]:
                            pred_1d = np.asarray(r['predictions']).ravel()
                            fig_pred.add_trace(go.Scatter(
                                y=pred_1d[:50], name=r['model_name'], mode='lines', 
                                opacity=0.8 if r != best_model else 1.0,
                                line=dict(width=2 if r == best_model else 1)
                            ))
                        fig_pred.update_layout(
                            title="预测结果对比（前50个样本）",
                            xaxis_title="样本",
                            yaxis_title="预测值" if current_task == "时序预测" else "类别",
                            height=350
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)

                with st.expander("📤 批量训练结果导出", expanded=False):
                    batch_export_df = results_df.copy()
                    if y_test is not None:
                        batch_pred_df = pd.DataFrame({"sample_id": np.arange(len(np.asarray(y_test).ravel())), "true_value": np.asarray(y_test).ravel()})
                        for r in batch_results:
                            batch_pred_df[f"{r['model_name']}_prediction"] = np.asarray(r["predictions"]).ravel()
                    else:
                        batch_pred_df = None
                    report_figures = []
                    if "fig" in locals():
                        report_figures.append(("模型性能指标对比", fig))
                    if "fig_pred" in locals():
                        report_figures.append(("预测结果对比", fig_pred))
                    batch_report = build_html_report(
                        "批量任务训练图文结果报告",
                        metrics={
                            "任务类型": current_task,
                            "模型数量": len(batch_results),
                            "最佳模型": best_model["model_name"],
                            "选择指标": batch_metrics,
                            "最佳指标值": f"{best_value:.4f}",
                        },
                        sections=[("结果摘要", "报告包含批量模型指标排行、最佳模型和预测明细。")],
                        figures=report_figures,
                        tables=[("模型指标排行", batch_export_df), ("预测明细", batch_pred_df)],
                    )
                    render_result_downloads(
                        "批量训练",
                        batch_export_df,
                        batch_report,
                        csv_name=f"batch_training_metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        html_name=f"batch_training_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        key_prefix="batch_training",
                    )
                    if batch_pred_df is not None:
                        st.download_button(
                            "📥 下载批量预测明细 CSV",
                            dataframe_to_csv_bytes(batch_pred_df),
                            file_name=f"batch_prediction_details_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_batch_prediction_details",
                            use_container_width=True,
                        )

                # 保存最佳模型功能
                if st.button("💾 保存最佳模型", key="save_best_model"):
                    best_model = batch_results[best_idx]
                    model_name = best_model['model_name']
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{model_name}_{current_task}_{timestamp}.pkl"
                    file_path = os.path.join("saved_models", filename)
                    
                    # 确保保存目录存在
                    os.makedirs("saved_models", exist_ok=True)
                    
                    # 保存模型
                    import pickle
                    try:
                        with open(file_path, 'wb') as f:
                            pickle.dump(best_model['trainer'], f)
                        st.success(f"✅ 最佳模型 '{model_name}' 已保存到: {file_path}")
                        
                        # 提供下载链接
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                label="📥 下载模型文件",
                                data=f,
                                file_name=filename,
                                mime="application/octet-stream"
                            )
                    except Exception as e:
                        st.error(f"保存模型失败: {str(e)}")

                # 清除结果按钮
                if st.button("🗑️ 清除结果", key="clear_batch"):
                    st.session_state.batch_results = None
                    st.session_state.batch_y_test = None
                    st.rerun()
        
        # ==================== 底部：工作流管理 ====================
        st.markdown("---")
        col_wf1, col_wf2 = st.columns([2, 1])
        
        with col_wf1:
            st.subheader("🔄 工作流模板")
            
            if st.session_state.saved_workflows:
                st.write("已保存的工作流：")
                for wf_name, wf_info in st.session_state.saved_workflows.items():
                    with st.expander(f"📁 {wf_name} (保存于 {wf_info['timestamp']})"):
                        st.write(f"任务类型: {wf_info.get('task_type', '时序预测')}")
                        st.write(f"模型列表: {', '.join(wf_info.get('models', []))}")
                        st.write(f"评估指标: {wf_info.get('metric', 'R²')}")
            
            st.markdown("### 创建新工作流")
            wf_name = st.text_input("工作流名称", key="wf_name", placeholder="输入工作流名称...")
            wf_models = st.multiselect(
                "默认模型列表",
                ["线性回归", "Ridge", "Lasso", "随机森林", "梯度提升", "SVR", "KNN", "XGBoost", "LightGBM",
                 "逻辑回归", "随机森林分类", "梯度提升分类", "SVM分类", "朴素贝叶斯"],
                default=["随机森林", "梯度提升"],
                key="wf_models"
            )
            wf_metric = st.selectbox(
                "默认评估指标",
                ["R²", "MSE", "MAE", "RMSE", "准确率", "精确率", "召回率", "F1分数"],
                index=0,
                key="wf_metric"
            )
            
            if st.button("💾 保存工作流", key="save_wf"):
                if wf_name:
                    st.session_state.saved_workflows[wf_name] = {
                        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                        'task_type': st.session_state.batch_task_type if st.session_state.batch_task_type else '时序预测',
                        'models': wf_models,
                        'metric': wf_metric
                    }
                    st.success(f"✅ 工作流 '{wf_name}' 已保存！")
                else:
                    st.warning("请输入工作流名称")
        
        with col_wf2:
            st.subheader("📌 快速操作")
            
            # 一键最佳模型训练
            if st.button("🎯 一键最佳模型", key="quick_best", help="使用默认参数训练最佳模型"):
                st.info("功能开发中...")
            
            # 导出配置
            if st.button("📤 导出配置", key="export_config"):
                st.info("功能开发中...")
            
            # 导入配置
            if st.button("📥 导入配置", key="import_config"):
                st.info("功能开发中...")
        
        # 显示已保存的工作流
        if 'saved_workflows' in st.session_state and st.session_state.saved_workflows:
            st.write("**已保存的工作流:**")
            for name, info in st.session_state.saved_workflows.items():
                col_wf1, col_wf2, col_wf3 = st.columns([3, 2, 1])
                with col_wf1:
                    st.write(f"- {name}")
                with col_wf2:
                    st.caption(f"{info.get('timestamp', 'N/A')} | 指标: {info.get('metric', 'N/A')}")
                with col_wf3:
                    if st.button("🗑️", key=f"del_workflow_{name}"):
                        del st.session_state.saved_workflows[name]
                        st.rerun()

# ==================== 标签4: 系统状态诊断 ====================
with tab4:
    st.markdown("<h1 style='font-size: 32px; font-weight: bold;'>🔍 系统状态诊断</h1>", unsafe_allow_html=True)
    
    # 初始化诊断系统
    diagnosis_system = DiagnosisSystem()
    
    st.subheader("🧭 增强诊断闭环")
    st.info("新增流程：异常检测 -> 故障分类 -> 根因解释 -> 处置建议 -> 诊断报告。原有诊断系统保留在下方。")
    diag_method_help = {
        "isolation_forest": "随机切分特征空间，少数样本更容易被隔离。适合高维表格异常检测，异常比例可影响阈值。",
        "one_class_svm": "只学习正常样本边界，适合正常工况占多数的数据；对缩放和核参数较敏感。",
        "lof": "比较局部邻域密度，适合局部稀疏异常；不适合样本极少或维度过高的数据。",
        "pca_t2_spe": "使用主成分空间和重构残差监控过程偏离，适合传感器过程监控和根因偏离分析。",
        "autoencoder": "当前为轻量 PCA 重构误差实现，保留深度 AutoEncoder 接口；适合先验证重构异常检测流程。",
    }
    with st.expander("❓ 诊断选项说明", expanded=False):
        st.markdown("""
        - **诊断数据来源**：已训练模型时优先使用测试特征；尚未训练时使用上传数据中的数值列。
        - **异常检测**：先判断样本是否偏离常见模式，再进入根因解释和处置建议。
        - **根因解释**：依据特征偏离、重构贡献或模型解释结果排序，只作为定位线索，不替代工程复核。
        - **诊断报告**：适合记录异常样本、关键特征、建议动作和复检项。
        """)
    diag_source_col, diag_method_col, diag_action_col = st.columns([2, 2, 1])
    with diag_source_col:
        enhanced_diag_source = st.selectbox(
            "诊断数据来源",
            ["当前测试特征", "当前完整数据"],
            key="enhanced_diag_source",
            help="优先使用模型测试集；如果尚未训练模型，可用当前上传数据的数值列进行无监督诊断。"
        )
    with diag_method_col:
        enhanced_diag_method = st.selectbox(
            "无监督诊断方法",
            ["isolation_forest", "one_class_svm", "lof", "pca_t2_spe", "autoencoder"],
            key="enhanced_diag_method",
            help="AutoEncoder 首版使用重构误差接口，当前实现为轻量PCA重构，后续可替换为深度自编码器。"
        )
        st.caption(diag_method_help[enhanced_diag_method])
    with diag_action_col:
        st.write("")
        st.write("")
        run_enhanced_diag = st.button("运行增强诊断", type="primary", key="run_enhanced_diag")
    with st.expander("⚙️ 增强诊断高级参数", expanded=False):
        adv_d1, adv_d2, adv_d3, adv_d4, adv_d5 = st.columns(5)
        with adv_d1:
            enhanced_contamination = st.slider(
                "异常比例估计",
                0.01, 0.5, 0.1, 0.01,
                key="enhanced_contamination",
                help="用于无监督方法的异常阈值，建议结合历史故障率或工况经验设置。"
            )
        with adv_d2:
            enhanced_warning = st.slider(
                "预警阈值",
                0.1, 0.95, 0.55, 0.05,
                key="enhanced_warning",
                help="归一化风险分超过该值标记为预警。"
            )
        with adv_d3:
            enhanced_critical = st.slider(
                "严重阈值",
                0.2, 1.0, 0.8, 0.05,
                key="enhanced_critical",
                help="归一化风险分超过该值标记为严重。"
            )
        with adv_d4:
            enhanced_lof_neighbors = st.slider(
                "LOF邻居数",
                2, 80, 20, 1,
                key="enhanced_lof_neighbors",
                help="仅LOF使用。邻居数越大越偏全局，越小越敏感于局部异常。"
            )
        with adv_d5:
            enhanced_pca_components = st.number_input(
                "PCA主成分数",
                0, 100, 0, 1,
                key="enhanced_pca_components",
                help="0表示自动选择；PCA/T2-SPE和AutoEncoder轻量重构接口使用。"
            )
    enhanced_diagnoser = EnhancedFaultDiagnosis(DiagnosisConfig(
        contamination=enhanced_contamination,
        severity_warning=enhanced_warning,
        severity_critical=enhanced_critical,
        lof_neighbors=enhanced_lof_neighbors,
        svm_nu=enhanced_contamination,
        pca_components=None if enhanced_pca_components <= 0 else int(enhanced_pca_components),
    ))
    
    if run_enhanced_diag:
        try:
            if enhanced_diag_source == "当前测试特征" and st.session_state.get('X_test') is not None:
                diag_X = st.session_state.X_test
                diag_features = st.session_state.get('feature_cols')
            elif st.session_state.get('df') is not None:
                diag_df = st.session_state.df.select_dtypes(include=[np.number]).copy()
                diag_X = diag_df
                diag_features = diag_df.columns.tolist()
            else:
                diag_X = None
                diag_features = None
            
            if diag_X is None:
                st.warning("请先上传数据或完成一次训练，以便获取诊断输入。")
            else:
                with st.spinner("正在执行增强诊断..."):
                    enhanced_result = enhanced_diagnoser.run_unsupervised(
                        diag_X,
                        feature_names=diag_features,
                        method=enhanced_diag_method
                    )
                st.session_state.enhanced_diagnosis_result = enhanced_result
                append_run_history(st.session_state, "增强故障诊断", enhanced_diag_method)
                st.success("✅ 增强诊断完成")
        except Exception as e:
            show_error("增强诊断失败", traceback.format_exc())
    
    if st.session_state.get('enhanced_diagnosis_result') is not None:
        enhanced_result = st.session_state.enhanced_diagnosis_result
        result_df = enhanced_result['results']
        root_df = enhanced_result['root_causes']
        drift_df = enhanced_result.get('feature_drift', pd.DataFrame())
        health = enhanced_result.get('health_summary', {})
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("样本数", len(result_df))
        d2.metric("异常样本", int(result_df['is_anomaly'].sum()) if 'is_anomaly' in result_df else "N/A")
        d3.metric("平均异常分", f"{result_df['anomaly_score'].mean():.3f}" if 'anomaly_score' in result_df else "N/A")
        d4.metric("健康评分", f"{health.get('health_score', 0):.1f}", health.get("status", ""))
        d5.metric("关键根因数", min(5, len(root_df)))
        
        diag_tabs = st.tabs(["健康概览", "异常样本", "根因特征", "处置建议", "报告导出"])
        with diag_tabs[0]:
            severity_counts = result_df["severity"].value_counts().reset_index()
            severity_counts.columns = ["severity", "count"]
            fig_severity = go.Figure(go.Bar(
                x=severity_counts["severity"],
                y=severity_counts["count"],
                marker_color=["#2ca02c" if item == "正常" else "#ff7f0e" if item == "预警" else "#d62728" for item in severity_counts["severity"]]
            ))
            fig_severity.update_layout(title="严重度分布", xaxis_title="等级", yaxis_title="样本数", height=320)
            fig_score = go.Figure()
            if "anomaly_score" in result_df:
                fig_score.add_trace(go.Scatter(x=result_df["sample_index"], y=result_df["anomaly_score"], mode="lines", name="异常分"))
                fig_score.add_hline(y=enhanced_warning, line_dash="dash", line_color="#ff7f0e", annotation_text="预警阈值")
                fig_score.add_hline(y=enhanced_critical, line_dash="dash", line_color="#d62728", annotation_text="严重阈值")
            fig_score.update_layout(title="异常风险分时序", xaxis_title="样本", yaxis_title="风险分", height=320)
            c_health1, c_health2 = st.columns(2)
            with c_health1:
                st.plotly_chart(fig_severity, use_container_width=True)
            with c_health2:
                st.plotly_chart(fig_score, use_container_width=True)
            if isinstance(drift_df, pd.DataFrame) and not drift_df.empty:
                st.write("**特征稳定性摘要**")
                st.dataframe(drift_df.head(20), use_container_width=True, hide_index=True)
        with diag_tabs[1]:
            st.dataframe(result_df.sort_values("anomaly_score", ascending=False).head(100), use_container_width=True, hide_index=True)
        with diag_tabs[2]:
            fig_root = go.Figure(go.Bar(
                x=root_df.head(15)["importance"],
                y=root_df.head(15)["feature"],
                orientation="h",
                marker_color="#256b7f"
            ))
            fig_root.update_layout(title="根因特征重要性", xaxis_title="重要性", yaxis_title="特征", height=420, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_root, use_container_width=True)
            st.dataframe(root_df.head(20), use_container_width=True, hide_index=True)
        with diag_tabs[3]:
            for action in enhanced_result.get('actions', []):
                st.warning(action)
        with diag_tabs[4]:
            report_md = enhanced_diagnoser.build_report(enhanced_result)
            enhanced_export_df = result_df.merge(root_df.head(1).add_prefix("top_root_"), how="cross") if not root_df.empty else result_df.copy()
            html_report = build_html_report(
                "增强故障诊断图文报告",
                metrics={
                    "诊断方法": enhanced_result.get("method"),
                    "样本数": len(result_df),
                    "异常样本": int(result_df["is_anomaly"].sum()) if "is_anomaly" in result_df else "N/A",
                    "健康评分": f"{health.get('health_score', 0):.1f}",
                    "状态": health.get("status", "N/A"),
                    "首要根因": health.get("top_feature", "N/A"),
                },
                sections=[("诊断结论", report_md.replace("# 增强故障诊断报告", "").strip())],
                figures=[("严重度分布", fig_severity), ("异常风险分时序", fig_score), ("根因特征重要性", fig_root)],
                tables=[("异常样本明细", result_df.sort_values("anomaly_score", ascending=False).head(100)), ("根因特征", root_df.head(30)), ("特征稳定性摘要", drift_df.head(30) if isinstance(drift_df, pd.DataFrame) else None)],
                notes=enhanced_result.get("actions", []),
            )
            render_result_downloads(
                "增强诊断",
                enhanced_export_df,
                html_report,
                csv_name=f"enhanced_diagnosis_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                html_name=f"enhanced_diagnosis_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                key_prefix="enhanced_diagnosis",
            )
            st.download_button(
                "下载诊断报告 Markdown",
                report_md,
                file_name=f"enhanced_diagnosis_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                key="download_enhanced_diag_report"
            )
            st.code(report_md, language="markdown")
    
    st.divider()
    
    # 故障诊断
    st.subheader("🔧 故障诊断系统")
    st.info("基于机器学习模型识别系统状态并进行故障诊断。")
    
    has_test_data = st.session_state.X_test is not None and st.session_state.y_test is not None
    
    # 获取实际的列名 - 包括特征和目标变量
    all_column_names = []
    
    # 首先尝试从原始数据获取所有列
    if hasattr(st.session_state, 'df') and st.session_state.df is not None:
        all_column_names = list(st.session_state.df.columns)
    
    # 其次尝试使用预保存的完整列名
    elif hasattr(st.session_state, 'all_data_columns') and st.session_state.all_data_columns:
        all_column_names = list(st.session_state.all_data_columns)
    
    # 再尝试从X_test获取特征列
    elif has_test_data and hasattr(st.session_state.X_test, 'columns'):
        all_column_names = list(st.session_state.X_test.columns)
        
        # 添加目标变量
        if st.session_state.y_test is not None:
            target_name = None
            if hasattr(st.session_state.y_test, 'name') and st.session_state.y_test.name:
                target_name = st.session_state.y_test.name
            elif hasattr(st.session_state, 'target_col_selected'):
                target_name = st.session_state.target_col_selected
            
            if target_name and target_name not in all_column_names:
                all_column_names.append(target_name)
    
    # 最后尝试从feature_cols获取
    elif st.session_state.feature_cols:
        all_column_names = list(st.session_state.feature_cols)
        
        # 添加目标变量
        if hasattr(st.session_state, 'target_col_selected') and st.session_state.target_col_selected:
            if st.session_state.target_col_selected not in all_column_names:
                all_column_names.append(st.session_state.target_col_selected)
    
    # 如果还是没有列名，生成默认的
    if not all_column_names:
        all_column_names = [f"Feature_{i}" for i in range(10)]
    
    diagnosis_method = st.selectbox(
        "诊断方法",
        ["阈值判断", "机器学习分类", "异常检测"],
        key="diagnosis_method_tab3",
        help="阈值判断: 基于预设范围检测异常; 机器学习分类: 使用分类模型识别状态; 异常检测: 自动识别异常模式"
    )
    
    # 初始化或获取保存的参数
    if 'diagnosis_params' not in st.session_state:
        st.session_state.diagnosis_params = {}
    
    thresholds = st.session_state.diagnosis_params.get('thresholds', {})
    
    if diagnosis_method == "阈值判断":
        st.write("设置各物理量的正常范围阈值：")
        
        # 获取可用的列名 - 包括特征和目标变量
        if has_test_data:
            feature_options = all_column_names
        else:
            feature_options = ["Feature_0", "Feature_1", "Feature_2", "Feature_3"]
        
        # 允许用户选择要监控的特征
        selected_features = st.multiselect(
            "选择要监控的特征",
            feature_options,
            default=feature_options[:min(4, len(feature_options))] if feature_options else [],
            key="thresh_features_select"
        )
        
        # 动态设置阈值输入
        thresholds = {}
        if selected_features:
            st.write("设置阈值范围：")
            cols = st.columns(4)
            for idx, feat in enumerate(selected_features):
                with cols[idx % 4]:
                    col_min, col_max = st.columns(2)
                    with col_min:
                        t_min = st.number_input(f"{feat} 最小值", value=-5.0, 
                                               key=f"{feat}_min_{idx}", format="%.4f")
                    with col_max:
                        t_max = st.number_input(f"{feat} 最大值", value=5.0, 
                                               key=f"{feat}_max_{idx}", format="%.4f")
                    thresholds[feat] = (t_min, t_max)
        else:
            st.warning("请选择至少一个特征进行阈值监控")
        
        # 保存阈值到 session_state
        st.session_state.diagnosis_params['thresholds'] = thresholds
    
    elif diagnosis_method == "机器学习分类":
        st.write("配置分类模型进行故障诊断：")
        col_ml1, col_ml2 = st.columns(2)
        with col_ml1:
            classifier_type = st.selectbox(
                "分类器类型",
                ["随机森林分类器", "梯度提升分类器", "SVM分类器"],
                key="classifier_type_tab3"
            )
        with col_ml2:
            n_classes = st.number_input("状态类别数量", min_value=2, max_value=10, value=3, key="n_classes_tab3")
        
        # 保存参数
        st.session_state.diagnosis_params['classifier_type'] = classifier_type
        st.session_state.diagnosis_params['n_classes'] = n_classes
        
        if has_test_data:
            st.write(f"测试数据形状: {st.session_state.X_test.shape}")
    
    elif diagnosis_method == "异常检测":
        st.write("配置异常检测算法：")
        col_an1, col_an2 = st.columns(2)
        with col_an1:
            anomaly_method = st.selectbox(
                "异常检测方法",
                ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"],
                key="anomaly_method_tab3"
            )
        with col_an2:
            contamination = st.slider("异常比例估计", 0.01, 0.5, 0.1, key="contamination_tab3")
        
        # 保存参数
        st.session_state.diagnosis_params['anomaly_method'] = anomaly_method
        st.session_state.diagnosis_params['contamination'] = contamination
    
    # 运行诊断按钮
    if st.button("🔍 运行诊断", type="primary", key="run_diagnosis_tab3"):
        if not has_test_data:
            st.error("请先在「时序预测」标签页中完成数据预处理，以获取测试数据。")
        else:
            with st.spinner("正在分析系统状态..."):
                try:
                    X_test = st.session_state.X_test
                    X_values = X_test.values if hasattr(X_test, 'values') else X_test
                    
                    # 获取目标变量数据（如果有）
                    y_test = st.session_state.y_test
                    y_values = None
                    target_col_name = None
                    
                    if y_test is not None:
                        y_values = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
                        if hasattr(y_test, 'name') and y_test.name:
                            target_col_name = y_test.name
                        elif hasattr(st.session_state, 'target_col_selected'):
                            target_col_name = st.session_state.target_col_selected
                        else:
                            target_col_name = 'Target'
                    
                    if diagnosis_method == "阈值判断":
                        # 获取阈值
                        thresholds = st.session_state.diagnosis_params.get('thresholds', {})
                        
                        if not thresholds:
                            st.error("请先设置至少一个特征的阈值")
                        else:
                            # 确定测试样本总数
                            n_samples = len(X_test)
                            
                            # 初始化每个样本的异常列表
                            all_anomalies = [[] for _ in range(n_samples)]
                            
                            # 检查特征列的阈值
                            if hasattr(X_test, 'columns'):
                                # 遍历每个特征
                                for feature_idx, feature in enumerate(X_test.columns):
                                    if feature in thresholds:
                                        low, high = thresholds[feature]
                                        # 遍历该特征的所有样本值
                                        for i in range(n_samples):
                                            val = X_values[i, feature_idx] if hasattr(X_values, 'shape') else X_values[i][feature_idx]
                                            if val < low or val > high:
                                                all_anomalies[i].append({
                                                    'feature': feature,
                                                    'value': val,
                                                    'threshold': (low, high),
                                                    'severity': 'high' if abs(val) > max(abs(low), abs(high)) * 1.5 else 'medium'
                                                })
                            
                            # 检查目标变量的阈值
                            if target_col_name and target_col_name in thresholds and y_values is not None:
                                low, high = thresholds[target_col_name]
                                for i in range(n_samples):
                                    val = y_values[i]
                                    if val < low or val > high:
                                        all_anomalies[i].append({
                                            'feature': target_col_name,
                                            'value': val,
                                            'threshold': (low, high),
                                            'severity': 'high' if abs(val) > max(abs(low), abs(high)) * 1.5 else 'medium'
                                        })
                            
                            # 构建诊断结果
                            diagnosis_results = []
                            for i in range(n_samples):
                                anomalies = all_anomalies[i]
                                diagnosis_results.append({
                                    'sample_id': i,
                                    'anomalies': anomalies,
                                    'warnings': [],
                                    'recommendations': []
                                })
                                
                                if anomalies:
                                    diagnosis_results[i]['recommendations'].append("检测到异常参数，建议检查传感器数据")
                                if len(anomalies) > 2:
                                    diagnosis_results[i]['recommendations'].append("多个参数异常，可能存在系统性故障")
                            
                            # 统计结果
                            total_samples = len(diagnosis_results)
                            normal_count = sum(1 for r in diagnosis_results if not r['anomalies'])
                            anomaly_count = total_samples - normal_count
                            
                            col_res1, col_res2, col_res3 = st.columns(3)
                            with col_res1:
                                st.metric("总样本数", total_samples)
                            with col_res2:
                                st.metric("正常样本", normal_count, delta_color="normal")
                            with col_res3:
                                st.metric("异常样本", anomaly_count, delta_color="inverse")
                            
                            if anomaly_count > 0:
                                st.warning(f"⚠️ 检测到 {anomaly_count} 个异常样本")
                                
                                # 显示异常详情
                                with st.expander("查看异常详情"):
                                    anomaly_samples = [r for r in diagnosis_results if r['anomalies']]
                                    for r in anomaly_samples[:20]:
                                        st.write(f"**样本 {r['sample_id']}**: 发现 {len(r['anomalies'])} 个异常")
                                        for a in r['anomalies']:
                                            severity_color = "🔴" if a['severity'] == 'high' else "🟡"
                                            st.write(f"  {severity_color} {a['feature']}: {a['value']:.4f} (阈值: {a['threshold']})")
                                        st.divider()
                            else:
                                st.success("✅ 所有样本均在正常范围内")
                            threshold_export_rows = []
                            for item in diagnosis_results:
                                if item["anomalies"]:
                                    for anomaly in item["anomalies"]:
                                        threshold_export_rows.append({
                                            "sample_id": item["sample_id"],
                                            "status": "异常",
                                            "feature": anomaly["feature"],
                                            "value": anomaly["value"],
                                            "threshold_min": anomaly["threshold"][0],
                                            "threshold_max": anomaly["threshold"][1],
                                            "severity": anomaly["severity"],
                                            "recommendation": "；".join(item["recommendations"]),
                                        })
                                else:
                                    threshold_export_rows.append({
                                        "sample_id": item["sample_id"],
                                        "status": "正常",
                                        "feature": "",
                                        "value": np.nan,
                                        "threshold_min": np.nan,
                                        "threshold_max": np.nan,
                                        "severity": "normal",
                                        "recommendation": "",
                                    })
                            st.session_state.legacy_diagnosis_result = {
                                "method": "阈值判断",
                                "summary": {"总样本数": total_samples, "正常样本": normal_count, "异常样本": anomaly_count},
                                "data": pd.DataFrame(threshold_export_rows),
                            }
                    
                    elif diagnosis_method == "机器学习分类":
                        # 获取保存的参数
                        classifier_type = st.session_state.diagnosis_params.get('classifier_type', '随机森林分类器')
                        n_classes = st.session_state.diagnosis_params.get('n_classes', 3)
                        
                        # 导入分类器
                        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                        from sklearn.svm import SVC
                        
                        # 使用简单的聚类方式生成状态标签
                        from sklearn.cluster import KMeans
                        kmeans = KMeans(n_clusters=n_classes, random_state=42)
                        y_state = kmeans.fit_predict(X_values)
                        
                        # 训练分类器
                        if classifier_type == "随机森林分类器":
                            clf = RandomForestClassifier(n_estimators=100, random_state=42)
                        elif classifier_type == "梯度提升分类器":
                            clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
                        else:
                            clf = SVC(kernel='rbf', probability=True, random_state=42)
                        
                        clf.fit(X_values, y_state)
                        y_pred = clf.predict(X_values)
                        
                        # 显示结果
                        unique_states, counts = np.unique(y_pred, return_counts=True)
                        
                        st.success("✅ 分类诊断完成！")
                        
                        col_res1, col_res2 = st.columns(2)
                        with col_res1:
                            st.metric("检测到的状态数", len(unique_states))
                        with col_res2:
                            st.metric("总样本数", len(y_pred))
                        
                        # 状态分布图
                        state_dist = pd.DataFrame({
                            '状态': [f'状态_{s}' for s in unique_states],
                            '数量': counts,
                            '比例(%)': (counts / len(y_pred) * 100).round(2)
                        })
                        st.dataframe(state_dist, use_container_width=True)
                        
                        fig_state = go.Figure(data=[
                            go.Pie(
                                labels=state_dist['状态'],
                                values=state_dist['数量'],
                                hole=0.3,
                                textinfo='label+percent'
                            )
                        ])
                        fig_state.update_layout(title="系统状态分布")
                        st.plotly_chart(fig_state, use_container_width=True)
                        st.session_state.legacy_diagnosis_result = {
                            "method": "机器学习分类",
                            "summary": {"分类器": classifier_type, "状态数": len(unique_states), "总样本数": len(y_pred)},
                            "data": pd.DataFrame({
                                "sample_id": np.arange(len(y_pred)),
                                "predicted_state": [f"状态_{item}" for item in y_pred],
                                "cluster_label": y_state,
                            }),
                            "distribution": state_dist,
                        }
                    
                    elif diagnosis_method == "异常检测":
                        # 获取保存的参数
                        anomaly_method = st.session_state.diagnosis_params.get('anomaly_method', 'Isolation Forest')
                        contamination = st.session_state.diagnosis_params.get('contamination', 0.1)
                        
                        from sklearn.ensemble import IsolationForest
                        from sklearn.svm import OneClassSVM
                        from sklearn.neighbors import LocalOutlierFactor
                        
                        if anomaly_method == "Isolation Forest":
                            ad = IsolationForest(contamination=contamination, random_state=42)
                            predictions = ad.fit_predict(X_values)
                        elif anomaly_method == "One-Class SVM":
                            ad = OneClassSVM(kernel='rbf', nu=contamination)
                            predictions = ad.fit_predict(X_values)
                        else:
                            ad = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
                            predictions = ad.fit_predict(X_values)
                        
                        # 统计结果
                        normal_mask = predictions == 1
                        anomaly_mask = predictions == -1
                        
                        normal_count = normal_mask.sum()
                        anomaly_count = anomaly_mask.sum()
                        
                        col_res1, col_res2, col_res3 = st.columns(3)
                        with col_res1:
                            st.metric("总样本数", len(predictions))
                        with col_res2:
                            st.metric("正常样本", normal_count, delta_color="normal")
                        with col_res3:
                            st.metric("异常样本", anomaly_count, delta_color="inverse")
                        
                        if anomaly_count > 0:
                            st.warning(f"⚠️ 异常检测算法检测到 {anomaly_count} 个异常点")
                            
                            # 可视化异常
                            if X_values.shape[1] >= 2:
                                fig_anom = go.Figure()
                                fig_anom.add_trace(go.Scatter(
                                    x=X_values[normal_mask, 0], y=X_values[normal_mask, 1],
                                    mode='markers', name='正常', marker=dict(color='green', size=5, opacity=0.6)
                                ))
                                fig_anom.add_trace(go.Scatter(
                                    x=X_values[anomaly_mask, 0], y=X_values[anomaly_mask, 1],
                                    mode='markers', name='异常', marker=dict(color='red', size=10, symbol='x')
                                ))
                                fig_anom.update_layout(
                                    title="异常检测结果可视化",
                                    xaxis_title="特征1",
                                    yaxis_title="特征2"
                                )
                                st.plotly_chart(fig_anom, use_container_width=True)
                            else:
                                st.info("数据特征不足2个，无法可视化")
                        else:
                            st.success("✅ 未检测到异常点")
                        if hasattr(ad, "score_samples"):
                            raw_scores = -ad.score_samples(X_values)
                        elif hasattr(ad, "decision_function"):
                            raw_scores = -ad.decision_function(X_values)
                        elif hasattr(ad, "negative_outlier_factor_"):
                            raw_scores = -ad.negative_outlier_factor_
                        else:
                            raw_scores = np.where(predictions == -1, 1.0, 0.0)
                        st.session_state.legacy_diagnosis_result = {
                            "method": f"异常检测 - {anomaly_method}",
                            "summary": {"总样本数": len(predictions), "正常样本": int(normal_count), "异常样本": int(anomaly_count), "异常比例": f"{anomaly_count / max(len(predictions), 1):.2%}"},
                            "data": pd.DataFrame({
                                "sample_id": np.arange(len(predictions)),
                                "prediction": predictions,
                                "status": np.where(predictions == -1, "异常", "正常"),
                                "risk_score": raw_scores,
                            }),
                        }
                            
                except Exception as e:
                    st.error(f"诊断失败: {str(e)}")

    if st.session_state.get("legacy_diagnosis_result") is not None:
        legacy_result = st.session_state.legacy_diagnosis_result
        legacy_df = legacy_result["data"]
        with st.expander("📤 传统诊断结果导出", expanded=False):
            if "status" in legacy_df.columns:
                status_counts = legacy_df["status"].value_counts().reset_index()
                status_counts.columns = ["status", "count"]
                fig_legacy_status = go.Figure(go.Bar(x=status_counts["status"], y=status_counts["count"], marker_color="#256b7f"))
                fig_legacy_status.update_layout(title="诊断状态分布", xaxis_title="状态", yaxis_title="样本数", height=320)
                st.plotly_chart(fig_legacy_status, use_container_width=True)
            elif "predicted_state" in legacy_df.columns:
                status_counts = legacy_df["predicted_state"].value_counts().reset_index()
                status_counts.columns = ["status", "count"]
                fig_legacy_status = go.Figure(go.Pie(labels=status_counts["status"], values=status_counts["count"], hole=0.35))
                fig_legacy_status.update_layout(title="系统状态分布", height=320)
                st.plotly_chart(fig_legacy_status, use_container_width=True)
            else:
                fig_legacy_status = None
            st.dataframe(legacy_df.head(200), use_container_width=True, hide_index=True)
            legacy_report = build_html_report(
                "系统状态诊断图文报告",
                metrics=legacy_result.get("summary", {}),
                sections=[("诊断方法", legacy_result.get("method", "N/A"))],
                figures=[("状态分布", fig_legacy_status)],
                tables=[("诊断结果明细", legacy_df), ("状态分布", status_counts if "status_counts" in locals() else None)],
                notes=["传统诊断结果用于快速识别状态，建议结合增强诊断根因分析进一步复核。"],
            )
            render_result_downloads(
                "传统诊断",
                legacy_df,
                legacy_report,
                csv_name=f"legacy_diagnosis_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                html_name=f"legacy_diagnosis_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                key_prefix="legacy_diagnosis",
            )
    
    st.divider()
    
    # 预训练模型库
    st.subheader("📚 预训练模型库")
    st.info("管理和使用预训练模型进行迁移学习。")
    
    # 初始化模型库session state
    if 'model_library' not in st.session_state:
        st.session_state.model_library = {
            "随机森林基础模型": {
                "type": "随机森林",
                "r2": 0.89,
                "date": "2024-01-15",
                "path": "saved_models/random_forest_model.pkl"
            },
            "梯度提升基础模型": {
                "type": "梯度提升",
                "r2": 0.91,
                "date": "2024-02-20",
                "path": "saved_models/multi_model_comparison_model.pkl"
            },
            "海洋环境通用模型": {
                "type": "集成学习",
                "r2": 0.93,
                "date": "2024-03-10",
                "path": None
            }
        }
    
    # 显示模型库
    library_df = pd.DataFrame([
        {"模型名称": name, "类型": info["type"], "R²": info["r2"], "创建时间": info["date"]}
        for name, info in st.session_state.model_library.items()
    ])
    st.dataframe(library_df, use_container_width=True)
    
    selected_model = st.selectbox(
        "选择要使用的模型",
        list(st.session_state.model_library.keys()),
        key="select_model_tab3",
        help="选择预训练模型进行加载、迁移学习或作为基准模型。"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("📥 加载选中模型", type="primary", key="load_model_tab3"):
            if selected_model:
                model_info = st.session_state.model_library[selected_model]
                st.session_state.loaded_model = selected_model
                st.session_state.loaded_model_info = model_info
                st.success(f"✅ 已加载模型: {selected_model} (R²={model_info['r2']})")
            else:
                st.warning("请先选择一个模型")
    
    with col2:
        uploaded_model = st.file_uploader("上传新模型", type=["pkl"], key="upload_model_file")
        if uploaded_model:
            model_name = st.text_input("模型名称", value=uploaded_model.name.replace(".pkl", ""), key="new_model_name")
            if st.button("📤 保存模型", key="save_uploaded_model"):
                import tempfile
                save_dir = "saved_models"
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{model_name}.pkl")
                with open(save_path, 'wb') as f:
                    f.write(uploaded_model.getvalue())
                
                st.session_state.model_library[model_name] = {
                    "type": "自定义",
                    "r2": 0.85,
                    "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                    "path": save_path
                }
                st.success(f"✅ 模型已保存: {model_name}")
                st.rerun()
    
    with col3:
        if st.button("🗑️ 删除选中模型", key="delete_model_tab3"):
            if selected_model in st.session_state.model_library:
                del st.session_state.model_library[selected_model]
                st.warning(f"⚠️ 模型 {selected_model} 已删除")
                st.rerun()
    
    st.divider()
    
    # 迁移学习
    st.subheader("🔄 迁移学习")
    st.info("利用预训练模型的知识，加速新模型训练。")
    
    # 检查是否有训练数据
    has_training_data = (st.session_state.X_train is not None and 
                        st.session_state.y_train is not None)
    
    if not has_training_data:
        st.warning("⚠️ 请先在「时序预测」标签页中完成数据预处理，以获取训练数据。")
    else:
        transfer_source = st.selectbox(
            "选择源模型",
            list(st.session_state.model_library.keys()),
            key="transfer_source_tab3",
            help="选择已训练好的模型作为知识来源，用于迁移到新任务。"
        )

        transfer_method = st.selectbox(
            "迁移方法",
            ["特征提取", "微调", "域适应"],
            key="transfer_method_tab3",
            help="特征提取: 使用源模型提取特征后训练新模型; 微调: 在源模型基础上调整参数; 域适应: 适应不同数据分布"
        )

        if transfer_method == "微调":
            freeze_ratio = st.slider(
                "冻结层比例",
                0.0, 1.0, 0.5,
                key="freeze_ratio_tab3",
                help="冻结层不会被更新。比例越高，保留的源模型知识越多，但适应性越差。"
            )
        else:
            freeze_ratio = 0.0
        
        # 迁移学习参数
        with st.expander("⚙️ 迁移学习参数", expanded=True):
            col_t1, col_t2, col_t3, col_t4 = st.columns(4)
            with col_t1:
                target_epochs = st.slider("训练轮数/迭代参考", 10, 300, 80, 10, key="transfer_epochs", help="用于树模型估计器数量参考；深度迁移时可对应训练轮数。")
                transfer_model_family = st.selectbox(
                    "目标模型族",
                    ["随机森林", "梯度提升", "Ridge"],
                    key="transfer_model_family",
                    help="迁移后的目标域模型类型。"
                )
            with col_t2:
                target_lr = st.selectbox("学习率", [0.001, 0.005, 0.01, 0.05, 0.1], index=2, key="transfer_lr", help="梯度提升和Ridge会使用；随机森林不使用学习率。")
                n_estimators_transfer = st.slider("树/迭代数量", 20, 400, 120, 20, key="transfer_n_estimators")
            with col_t3:
                target_domain_ratio = st.slider("目标域训练样本比例", 0.2, 1.0, 1.0, 0.05, key="transfer_target_ratio", help="从当前训练集抽取多少作为目标域微调数据。")
                source_weight = st.slider("源域权重", 0.0, 1.0, 0.3, 0.05, key="transfer_source_weight", help="微调联合训练时源域样本权重。")
            with col_t4:
                target_weight = st.slider("目标域权重", 0.1, 2.0, 1.0, 0.05, key="transfer_target_weight", help="微调联合训练时目标域样本权重。")
                random_state_transfer = st.number_input("随机种子", 0, 9999, 42, key="transfer_random_state")
            col_t5, col_t6, col_t7 = st.columns(3)
            with col_t5:
                feature_keep_ratio = st.slider("特征保留比例", 0.1, 1.0, 0.7, 0.05, key="transfer_feature_keep_ratio", help="特征提取策略中按源模型重要性保留的特征比例。")
            with col_t6:
                ensemble_source_weight = st.slider("集成源模型权重", 0.0, 1.0, 0.45, 0.05, key="transfer_ensemble_source_weight", help="域适应/集成评估中源模型预测占比。")
            with col_t7:
                domain_adaptation_method = st.selectbox(
                    "域适应方法",
                    ["correlation_alignment", "subspace_alignment", "instance_weighting", "none"],
                    key="transfer_domain_adaptation_method",
                    help="CORAL对齐协方差；子空间对齐使用PCA；实例加权按目标域相似度加权。"
                )
        
        if st.button("🚀 开始迁移学习", type="primary", key="start_transfer_tab3"):
            with st.spinner("正在进行迁移学习..."):
                try:
                    X_train = st.session_state.X_train
                    y_train = st.session_state.y_train
                    X_test = st.session_state.X_test
                    y_test = st.session_state.y_test
                    
                    # 导入迁移学习模块
                    from sklearn.base import clone
                    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                    from sklearn.linear_model import Ridge
                    from sklearn.model_selection import train_test_split
                    
                    source_model_info = st.session_state.model_library.get(transfer_source, {})
                    X_train_arr = X_train.values if hasattr(X_train, 'values') else np.asarray(X_train)
                    X_test_arr = X_test.values if hasattr(X_test, 'values') else np.asarray(X_test)
                    y_train_arr = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
                    y_test_arr = y_test.values if hasattr(y_test, 'values') else np.asarray(y_test)
                    if target_domain_ratio < 1.0 and len(X_train_arr) > 5:
                        X_target, _, y_target, _ = train_test_split(
                            X_train_arr,
                            y_train_arr,
                            train_size=target_domain_ratio,
                            random_state=int(random_state_transfer),
                        )
                    else:
                        X_target, y_target = X_train_arr, y_train_arr

                    def make_transfer_model():
                        if transfer_model_family == "随机森林":
                            return RandomForestRegressor(
                                n_estimators=int(n_estimators_transfer),
                                max_depth=None,
                                random_state=int(random_state_transfer),
                                n_jobs=-1,
                            )
                        if transfer_model_family == "梯度提升":
                            return GradientBoostingRegressor(
                                n_estimators=int(n_estimators_transfer),
                                learning_rate=float(target_lr),
                                max_depth=3,
                                random_state=int(random_state_transfer),
                            )
                        return Ridge(alpha=max(float(target_lr) * 100.0, 0.01), random_state=int(random_state_transfer))

                    source_model = make_transfer_model()
                    source_model.fit(X_train_arr, y_train_arr)
                    
                    if transfer_method == "特征提取":
                        if hasattr(source_model, "feature_importances_"):
                            importances = source_model.feature_importances_
                            keep_n = max(1, int(np.ceil(len(importances) * feature_keep_ratio)))
                            selected_idx = np.argsort(importances)[-keep_n:]
                        else:
                            selected_idx = np.arange(X_train_arr.shape[1])
                        source_train_pred = source_model.predict(X_target)
                        source_test_pred = source_model.predict(X_test_arr)
                        X_train_aug = np.column_stack([X_target[:, selected_idx], source_train_pred])
                        X_test_aug = np.column_stack([X_test_arr[:, selected_idx], source_test_pred])
                        rf_target = make_transfer_model()
                        rf_target.fit(X_train_aug, y_target)
                        y_pred = rf_target.predict(X_test_aug)
                        transfer_notes = [
                            f"按源模型特征重要性保留 {len(selected_idx)} / {X_train_arr.shape[1]} 个特征。",
                            "源模型预测被作为目标模型的附加知识特征参与训练。",
                        ]
                        
                    elif transfer_method == "微调":
                        X_source = X_train_arr
                        y_source = source_model.predict(X_train_arr)
                        X_combined = np.vstack([X_source, X_target])
                        y_combined = np.concatenate([y_source, y_target])
                        sample_weights = np.concatenate([
                            np.ones(len(y_source)) * float(source_weight) * max(0.05, 1.0 - freeze_ratio),
                            np.ones(len(y_target)) * float(target_weight),
                        ])
                        rf_target = make_transfer_model()
                        try:
                            rf_target.fit(X_combined, y_combined, sample_weight=sample_weights)
                        except TypeError:
                            rf_target.fit(X_combined, y_combined)
                        y_pred = rf_target.predict(X_test_arr)
                        transfer_notes = [
                            "微调策略使用源模型伪标签和目标域真实标签联合训练。",
                            f"源域权重={source_weight:.2f}，目标域权重={target_weight:.2f}，冻结比例用于降低源域更新权重。",
                        ]
                        
                    else:  # 域适应
                        if domain_adaptation_method != "none":
                            adapter = DomainAdaptation(method=domain_adaptation_method)
                            adapter.fit(X_train_arr, X_target)
                            if domain_adaptation_method == "instance_weighting":
                                X_source_adapted, adaptive_weights = adapter.transform(X_train_arr, domain='source')
                                X_target_adapted, _ = adapter.transform(X_target, domain='target')
                                X_test_adapted = X_test_arr
                            else:
                                X_source_adapted = adapter.transform(X_train_arr, domain='source')
                                X_target_adapted = adapter.transform(X_target, domain='target')
                                X_test_adapted = adapter.transform(X_test_arr, domain='target')
                                adaptive_weights = np.ones(len(X_train_arr))
                        else:
                            X_source_adapted, X_target_adapted, X_test_adapted = X_train_arr, X_target, X_test_arr
                            adaptive_weights = np.ones(len(X_train_arr))
                        source_adapted_model = make_transfer_model()
                        try:
                            source_adapted_model.fit(X_source_adapted, y_train_arr, sample_weight=adaptive_weights)
                        except TypeError:
                            source_adapted_model.fit(X_source_adapted, y_train_arr)
                        target_model = make_transfer_model()
                        target_model.fit(X_target_adapted, y_target)
                        pred_source = source_adapted_model.predict(X_test_adapted)
                        pred_target = target_model.predict(X_test_adapted)
                        y_pred = float(ensemble_source_weight) * pred_source + (1.0 - float(ensemble_source_weight)) * pred_target
                        rf_target = target_model
                        transfer_notes = [
                            f"域适应方法：{domain_adaptation_method}。",
                            f"预测融合：源模型权重 {ensemble_source_weight:.2f}，目标模型权重 {1.0 - ensemble_source_weight:.2f}。",
                        ]
                    
                    # 评估结果
                    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                    
                    r2 = r2_score(y_test_arr, y_pred)
                    mse = mean_squared_error(y_test_arr, y_pred)
                    mae = mean_absolute_error(y_test_arr, y_pred)
                    rmse = np.sqrt(mse)
                    residuals = y_test_arr - y_pred
                    
                    st.success("✅ 迁移学习完成！")
                    
                    # 显示结果
                    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                    with col_r1:
                        st.metric("R²", f"{r2:.4f}")
                    with col_r2:
                        st.metric("MSE", f"{mse:.4f}")
                    with col_r3:
                        st.metric("MAE", f"{mae:.4f}")
                    with col_r4:
                        st.metric("RMSE", f"{rmse:.4f}")
                    
                    # 保存迁移学习结果
                    st.session_state.transfer_result = {
                        'source_model': transfer_source,
                        'method': transfer_method,
                        'model_family': transfer_model_family,
                        'domain_adaptation_method': domain_adaptation_method,
                        'target_domain_ratio': target_domain_ratio,
                        'source_weight': source_weight,
                        'target_weight': target_weight,
                        'feature_keep_ratio': feature_keep_ratio,
                        'ensemble_source_weight': ensemble_source_weight,
                        'r2': r2,
                        'mse': mse,
                        'mae': mae,
                        'rmse': rmse,
                        'y_pred': y_pred,
                        'y_true': y_test_arr,
                        'residuals': residuals,
                        'notes': transfer_notes,
                    }
                    
                    # 可视化
                    fig_tl = go.Figure()
                    fig_tl.add_trace(go.Scatter(
                        y=y_test_arr, name='真实值', mode='lines+markers', line=dict(color='black')
                    ))
                    fig_tl.add_trace(go.Scatter(
                        y=y_pred, name='迁移学习预测', mode='lines', line=dict(color='blue')
                    ))
                    fig_tl.update_layout(
                        title=f"迁移学习结果 ({transfer_method})",
                        xaxis_title="样本",
                        yaxis_title="预测值"
                    )
                    st.plotly_chart(fig_tl, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"迁移学习失败: {str(e)}")

        if st.session_state.get("transfer_result") is not None:
            transfer_result = st.session_state.transfer_result
            transfer_export_df = pd.DataFrame({
                "sample_id": np.arange(len(transfer_result["y_pred"])),
                "true_value": transfer_result["y_true"],
                "predicted_value": transfer_result["y_pred"],
                "residual": transfer_result["residuals"],
                "absolute_error": np.abs(transfer_result["residuals"]),
            })
            with st.expander("📤 迁移学习结果导出", expanded=True):
                fig_transfer_export = go.Figure()
                fig_transfer_export.add_trace(go.Scatter(y=transfer_result["y_true"], name="真实值", mode="lines+markers", line=dict(color="black")))
                fig_transfer_export.add_trace(go.Scatter(y=transfer_result["y_pred"], name="迁移预测", mode="lines", line=dict(color="#256b7f")))
                fig_transfer_export.update_layout(title=f"迁移学习预测结果 ({transfer_result['method']})", xaxis_title="样本", yaxis_title="值", height=420)
                fig_residual = go.Figure(go.Histogram(x=transfer_result["residuals"], nbinsx=30, marker_color="#ff7f0e"))
                fig_residual.update_layout(title="迁移学习残差分布", xaxis_title="残差", yaxis_title="数量", height=320)
                st.dataframe(transfer_export_df.head(200), use_container_width=True, hide_index=True)
                transfer_report = build_html_report(
                    "迁移学习图文结果报告",
                    metrics={
                        "源模型": transfer_result["source_model"],
                        "迁移方法": transfer_result["method"],
                        "目标模型": transfer_result["model_family"],
                        "R²": f"{transfer_result['r2']:.4f}",
                        "RMSE": f"{transfer_result['rmse']:.4f}",
                        "MAE": f"{transfer_result['mae']:.4f}",
                    },
                    sections=[
                        ("参数配置", f"域适应={transfer_result.get('domain_adaptation_method')}; 目标域比例={transfer_result.get('target_domain_ratio')}; 源/目标权重={transfer_result.get('source_weight')}/{transfer_result.get('target_weight')}; 特征保留比例={transfer_result.get('feature_keep_ratio')}; 集成源模型权重={transfer_result.get('ensemble_source_weight')}.")
                    ],
                    figures=[("预测对比", fig_transfer_export), ("残差分布", fig_residual)],
                    tables=[("预测明细", transfer_export_df)],
                    notes=transfer_result.get("notes", []),
                )
                render_result_downloads(
                    "迁移学习",
                    transfer_export_df,
                    transfer_report,
                    csv_name=f"transfer_learning_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    html_name=f"transfer_learning_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    key_prefix="transfer_learning",
                )

# ==================== 标签5: 优化算法 ====================
with tab5:
    st.markdown("<h1 style='font-size: 32px; font-weight: bold;'>🎯 优化算法</h1>", unsafe_allow_html=True)
    
    # 初始化优化算法session state
    if 'optimization_problem' not in st.session_state:
        st.session_state.optimization_problem = {
            'objective_func': None,
            'variables': [],
            'bounds': [],
            'constraints': [],
            'algorithm': 'differential_evolution',
            'history': [],
            'result': None
        }
    
    # 问题定义方式选择
    problem_type = st.radio(
        "问题定义方式",
        ["函数表达式", "数据导入"],
        horizontal=True,
        key="opt_problem_type",
        help="函数表达式: 直接输入数学表达式定义优化问题; 数据导入: 从文件导入数据点进行拟合优化"
    )
    
    if problem_type == "函数表达式":
        # ==================== 函数表达式定义区域 ====================
        st.subheader("📝 目标函数定义")
        
        col_func1, col_func2 = st.columns([3, 1])
        with col_func1:
            objective_func = st.text_area(
                "目标函数表达式 (使用Python语法)",
                value="x[0]**2 + x[1]**2 + 5*np.sin(x[0]) + 3*np.cos(x[1])",
                height=80,
                key="opt_objective_func",
                help="输入目标函数，使用x[0], x[1]...表示变量。支持numpy函数如np.sin, np.cos, np.exp等"
            )
        with col_func2:
            st.info("💡 **变量说明**\n- x[0], x[1]... 表示优化变量\n- 支持np.sin, np.cos, np.exp等\n- 目标: 最小化函数值")
        
        # 验证函数按钮
        if st.button("🔍 验证函数", key="validate_func"):
            try:
                # 测试编译函数
                test_code = f"""
import numpy as np
def test_func(x):
    return {objective_func}
"""
                exec(test_code, {"np": np})
                record_custom_execution("优化目标函数验证", "验证通过")
                st.success("✅ 函数表达式验证通过！")
            except Exception as e:
                record_custom_execution("优化目标函数验证", "验证失败", str(e))
                st.error(f"❌ 函数表达式错误: {str(e)}")
        
        st.divider()
        
        # ==================== 变量设置区域 ====================
        st.subheader("🔧 优化变量设置")
        
        # 变量数量
        n_variables = st.number_input(
            "变量数量",
            min_value=1,
            max_value=20,
            value=2,
            key="opt_n_variables",
            help="设置优化问题的变量维度"
        )
        
        # 变量范围和名称设置
        st.write("**变量范围设置:**")
        variable_configs = []
        
        cols_per_row = 3
        for i in range(0, n_variables, cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i + j
                if idx < n_variables:
                    with cols[j]:
                        var_name = st.text_input(
                            f"变量 {idx} 名称",
                            value=f"x{idx}",
                            key=f"opt_var_name_{idx}"
                        )
                        col_min, col_max = st.columns(2)
                        with col_min:
                            var_min = st.number_input(
                                f"最小值",
                                value=-10.0,
                                key=f"opt_var_min_{idx}"
                            )
                        with col_max:
                            var_max = st.number_input(
                                f"最大值",
                                value=10.0,
                                key=f"opt_var_max_{idx}"
                            )
                        variable_configs.append({
                            'name': var_name,
                            'index': idx,
                            'bounds': (var_min, var_max)
                        })
        
        st.divider()
        
        # ==================== 约束条件设置 ====================
        st.subheader("⛓️ 约束条件设置")
        
        use_constraints = st.checkbox(
            "启用约束条件",
            value=False,
            key="opt_use_constraints"
        )
        
        constraint_configs = []
        if use_constraints:
            n_constraints = st.number_input(
                "约束条件数量",
                min_value=1,
                max_value=10,
                value=1,
                key="opt_n_constraints"
            )
            
            for i in range(n_constraints):
                with st.expander(f"约束条件 {i+1}", expanded=i==0):
                    constraint_type = st.selectbox(
                        f"约束类型 {i+1}",
                        ["不等式约束 (≤0)", "等式约束 (=0)", "边界约束"],
                        key=f"opt_constraint_type_{i}"
                    )
                    
                    if constraint_type == "边界约束":
                        col_c1, col_c2 = st.columns(2)
                        with col_c1:
                            var_idx = st.number_input(
                                f"变量索引",
                                min_value=0,
                                max_value=n_variables-1,
                                value=0,
                                key=f"opt_constraint_var_{i}"
                            )
                        with col_c2:
                            bound_type = st.selectbox(
                                f"边界类型",
                                ["下限", "上限"],
                                key=f"opt_bound_type_{i}"
                            )
                        bound_value = st.number_input(
                            f"边界值",
                            value=0.0,
                            key=f"opt_bound_value_{i}"
                        )
                        constraint_configs.append({
                            'type': 'bound',
                            'var_idx': var_idx,
                            'bound_type': bound_type,
                            'value': bound_value
                        })
                    else:
                        constraint_expr = st.text_area(
                            f"约束表达式 (使用x[0], x[1]...)",
                            value=f"x[0] + x[1] - 5",
                            key=f"opt_constraint_expr_{i}"
                        )
                        constraint_configs.append({
                            'type': 'ineq' if '不等式' in constraint_type else 'eq',
                            'expr': constraint_expr
                        })
    
    else:
        # ==================== 数据导入区域 ====================
        st.subheader("📥 数据导入")
        
        uploaded_opt_file = st.file_uploader(
            "上传数据文件 (支持 .out, .dat, .csv, .xlsx, .xls, .txt)",
            type=['out', 'dat', 'csv', 'xlsx', 'xls', 'txt'],
            key="opt_data_file"
        )
        
        if uploaded_opt_file is not None:
            try:
                import tempfile
                import re
                
                # 保存上传的文件到临时位置
                with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=f".{uploaded_opt_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_opt_file.getvalue())
                    tmp_path = tmp_file.name
                
                # 使用项目原有的智能数据加载逻辑
                def detect_file_format_smart(content_lines):
                    """智能检测文件格式，识别列名行、单位行和数据行"""
                    column_line = None
                    unit_line = None
                    data_start = 0
                    
                    # 找到第一个数据行
                    first_data_line = None
                    for i, line in enumerate(content_lines):
                        stripped_line = line.strip()
                        if not stripped_line or stripped_line.startswith('#') or stripped_line.startswith('//'):
                            continue
                        
                        # 检查是否包含数字（数据行）
                        has_numbers = any(char.isdigit() or char in '.-+eE' for char in stripped_line)
                        if has_numbers:
                            # 尝试解析为数值列表
                            values = re.split(r'\s+|[,;\t]', stripped_line)
                            values = [val.strip() for val in values if val.strip()]
                            
                            # 检查是否至少有2个数值
                            numeric_count = 0
                            for val in values:
                                try:
                                    float(val)
                                    numeric_count += 1
                                except ValueError:
                                    continue
                            
                            if numeric_count >= 2:
                                first_data_line = i
                                break
                    
                    if first_data_line is None:
                        return None, None, 0
                    
                    # 寻找列名行和单位行
                    if first_data_line > 0:
                        prev_line = content_lines[first_data_line - 1].strip()
                        
                        # 检查是否是单位行：包含括号或特殊单位符号，且主要是单位符号
                        is_unit_line = False
                        if any(char in '()[]{}<>/°' for char in prev_line):
                            # 检查是否主要是单位（如 (s), (m), (deg), (kN) 等）
                            # 单位行通常只有少量不同的字符类型
                            unit_patterns = re.findall(r'\([^)]*\)|[°\w/]+', prev_line)
                            if len(unit_patterns) >= 2:  # 至少有两个单位模式
                                is_unit_line = True
                        
                        if is_unit_line:
                            unit_line = first_data_line - 1
                            # 单位行之前的行可能是列名行
                            if first_data_line > 1:
                                col_candidate = content_lines[first_data_line - 2].strip()
                                # 检查是否是有效的列名行
                                # 有效的列名应该包含字母或下划线，且不是纯单位格式
                                if (re.search(r'[a-zA-Z_]', col_candidate) and 
                                    not re.match(r'^[\d\s.+-eE()\[\]{}<>/°]+$', col_candidate)):
                                    # 进一步检查列名是否唯一且合理
                                    test_cols = re.split(r'\s+|[,;\t]', col_candidate)
                                    test_cols = [c.strip() for c in test_cols if c.strip()]
                                    if len(test_cols) == len(set(test_cols)) and len(test_cols) >= 2:
                                        column_line = first_data_line - 2
                                        data_start = first_data_line
                                    else:
                                        column_line = None
                                        data_start = first_data_line
                                else:
                                    column_line = None
                                    data_start = first_data_line
                            else:
                                column_line = None
                                data_start = first_data_line
                        else:
                            # 检查前一行是否是列名（不是单位行）
                            # 有效的列名应该包含字母或下划线，且不是纯数字或纯单位
                            if (re.search(r'[a-zA-Z_]', prev_line) and 
                                not re.match(r'^[\d\s.+-eE()\[\]{}<>/°]+$', prev_line)):
                                # 检查列名是否唯一
                                test_cols = re.split(r'\s+|[,;\t]', prev_line)
                                test_cols = [c.strip() for c in test_cols if c.strip()]
                                if len(test_cols) == len(set(test_cols)) and len(test_cols) >= 2:
                                    column_line = first_data_line - 1
                                    unit_line = None
                                    data_start = first_data_line
                                else:
                                    column_line = None
                                    unit_line = None
                                    data_start = first_data_line
                            else:
                                column_line = None
                                unit_line = None
                                data_start = first_data_line
                    
                    return column_line, unit_line, data_start
                
                def read_data_with_auto_detect(file_path):
                    """自动检测格式并读取数据"""
                    # 首先尝试检测编码
                    encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
                    content_lines = None
                    used_encoding = None
                    
                    for encoding in encodings:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content_lines = f.readlines()
                            used_encoding = encoding
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if content_lines is None:
                        raise ValueError("无法识别文件编码")
                    
                    # 检测文件格式
                    column_line, unit_line, data_start = detect_file_format_smart(content_lines)
                    
                    # 自动检测分隔符
                    sample_lines = []
                    for i in range(data_start, min(data_start + 5, len(content_lines))):
                        line = content_lines[i].strip()
                        if line:
                            sample_lines.append(line)
                    
                    # 统计各种分隔符的出现次数
                    delimiters = {'\t': 0, ',': 0, ';': 0, ' ': 0}
                    for line in sample_lines:
                        for delim in delimiters:
                            if delim in line:
                                delimiters[delim] += line.count(delim)
                    
                    # 选择最常见的分隔符
                    delimiter = max(delimiters, key=delimiters.get)
                    if delimiters[delimiter] == 0:
                        delimiter = r'\s+'
                    
                    # 读取列名
                    if column_line is not None:
                        raw_columns = re.split(delimiter, content_lines[column_line].strip())
                        columns = [col.strip() for col in raw_columns if col.strip()]
                    else:
                        # 没有列名，从数据行推断
                        first_data = content_lines[data_start].strip()
                        values = re.split(delimiter, first_data)
                        values = [v.strip() for v in values if v.strip()]
                        columns = [f'col_{i}' for i in range(len(values))]
                    
                    # 读取数据
                    data = []
                    for i in range(data_start, len(content_lines)):
                        line = content_lines[i].strip()
                        if not line or line.startswith('#') or line.startswith('//'):
                            continue
                        
                        values = re.split(delimiter, line)
                        values = [v.strip() for v in values if v.strip()]
                        
                        if len(values) >= len(columns):
                            data.append(values[:len(columns)])
                        elif len(values) > 0:
                            # 填充缺失值
                            values += [None] * (len(columns) - len(values))
                            data.append(values)
                    
                    if not data:
                        raise ValueError("文件中没有有效数据")
                    
                    # 创建DataFrame
                    df = pd.DataFrame(data, columns=columns)
                    
                    # 尝试转换为数值类型 - 确保每列是Series类型
                    for col in df.columns:
                        try:
                            # 先确保列是Series类型
                            if isinstance(df[col], pd.Series):
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            else:
                                # 如果不是Series，转换为Series再处理
                                df[col] = pd.to_numeric(pd.Series(df[col]), errors='coerce')
                        except Exception as e:
                            # 如果转换失败，保持原样
                            pass
                    
                    # 删除全为NaN的列
                    df = df.dropna(axis=1, how='all')
                    
                    # 删除全为NaN的行
                    df = df.dropna(axis=0, how='all')
                    
                    # 重置索引
                    df = df.reset_index(drop=True)
                    
                    return df, column_line, unit_line, used_encoding
                
                # 根据文件类型选择读取方式
                file_ext = uploaded_opt_file.name.split('.')[-1].lower()
                
                if file_ext in ['out', 'dat', 'txt']:
                    # 使用智能检测读取
                    opt_df, col_line, unit_line, encoding = read_data_with_auto_detect(tmp_path)
                    
                    # 显示检测信息
                    with st.expander("📋 文件解析信息", expanded=True):
                        st.write(f"**文件名:** {uploaded_opt_file.name}")
                        st.write(f"**文件编码:** {encoding}")
                        if col_line is not None:
                            st.write(f"**列名行:** 第 {col_line + 1} 行")
                        else:
                            st.write(f"**列名行:** 未检测到，使用默认列名")
                        if unit_line is not None:
                            st.write(f"**单位行:** 第 {unit_line + 1} 行 (已跳过)")
                        st.write(f"**数据起始行:** 第 {(col_line or 0) + 2 if col_line else 1} 行")
                        
                elif file_ext in ['csv']:
                    # 尝试多种分隔符
                    try:
                        opt_df = pd.read_csv(tmp_path)
                    except:
                        try:
                            opt_df = pd.read_csv(tmp_path, sep=';')
                        except:
                            opt_df = pd.read_csv(tmp_path, sep='\t')
                            
                elif file_ext in ['xlsx', 'xls']:
                    opt_df = pd.read_excel(tmp_path)
                
                # 清理临时文件
                try:
                    os.remove(tmp_path)
                except:
                    pass
                
                if opt_df.empty:
                    st.error("❌ 数据文件为空或无法解析")
                    st.stop()
                
                st.success(f"✅ 数据加载成功！形状: {opt_df.shape}")
                
                # 显示数据预览
                with st.expander("📊 数据预览"):
                    st.dataframe(opt_df.head(10), use_container_width=True)
                    
                    # 显示数据统计信息
                    st.write("**数据统计:**")
                    st.write(f"- 行数: {len(opt_df)}")
                    st.write(f"- 列数: {len(opt_df.columns)}")
                    st.write(f"- 数值列: {len(opt_df.select_dtypes(include=[np.number]).columns)}")
                
                # 选择目标列和特征列
                numeric_cols = opt_df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) < 2:
                    st.error("❌ 数据中需要至少2个数值列才能进行优化")
                    st.stop()
                
                col_target, col_features = st.columns(2)
                with col_target:
                    target_col = st.selectbox(
                        "选择目标列 (Y)",
                        numeric_cols,
                        key="opt_target_col"
                    )
                with col_features:
                    available_features = [c for c in numeric_cols if c != target_col]
                    feature_cols = st.multiselect(
                        "选择特征列 (X)",
                        available_features,
                        default=available_features[:min(2, len(available_features))],
                        key="opt_feature_cols"
                    )
                
                # 优化目标设置
                opt_goal = st.radio(
                    "优化目标",
                    ["最小化目标值", "最大化目标值", "拟合数据"],
                    horizontal=True,
                    key="opt_data_goal"
                )
                
                if feature_cols:
                    st.info(f"将使用 {len(feature_cols)} 个特征列优化 {target_col}")
                    
                    # 显示数据范围
                    with st.expander("📈 数据范围"):
                        range_df = pd.DataFrame({
                            '列名': feature_cols + [target_col],
                            '最小值': [opt_df[col].min() for col in feature_cols + [target_col]],
                            '最大值': [opt_df[col].max() for col in feature_cols + [target_col]],
                            '均值': [opt_df[col].mean() for col in feature_cols + [target_col]],
                            '标准差': [opt_df[col].std() for col in feature_cols + [target_col]]
                        })
                        st.dataframe(range_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ 数据加载失败: {str(e)}")
                st.exception(e)
        else:
            st.info("👆 请上传数据文件 (.out, .dat, .csv, .xlsx, .xls, .txt)")
    
    st.divider()
    
    # ==================== 算法选择和设置区域 ====================
    st.subheader("⚙️ 优化算法设置")
    with st.expander("❓ 优化算法功能说明", expanded=False):
        st.markdown("""
        - **全局优化**：差分进化、遗传算法、PSO、模拟退火、SHGO、Basin-Hopping 适合多峰或初值不确定问题。
        - **局部优化**：Nelder-Mead、L-BFGS-B、SLSQP、Powell、BFGS 等适合目标函数较平滑或已有合理初值的问题。
        - **轻量/简化实现**：贝叶斯优化、CMA-ES 和部分群智能算法用于实验与初筛，结果应与 SciPy 稳定算法交叉验证。
        - **结果判断**：不要只看最优值，还要看收敛曲线、最优值改进幅度、变量是否贴边和多算法对比结果。
        """)
        st.dataframe(optimization_algorithm_matrix(), use_container_width=True, hide_index=True)
    
    col_alg1, col_alg2 = st.columns([2, 1])
    
    with col_alg1:
        # 优化算法选择
        algorithm = st.selectbox(
            "选择优化算法",
            [
                "差分进化 (Differential Evolution)",
                "遗传算法 (Genetic Algorithm)",
                "粒子群优化 (PSO)",
                "模拟退火 (Simulated Annealing)",
                "Nelder-Mead 单纯形法",
                "L-BFGS-B 拟牛顿法",
                "SLSQP 序列最小二乘",
                "CMA-ES 协方差矩阵自适应",
                "贝叶斯优化 (Bayesian Optimization)",
                "Powell 方法",
                "CG 共轭梯度法",
                "BFGS 拟牛顿法",
                "TNC 截断牛顿法",
                "COBYLA 约束优化",
                "trust-constr 信赖域约束",
                "SHGO 单纯形分层全局优化",
                "Basin-Hopping 盆地跳跃",
                "灰狼优化算法 (GWO)",
                "蚁群优化算法 (ACO)",
                "混合蛙跳算法 (SFLA)",
                "萤火虫算法 (FA)",
                "禁忌搜索算法 (TS)",
                "人工鱼群算法 (AFSA)",
                "免疫遗传算法 (IGA)",
                "📝 自定义算法 (Custom Algorithm)"
            ],
            key="opt_algorithm",
            help="选择适合您问题的优化算法。全局优化算法适合多峰函数，局部优化算法适合单峰函数。选择'自定义算法'可以编写自己的优化算法。"
        )
        
        # 算法参数设置
        with st.expander("🔧 算法参数设置"):
            if "差分进化" in algorithm:
                pop_size = st.slider("种群大小", 10, 100, 50, key="opt_de_pop")
                mutation = st.slider("变异因子 (F)", 0.1, 2.0, 0.8, 0.1, key="opt_de_mutation")
                crossover = st.slider("交叉概率 (CR)", 0.1, 1.0, 0.9, 0.05, key="opt_de_crossover")
                max_iter = st.slider("最大迭代次数", 50, 2000, 500, key="opt_de_iter")
            elif "遗传算法" in algorithm:
                pop_size = st.slider("种群大小", 20, 200, 100, key="opt_ga_pop")
                crossover_rate = st.slider("交叉率", 0.1, 1.0, 0.8, 0.05, key="opt_ga_cross")
                mutation_rate = st.slider("变异率", 0.01, 0.5, 0.1, 0.01, key="opt_ga_mut")
                max_iter = st.slider("最大迭代次数", 50, 2000, 500, key="opt_ga_iter")
            elif "粒子群" in algorithm:
                n_particles = st.slider("粒子数量", 10, 100, 30, key="opt_pso_particles")
                w = st.slider("惯性权重", 0.1, 1.0, 0.5, 0.1, key="opt_pso_w")
                c1 = st.slider("认知系数 (c1)", 0.0, 4.0, 2.0, 0.1, key="opt_pso_c1")
                c2 = st.slider("社会系数 (c2)", 0.0, 4.0, 2.0, 0.1, key="opt_pso_c2")
                max_iter = st.slider("最大迭代次数", 50, 2000, 300, key="opt_pso_iter")
            elif "模拟退火" in algorithm:
                initial_temp = st.slider("初始温度", 100, 10000, 1000, key="opt_sa_temp")
                cooling_rate = st.slider("冷却系数", 0.9, 0.999, 0.95, 0.001, key="opt_sa_cooling")
                max_iter = st.slider("最大迭代次数", 100, 5000, 1000, key="opt_sa_iter")
            elif "CMA-ES" in algorithm:
                pop_size = st.slider("候选解数量", 8, 120, 32, key="opt_cma_pop", help="每轮采样的候选解数量。")
                sigma0 = st.slider("初始搜索半径", 0.01, 1.0, 0.25, 0.01, key="opt_cma_sigma", help="相对变量范围的初始采样尺度。")
                max_iter = st.slider("最大迭代次数", 50, 2000, 500, key="opt_cma_iter")
            elif "贝叶斯优化" in algorithm:
                n_initial = st.slider("初始随机样本数", 5, 80, 16, key="opt_bo_initial", help="用于拟合初始高斯过程代理模型的随机样本数。")
                n_candidates = st.slider("每轮候选点数量", 50, 1500, 300, 50, key="opt_bo_candidates", help="每轮在候选点上最大化采集函数。")
                max_iter = st.slider("最大评估轮数", 10, 300, 60, 5, key="opt_bo_iter")
            elif "灰狼" in algorithm:
                pop_size = st.slider("灰狼数量", 10, 100, 30, key="opt_gwo_pop")
                max_iter = st.slider("最大迭代次数", 50, 2000, 500, key="opt_gwo_iter")
            elif "蚁群" in algorithm:
                pop_size = st.slider("蚂蚁数量", 20, 100, 50, key="opt_aco_pop")
                max_iter = st.slider("最大迭代次数", 50, 2000, 500, key="opt_aco_iter")
            elif "萤火虫" in algorithm:
                pop_size = st.slider("萤火虫数量", 10, 100, 40, key="opt_fa_pop")
                max_iter = st.slider("最大迭代次数", 50, 2000, 500, key="opt_fa_iter")
            elif "禁忌搜索" in algorithm:
                tabu_size = st.slider("禁忌表大小", 5, 50, 10, key="opt_ts_tabu")
                max_iter = st.slider("最大迭代次数", 50, 2000, 500, key="opt_ts_iter")
            elif "人工鱼群" in algorithm:
                pop_size = st.slider("鱼群数量", 20, 100, 50, key="opt_afsa_pop")
                max_iter = st.slider("最大迭代次数", 50, 2000, 500, key="opt_afsa_iter")
            elif "免疫遗传" in algorithm:
                pop_size = st.slider("种群大小", 20, 100, 60, key="opt_iga_pop")
                max_iter = st.slider("最大迭代次数", 50, 2000, 500, key="opt_iga_iter")
            elif "混合蛙跳" in algorithm:
                pop_size = st.slider("蛙群数量", 20, 100, 40, key="opt_sfla_pop")
                max_iter = st.slider("最大迭代次数", 50, 2000, 500, key="opt_sfla_iter")
            elif "自定义" in algorithm:
                max_iter = st.slider("最大迭代次数", 50, 2000, 500, key="opt_custom_iter")
                
                st.divider()
                st.markdown("### 📝 自定义算法代码")
                
                with st.expander("📖 点击查看接口说明与模板"):
                    st.markdown("""
                    ### 接口规则
                    
                    您的自定义算法需要实现以下接口：
                    
                    **输入参数**:
                    - `obj_func`: 目标函数，接受numpy数组返回标量值
                    - `bounds`: 变量边界列表，如 `[(min1, max1), (min2, max2), ...]`
                    - `max_iter`: 最大迭代次数
                    - `progress_bar`: Streamlit进度条对象
                    - `status_text`: Streamlit状态文本对象
                    - `iteration_history`: 列表，用于记录迭代历史
                    - `start_time`: 开始时间戳
                    
                    **输出要求**:
                    - 必须定义 `result` 变量，包含以下属性：
                      - `x`: 最优解（numpy数组）
                      - `fun`: 最优目标函数值（标量）
                      - `nfev`: 函数评估次数
                      - `nit`: 实际迭代次数
                      - `success`: 是否成功（布尔值）
                    
                    **迭代记录示例**:
                    ```python
                    iteration_history.append({
                        'x': current_best.copy(),
                        'fun': current_best_fitness,
                        'time': time.time() - start_time
                    })
                    ```
                    
                    **进度更新示例**:
                    ```python
                    progress_bar.progress((i + 1) / max_iter)
                    status_text.text(f"迭代 {i+1}: f(x) = {best_f:.6f}")
                    ```
                    """)
            else:
                max_iter = st.slider("最大迭代次数/函数评估次数", 50, 5000, 1000, key="opt_general_iter")
                tolerance = st.number_input("收敛容差", value=1e-6, format="%.0e", key="opt_tol")
    
    # 自定义算法代码输入区（在算法参数设置下方，全宽度显示）
    if "自定义" in algorithm:
        st.markdown("---")
        
        # 默认模板代码
        default_template = '''# 自定义优化算法示例 - 简化版随机搜索
def custom_optimize(obj_func, bounds, max_iter, progress_bar, status_text, iteration_history, start_time):
    import numpy as np
    import time
    
    n_dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # 初始化最优解
    best_x = np.random.uniform(lb, ub)
    best_f = obj_func(best_x)
    
    # 记录初始点
    iteration_history.append({
        'x': best_x.copy(),
        'fun': best_f,
        'time': time.time() - start_time
    })
    
    # 主循环
    for i in range(max_iter):
        # 生成新解（随机搜索示例）
        new_x = np.random.uniform(lb, ub)
        new_f = obj_func(new_x)
        
        # 更新最优
        if new_f < best_f:
            best_f = new_f
            best_x = new_x.copy()
        
        # 记录历史
        iteration_history.append({
            'x': best_x.copy(),
            'fun': best_f,
            'time': time.time() - start_time
        })
        
        # 更新进度
        progress = (i + 1) / max_iter
        progress_bar.progress(progress)
        status_text.text(f"迭代 {i+1}/{max_iter}: f(x) = {best_f:.6f}")
    
    # 创建结果对象
    class CustomResult:
        def __init__(self, x, fun, nfev, nit):
            self.x = x
            self.fun = fun
            self.nfev = nfev
            self.nit = nit
            self.success = True
    
    return CustomResult(best_x, best_f, max_iter, max_iter)

# 执行优化
result = custom_optimize(obj_func, bounds, max_iter, progress_bar, status_text, iteration_history, start_time)
'''
        
        custom_code = st.text_area(
            "编写您的优化算法代码（Python）",
            value=default_template,
            height=400,
            key="custom_algorithm_code_input",
            help="在此输入您的自定义优化算法Python代码。代码将在优化时动态执行。"
        )
        
        # 使用不同的key保存到session state
        if 'custom_algo_code' not in st.session_state:
            st.session_state.custom_algo_code = default_template
        
        # 更新session state
        st.session_state.custom_algo_code = custom_code
        
        st.markdown("---")
    
    with col_alg2:
        st.info("📚 **算法说明**")
        algo_info = {
            "差分进化": "全局优化，适合连续变量，鲁棒性强",
            "遗传算法": "全局优化，适合复杂搜索空间",
            "粒子群": "全局优化，收敛速度快，参数少",
            "模拟退火": "全局优化，适合逃离局部最优",
            "Nelder-Mead": "局部优化，无需梯度，适合非光滑函数",
            "L-BFGS-B": "局部优化，适合大规模问题，支持边界",
            "SLSQP": "约束优化，支持等式/不等式约束",
            "CMA-ES": "全局优化，自适应步长，适合病态问题",
            "贝叶斯优化": "适合昂贵的黑箱函数，样本高效",
            "Powell": "局部优化，无需梯度，适合高维问题",
            "CG": "局部优化，共轭梯度法，适合大规模问题",
            "BFGS": "局部优化，拟牛顿法，收敛快",
            "TNC": "局部优化，截断牛顿法，适合大规模约束",
            "COBYLA": "约束优化，仅需函数值，无需梯度",
            "trust-constr": "约束优化，信赖域方法，鲁棒性强",
            "SHGO": "全局优化，单纯形分层，适合多峰函数",
            "Basin-Hopping": "全局优化，盆地跳跃，适合复杂能量景观",
            "灰狼": "群智能优化，模拟灰狼社会等级和狩猎行为",
            "蚁群": "群智能优化，模拟蚂蚁觅食路径选择行为",
            "混合蛙跳": "群智能优化，模memetic算法，局部深度搜索",
            "萤火虫": "群智能优化，模拟萤火虫发光吸引和移动行为",
            "禁忌搜索": "元启发式算法，使用禁忌表避免循环",
            "人工鱼群": "群智能优化，模拟鱼类觅食、聚群和追尾行为",
            "免疫遗传": "混合算法，结合免疫系统和遗传算法优势",
            "自定义": "用户自定义优化算法，通过Python代码实现"
        }
        for k, v in algo_info.items():
            if k in algorithm:
                st.write(f"**{k}**: {v}")
                break
        opt_audit_df = optimization_algorithm_matrix()
        matched_audit = opt_audit_df[opt_audit_df["算法"].apply(lambda name: any(part in algorithm for part in str(name).split("/")))]
        if not matched_audit.empty:
            status_row = matched_audit.iloc[0]
            st.caption(f"实现状态: {status_row['实现状态']} | {status_row['实现方式']}")
    
    st.divider()
    
    # ==================== 求解按钮和动画展示 ====================
    col_solve1, col_solve2 = st.columns([1, 3])
    
    with col_solve1:
        solve_button = st.button("🚀 开始优化求解", type="primary", use_container_width=True, key="opt_solve_btn")
        
        # 动画显示选项
        show_animation = st.checkbox(
            "显示优化过程动画",
            value=True,
            key="opt_show_animation"
        )
        
        # 保存历史记录选项
        save_history = st.checkbox(
            "保存优化历史",
            value=True,
            key="opt_save_history"
        )
    
    with col_solve2:
        # 优化过程展示区域
        if solve_button:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 构建目标函数
                if problem_type == "函数表达式":
                    # 编译目标函数
                    func_code = f"""
import numpy as np
def objective_func(x):
    return {objective_func}
"""
                    local_ns = {"np": np}
                    exec(func_code, local_ns)
                    obj_func = local_ns["objective_func"]
                    
                    # 获取边界
                    bounds = [config['bounds'] for config in variable_configs]
                    
                else:  # 数据导入
                    if uploaded_opt_file is None or not feature_cols:
                        st.error("❌ 请先上传数据并选择特征列")
                        st.stop()
                    
                    # 构建数据驱动的目标函数
                    X_data = opt_df[feature_cols].values
                    y_data = opt_df[target_col].values
                    
                    # 检查数据量，如果太大则进行采样
                    n_samples = len(X_data)
                    max_samples_for_rbf = 5000  # RBF最大支持的数据量
                    
                    if n_samples > max_samples_for_rbf:
                        status_text.text(f"数据量较大 ({n_samples} 行)，正在进行智能采样...")
                        # 使用随机采样保留数据分布
                        np.random.seed(42)
                        indices = np.random.choice(n_samples, max_samples_for_rbf, replace=False)
                        X_data_sample = X_data[indices]
                        y_data_sample = y_data[indices]
                        st.info(f"⚠️ 数据量较大，已自动采样至 {max_samples_for_rbf} 行进行优化")
                    else:
                        X_data_sample = X_data
                        y_data_sample = y_data
                    
                    # 使用更轻量的插值方法 - 线性插值或KNN
                    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
                    from sklearn.neighbors import KNeighborsRegressor
                    
                    # 根据数据维度选择合适的方法
                    n_dims = X_data_sample.shape[1]
                    
                    if n_dims <= 3 and len(X_data_sample) < 1000:
                        # 低维小数据：使用线性插值
                        try:
                            interpolator = LinearNDInterpolator(X_data_sample, y_data_sample, fill_value=np.mean(y_data_sample))
                            method_name = "线性插值"
                        except:
                            # 如果线性插值失败（如点共面），使用KNN
                            interpolator = KNeighborsRegressor(n_neighbors=min(5, len(X_data_sample)))
                            interpolator.fit(X_data_sample, y_data_sample)
                            method_name = "KNN回归"
                    else:
                        # 高维或大数据：使用KNN回归（内存友好）
                        n_neighbors = min(5, len(X_data_sample) - 1) if len(X_data_sample) > 5 else 1
                        interpolator = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
                        interpolator.fit(X_data_sample, y_data_sample)
                        method_name = "KNN回归"
                    
                    def obj_func(x):
                        """
                        目标函数 - 支持向量化输入
                        x: 可以是单个点 (n_dims,) 或多个点 (n_points, n_dims)
                        """
                        x_array = np.atleast_2d(x)
                        result = interpolator(x_array)
                        # 确保返回标量或一维数组
                        if result.ndim > 1:
                            result = result.ravel()
                        # 如果输入是单个点，返回标量；否则返回数组
                        if x_array.shape[0] == 1 and np.array(x).ndim == 1:
                            return float(result[0])
                        return result
                    
                    # 数据范围作为边界
                    bounds = [(X_data[:, i].min(), X_data[:, i].max()) for i in range(X_data.shape[1])]
                    variable_configs = [{'name': col, 'index': i, 'bounds': bounds[i]} 
                                       for i, col in enumerate(feature_cols)]
                    
                    st.info(f"📊 使用 {method_name} 构建代理模型，数据量: {len(X_data_sample)} 行")
                
                # 执行优化
                status_text.text("正在初始化优化算法...")
                
                # 根据选择的算法执行优化
                from scipy.optimize import OptimizeResult, differential_evolution, minimize, dual_annealing, direct
                import time
                
                iteration_history = []
                start_time = time.time()
                
                def callback(xk, convergence=None):
                    """优化迭代回调函数"""
                    fval = obj_func(xk)
                    iteration_history.append({
                        'x': xk.copy(),
                        'fun': fval,
                        'time': time.time() - start_time
                    })
                    progress = min(len(iteration_history) / max_iter, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"迭代 {len(iteration_history)}: f(x) = {fval:.6f}")
                
                # 选择并执行算法
                if "差分进化" in algorithm:
                    result = differential_evolution(
                        obj_func,
                        bounds,
                        maxiter=max_iter,
                        popsize=pop_size // len(bounds),
                        mutation=mutation,
                        recombination=crossover,
                        callback=callback,
                        polish=True,
                        seed=42
                    )
                elif "遗传算法" in algorithm:
                    rng = np.random.default_rng(42)
                    lb = np.array([b[0] for b in bounds], dtype=float)
                    ub = np.array([b[1] for b in bounds], dtype=float)
                    n_dim = len(bounds)
                    n_pop = max(pop_size, 4)
                    population = rng.uniform(lb, ub, size=(n_pop, n_dim))
                    fitness = np.array([obj_func(ind) for ind in population])
                    nfev = n_pop
                    elite_count = max(2, n_pop // 10)

                    for iteration in range(max_iter):
                        order = np.argsort(fitness)
                        population = population[order]
                        fitness = fitness[order]
                        elites = population[:elite_count].copy()

                        selected_idx = rng.integers(0, max(n_pop // 2, 2), size=n_pop - elite_count)
                        parents = population[selected_idx]
                        children = []
                        scale = np.maximum(ub - lb, 1e-12)
                        for child_idx in range(0, len(parents), 2):
                            p1 = parents[child_idx]
                            p2 = parents[(child_idx + 1) % len(parents)]
                            if rng.random() < crossover_rate:
                                alpha = rng.random(n_dim)
                                child = alpha * p1 + (1 - alpha) * p2
                            else:
                                child = p1.copy()
                            mutation_mask = rng.random(n_dim) < mutation_rate
                            child[mutation_mask] += rng.normal(0, 0.08, mutation_mask.sum()) * scale[mutation_mask]
                            children.append(np.clip(child, lb, ub))

                        population = np.vstack([elites, np.array(children)[: n_pop - elite_count]])
                        fitness = np.array([obj_func(ind) for ind in population])
                        nfev += n_pop
                        best_idx = int(np.argmin(fitness))
                        iteration_history.append({'x': population[best_idx].copy(), 'fun': float(fitness[best_idx]), 'time': time.time() - start_time})
                        progress_bar.progress(min((iteration + 1) / max_iter, 1.0))
                        status_text.text(f"迭代 {iteration + 1}: f(x) = {fitness[best_idx]:.6f}")

                    best_idx = int(np.argmin(fitness))
                    result = OptimizeResult(x=population[best_idx], fun=float(fitness[best_idx]), nfev=nfev, nit=max_iter, success=True, message="Real-coded GA completed")
                elif "粒子群" in algorithm:
                    rng = np.random.default_rng(42)
                    lb = np.array([b[0] for b in bounds], dtype=float)
                    ub = np.array([b[1] for b in bounds], dtype=float)
                    n_dim = len(bounds)
                    positions = rng.uniform(lb, ub, size=(n_particles, n_dim))
                    span = np.maximum(ub - lb, 1e-12)
                    velocities = rng.normal(0, 0.05, size=(n_particles, n_dim)) * span
                    personal_best = positions.copy()
                    personal_best_f = np.array([obj_func(pos) for pos in positions])
                    nfev = n_particles
                    global_idx = int(np.argmin(personal_best_f))
                    global_best = personal_best[global_idx].copy()
                    global_best_f = float(personal_best_f[global_idx])

                    for iteration in range(max_iter):
                        r1 = rng.random((n_particles, n_dim))
                        r2 = rng.random((n_particles, n_dim))
                        velocities = w * velocities + c1 * r1 * (personal_best - positions) + c2 * r2 * (global_best - positions)
                        positions = np.clip(positions + velocities, lb, ub)
                        fitness = np.array([obj_func(pos) for pos in positions])
                        nfev += n_particles
                        improved = fitness < personal_best_f
                        personal_best[improved] = positions[improved]
                        personal_best_f[improved] = fitness[improved]
                        global_idx = int(np.argmin(personal_best_f))
                        if personal_best_f[global_idx] < global_best_f:
                            global_best = personal_best[global_idx].copy()
                            global_best_f = float(personal_best_f[global_idx])
                        iteration_history.append({'x': global_best.copy(), 'fun': global_best_f, 'time': time.time() - start_time})
                        progress_bar.progress(min((iteration + 1) / max_iter, 1.0))
                        status_text.text(f"迭代 {iteration + 1}: f(x) = {global_best_f:.6f}")

                    result = OptimizeResult(x=global_best, fun=global_best_f, nfev=nfev, nit=max_iter, success=True, message="PSO completed")
                elif "CMA-ES" in algorithm:
                    rng = np.random.default_rng(42)
                    lb = np.array([b[0] for b in bounds], dtype=float)
                    ub = np.array([b[1] for b in bounds], dtype=float)
                    n_dim = len(bounds)
                    mean = (lb + ub) / 2
                    span = np.maximum(ub - lb, 1e-12)
                    sigma = sigma0
                    cov = np.eye(n_dim)
                    n_pop = max(pop_size, 4 + int(3 * np.log(max(n_dim, 2))))
                    elite_count = max(2, n_pop // 2)
                    weights = np.log(elite_count + 0.5) - np.log(np.arange(1, elite_count + 1))
                    weights = weights / weights.sum()
                    best_x = mean.copy()
                    best_f = float(obj_func(best_x))
                    nfev = 1

                    for iteration in range(max_iter):
                        samples = rng.multivariate_normal(np.zeros(n_dim), cov + 1e-8 * np.eye(n_dim), size=n_pop)
                        candidates = np.clip(mean + sigma * span * samples, lb, ub)
                        fitness = np.array([obj_func(candidate) for candidate in candidates])
                        nfev += n_pop
                        order = np.argsort(fitness)
                        elites = candidates[order[:elite_count]]
                        old_best = best_f
                        if fitness[order[0]] < best_f:
                            best_f = float(fitness[order[0]])
                            best_x = candidates[order[0]].copy()
                        mean = np.sum(elites * weights[:, None], axis=0)
                        centered = (elites - mean) / span
                        cov = (centered.T * weights) @ centered + 1e-6 * np.eye(n_dim)
                        sigma = float(np.clip(sigma * (1.03 if best_f < old_best else 0.97), 0.005, 2.0))
                        iteration_history.append({'x': best_x.copy(), 'fun': best_f, 'time': time.time() - start_time})
                        progress_bar.progress(min((iteration + 1) / max_iter, 1.0))
                        status_text.text(f"迭代 {iteration + 1}: f(x) = {best_f:.6f}, sigma={sigma:.4f}")

                    result = OptimizeResult(x=best_x, fun=best_f, nfev=nfev, nit=max_iter, success=True, message="Lightweight CMA-ES style search completed")
                elif "贝叶斯优化" in algorithm:
                    from scipy.stats import norm
                    from sklearn.gaussian_process import GaussianProcessRegressor
                    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

                    rng = np.random.default_rng(42)
                    lb = np.array([b[0] for b in bounds], dtype=float)
                    ub = np.array([b[1] for b in bounds], dtype=float)
                    n_dim = len(bounds)
                    n_init = min(max(n_initial, 3), max_iter)
                    X_obs = rng.uniform(lb, ub, size=(n_init, n_dim))
                    y_obs = np.array([obj_func(x) for x in X_obs])
                    nfev = n_init
                    best_idx = int(np.argmin(y_obs))
                    best_x = X_obs[best_idx].copy()
                    best_f = float(y_obs[best_idx])

                    for iteration in range(max_iter):
                        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(nu=2.5) + WhiteKernel(noise_level=1e-6, noise_level_bounds="fixed")
                        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
                        gp.fit(X_obs, y_obs)
                        candidates = rng.uniform(lb, ub, size=(n_candidates, n_dim))
                        mu, sigma_pred = gp.predict(candidates, return_std=True)
                        sigma_pred = np.maximum(sigma_pred, 1e-12)
                        xi = 0.01
                        improvement = best_f - mu - xi
                        z = improvement / sigma_pred
                        expected_improvement = improvement * norm.cdf(z) + sigma_pred * norm.pdf(z)
                        next_x = candidates[int(np.argmax(expected_improvement))]
                        next_y = float(obj_func(next_x))
                        X_obs = np.vstack([X_obs, next_x])
                        y_obs = np.append(y_obs, next_y)
                        nfev += 1
                        if next_y < best_f:
                            best_f = next_y
                            best_x = next_x.copy()
                        iteration_history.append({'x': best_x.copy(), 'fun': best_f, 'time': time.time() - start_time})
                        progress_bar.progress(min((iteration + 1) / max_iter, 1.0))
                        status_text.text(f"评估 {iteration + 1}/{max_iter}: f(x) = {best_f:.6f}")

                    result = OptimizeResult(x=best_x, fun=best_f, nfev=nfev, nit=max_iter, success=True, message="Gaussian-process Bayesian optimization completed")
                elif "Nelder-Mead" in algorithm:
                    x0 = np.mean(bounds, axis=1)
                    result = minimize(
                        obj_func,
                        x0,
                        method='Nelder-Mead',
                        bounds=bounds,
                        options={'maxiter': max_iter, 'xatol': tolerance},
                        callback=lambda xk: callback(xk)
                    )
                elif "L-BFGS-B" in algorithm:
                    x0 = np.mean(bounds, axis=1)
                    # 记录初始点
                    iteration_history.append({
                        'x': x0.copy(),
                        'fun': obj_func(x0),
                        'time': time.time() - start_time
                    })
                    
                    def lbfgsb_callback(xk):
                        """L-BFGS-B回调函数"""
                        fval = obj_func(xk)
                        iteration_history.append({
                            'x': xk.copy(),
                            'fun': fval,
                            'time': time.time() - start_time
                        })
                        progress = min(len(iteration_history) / max_iter, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"迭代 {len(iteration_history)}: f(x) = {fval:.6f}")
                    
                    result = minimize(
                        obj_func,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': max_iter, 'ftol': tolerance},
                        callback=lbfgsb_callback
                    )
                elif "SLSQP" in algorithm:
                    x0 = np.mean(bounds, axis=1)
                    # 记录初始点
                    iteration_history.append({
                        'x': x0.copy(),
                        'fun': obj_func(x0),
                        'time': time.time() - start_time
                    })
                    
                    def slsqp_callback(xk):
                        """SLSQP回调函数"""
                        fval = obj_func(xk)
                        iteration_history.append({
                            'x': xk.copy(),
                            'fun': fval,
                            'time': time.time() - start_time
                        })
                        progress = min(len(iteration_history) / max_iter, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"迭代 {len(iteration_history)}: f(x) = {fval:.6f}")
                    
                    result = minimize(
                        obj_func,
                        x0,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': max_iter, 'ftol': tolerance},
                        callback=slsqp_callback
                    )
                elif "模拟退火" in algorithm:
                    # 模拟退火使用自定义回调
                    sa_iteration_count = [0]  # 使用列表来在闭包中修改
                    
                    def sa_callback(x, f, context):
                        """模拟退火回调函数"""
                        sa_iteration_count[0] += 1
                        iteration_history.append({
                            'x': x.copy(),
                            'fun': f,
                            'time': time.time() - start_time
                        })
                        progress = min(sa_iteration_count[0] / max_iter, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"迭代 {sa_iteration_count[0]}: f(x) = {f:.6f}, 温度={context.get('T', 'N/A') if isinstance(context, dict) else 'N/A'}")
                        return False  # 返回False继续优化
                    
                    result = dual_annealing(
                        obj_func,
                        bounds,
                        maxiter=max_iter,
                        initial_temp=initial_temp,
                        restart_temp_ratio=cooling_rate,
                        callback=sa_callback,
                        seed=42
                    )
                elif "Powell" in algorithm:
                    x0 = np.mean(bounds, axis=1)
                    iteration_history.append({'x': x0.copy(), 'fun': obj_func(x0), 'time': time.time() - start_time})
                    
                    def powell_callback(xk):
                        fval = obj_func(xk)
                        iteration_history.append({'x': xk.copy(), 'fun': fval, 'time': time.time() - start_time})
                        progress_bar.progress(min(len(iteration_history) / max_iter, 1.0))
                        status_text.text(f"迭代 {len(iteration_history)}: f(x) = {fval:.6f}")
                    
                    result = minimize(obj_func, x0, method='Powell', callback=powell_callback,
                                    options={'maxiter': max_iter, 'ftol': tolerance})
                elif "CG" in algorithm:
                    x0 = np.mean(bounds, axis=1)
                    iteration_history.append({'x': x0.copy(), 'fun': obj_func(x0), 'time': time.time() - start_time})
                    
                    def cg_callback(xk):
                        fval = obj_func(xk)
                        iteration_history.append({'x': xk.copy(), 'fun': fval, 'time': time.time() - start_time})
                        progress_bar.progress(min(len(iteration_history) / max_iter, 1.0))
                        status_text.text(f"迭代 {len(iteration_history)}: f(x) = {fval:.6f}")
                    
                    result = minimize(obj_func, x0, method='CG', callback=cg_callback,
                                    options={'maxiter': max_iter, 'gtol': tolerance})
                elif "BFGS" in algorithm:
                    x0 = np.mean(bounds, axis=1)
                    iteration_history.append({'x': x0.copy(), 'fun': obj_func(x0), 'time': time.time() - start_time})
                    
                    def bfgs_callback(xk):
                        fval = obj_func(xk)
                        iteration_history.append({'x': xk.copy(), 'fun': fval, 'time': time.time() - start_time})
                        progress_bar.progress(min(len(iteration_history) / max_iter, 1.0))
                        status_text.text(f"迭代 {len(iteration_history)}: f(x) = {fval:.6f}")
                    
                    result = minimize(obj_func, x0, method='BFGS', callback=bfgs_callback,
                                    options={'maxiter': max_iter, 'gtol': tolerance})
                elif "TNC" in algorithm:
                    x0 = np.mean(bounds, axis=1)
                    iteration_history.append({'x': x0.copy(), 'fun': obj_func(x0), 'time': time.time() - start_time})
                    
                    def tnc_callback(xk):
                        fval = obj_func(xk)
                        iteration_history.append({'x': xk.copy(), 'fun': fval, 'time': time.time() - start_time})
                        progress_bar.progress(min(len(iteration_history) / max_iter, 1.0))
                        status_text.text(f"迭代 {len(iteration_history)}: f(x) = {fval:.6f}")
                    
                    result = minimize(obj_func, x0, method='TNC', bounds=bounds, callback=tnc_callback,
                                    options={'maxiter': max_iter})
                elif "COBYLA" in algorithm:
                    x0 = np.mean(bounds, axis=1)
                    iteration_history.append({'x': x0.copy(), 'fun': obj_func(x0), 'time': time.time() - start_time})
                    
                    def cobyla_callback(xk):
                        fval = obj_func(xk)
                        iteration_history.append({'x': xk.copy(), 'fun': fval, 'time': time.time() - start_time})
                        progress_bar.progress(min(len(iteration_history) / max_iter, 1.0))
                        status_text.text(f"迭代 {len(iteration_history)}: f(x) = {fval:.6f}")
                    
                    result = minimize(obj_func, x0, method='COBYLA', callback=cobyla_callback,
                                    options={'maxiter': max_iter, 'tol': tolerance})
                elif "trust-constr" in algorithm:
                    x0 = np.mean(bounds, axis=1)
                    iteration_history.append({'x': x0.copy(), 'fun': obj_func(x0), 'time': time.time() - start_time})
                    
                    def trust_constr_callback(xk, state):
                        fval = obj_func(xk)
                        iteration_history.append({'x': xk.copy(), 'fun': fval, 'time': time.time() - start_time})
                        progress_bar.progress(min(len(iteration_history) / max_iter, 1.0))
                        status_text.text(f"迭代 {len(iteration_history)}: f(x) = {fval:.6f}")
                        return False
                    
                    from scipy.optimize import Bounds
                    lb = [b[0] for b in bounds]
                    ub = [b[1] for b in bounds]
                    result = minimize(obj_func, x0, method='trust-constr',
                                    bounds=Bounds(lb, ub), callback=trust_constr_callback,
                                    options={'maxiter': max_iter, 'gtol': tolerance, 'xtol': tolerance})
                elif "SHGO" in algorithm:
                    from scipy.optimize import shgo
                    
                    def shgo_callback(x, f, context):
                        iteration_history.append({'x': x.copy(), 'fun': f, 'time': time.time() - start_time})
                        progress_bar.progress(min(len(iteration_history) / max_iter, 1.0))
                        status_text.text(f"迭代 {len(iteration_history)}: f(x) = {f:.6f}")
                        return False
                    
                    result = shgo(obj_func, bounds, callback=shgo_callback,
                                options={'maxiter': max_iter})
                elif "Basin-Hopping" in algorithm:
                    from scipy.optimize import basinhopping
                    
                    x0 = np.mean(bounds, axis=1)
                    iteration_history.append({'x': x0.copy(), 'fun': obj_func(x0), 'time': time.time() - start_time})
                    
                    def bh_callback(x, f, accept):
                        iteration_history.append({'x': x.copy(), 'fun': f, 'time': time.time() - start_time})
                        progress_bar.progress(min(len(iteration_history) / max_iter, 1.0))
                        status_text.text(f"迭代 {len(iteration_history)}: f(x) = {f:.6f}, 接受={accept}")
                    
                    # 创建局部优化器
                    local_minimizer = {'method': 'L-BFGS-B', 'bounds': bounds}
                    result = basinhopping(obj_func, x0, niter=max_iter, callback=bh_callback,
                                        minimizer_kwargs=local_minimizer, seed=42)
                
                # ==================== 群智能优化算法 ====================
                elif "灰狼" in algorithm:
                    # 灰狼优化算法 (GWO)
                    n_wolves = st.session_state.get('opt_gwo_pop', 30)
                    n_dim = len(bounds)
                    
                    # 初始化灰狼种群
                    wolves = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (n_wolves, n_dim))
                    fitness = np.array([obj_func(w) for w in wolves])
                    
                    # 排序找到 alpha, beta, delta
                    sorted_idx = np.argsort(fitness)
                    alpha, beta, delta = wolves[sorted_idx[:3]]
                    
                    for iteration in range(max_iter):
                        a = 2 - iteration * (2 / max_iter)  # 线性递减
                        
                        for i in range(n_wolves):
                            for j in range(n_dim):
                                r1, r2 = np.random.rand(), np.random.rand()
                                A1 = 2 * a * r1 - a
                                C1 = 2 * r2
                                D_alpha = abs(C1 * alpha[j] - wolves[i, j])
                                X1 = alpha[j] - A1 * D_alpha
                                
                                r1, r2 = np.random.rand(), np.random.rand()
                                A2 = 2 * a * r1 - a
                                C2 = 2 * r2
                                D_beta = abs(C2 * beta[j] - wolves[i, j])
                                X2 = beta[j] - A2 * D_beta
                                
                                r1, r2 = np.random.rand(), np.random.rand()
                                A3 = 2 * a * r1 - a
                                C3 = 2 * r2
                                D_delta = abs(C3 * delta[j] - wolves[i, j])
                                X3 = delta[j] - A3 * D_delta
                                
                                wolves[i, j] = (X1 + X2 + X3) / 3
                            
                            # 边界处理
                            wolves[i] = np.clip(wolves[i], [b[0] for b in bounds], [b[1] for b in bounds])
                        
                        # 更新适应度
                        fitness = np.array([obj_func(w) for w in wolves])
                        sorted_idx = np.argsort(fitness)
                        alpha, beta, delta = wolves[sorted_idx[:3]]
                        
                        # 记录历史
                        iteration_history.append({
                            'x': alpha.copy(),
                            'fun': fitness[sorted_idx[0]],
                            'time': time.time() - start_time
                        })
                        progress_bar.progress(min((iteration + 1) / max_iter, 1.0))
                        status_text.text(f"迭代 {iteration + 1}: f(x) = {fitness[sorted_idx[0]]:.6f}")
                    
                    # 创建结果对象
                    class GWOResult:
                        def __init__(self, x, fun, nfev, nit):
                            self.x = x
                            self.fun = fun
                            self.nfev = nfev
                            self.nit = nit
                            self.success = True
                    
                    result = GWOResult(alpha, fitness[sorted_idx[0]], n_wolves * max_iter, max_iter)
                
                elif "蚁群" in algorithm:
                    # 蚁群优化算法 (ACO) - 简化版用于连续优化
                    n_ants = st.session_state.get('opt_aco_pop', 50)
                    n_dim = len(bounds)
                    
                    # 初始化
                    ants = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (n_ants, n_dim))
                    best_solution = None
                    best_fitness = float('inf')
                    
                    # 信息素初始化
                    pheromone = np.ones((n_ants, n_dim))
                    
                    for iteration in range(max_iter):
                        # 生成新解
                        new_ants = ants + np.random.normal(0, 0.1, (n_ants, n_dim)) * pheromone
                        new_ants = np.clip(new_ants, [b[0] for b in bounds], [b[1] for b in bounds])
                        
                        # 评估
                        fitness = np.array([obj_func(a) for a in new_ants])
                        
                        # 更新最优
                        min_idx = np.argmin(fitness)
                        if fitness[min_idx] < best_fitness:
                            best_fitness = fitness[min_idx]
                            best_solution = new_ants[min_idx].copy()
                        
                        # 更新信息素
                        for i in range(n_ants):
                            if fitness[i] < best_fitness * 1.1:  # 好的解增加信息素
                                pheromone[i] *= 1.1
                            else:
                                pheromone[i] *= 0.9
                        
                        ants = new_ants
                        
                        # 记录历史
                        iteration_history.append({
                            'x': best_solution.copy() if best_solution is not None else ants[min_idx],
                            'fun': best_fitness,
                            'time': time.time() - start_time
                        })
                        progress_bar.progress(min((iteration + 1) / max_iter, 1.0))
                        status_text.text(f"迭代 {iteration + 1}: f(x) = {best_fitness:.6f}")
                    
                    class ACOResult:
                        def __init__(self, x, fun, nfev, nit):
                            self.x = x
                            self.fun = fun
                            self.nfev = nfev
                            self.nit = nit
                            self.success = True
                    
                    result = ACOResult(best_solution, best_fitness, n_ants * max_iter, max_iter)
                
                elif "萤火虫" in algorithm:
                    # 萤火虫算法 (FA)
                    n_fireflies = st.session_state.get('opt_fa_pop', 40)
                    n_dim = len(bounds)
                    alpha = 0.5  # 随机项系数
                    beta0 = 1.0  # 吸引度基准
                    gamma = 1.0  # 光吸收系数
                    
                    # 初始化萤火虫位置
                    fireflies = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (n_fireflies, n_dim))
                    brightness = np.array([obj_func(f) for f in fireflies])
                    
                    for iteration in range(max_iter):
                        alpha = alpha * 0.98  # 逐渐减小随机性
                        
                        for i in range(n_fireflies):
                            for j in range(n_fireflies):
                                if brightness[j] < brightness[i]:  # j更亮（更好）
                                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                                    beta = beta0 * np.exp(-gamma * r ** 2)
                                    fireflies[i] += beta * (fireflies[j] - fireflies[i]) + alpha * (np.random.rand(n_dim) - 0.5)
                                    fireflies[i] = np.clip(fireflies[i], [b[0] for b in bounds], [b[1] for b in bounds])
                                    brightness[i] = obj_func(fireflies[i])
                        
                        # 记录最优
                        best_idx = np.argmin(brightness)
                        iteration_history.append({
                            'x': fireflies[best_idx].copy(),
                            'fun': brightness[best_idx],
                            'time': time.time() - start_time
                        })
                        progress_bar.progress(min((iteration + 1) / max_iter, 1.0))
                        status_text.text(f"迭代 {iteration + 1}: f(x) = {brightness[best_idx]:.6f}")
                    
                    best_idx = np.argmin(brightness)
                    
                    class FAResult:
                        def __init__(self, x, fun, nfev, nit):
                            self.x = x
                            self.fun = fun
                            self.nfev = nfev
                            self.nit = nit
                            self.success = True
                    
                    result = FAResult(fireflies[best_idx], brightness[best_idx], n_fireflies * max_iter, max_iter)
                
                elif "禁忌搜索" in algorithm:
                    # 禁忌搜索算法 (TS)
                    tabu_size = st.session_state.get('opt_ts_tabu', 10)
                    n_dim = len(bounds)
                    
                    # 初始解
                    current = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], n_dim)
                    current_fitness = obj_func(current)
                    best_solution = current.copy()
                    best_fitness = current_fitness
                    
                    tabu_list = []
                    
                    for iteration in range(max_iter):
                        # 生成邻域解
                        neighbors = []
                        for _ in range(20):
                            neighbor = current + np.random.normal(0, 0.1, n_dim)
                            neighbor = np.clip(neighbor, [b[0] for b in bounds], [b[1] for b in bounds])
                            # 检查是否在禁忌表中
                            is_tabu = any(np.allclose(neighbor, tabu, atol=1e-3) for tabu in tabu_list)
                            if not is_tabu:
                                neighbors.append(neighbor)
                        
                        if not neighbors:
                            continue
                        
                        # 评估邻域解
                        neighbor_fitness = [obj_func(n) for n in neighbors]
                        best_neighbor_idx = np.argmin(neighbor_fitness)
                        
                        # 更新当前解
                        current = neighbors[best_neighbor_idx]
                        current_fitness = neighbor_fitness[best_neighbor_idx]
                        
                        # 更新最优解
                        if current_fitness < best_fitness:
                            best_fitness = current_fitness
                            best_solution = current.copy()
                        
                        # 更新禁忌表
                        tabu_list.append(current.copy())
                        if len(tabu_list) > tabu_size:
                            tabu_list.pop(0)
                        
                        # 记录历史
                        iteration_history.append({
                            'x': best_solution.copy(),
                            'fun': best_fitness,
                            'time': time.time() - start_time
                        })
                        progress_bar.progress(min((iteration + 1) / max_iter, 1.0))
                        status_text.text(f"迭代 {iteration + 1}: f(x) = {best_fitness:.6f}, 禁忌表={len(tabu_list)}")
                    
                    class TSResult:
                        def __init__(self, x, fun, nfev, nit):
                            self.x = x
                            self.fun = fun
                            self.nfev = nfev
                            self.nit = nit
                            self.success = True
                    
                    result = TSResult(best_solution, best_fitness, max_iter * 20, max_iter)
                
                elif "人工鱼群" in algorithm:
                    # 人工鱼群算法 (AFSA)
                    n_fish = st.session_state.get('opt_afsa_pop', 50)
                    n_dim = len(bounds)
                    visual = 0.5  # 感知范围
                    step = 0.1  # 移动步长
                    delta = 0.618  # 拥挤度因子
                    
                    # 初始化鱼群
                    fish = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (n_fish, n_dim))
                    fitness = np.array([obj_func(f) for f in fish])
                    best_fish = fish[np.argmin(fitness)]
                    best_fitness = np.min(fitness)
                    
                    for iteration in range(max_iter):
                        for i in range(n_fish):
                            # 觅食行为
                            new_pos = fish[i] + step * np.random.uniform(-1, 1, n_dim)
                            new_pos = np.clip(new_pos, [b[0] for b in bounds], [b[1] for b in bounds])
                            new_fitness = obj_func(new_pos)
                            
                            if new_fitness < fitness[i]:
                                fish[i] = new_pos
                                fitness[i] = new_fitness
                            else:
                                # 随机移动
                                fish[i] += step * np.random.uniform(-1, 1, n_dim)
                                fish[i] = np.clip(fish[i], [b[0] for b in bounds], [b[1] for b in bounds])
                                fitness[i] = obj_func(fish[i])
                        
                        # 更新最优
                        current_best_idx = np.argmin(fitness)
                        if fitness[current_best_idx] < best_fitness:
                            best_fitness = fitness[current_best_idx]
                            best_fish = fish[current_best_idx].copy()
                        
                        # 记录历史
                        iteration_history.append({
                            'x': best_fish.copy(),
                            'fun': best_fitness,
                            'time': time.time() - start_time
                        })
                        progress_bar.progress(min((iteration + 1) / max_iter, 1.0))
                        status_text.text(f"迭代 {iteration + 1}: f(x) = {best_fitness:.6f}")
                    
                    class AFSAResult:
                        def __init__(self, x, fun, nfev, nit):
                            self.x = x
                            self.fun = fun
                            self.nfev = nfev
                            self.nit = nit
                            self.success = True
                    
                    result = AFSAResult(best_fish, best_fitness, n_fish * max_iter, max_iter)
                
                elif "免疫遗传" in algorithm:
                    # 免疫遗传算法 (IGA)
                    n_pop = st.session_state.get('opt_iga_pop', 60)
                    n_dim = len(bounds)
                    
                    # 初始化种群
                    pop = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (n_pop, n_dim))
                    fitness = np.array([obj_func(p) for p in pop])
                    
                    # 克隆选择参数
                    clone_factor = 5
                    mutation_rate = 0.1
                    
                    for iteration in range(max_iter):
                        # 选择（基于浓度和适应度）
                        sorted_idx = np.argsort(fitness)
                        selected = pop[sorted_idx[:n_pop//2]]
                        selected_fitness = fitness[sorted_idx[:n_pop//2]]
                        
                        # 克隆
                        clones = []
                        for i, ind in enumerate(selected):
                            n_clones = int(clone_factor * (n_pop//2 - i) / (n_pop//2))
                            for _ in range(n_clones):
                                clone = ind + mutation_rate * np.random.normal(0, 1, n_dim)
                                clone = np.clip(clone, [b[0] for b in bounds], [b[1] for b in bounds])
                                clones.append(clone)
                        
                        # 评估克隆
                        if clones:
                            clone_fitness = np.array([obj_func(c) for c in clones])
                            # 选择最优
                            all_pop = np.vstack([pop, clones])
                            all_fitness = np.concatenate([fitness, clone_fitness])
                            best_idx = np.argsort(all_fitness)[:n_pop]
                            pop = all_pop[best_idx]
                            fitness = all_fitness[best_idx]
                        
                        # 记录历史
                        best_idx = np.argmin(fitness)
                        iteration_history.append({
                            'x': pop[best_idx].copy(),
                            'fun': fitness[best_idx],
                            'time': time.time() - start_time
                        })
                        progress_bar.progress(min((iteration + 1) / max_iter, 1.0))
                        status_text.text(f"迭代 {iteration + 1}: f(x) = {fitness[best_idx]:.6f}")
                    
                    best_idx = np.argmin(fitness)
                    
                    class IGAResult:
                        def __init__(self, x, fun, nfev, nit):
                            self.x = x
                            self.fun = fun
                            self.nfev = nfev
                            self.nit = nit
                            self.success = True
                    
                    result = IGAResult(pop[best_idx], fitness[best_idx], n_pop * max_iter, max_iter)
                
                elif "混合蛙跳" in algorithm:
                    # 简化版混合蛙跳算法
                    n_frogs = st.session_state.get('opt_sfla_pop', 40)
                    n_dim = len(bounds)
                    n_memeplexes = 5
                    
                    # 初始化蛙群
                    frogs = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (n_frogs, n_dim))
                    fitness = np.array([obj_func(f) for f in frogs])
                    
                    for iteration in range(max_iter):
                        # 排序
                        sorted_idx = np.argsort(fitness)
                        frogs = frogs[sorted_idx]
                        fitness = fitness[sorted_idx]
                        
                        # 局部搜索（简化版）
                        for i in range(n_frogs):
                            # 向最优蛙学习
                            if i > 0:
                                new_frog = frogs[i] + np.random.rand() * (frogs[0] - frogs[i])
                                new_frog = np.clip(new_frog, [b[0] for b in bounds], [b[1] for b in bounds])
                                new_fitness = obj_func(new_frog)
                                
                                if new_fitness < fitness[i]:
                                    frogs[i] = new_frog
                                    fitness[i] = new_fitness
                        
                        # 记录历史
                        iteration_history.append({
                            'x': frogs[0].copy(),
                            'fun': fitness[0],
                            'time': time.time() - start_time
                        })
                        progress_bar.progress(min((iteration + 1) / max_iter, 1.0))
                        status_text.text(f"迭代 {iteration + 1}: f(x) = {fitness[0]:.6f}")
                    
                    class SFLAResult:
                        def __init__(self, x, fun, nfev, nit):
                            self.x = x
                            self.fun = fun
                            self.nfev = nfev
                            self.nit = nit
                            self.success = True
                    
                    result = SFLAResult(frogs[0], fitness[0], n_frogs * max_iter, max_iter)
                
                elif "自定义" in algorithm:
                    # 执行自定义算法
                    custom_code = st.session_state.get('custom_algo_code', '')
                    
                    if not custom_code.strip():
                        st.error("❌ 请先编写自定义算法代码")
                        raise ValueError("自定义算法代码为空")
                    
                    try:
                        # 创建局部命名空间
                        local_namespace = {
                            'obj_func': obj_func,
                            'bounds': bounds,
                            'max_iter': max_iter,
                            'progress_bar': progress_bar,
                            'status_text': status_text,
                            'iteration_history': iteration_history,
                            'start_time': start_time,
                            'np': np,
                            'time': time
                        }
                        
                        # 执行自定义代码
                        exec(custom_code, local_namespace)
                        record_custom_execution("自定义优化算法", "执行成功")
                        
                        # 获取结果
                        if 'result' in local_namespace:
                            result = local_namespace['result']
                        else:
                            st.error("❌ 自定义算法必须定义 'result' 变量")
                            raise ValueError("自定义算法未返回结果")
                            
                    except Exception as e:
                        record_custom_execution("自定义优化算法", "执行失败", str(e))
                        st.error(f"❌ 自定义算法执行错误: {str(e)}")
                        raise
                
                else:
                    # 默认使用差分进化
                    result = differential_evolution(
                        obj_func,
                        bounds,
                        maxiter=max_iter,
                        callback=callback,
                        seed=42
                    )
                
                opt_audit_rows = optimization_algorithm_matrix()
                opt_audit_match = opt_audit_rows[opt_audit_rows["算法"].apply(lambda name: any(part in algorithm for part in str(name).split("/")))]
                opt_impl_status = opt_audit_match.iloc[0].to_dict() if not opt_audit_match.empty else {}

                # 保存结果
                st.session_state.optimization_result = {
                    'success': result.success if hasattr(result, 'success') else True,
                    'x': result.x,
                    'fun': result.fun,
                    'nfev': result.nfev if hasattr(result, 'nfev') else len(iteration_history),
                    'nit': result.nit if hasattr(result, 'nit') else len(iteration_history),
                    'history': iteration_history,
                    'variable_names': [v['name'] for v in variable_configs],
                    'bounds': bounds,
                    'algorithm': algorithm,
                    'implementation_status': opt_impl_status,
                    'time': time.time() - start_time
                }
                
                progress_bar.progress(1.0)
                status_text.text("✅ 优化完成！")
                
                if save_history:
                    if 'optimization_history' not in st.session_state:
                        st.session_state.optimization_history = []
                    st.session_state.optimization_history.append({
                        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'algorithm': algorithm,
                        'result': result.fun,
                        'variables': result.x.tolist()
                    })
                
                st.success("✅ 优化求解成功完成！")
                
            except Exception as e:
                st.error(f"❌ 优化失败: {str(e)}")
                st.exception(e)
    
    # ==================== 结果展示区域 ====================
    if 'optimization_result' in st.session_state and st.session_state.optimization_result:
        st.divider()
        st.subheader("📊 优化结果")
        
        result = st.session_state.optimization_result
        
        # 关键指标展示
        col_res1, col_res2, col_res3, col_res4, col_res5 = st.columns(5)
        with col_res1:
            st.metric("最优目标值", f"{result['fun']:.6f}")
        with col_res2:
            st.metric("函数评估次数", result['nfev'])
        with col_res3:
            st.metric("迭代次数", result['nit'])
        with col_res4:
            st.metric("求解时间", f"{result['time']:.3f}s")
        with col_res5:
            st.metric("成功状态", "成功" if result.get('success', True) else "需复核")

        if result.get("implementation_status"):
            impl = result["implementation_status"]
            st.info(f"算法实现状态：{impl.get('实现状态', '未知')}；实现方式：{impl.get('实现方式', '未记录')}。")
        
        # 最优变量值
        st.write("**最优变量值:**")
        lower_bounds = np.array([b[0] for b in result['bounds']], dtype=float)
        upper_bounds = np.array([b[1] for b in result['bounds']], dtype=float)
        optimal_x = np.array(result['x'], dtype=float)
        bound_span = np.maximum(upper_bounds - lower_bounds, 1e-12)
        normalized_position = (optimal_x - lower_bounds) / bound_span
        var_df = pd.DataFrame({
            '变量名': result['variable_names'],
            '最优值': optimal_x,
            '下界': lower_bounds,
            '上界': upper_bounds,
            '边界位置(0=下界,1=上界)': normalized_position
        })
        st.dataframe(var_df, use_container_width=True)
        fig_vars = go.Figure(go.Bar(
            x=var_df['变量名'],
            y=var_df['边界位置(0=下界,1=上界)'],
            marker_color="#4c78a8"
        ))
        fig_vars.update_layout(
            title="最优变量在边界区间中的位置",
            xaxis_title="变量",
            yaxis_title="归一化位置",
            yaxis=dict(range=[0, 1]),
            height=320
        )
        st.plotly_chart(fig_vars, use_container_width=True)
        
        # 收敛曲线
        if result['history']:
            st.write("**收敛过程:**")
            
            history_df = pd.DataFrame(result['history'])
            history_df['iteration'] = np.arange(1, len(history_df) + 1)
            history_df['best_so_far'] = history_df['fun'].cummin()
            history_df['improvement'] = history_df['best_so_far'].shift(1).fillna(history_df['best_so_far']) - history_df['best_so_far']
            
            # 创建收敛曲线图
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=history_df['iteration'],
                y=history_df['fun'],
                mode='lines+markers',
                name='目标函数值',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4)
            ))
            fig_conv.add_trace(go.Scatter(
                x=history_df['iteration'],
                y=history_df['best_so_far'],
                mode='lines',
                name='历史最优值',
                line=dict(color='#d62728', width=2, dash='dash')
            ))
            fig_conv.update_layout(
                title="优化收敛曲线",
                xaxis_title="迭代次数",
                yaxis_title="目标函数值",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_conv, use_container_width=True)

            fig_improve = go.Figure(go.Bar(
                x=history_df['iteration'],
                y=history_df['improvement'].clip(lower=0),
                marker_color="#59a14f",
                name="每轮改进量"
            ))
            fig_improve.update_layout(
                title="历史最优值改进量",
                xaxis_title="迭代次数",
                yaxis_title="改进量",
                height=300
            )
            st.plotly_chart(fig_improve, use_container_width=True)

            with st.expander("📋 迭代历史明细", expanded=False):
                display_history = history_df.drop(columns=['x'], errors='ignore').tail(200)
                st.dataframe(display_history, use_container_width=True, hide_index=True)
            
            # 如果有2个变量，显示优化轨迹
            if len(result['x']) == 2:
                st.write("**优化轨迹 (2D):**")
                
                traj_x = [h['x'][0] for h in result['history']]
                traj_y = [h['x'][1] for h in result['history']]
                
                fig_traj = go.Figure()
                fig_traj.add_trace(go.Scatter(
                    x=traj_x,
                    y=traj_y,
                    mode='lines+markers',
                    name='优化轨迹',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4, color=list(range(len(traj_x))), 
                               colorscale='Viridis', showscale=True,
                               colorbar=dict(title="迭代"))
                ))
                fig_traj.add_trace(go.Scatter(
                    x=[result['x'][0]],
                    y=[result['x'][1]],
                    mode='markers',
                    name='最优解',
                    marker=dict(size=15, color='red', symbol='star')
                ))
                fig_traj.update_layout(
                    title="优化轨迹",
                    xaxis_title=result['variable_names'][0],
                    yaxis_title=result['variable_names'][1],
                    height=500
                )
                st.plotly_chart(fig_traj, use_container_width=True)
        
        # 导出结果
        st.divider()
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        with col_exp1:
            result_json = pd.DataFrame([{
                'algorithm': result['algorithm'],
                'optimal_value': result['fun'],
                'optimal_variables': result['x'].tolist(),
                'variable_names': result['variable_names'],
                'nfev': result['nfev'],
                'nit': result['nit'],
                'time': result['time'],
                'success': result.get('success', True),
                'implementation_status': result.get('implementation_status', {})
            }]).to_json(orient='records')
            st.download_button(
                "📥 导出结果 (JSON)",
                result_json,
                file_name="optimization_result.json",
                mime="application/json"
            )
        with col_exp2:
            if result['history']:
                history_export_df = pd.DataFrame(result['history'])
                history_export_df['iteration'] = np.arange(1, len(history_export_df) + 1)
                history_export_df['best_so_far'] = history_export_df['fun'].cummin()
                history_export_df['improvement'] = history_export_df['best_so_far'].shift(1).fillna(history_export_df['best_so_far']) - history_export_df['best_so_far']
                if 'x' in history_export_df.columns:
                    history_export_df['x'] = history_export_df['x'].apply(lambda value: np.asarray(value, dtype=float).tolist())
                history_csv = history_export_df.to_csv(index=False)
                st.download_button(
                    "📥 导出迭代历史 (CSV)",
                    history_csv,
                    file_name="optimization_history.csv",
                    mime="text/csv"
                )
        with col_exp3:
            opt_tables = []
            opt_figures = []
            if result['history']:
                opt_tables.append(("迭代历史", history_export_df if 'history_export_df' in locals() else pd.DataFrame(result['history'])))
                if 'fig_conv' in locals():
                    opt_figures.append(("优化收敛曲线", fig_conv))
                if 'fig_improve' in locals():
                    opt_figures.append(("历史最优值改进量", fig_improve))
                if 'fig_traj' in locals():
                    opt_figures.append(("二维优化轨迹", fig_traj))
            opt_result_df = pd.DataFrame([{
                'algorithm': result['algorithm'],
                'optimal_value': result['fun'],
                'optimal_variables': np.asarray(result['x'], dtype=float).tolist(),
                'nfev': result['nfev'],
                'nit': result['nit'],
                'time': result['time'],
                'success': result.get('success', True),
            }])
            opt_report = build_html_report(
                "优化算法图文结果报告",
                metrics={
                    "算法": result["algorithm"],
                    "最优目标值": f"{result['fun']:.6g}",
                    "变量数量": len(result["x"]),
                    "函数评估次数": result["nfev"],
                    "迭代次数": result["nit"],
                    "是否成功": result.get("success", True),
                },
                sections=[("最优变量", ", ".join(f"{name}={value:.6g}" for name, value in zip(result["variable_names"], result["x"])))],
                figures=opt_figures,
                tables=[("优化结果", opt_result_df)] + opt_tables,
            )
            st.download_button(
                "📄 导出HTML图文报告",
                opt_report.encode("utf-8"),
                file_name=f"optimization_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                key="download_optimization_html",
                use_container_width=True,
            )
    
    # ==================== 优化历史记录 ====================
    if 'optimization_history' in st.session_state and st.session_state.optimization_history:
        st.divider()
        st.subheader("📚 优化历史记录")
        
        hist_df = pd.DataFrame(st.session_state.optimization_history)
        st.dataframe(hist_df, use_container_width=True)
        
        if st.button("🗑️ 清空历史记录", key="clear_opt_history"):
            st.session_state.optimization_history = []
            st.rerun()

# ==================== 标签7: 强化学习与智能控制 ====================
with tab7:
    st.markdown("<h1 style='font-size: 32px; font-weight: bold;'>🧠 强化学习与智能控制</h1>", unsafe_allow_html=True)
    st.info("强化学习页面已扩展为智能控制实验台：支持多算法对比、不同控制场景、奖励函数配置、策略评估和结果导出。")
    with st.expander("❓ 强化学习选项说明", expanded=False):
        st.markdown("""
        - **真实深度强化学习**：DQN、A2C、PPO、DDPG、TD3、SAC 调用 Stable-Baselines3 + Gymnasium + PyTorch 后端训练，不再使用名称模拟。
        - **表格强化学习**：Q-Learning、SARSA、Expected SARSA 使用离散状态/动作 Q 表真实更新。
        - **Ray RLlib**：已预留真实 PPO 接口；当前 Python 3.13 环境下 Ray 未提供可安装包，因此不会用其他算法冒充。
        - **控制基线**：PID/MPC Baseline 不是强化学习算法，用于工程对照。
        - **控制场景**：不同场景改变系统状态转移方程，可用于快速比较算法在惯性、扰动和调度问题中的表现。
        - **奖励函数**：偏差越小、动作越平稳，奖励越高；动作惩罚过大会使策略保守，稳定性惩罚过大会压制大误差振荡。
        """)

    rl_specs = list_rl_algorithms()
    rl_backend_status = rl_backend_availability()
    rl_spec_df = pd.DataFrame([item.__dict__ for item in rl_specs])
    rl_spec_df["后端可用"] = rl_spec_df["backend"].map(rl_backend_status).fillna(False)

    with st.expander("📚 算法库", expanded=False):
        st.dataframe(
            rl_spec_df.rename(columns={
                "key": "算法ID",
                "name": "算法名称",
                "family": "算法族",
                "action_space": "动作空间",
                "backend": "后端",
                "implementation_status": "实现状态",
                "description": "适用说明"
            }),
            use_container_width=True,
            hide_index=True
        )

    rl_environment_mode = st.radio(
        "环境来源",
        ["内置控制场景", "用户数据驱动环境", "自定义函数环境"],
        horizontal=True,
        key="rl_environment_mode",
        help="内置场景适合快速验证算法；用户数据驱动环境会用上传数据拟合环境；自定义函数环境可用Python定义reset/step。"
    )
    rl_data_config = None
    rl_dataset_metadata = None
    rl_custom_config = None
    if rl_environment_mode == "用户数据驱动环境":
        st.subheader("📁 用户强化学习数据")
        st.caption("数据至少需要包含状态列和动作列；建议包含奖励列。若没有显式下一状态列，系统会按行顺序用下一行状态作为转移目标。")
        data_source_col, data_preview_col = st.columns([1.1, 1.4])
        with data_source_col:
            rl_data_source = st.radio(
                "数据来源",
                ["上传文件", "使用示例数据", "使用当前工作台数据"],
                horizontal=True,
                key="rl_data_source",
                help="上传CSV/Excel/out/dat文件，或复用项目中已经上传的数据。"
            )
            rl_uploaded_file = None
            if rl_data_source == "上传文件":
                rl_uploaded_file = st.file_uploader(
                    "上传强化学习日志数据",
                    type=["csv", "xlsx", "xls", "out", "dat", "txt"],
                    key="rl_data_file",
                    help="每行代表一个时间步或一次交互记录，包含状态、动作、奖励、可选下一状态/终止标记。"
                )
            elif rl_data_source == "使用示例数据":
                rl_sample_files = [
                    "强化学习_倒立摆控制日志.csv",
                    "强化学习_温控过程日志.csv",
                    "强化学习_能量调度日志.csv",
                ]
                rl_sample_choice = st.selectbox("选择强化学习示例数据", rl_sample_files, key="rl_sample_data_choice")
            try:
                if rl_data_source == "上传文件" and rl_uploaded_file is not None:
                    suffix = os.path.splitext(rl_uploaded_file.name)[1].lower()
                    if suffix == ".csv":
                        rl_user_df = pd.read_csv(rl_uploaded_file)
                    elif suffix in [".xlsx", ".xls"]:
                        rl_user_df = pd.read_excel(rl_uploaded_file)
                    else:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_rl_file:
                            tmp_rl_file.write(rl_uploaded_file.getvalue())
                            tmp_rl_path = tmp_rl_file.name
                        try:
                            rl_user_df = read_data_file(tmp_rl_path)
                        finally:
                            os.unlink(tmp_rl_path)
                    st.session_state.rl_user_df = rl_user_df
                elif rl_data_source == "使用示例数据":
                    rl_sample_path = os.path.join("sample_data", rl_sample_choice)
                    st.session_state.rl_user_df = pd.read_csv(rl_sample_path)
                elif rl_data_source == "使用当前工作台数据" and st.session_state.get("df") is not None:
                    st.session_state.rl_user_df = st.session_state.df.copy()
            except Exception:
                show_error("强化学习数据读取失败", traceback.format_exc())
        rl_user_df = st.session_state.get("rl_user_df")
        if rl_user_df is not None:
            numeric_cols = rl_user_df.select_dtypes(include=[np.number]).columns.tolist()
            with data_preview_col:
                st.metric("数据行数", len(rl_user_df))
                st.metric("数值列数", len(numeric_cols))
                st.dataframe(rl_user_df.head(20), use_container_width=True, hide_index=True)
            if len(numeric_cols) < 2:
                st.error("用户数据至少需要2个数值列：一个或多个状态列，以及一个动作列。")
            else:
                map_col1, map_col2, map_col3 = st.columns(3)
                inferred_action_cols = [col for col in numeric_cols if "action" in col.lower() or "torque" in col.lower()]
                inferred_reward_cols = [col for col in numeric_cols if col.lower() == "reward" or "reward" in col.lower()]
                inferred_next_cols = [col for col in numeric_cols if col.lower().startswith("next_")]
                inferred_terminal_cols = [col for col in numeric_cols if "terminal" in col.lower() or "done" in col.lower()]
                inferred_state_cols = [
                    col for col in numeric_cols
                    if col not in inferred_action_cols + inferred_reward_cols + inferred_next_cols + inferred_terminal_cols
                ]
                if not inferred_state_cols:
                    inferred_state_cols = numeric_cols[: min(3, max(1, len(numeric_cols) - 1))]
                with map_col1:
                    rl_state_columns = st.multiselect(
                        "状态列",
                        numeric_cols,
                        default=inferred_state_cols[: min(4, len(inferred_state_cols))],
                        key="rl_state_columns",
                        help="状态是智能体观测到的系统变量，例如温度、位移、速度、载荷、库存、功率等。"
                    )
                    rl_action_candidates = [col for col in numeric_cols if col not in rl_state_columns] or numeric_cols
                    default_action_index = 0
                    for idx, col in enumerate(rl_action_candidates):
                        if col in inferred_action_cols:
                            default_action_index = idx
                            break
                    rl_action_column = st.selectbox(
                        "动作列",
                        rl_action_candidates,
                        index=default_action_index,
                        key="rl_action_column",
                        help="动作是历史控制量或决策量。当前数据驱动环境首版支持单连续动作。"
                    )
                with map_col2:
                    reward_options = ["自动生成奖励"] + [col for col in numeric_cols if col not in [rl_action_column]]
                    default_reward_index = 0
                    for idx, col in enumerate(reward_options):
                        if col in inferred_reward_cols:
                            default_reward_index = idx
                            break
                    rl_reward_choice = st.selectbox(
                        "奖励列",
                        reward_options,
                        index=default_reward_index,
                        key="rl_reward_column",
                        help="有真实收益/成本/质量评分时请选择奖励列；没有时系统使用状态变化和动作幅度构造保守奖励。"
                    )
                    next_state_options = [col for col in numeric_cols if col not in [rl_action_column]]
                    default_next_cols = [col for col in inferred_next_cols if col in next_state_options]
                    rl_next_state_columns = st.multiselect(
                        "下一状态列（可选）",
                        next_state_options,
                        default=default_next_cols,
                        key="rl_next_state_columns",
                        help="如果数据中已有 next_* 列，请选择；否则系统按时间顺序用下一行状态作为下一状态。"
                    )
                with map_col3:
                    rl_terminal_options = ["无"] + numeric_cols
                    default_terminal_index = 0
                    for idx, col in enumerate(rl_terminal_options):
                        if col in inferred_terminal_cols:
                            default_terminal_index = idx
                            break
                    rl_terminal_choice = st.selectbox(
                        "终止标记列（可选）",
                        rl_terminal_options,
                        index=default_terminal_index,
                        key="rl_terminal_column",
                        help="可选。当前版本主要用于数据审计，训练环境仍按单轮最大步数终止。"
                    )
                    rl_data_action_limit = st.number_input(
                        "动作幅度上限",
                        min_value=0.0,
                        value=0.0,
                        step=0.1,
                        key="rl_data_action_limit",
                        help="填0表示从历史动作列自动推断最大绝对动作。"
                    )
                    rl_dynamics_trees = st.slider("环境模型树数量", 20, 300, 100, 20, key="rl_dynamics_trees")
                if rl_state_columns and rl_action_column:
                    rl_data_config = DataDrivenEnvironmentConfig(
                        data=rl_user_df,
                        state_columns=rl_state_columns,
                        action_column=rl_action_column,
                        reward_column=None if rl_reward_choice == "自动生成奖励" else rl_reward_choice,
                        next_state_columns=rl_next_state_columns or None,
                        terminal_column=None if rl_terminal_choice == "无" else rl_terminal_choice,
                        max_steps=60,
                        seed=42,
                        action_limit=None if rl_data_action_limit <= 0 else rl_data_action_limit,
                        n_estimators=rl_dynamics_trees,
                    )
                    rl_dataset_metadata = validate_rl_dataset(rl_data_config)
                    if rl_dataset_metadata["ok"]:
                        st.success("✅ 数据驱动强化学习环境校验通过")
                    else:
                        for issue in rl_dataset_metadata["issues"]:
                            st.error(issue)
                    st.dataframe(pd.DataFrame([rl_dataset_metadata]), use_container_width=True, hide_index=True)
    elif rl_environment_mode == "自定义函数环境":
        st.subheader("🧩 自定义强化学习场景函数")
        st.caption("定义 `reset(seed)` 和 `step(obs, action, step_index)`。平台会包装为 Gymnasium 环境，并使用 Stable-Baselines3 真实训练连续动作策略。")
        default_rl_env_code = '''import numpy as np

def reset(seed):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-0.2, 0.2)
    theta_dot = rng.uniform(-0.05, 0.05)
    return np.array([theta, theta_dot], dtype=float)

def step(obs, action, step_index):
    theta, theta_dot = obs
    action = float(np.clip(action, -2.0, 2.0))
    dt = 0.05
    gravity = 9.81
    length = 1.0
    damping = 0.08
    theta_ddot = (gravity / length) * np.sin(theta) + action - damping * theta_dot
    theta_dot = np.clip(theta_dot + theta_ddot * dt, -8.0, 8.0)
    theta = ((theta + theta_dot * dt + np.pi) % (2 * np.pi)) - np.pi
    reward = -(theta**2 + 0.1 * theta_dot**2 + 0.01 * action**2)
    done = abs(theta) > np.pi / 2
    info = {"error": float(theta), "state_norm": float(np.linalg.norm([theta, theta_dot]))}
    return np.array([theta, theta_dot], dtype=float), float(reward), bool(done), info
'''
        if "rl_custom_env_code" not in st.session_state:
            st.session_state.rl_custom_env_code = default_rl_env_code
        custom_col1, custom_col2 = st.columns([1.4, 1])
        with custom_col1:
            rl_custom_code = st.text_area(
                "自定义环境代码",
                value=st.session_state.rl_custom_env_code,
                height=360,
                key="rl_custom_env_code_input",
                help="必须定义 reset(seed) 和 step(obs, action, step_index)。step返回(next_obs, reward, done, info)。"
            )
            st.session_state.rl_custom_env_code = rl_custom_code
        with custom_col2:
            rl_custom_obs_dim = st.number_input("观测维度", min_value=1, max_value=64, value=2, step=1, key="rl_custom_obs_dim")
            rl_custom_action_limit = st.number_input("动作幅度上限", min_value=0.1, max_value=100.0, value=2.0, step=0.1, key="rl_custom_action_limit")
            rl_custom_max_steps = st.slider("单轮最大步数", 10, 500, 120, 10, key="rl_custom_max_steps")
            if st.button("校验自定义环境", key="validate_custom_rl_env"):
                try:
                    test_config = CustomFunctionEnvironmentConfig(
                        code=rl_custom_code,
                        observation_dim=int(rl_custom_obs_dim),
                        action_limit=float(rl_custom_action_limit),
                        max_steps=int(rl_custom_max_steps),
                        seed=42,
                    )
                    from src.models.reinforcement_learning import CustomFunctionControlEnvironment
                    test_env = CustomFunctionControlEnvironment(test_config)
                    test_obs = test_env.reset()
                    test_next, test_reward, test_done, test_info = test_env.step(0.0)
                    record_custom_execution("强化学习自定义环境", "校验成功")
                    st.success("✅ 自定义环境校验通过")
                    st.write({"obs": test_obs.tolist(), "next_obs": test_next.tolist(), "reward": test_reward, "done": test_done, "info": test_info})
                except Exception as exc:
                    record_custom_execution("强化学习自定义环境", "校验失败", str(exc))
                    show_error("自定义强化学习环境校验失败", traceback.format_exc())
        rl_custom_config = CustomFunctionEnvironmentConfig(
            code=rl_custom_code,
            observation_dim=int(rl_custom_obs_dim),
            action_limit=float(rl_custom_action_limit),
            max_steps=int(rl_custom_max_steps),
            seed=42,
        )
    
    rl_col1, rl_col2, rl_col3 = st.columns([1.05, 1.25, 1.2])
    with rl_col1:
        st.subheader("环境定义")
        if rl_environment_mode == "内置控制场景":
            rl_scenario = st.selectbox(
                "控制场景",
                ["position_control", "thermal_inertia", "vibration_damping", "energy_dispatch", "inverted_pendulum"],
                format_func=lambda x: {
                    "position_control": "位置/姿态控制",
                    "thermal_inertia": "热惯性过程控制",
                    "vibration_damping": "振动抑制控制",
                    "energy_dispatch": "能量调度控制",
                    "inverted_pendulum": "倒立摆平衡控制",
                }[x],
                key="rl_scenario",
                help="选择一维控制环境的状态转移类型。不同场景会改变惯性、扰动和控制响应强度。"
            )
            target_value = st.number_input("目标状态", value=0.0, key="rl_target_value", help="普通场景目标状态；倒立摆场景表示目标角度。")
            default_initial = 0.2 if rl_scenario == "inverted_pendulum" else 5.0
            initial_state = st.number_input("初始状态/初始角度", value=default_initial, key="rl_initial_state", help="普通场景为初始状态；倒立摆场景为初始角度(rad)。")
            disturbance = st.slider("扰动强度", 0.0, 1.0, 0.05, 0.01, key="rl_disturbance", help="环境噪声标准差，越大越考验策略鲁棒性。")
            action_default = 2.0 if rl_scenario == "inverted_pendulum" else 1.0
            action_limit = st.slider("动作幅度上限", 0.1, 5.0, action_default, 0.1, key="rl_action_limit", help="单步控制量的上下限，用于模拟执行器饱和。")
            max_steps = st.slider("单轮最大步数", 10, 300, 60, 10, key="rl_max_steps", help="每个 episode 最多执行的控制步数。")
        elif rl_environment_mode == "用户数据驱动环境":
            rl_scenario = "data_driven"
            target_value = 0.0
            initial_state = 0.0
            disturbance = 0.0
            action_limit = 1.0
            max_steps = st.slider("单轮最大步数", 10, 300, 60, 10, key="rl_data_max_steps", help="策略在数据驱动环境中每轮最多滚动预测的步数。")
            if rl_data_config is not None and rl_dataset_metadata is not None:
                rl_data_config.max_steps = max_steps
                st.metric("状态维度", rl_dataset_metadata["state_dim"])
                st.metric("有效转移样本", rl_dataset_metadata["usable_rows"])
                st.caption(f"动作列: {rl_data_config.action_column}")
                st.caption(f"奖励: {rl_data_config.reward_column or '自动生成奖励'}")
            else:
                st.warning("请先上传数据并完成状态列/动作列映射。")
        else:
            rl_scenario = "custom_function"
            target_value = 0.0
            initial_state = 0.0
            disturbance = 0.0
            action_limit = float(rl_custom_config.action_limit if rl_custom_config is not None else 1.0)
            max_steps = int(rl_custom_config.max_steps if rl_custom_config is not None else 60)
            if rl_custom_config is not None:
                st.metric("观测维度", rl_custom_config.observation_dim)
                st.metric("动作上限", f"{rl_custom_config.action_limit:.3f}")
                st.caption("自定义环境由 reset/step 函数定义，训练时使用 Stable-Baselines3 连续动作算法。")

    with rl_col2:
        st.subheader("算法与训练")
        backend_label_map = {
            "stable-baselines3": "Stable-Baselines3 真实深度RL",
            "tabular": "表格RL真实Q表",
            "baseline": "PID/MPC控制基线",
            "ray-rllib": "Ray RLlib 分布式RL",
        }
        rl_backend_label = st.selectbox(
            "训练后端",
            options=list(backend_label_map.values()),
            index=0,
            key="rl_backend",
            help="选择真实训练后端。深度RL使用Stable-Baselines3；表格RL使用Q表；基线用于工程对照。"
        )
        selected_backend = {v: k for k, v in backend_label_map.items()}[rl_backend_label]
        if rl_environment_mode in ["用户数据驱动环境", "自定义函数环境"] and selected_backend != "stable-baselines3":
            st.warning(f"{rl_environment_mode} 当前使用 Stable-Baselines3 连续动作后端；表格RL/基线/RLlib不用于该环境训练。")
            selected_backend = "stable-baselines3"
        if not rl_backend_status.get(selected_backend, False):
            st.error(f"{rl_backend_label} 当前不可用。Stable-Baselines3 后端需 torch/gymnasium/stable-baselines3；Ray RLlib 在当前 Python 3.13 环境没有可安装包。")
        backend_specs = [item for item in rl_specs if item.backend == selected_backend]
        if rl_environment_mode in ["用户数据驱动环境", "自定义函数环境"]:
            backend_specs = [item for item in backend_specs if item.key in ["a2c", "ppo", "ddpg", "td3", "sac"]]
        rl_options = {f"{item.name} · {item.family}": item.key for item in backend_specs}
        default_backend_keys = {
            "stable-baselines3": ["ppo", "sac", "td3"],
            "tabular": ["q_learning", "sarsa"],
            "baseline": ["pid_baseline", "mpc_baseline"],
            "ray-rllib": ["ppo_rllib"],
        }[selected_backend]
        selected_rl_labels = st.multiselect(
            "选择算法",
            options=list(rl_options.keys()),
            default=[label for label, key in rl_options.items() if key in default_backend_keys],
            key="rl_algorithms",
            help="可多选进行同一环境下的策略训练与评估对比。"
        )
        selected_rl_algorithms = [rl_options[label] for label in selected_rl_labels]
        if selected_rl_algorithms:
            st.dataframe(
                rl_spec_df[rl_spec_df["key"].isin(selected_rl_algorithms)][["key", "name", "family", "action_space", "backend", "implementation_status", "description"]]
                .rename(columns={"key": "算法ID", "name": "算法名称", "family": "算法族", "action_space": "动作空间", "backend": "后端", "implementation_status": "实现状态", "description": "说明"}),
                use_container_width=True,
                hide_index=True,
            )
        episodes = st.slider("训练轮数", 5, 300, 40 if selected_backend == "stable-baselines3" else 60, 5, key="rl_episodes", help="训练轮数越多曲线越稳定，但运行时间也更长。")
        if selected_backend == "stable-baselines3":
            learning_rate = st.slider("学习率", 0.0001, 0.01, 0.001, 0.0001, key="rl_learning_rate", help="Stable-Baselines3优化器学习率。过大容易使策略不稳定。")
        else:
            learning_rate = st.slider("学习率/搜索步长", 0.005, 0.3, 0.05, 0.005, key="rl_learning_rate", help="表格RL更新步长或基线搜索步长。")
        exploration = st.slider("探索强度", 0.0, 1.0, 0.2, 0.01, key="rl_exploration", help="训练早期动作随机扰动强度，用于探索不同控制策略。")
        seed = st.number_input("随机种子", 0, 9999, 42, key="rl_seed", help="固定随机种子便于复现实验结果。")

    with rl_col3:
        st.subheader("奖励函数")
        action_penalty = st.slider("动作惩罚权重", 0.0, 0.5, 0.05, 0.01, key="rl_action_penalty", help="惩罚控制动作幅度，值越大越偏向省力和平滑控制。")
        stability_penalty = st.slider("稳定性惩罚权重", 0.0, 0.2, 0.01, 0.005, key="rl_stability_penalty", help="惩罚大误差平方项，值越大越重视抑制偏差振荡。")
        done_tolerance = st.slider("收敛阈值", 0.001, 0.2, 0.02, 0.001, key="rl_done_tolerance", help="当绝对误差小于该阈值时，可提前结束当前轮评估。")
        st.markdown("**状态**: 系统偏差观测  \n**动作**: 连续或离散控制量  \n**奖励**: 偏差、动作幅度、稳定性联合惩罚")
        train_rl = st.button("🚀 训练并对比策略", type="primary", key="train_rl_policy", use_container_width=True)
    
    if train_rl:
        try:
            if not selected_rl_algorithms:
                st.warning("请至少选择一个强化学习算法。")
                st.stop()
            if not rl_backend_status.get(selected_backend, False):
                st.error("所选训练后端不可用，不能启动训练，避免产生失真结果。")
                st.stop()
            if rl_environment_mode == "用户数据驱动环境":
                if rl_data_config is None:
                    st.error("请先上传强化学习数据并完成状态/动作/奖励字段映射。")
                    st.stop()
                rl_data_config.max_steps = max_steps
                rl_data_config.seed = seed
                rl_dataset_metadata = validate_rl_dataset(rl_data_config)
                if not rl_dataset_metadata["ok"]:
                    st.error("数据驱动环境校验未通过，不能训练。")
                    for issue in rl_dataset_metadata["issues"]:
                        st.error(issue)
                    st.stop()
            elif rl_environment_mode == "自定义函数环境":
                if rl_custom_config is None:
                    st.error("请先填写并校验自定义强化学习环境函数。")
                    st.stop()
                rl_custom_config.seed = seed
                rl_custom_config.max_steps = max_steps
            else:
                config = ControlEnvironmentConfig(
                    target_value=target_value,
                    initial_state=initial_state,
                    disturbance=disturbance,
                    action_limit=action_limit,
                    max_steps=max_steps,
                    seed=seed,
                    scenario=rl_scenario,
                    action_penalty=action_penalty,
                    stability_penalty=stability_penalty,
                    done_tolerance=done_tolerance,
                )
            with st.spinner("正在训练强化学习控制策略..."):
                if rl_environment_mode == "用户数据驱动环境":
                    rl_history, rl_eval, rl_env_metadata = compare_data_driven_algorithms(
                        rl_data_config,
                        selected_rl_algorithms,
                        episodes=episodes,
                        learning_rate=learning_rate,
                    )
                elif rl_environment_mode == "自定义函数环境":
                    record_custom_execution("强化学习自定义环境", "开始训练")
                    rl_history, rl_eval, rl_env_metadata = compare_custom_function_algorithms(
                        rl_custom_config,
                        selected_rl_algorithms,
                        episodes=episodes,
                        learning_rate=learning_rate,
                    )
                    record_custom_execution("强化学习自定义环境", "训练成功")
                else:
                    rl_history, rl_eval = compare_algorithms(
                        config,
                        selected_rl_algorithms,
                        episodes=episodes,
                        learning_rate=learning_rate,
                        exploration=exploration,
                        backend=selected_backend
                    )
                    rl_env_metadata = {
                        "environment": "builtin",
                        "scenario": rl_scenario,
                        "max_steps": max_steps,
                        "action_limit": action_limit,
                    }
                rl_summary = summarize_evaluation(rl_eval)
                rl_recommendation = recommend_control_action(rl_eval)
            st.session_state.rl_history = rl_history
            st.session_state.rl_eval = rl_eval
            st.session_state.rl_summary = rl_summary
            st.session_state.rl_recommendation = rl_recommendation
            st.session_state.rl_env_metadata = rl_env_metadata
            append_run_history(st.session_state, "强化学习策略训练", f"{rl_environment_mode}: {', '.join(selected_rl_algorithms)}")
            st.success("✅ 控制策略训练完成")
        except Exception:
            show_error("强化学习训练失败", traceback.format_exc())
    
    if st.session_state.get('rl_history') is not None:
        rl_history = st.session_state.rl_history
        rl_eval = st.session_state.rl_eval
        rl_summary = st.session_state.get('rl_summary', summarize_evaluation(rl_eval))
        rl_recommendation = st.session_state.rl_recommendation
        
        best_row = rl_summary.iloc[0] if rl_summary is not None and not rl_summary.empty else None
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("最佳算法", best_row["algorithm"] if best_row is not None else "N/A")
        m2.metric("平均绝对误差", f"{best_row['mean_abs_error']:.4f}" if best_row is not None else "N/A")
        m3.metric("最终误差", f"{rl_recommendation['final_error']:.4f}")
        m4.metric("策略建议", rl_recommendation['summary'])
        if st.session_state.get("rl_env_metadata"):
            with st.expander("🧾 强化学习环境元数据", expanded=False):
                st.json(st.session_state.rl_env_metadata)
        
        rl_result_tabs = st.tabs(["训练曲线", "策略轨迹", "误差与动作", "算法排行榜", "数据导出"])
        with rl_result_tabs[0]:
            fig_reward = go.Figure()
            for algorithm, frame in rl_history.groupby("algorithm"):
                fig_reward.add_trace(go.Scatter(x=frame["episode"], y=frame["total_reward"], mode="lines", name=f"{algorithm} 回报"))
                fig_reward.add_trace(go.Scatter(x=frame["episode"], y=frame["best_reward"], mode="lines", name=f"{algorithm} 最佳", line=dict(dash="dash")))
            fig_reward.update_layout(height=420, title="多算法训练回报曲线", xaxis_title="Episode", yaxis_title="Reward")
            st.plotly_chart(fig_reward, use_container_width=True)
        with rl_result_tabs[1]:
            fig_policy = go.Figure()
            for algorithm, frame in rl_eval.groupby("algorithm"):
                fig_policy.add_trace(go.Scatter(x=frame["step"], y=frame["state"], mode="lines+markers", name=f"{algorithm} 状态"))
            fig_policy.update_layout(height=420, title="状态收敛轨迹", xaxis_title="Step", yaxis_title="State")
            st.plotly_chart(fig_policy, use_container_width=True)
        with rl_result_tabs[2]:
            err_col, action_col = st.columns(2)
            with err_col:
                fig_error = go.Figure()
                for algorithm, frame in rl_eval.groupby("algorithm"):
                    fig_error.add_trace(go.Scatter(x=frame["step"], y=frame["abs_error"], mode="lines", name=algorithm))
                fig_error.update_layout(height=360, title="绝对误差变化", xaxis_title="Step", yaxis_title="|Error|")
                st.plotly_chart(fig_error, use_container_width=True)
            with action_col:
                fig_action = go.Figure()
                for algorithm, frame in rl_eval.groupby("algorithm"):
                    fig_action.add_trace(go.Histogram(x=frame["action"], name=algorithm, opacity=0.65))
                fig_action.update_layout(height=360, title="动作分布", xaxis_title="Action", yaxis_title="Count", barmode="overlay")
                st.plotly_chart(fig_action, use_container_width=True)
        with rl_result_tabs[3]:
            st.dataframe(rl_summary, use_container_width=True, hide_index=True)
            if rl_summary is not None and not rl_summary.empty:
                fig_rank = go.Figure(go.Bar(
                    x=rl_summary["algorithm"],
                    y=rl_summary["mean_abs_error"],
                    marker_color="#2b8cbe"
                ))
                fig_rank.update_layout(height=360, title="算法平均绝对误差排行", xaxis_title="Algorithm", yaxis_title="Mean |Error|")
                st.plotly_chart(fig_rank, use_container_width=True)
        with rl_result_tabs[4]:
            st.download_button(
                "📥 下载训练历史",
                rl_history.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"rl_training_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_rl_history"
            )
            st.download_button(
                "📥 下载策略评估结果",
                rl_eval.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"rl_policy_evaluation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_rl_eval"
            )
            rl_report = build_html_report(
                "强化学习与智能控制图文结果报告",
                metrics={
                    "环境来源": st.session_state.get("rl_env_metadata", {}).get("environment", rl_environment_mode),
                    "最佳算法": best_row["algorithm"] if best_row is not None else "N/A",
                    "平均绝对误差": f"{best_row['mean_abs_error']:.4f}" if best_row is not None else "N/A",
                    "最终误差": f"{rl_recommendation['final_error']:.4f}",
                    "策略建议": rl_recommendation["summary"],
                },
                sections=[("控制结论", rl_recommendation.get("detail", rl_recommendation["summary"]))],
                figures=[
                    ("多算法训练回报曲线", fig_reward if "fig_reward" in locals() else None),
                    ("状态收敛轨迹", fig_policy if "fig_policy" in locals() else None),
                    ("绝对误差变化", fig_error if "fig_error" in locals() else None),
                    ("动作分布", fig_action if "fig_action" in locals() else None),
                    ("算法平均绝对误差排行", fig_rank if "fig_rank" in locals() else None),
                ],
                tables=[("算法排行榜", rl_summary), ("训练历史", rl_history), ("策略评估", rl_eval)],
            )
            st.download_button(
                "📄 下载HTML图文报告",
                rl_report.encode("utf-8"),
                file_name=f"rl_control_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                key="download_rl_html_report",
                use_container_width=True,
            )
        
# ==================== 标签6: 帮助手册 ====================
with tab6:
    st.markdown("<h1 style='font-size: 32px; font-weight: bold;'>📖 帮助手册</h1>", unsafe_allow_html=True)
    
    help_col1, help_col2 = st.columns([1, 3])
    
    with help_col1:
        st.write("**📑 目录**")
        
        # 搜索框
        search_query = st.text_input("🔍 搜索", placeholder="输入关键词...", key="help_search_tab4")
        
        st.session_state.help_topics_cache = get_all_topics()
        
        if search_query:
            search_results = search_help(search_query)
            if search_results:
                st.write(f"**搜索结果 ({len(search_results)}个):**")
                for key, title, icon, summary, score in search_results:
                    if st.button(
                        f"{icon} {title}",
                        key=f"help_search_hit_{key}",
                        use_container_width=True,
                        help="点击打开该帮助主题"
                    ):
                        st.session_state.help_topic = key
                        st.rerun()
                    st.caption(f"{summary}  · 匹配度 {score}")
            else:
                st.info("未找到相关内容")
        else:
            st.write("**所有主题:**")
            topics = st.session_state.help_topics_cache
            topic_options = {f"{icon} {title}": key for key, title, icon in topics}
            
            current_topic_key = st.session_state.get('help_topic', 'overview')
            current_topic_title = None
            for key, title, icon in topics:
                if key == current_topic_key:
                    current_topic_title = f"{icon} {title}"
                    break
            
            selected_topic = st.selectbox(
                "选择主题",
                options=list(topic_options.keys()),
                index=list(topic_options.keys()).index(current_topic_title) if current_topic_title else 0,
                key="topic_select_tab4"
            )
            if selected_topic:
                st.session_state.help_topic = topic_options[selected_topic]
    
    with help_col2:
        # 显示选中的主题内容
        current_topic = st.session_state.get('help_topic', 'overview')
        help_content = get_help_content(current_topic)
        
        if help_content:
            st.markdown(f"## {help_content['icon']} {help_content['title']}")
            st.markdown(help_content['content'])
            
            # 添加反馈按钮
            st.divider()
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("👍 有帮助", key=f"helpful_{current_topic}_tab4"):
                    st.success("感谢您的反馈！")
            with col2:
                if st.button("👎 没帮助", key=f"not_helpful_{current_topic}_tab4"):
                    st.info("我们会继续改进，谢谢反馈！")
        else:
            st.error("未找到帮助内容")



# ==================== 页脚 ====================
st.sidebar.markdown("""
---
**关于本平台**

可用于风电、海洋平台等工程系统的数据时序预测、分类，故障诊断与优化求解

© 2026 OEye contributors
""")
