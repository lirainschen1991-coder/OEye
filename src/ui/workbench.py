"""Project workbench helpers for the Streamlit application."""

from __future__ import annotations

import datetime as _dt
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _shape_label(value: Any) -> str:
    if value is None:
        return "未准备"
    shape = getattr(value, "shape", None)
    if shape is None:
        try:
            return f"{len(value)}"
        except TypeError:
            return "已准备"
    if len(shape) == 1:
        return f"{shape[0]} 样本"
    return f"{shape[0]} x {shape[1]}"


def build_workbench_summary(state: Any) -> dict[str, Any]:
    """Collect a compact, UI-ready snapshot from Streamlit session state."""
    df = state.get("df")
    y_pred = state.get("y_pred")
    error_log = state.get("error_log", [])
    run_history = state.get("run_history", [])

    if isinstance(df, pd.DataFrame):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        missing_cells = int(df.isna().sum().sum())
        data_status = "已上传"
        data_shape = f"{len(df)} 行 / {len(df.columns)} 列"
    else:
        numeric_cols = []
        missing_cells = 0
        data_status = "等待上传"
        data_shape = "无数据"

    if y_pred is not None:
        prediction_status = f"已生成 {_shape_label(y_pred)}"
    else:
        prediction_status = "未预测"

    trained = bool(state.get("model_trained", False))
    model_type = state.get("model_type_trained") or "未训练"
    diagnosis = state.get("enhanced_diagnosis_result")
    diagnosis_status = "已完成" if diagnosis else "待诊断"

    return {
        "data_status": data_status,
        "data_shape": data_shape,
        "numeric_columns": len(numeric_cols),
        "missing_cells": missing_cells,
        "task_type": state.get("batch_task_type") or state.get("task_type") or "未选择",
        "training_status": "已训练" if trained else "未训练",
        "model_type": model_type,
        "prediction_status": prediction_status,
        "diagnosis_status": diagnosis_status,
        "error_count": len(error_log),
        "run_count": len(run_history),
    }


def next_step_suggestions(state: Any) -> list[str]:
    suggestions = []
    if state.get("df") is None:
        suggestions.append("上传或加载示例数据，先完成数据准备。")
    if state.get("df") is not None and not state.get("feature_cols"):
        suggestions.append("选择目标列和特征列，锁定训练任务定义。")
    if state.get("feature_cols") and not state.get("model_trained", False):
        suggestions.append("训练一个基线模型，再用 AutoML 或深度学习模型做对比。")
    if state.get("model_trained", False) and state.get("y_pred") is None:
        suggestions.append("运行预测并查看残差、置信度和异常样本。")
    if state.get("model_trained", False) and not state.get("enhanced_diagnosis_result"):
        suggestions.append("运行增强故障诊断，生成根因解释和处置建议。")
    if not suggestions:
        suggestions.append("导出训练配置、诊断报告和预测结果，沉淀为可复现实验。")
    return suggestions


def render_project_workbench(st, state: Any) -> None:
    """Render the top-level project workbench."""
    summary = build_workbench_summary(state)

    st.markdown("<h1 style='font-size: 32px; font-weight: bold;'>🏠 项目工作台</h1>", unsafe_allow_html=True)
    st.caption("把数据、训练、预测、诊断和优化状态放在同一屏，方便判断下一步该做什么。")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("数据状态", summary["data_status"], summary["data_shape"])
    c2.metric("训练状态", summary["training_status"], summary["model_type"])
    c3.metric("预测状态", summary["prediction_status"])
    c4.metric("诊断状态", summary["diagnosis_status"], f"{summary['error_count']} 个错误日志")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("📌 当前项目概况")
        overview = pd.DataFrame(
            [
                {"项目": "任务类型", "状态": summary["task_type"]},
                {"项目": "数值列", "状态": summary["numeric_columns"]},
                {"项目": "缺失单元格", "状态": summary["missing_cells"]},
                {"项目": "运行记录", "状态": summary["run_count"]},
            ]
        )
        st.dataframe(overview, use_container_width=True, hide_index=True)

        y_pred = state.get("y_pred")
        y_test = state.get("y_test")
        if y_pred is not None and y_test is not None:
            try:
                y_pred_arr = np.asarray(y_pred).ravel()
                y_test_arr = np.asarray(y_test).ravel()
                n = min(len(y_pred_arr), len(y_test_arr), 200)
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=y_test_arr[:n], name="真实值", mode="lines"))
                fig.add_trace(go.Scatter(y=y_pred_arr[:n], name="预测值", mode="lines"))
                fig.update_layout(height=320, title="最近预测对比", margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("预测结果已存在，但当前格式暂不适合在工作台绘图。")

    with right:
        st.subheader("➡️ 下一步建议")
        for suggestion in next_step_suggestions(state):
            st.info(suggestion)

        st.subheader("🧾 最近运行")
        run_history = state.get("run_history", [])
        if run_history:
            for item in run_history[-5:][::-1]:
                st.write(f"**{item.get('name', '运行')}**")
                st.caption(item.get("time", ""))
        else:
            st.caption("暂无运行记录")


def append_run_history(state: Any, name: str, detail: str | None = None) -> None:
    history = state.get("run_history", [])
    history.append(
        {
            "name": name,
            "detail": detail or "",
            "time": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    state.run_history = history[-50:]
