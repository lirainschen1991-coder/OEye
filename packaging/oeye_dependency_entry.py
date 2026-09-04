from __future__ import annotations

import subprocess
import sys
import shutil
from pathlib import Path

import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "requirements-standard.txt").exists() else SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dependency_manager import (  # noqa: E402
    DependencyStatus,
    build_pip_install_command,
    check_profile,
    create_managed_venv,
    default_base_python,
    discover_python_candidates,
    estimate_missing,
    install_list,
    load_config,
    log_dir,
    managed_python,
    run_install_command,
    save_config,
    subprocess_creationflags,
)


STATUS_LABELS = {
    DependencyStatus.SATISFIED: "已满足",
    DependencyStatus.MISSING: "未安装",
    DependencyStatus.OUTDATED: "版本过低",
    DependencyStatus.UNSUPPORTED: "环境不支持",
}


def _checks_to_df(checks):
    return [
        {
            "依赖": item.package_name,
            "版本要求": item.specifier or "任意",
            "当前版本": item.installed_version or "-",
            "状态": STATUS_LABELS[item.status],
            "说明": item.reason,
            "预计下载(MB)": item.estimated_download_mb,
            "预计安装后(MB)": item.estimated_install_mb,
        }
        for item in checks
    ]


def _profile_panel(profile: str, title: str, target_python: str, index_url: str):
    st.subheader(title)
    checks = check_profile(profile, root=PROJECT_ROOT, python_executable=target_python)
    missing = install_list(checks)
    download_mb, install_mb = estimate_missing(checks)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("依赖总数", len(checks))
    c2.metric("已满足", sum(item.status == DependencyStatus.SATISFIED for item in checks))
    c3.metric("需补齐/升级", len(missing))
    c4.metric("预计新增下载", f"{download_mb} MB")
    st.caption(f"预计安装后占用约 {install_mb} MB；实际大小会随 CPU/GPU 版本、pip 缓存和系统平台变化。")

    st.dataframe(_checks_to_df(checks), width="stretch", hide_index=True)

    if missing:
        with st.expander("将安装或升级的精确清单", expanded=True):
            st.code("\n".join(missing), language="text")
    else:
        st.success("该档位依赖已满足，不会触发重复下载。")

    python_exists = Path(target_python).exists() or shutil.which(target_python) is not None
    if not python_exists:
        st.warning("目标 Python 不存在。请先创建 OEye 托管 venv，或选择一个可用 Python。")

    if st.button(f"一键补齐 {title}", disabled=not bool(missing) or not python_exists, key=f"install_{profile}"):
        command = build_pip_install_command(target_python, missing, index_url=index_url)
        st.info("正在执行安装，日志会写入本地 OEye 日志目录。")
        st.code(" ".join(command), language="powershell")
        code = run_install_command(command, log_dir() / "dependency-install.log")
        if code == 0:
            st.success("安装完成，已自动重新检测。")
            st.rerun()
        else:
            st.error(f"安装命令返回 {code}，请查看日志：{log_dir() / 'dependency-install.log'}")

    return checks


def _launch_main_app(target_python: str):
    if not target_python or (not Path(target_python).exists() and shutil.which(target_python) is None):
        st.error("目标 Python 不存在，无法启动主应用。请先选择一个已安装的 Python，或创建 OEye 托管 venv。")
        return
    launcher = SCRIPT_DIR / "oeye_launcher.py"
    command = [target_python, str(launcher), "--main-app"]
    try:
        process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), creationflags=subprocess_creationflags())
    except FileNotFoundError as exc:
        st.error(f"无法启动主应用：目标 Python 不存在或不可执行。{exc}")
        return
    except Exception as exc:
        st.error(f"无法启动主应用：{exc}")
        return
    st.session_state["oeye_main_process_pid"] = process.pid
    st.success(f"主应用已启动，进程 PID：{process.pid}。如果没有自动打开浏览器，请查看日志目录。")


def _python_exists(value: str | Path | None) -> bool:
    if not value:
        return False
    value_str = str(value)
    return Path(value_str).exists() or shutil.which(value_str) is not None


def _python_key(value: str | Path) -> str:
    value_str = str(value)
    resolved = shutil.which(value_str) or value_str
    try:
        return str(Path(resolved).resolve()).casefold()
    except Exception:
        return str(Path(resolved)).casefold()


def _default_target_python(config: dict, base_candidates: list[str]) -> str:
    candidates: list[str] = []

    configured = str(config.get("python_executable") or "")
    if _python_exists(configured):
        candidates.append(configured)

    managed = managed_python()
    if managed.exists():
        candidates.append(str(managed))

    candidates.extend(candidate for candidate in base_candidates if _python_exists(candidate))

    deduped: list[str] = []
    for candidate in candidates:
        normalized = _python_key(candidate)
        if normalized not in {_python_key(item) for item in deduped}:
            deduped.append(candidate)

    if not deduped:
        fallback = default_base_python()
        return fallback or str(managed)

    def score(candidate: str) -> tuple[int, int]:
        checks = check_profile("standard", root=PROJECT_ROOT, python_executable=candidate)
        satisfied = sum(item.status == DependencyStatus.SATISFIED for item in checks)
        runtime_checks = check_profile("runtime", root=PROJECT_ROOT, python_executable=candidate)
        runtime_satisfied = sum(item.status == DependencyStatus.SATISFIED for item in runtime_checks)
        return satisfied, runtime_satisfied

    return max(deduped, key=score)


def _candidate_rows(candidates: list[str], selected_python: str) -> list[dict]:
    rows = []
    for candidate in candidates:
        if not _python_exists(candidate):
            continue
        standard = check_profile("standard", root=PROJECT_ROOT, python_executable=candidate)
        runtime = check_profile("runtime", root=PROJECT_ROOT, python_executable=candidate)
        rows.append(
            {
                "Python": candidate,
                "轻型满足": f"{sum(item.status == DependencyStatus.SATISFIED for item in standard)} / {len(standard)}",
                "完整满足": f"{sum(item.status == DependencyStatus.SATISFIED for item in runtime)} / {len(runtime)}",
                "当前选择": "是" if _python_key(candidate) == _python_key(selected_python) else "",
            }
        )
    return rows


def main():
    st.set_page_config(page_title="OEye 依赖检测与启动", page_icon="OEye", layout="wide")
    st.title("OEye 依赖检测与启动")

    config = load_config()
    default_index = str(config.get("pip_index_url") or "")
    base_candidates = discover_python_candidates()
    default_base = str(config.get("base_python") or default_base_python())
    default_python = _default_target_python(config, base_candidates)
    known_candidates = []
    for candidate in [str(config.get("python_executable") or ""), str(managed_python()), *base_candidates]:
        if candidate and _python_exists(candidate) and _python_key(candidate) not in {_python_key(item) for item in known_candidates}:
            known_candidates.append(candidate)

    with st.sidebar:
        st.header("运行环境")
        with st.expander("使用步骤与原理", expanded=True):
            st.markdown(
                """
1. 双击 `OEye.exe` 后，先进入依赖检测与启动界面，不会直接训练模型。
2. 先看“目标 Python”。系统会扫描本机 Python，并优先选择依赖满足度最高的环境。
3. 如果轻型/完整依赖已经满足，直接点击“启动 OEye 主应用”。
4. 如果缺少依赖，按需要点击“一键补齐轻型依赖”或“一键补齐完整依赖”。安装前会生成缺失清单，只安装未满足项。
5. “创建/检查 OEye 托管 venv”是可选项，用于想给 OEye 单独建一个 Python 环境的机器。创建 venv 本身不下载算法依赖，也不会自动切换目标 Python。
6. 主应用会在本机 `127.0.0.1` 动态端口启动；端口被占用时会自动换下一个可用端口。

原理：`OEye.exe` 已内置打开检测界面所需的 Python、Streamlit 和 PyWebView；算法库按“已有优先、缺失补齐”的方式检查目标 Python，避免重复下载已满足依赖。
                """.strip()
            )
        if known_candidates:
            selected_index = next(
                (idx for idx, item in enumerate(known_candidates) if _python_key(item) == _python_key(default_python)),
                0,
            )
            target_python = st.selectbox(
                "目标 Python",
                options=known_candidates,
                index=selected_index,
                help="默认选择依赖满足度最高的已有 Python；只有一键补齐会向该环境安装缺失依赖。",
            )
        else:
            target_python = st.text_input("目标 Python", value=default_python, help="未发现可用 Python 时可手动填写 python.exe 路径。")
        target_exists = _python_exists(target_python)
        if target_exists:
            st.success(f"当前检测目标：{target_python}")
        else:
            st.error("当前目标 Python 不存在；依赖检测结果会不准确，主应用也无法启动。")
        if known_candidates:
            with st.expander("已发现 Python 环境", expanded=False):
                st.dataframe(_candidate_rows(known_candidates, target_python), width="stretch", hide_index=True)
        else:
            st.warning("未检测到可用于安装依赖的系统 Python。当前启动包可打开检测 UI，但补齐算法依赖需要一个普通 Python 解释器。")
        base_python = st.text_input("创建托管环境所用 Python", value=default_base, help="首次创建 OEye 托管 venv 时使用；不会使用 OEye.exe 自身作为 pip 安装目标。")
        index_url = st.text_input("pip 镜像源", value=default_index, help="仅用于本应用安装命令，不修改用户全局 pip 配置。")

        if st.button("保存设置"):
            save_config({"python_executable": target_python, "base_python": base_python, "pip_index_url": index_url})
            st.success("设置已保存。")

        if st.button("创建/检查 OEye 托管 venv（可选）", help="这是给运行机器准备独立环境的功能。创建 venv 本身不会下载算法依赖；补依赖需另点一键补齐。"):
            try:
                created = create_managed_venv(base_python)
                save_config({"python_executable": target_python, "base_python": base_python, "pip_index_url": index_url})
                st.success(f"托管环境就绪：{created}。当前目标 Python 未自动切换；需要使用托管环境时请在目标 Python 下拉框中选择它。")
                st.rerun()
            except Exception as exc:
                st.error(f"托管环境创建失败：{exc}")

        st.divider()
        st.caption(f"日志目录：{log_dir()}")

    st.info("启动包内置最小 UI 能力；传统机器学习使用轻型依赖，深度学习/强化学习完整训练使用完整依赖。")

    tab_overview, tab_standard, tab_runtime, tab_logs = st.tabs(["总览", "轻型依赖", "完整依赖", "日志与启动"])

    with tab_standard:
        standard_checks = _profile_panel("standard", "轻型依赖", target_python, index_url)

    with tab_runtime:
        runtime_checks = _profile_panel("runtime", "完整依赖", target_python, index_url)

    with tab_overview:
        all_checks = check_profile("runtime", root=PROJECT_ROOT, python_executable=target_python)
        needed = install_list(all_checks)
        download_mb, install_mb = estimate_missing(all_checks)
        st.metric("完整能力仍需补齐/升级", len(needed))
        st.metric("完整能力预计新增下载", f"{download_mb} MB")
        st.metric("完整能力预计安装后占用", f"{install_mb} MB")
        st.dataframe(_checks_to_df(all_checks), width="stretch", hide_index=True)

    with tab_logs:
        if st.button("启动 OEye 主应用", disabled=not target_exists):
            _launch_main_app(target_python)

        install_log = log_dir() / "dependency-install.log"
        if install_log.exists():
            st.text_area("最近安装日志", install_log.read_text(encoding="utf-8", errors="replace")[-12000:], height=320)
        else:
            st.caption("暂无安装日志。")

        st.caption("主应用会作为独立 Streamlit 子进程运行在 127.0.0.1 动态端口。")


if __name__ == "__main__":
    main()
