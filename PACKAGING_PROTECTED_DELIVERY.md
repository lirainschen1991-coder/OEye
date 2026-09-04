# OEye 轻量受保护交付说明

本方案用于生成 Windows 桌面交付包：包内包含最小启动能力，业务代码以 `.pyc` 字节码放入 `protected_sources`，不明文交付 `app.py`、`src/**/*.py`、`custom_models/**/*.py`。

## 架构

- `OEye.exe` 启动 `packaging/oeye_launcher.py`，打开依赖检测与启动 UI。
- 依赖 UI 使用包内 Streamlit/PyWebView 运行，客户无需预装 Streamlit 才能打开检测界面。
- 轻型依赖来自 `requirements-standard.txt`，覆盖传统机器学习、诊断、优化和报告导出。
- 完整依赖来自 `requirements-runtime.txt`，增加 TensorFlow、PyTorch、CatBoost、Gymnasium、Stable-Baselines3 等深度学习和强化学习能力。
- 依赖安装目标是 OEye 托管 venv 或用户指定 Python。检测先读当前版本，只对缺失或版本不满足的包执行 pip 安装。
- Ray RLlib 在 `requirements-runtime.txt` 中带有 `python_version < "3.13"` 标记；Python 3.13 会显示环境不支持，不进入安装命令。

## 构建

```powershell
pip install -r requirements-bootstrap.txt
powershell -ExecutionPolicy Bypass -File packaging\build_oeye.ps1
```

仅检查本机打包环境：

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_oeye.ps1 -CheckOnly
```

仅检查字节码保护构建：

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_oeye.ps1 -SkipPyInstaller
```

跳过冻结包 UI 冒烟测试：

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_oeye.ps1 -SkipSmokeTest
```

当前优化版参考 AIPlot 的成功打包经验，使用 `streamlit.web.bootstrap.run()` 传入端口和 headless 配置，并在构建后直接访问 `/_stcore/health` 验证依赖检测 UI 是否可启动。`polars`、`duckdb` 等本依赖检测 UI 不需要的旁支依赖会被显式排除，避免把启动层打成完整数据科学运行时。

交付包会使用 `packaging/OEye.ico` 作为 `OEye.exe` 图标，并在构建 smoke test 中验证冻结包可以导入 `streamlit.runtime.scriptrunner.magic_funcs`，避免 Streamlit 页面打开后才暴露隐式模块缺失。

## 交付检查

构建脚本会检查：

- dist 内无明文业务源码：`app.py`、`src/**/*.py`、`custom_models/**/*.py`。
- dist 内无大型 `.dat/.out`、`saved_models`、`.venv`、pip 缓存。
- 仅复制 `sample_data` 下小于 5 MB 的 `.csv/.json/.md/.txt` 文件。
- 冻结后的依赖检测 UI 能通过本地 health check。

## 运行日志

日志写入：

```text
%LOCALAPPDATA%\OEye\logs
```

包括依赖检测、依赖安装、启动器和主应用 Streamlit 日志。
