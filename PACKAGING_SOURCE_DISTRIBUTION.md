# OEye 源码轻量交付说明

## 交付思路

本项目不建议直接把 Python、TensorFlow、PyTorch、CatBoost 等依赖一起打进安装包。推荐交付：

- 源码
- 必要小型样例数据 `sample_data`
- 依赖清单
- 一键安装脚本
- 一键启动脚本

这样交付包通常只有几 MB；客户首次安装时再联网下载依赖。

## 推荐客户环境

- Windows 10/11
- Python 3.11 或 3.12 优先
- 网络可访问 PyPI 或企业内部 pip 镜像
- 完整运行环境建议预留 6GB 以上磁盘空间

说明：
- Python 3.13 可运行当前大部分功能，但 `ray[rllib]` 依赖在 `requirements-runtime.txt` 中按 `python_version < "3.13"` 安装。
- 如果必须启用 Ray RLlib，请使用 Python 3.11/3.12，并按实际环境测试。

## 安装依赖

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_environment.ps1 -Profile runtime
```

也可以直接双击：

```text
install_runtime.bat
```

如果客户只需要标准机器学习、诊断、优化和报告功能，不需要 TensorFlow/PyTorch/强化学习深度后端，可先试轻量依赖：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_environment.ps1 -Profile standard
```

如果需要使用国内或企业 pip 镜像：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_environment.ps1 -Profile runtime -IndexUrl https://pypi.tuna.tsinghua.edu.cn/simple
```

## 启动应用

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1
```

也可以直接双击：

```text
run_app.bat
```

默认地址：

```text
http://localhost:8620
```

## 生成源码轻量包

仅在准备交付 ZIP 时执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\make_source_package.ps1
```

脚本会生成 `release/OEye_source_minimal_时间戳.zip`，包含源码、测试、轻型样例数据、许可证、说明文档和依赖清单，默认排除：

- `.git`
- `__pycache__`、`.pytest_cache`
- `.venv`
- `saved_models`
- 大型原始 `.dat/.out/.txt` 数据
- 训练日志和临时产物

保护版 ZIP 不放进源码 ZIP 或 Git 提交。由于保护版 ZIP 通常超过 GitHub 普通仓库的单文件限制，建议将它作为 GitHub Release 资产上传。

## 依赖配置区别

- `requirements-runtime.txt`：完整运行依赖，包含 TensorFlow、PyTorch、Stable-Baselines3、CatBoost 等。
- `requirements-standard.txt`：轻量运行依赖，不含深度学习和强化学习重依赖。
- `requirements.txt`：开发/测试依赖，包含 `pytest`。

## 体积预估

- 源码轻量 ZIP：约 3MB - 6MB
- 客户本机完整依赖安装后：约 4GB - 7GB
- 客户本机标准依赖安装后：约 1GB - 2GB

实际大小会受 Python 版本、CPU/GPU 包、pip 缓存和系统已有依赖影响。
