# OEye AItrainFramework

OEye 是一个基于 Streamlit 的数据学习、预测、诊断与优化平台，覆盖：

- 时序预测与数据预处理
- 分类任务与批量训练
- 传统机器学习、深度学习扩展与迁移学习
- 故障异常检测、故障分类、根因解释和诊断报告
- 强化学习环境配置、策略训练与控制结果分析
- 多种优化算法、过程曲线和结果导出

## 快速开始

推荐使用 Python 3.11 或 3.12。标准机器学习、诊断、优化功能可使用轻量依赖；深度学习和强化学习功能需要完整依赖。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_environment.ps1 -Profile standard
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1
```

需要启用 TensorFlow、PyTorch、Stable-Baselines3、CatBoost 或 Ray RLlib 时：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_environment.ps1 -Profile runtime
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1
```

如果需要使用企业或国内镜像，可以给安装脚本传入 `-IndexUrl`。脚本会把依赖安装到项目 `.venv`，不会把依赖目录提交到仓库。

## 依赖清单

- `requirements.txt`：完整运行和开发测试依赖
- `requirements-standard.txt`：标准机器学习、诊断、优化和报告功能
- `requirements-runtime.txt`：完整运行能力，包括深度学习和强化学习后端
- `requirements-bootstrap.txt`：受保护交付的启动环境依赖

依赖清单只记录包名和版本约束，不携带任何第三方依赖文件。

## 样例数据

`sample_data` 中提供可直接用于时序预测、分类、故障诊断、强化学习日志和图像分类演示的小型样例。项目根目录中的大型工程数据、模型产物、缓存和运行输出不属于源码发行包。

## 源码轻量包

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\make_source_package.ps1
```

脚本会在 `release` 目录生成带时间戳的源码 ZIP，并在打包前后检查清单。该 ZIP 不包含 Python 环境、第三方依赖、Git 历史、构建目录、大型原始数据或模型文件。

## 保护版交付包

保护版构建说明见 `PACKAGING_PROTECTED_DELIVERY.md`。保护版包含用于打开依赖检测 UI 的最小运行环境；算法依赖仍由用户按需检查和安装。保护版 ZIP 应作为 GitHub Release 资产提供，不应提交到 Git 源码历史。

## 许可证

本项目采用 GNU General Public License v3.0 或更高版本（GPL-3.0-or-later），详见 [LICENSE](LICENSE)。
