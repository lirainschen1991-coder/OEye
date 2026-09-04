# OEye

OEye 是一个基于 Streamlit 的数据学习、预测、故障诊断与优化平台。它把数据预处理、模型训练、评估解释和结果导出集中在一个可操作的界面中，适合快速验证数据分析方案，也便于继续扩展算法和业务流程。

当前功能包括：

- 时序预测、分类任务和批量训练
- 传统机器学习、深度学习与迁移学习扩展
- 异常检测、故障分类、根因解释和诊断报告
- 强化学习环境配置、策略训练和控制结果分析
- 优化算法、过程曲线和结果导出

## 开始使用

建议使用 Python 3.11 或 3.12。创建虚拟环境并安装标准依赖：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_environment.ps1 -Profile standard
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1
```

需要深度学习或强化学习后端时，安装完整依赖：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_environment.ps1 -Profile runtime
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1
```

安装脚本会在项目目录创建 `.venv`。使用企业或国内镜像时，可以通过 `-IndexUrl` 指定 pip 源。

## 依赖清单

- `requirements-standard.txt`：标准机器学习、诊断、优化和报告功能
- `requirements-runtime.txt`：在标准能力上增加深度学习和强化学习后端
- `requirements.txt`：完整运行依赖与开发测试依赖

这些文件只列出包名和版本约束，第三方库由安装脚本从 pip 源获取。

## 样例数据

`sample_data` 提供时序预测、分类、故障诊断和强化学习日志等小型数据集，也包含图像分类示例。下载仓库后即可在应用中选择这些文件进行试跑，或将其作为自定义数据格式的参考。

## 代码结构

- `app.py`：Streamlit 应用入口和页面流程
- `src/data`：数据读取与预处理
- `src/models`：训练、诊断、解释、迁移学习和强化学习模块
- `src/ui`、`src/utils`、`src/visualization`：界面、帮助内容和可视化组件
- `custom_models`：自定义模型与函数模板
- `tests`：快速回归测试

## 测试

```powershell
pytest tests -q
```

## 许可证

本项目采用 GNU General Public License v3.0 或更高版本（GPL-3.0-or-later），详见 [LICENSE](LICENSE)。
