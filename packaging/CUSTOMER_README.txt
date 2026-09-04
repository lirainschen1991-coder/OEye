OEye 客户交付包使用说明

1. 双击 OEye.exe 打开“OEye 依赖检测与启动”界面。
2. 界面会扫描客户电脑已有 Python，并默认选择依赖满足度最高的“目标 Python”。
3. 如果客户电脑已有完整 Python 环境，直接使用自动选择的目标 Python，不要点击托管 venv。
4. “创建/检查 OEye 托管 venv”是可选功能，用于客户希望新建独立环境时使用；创建 venv 本身不会安装算法依赖，也不会自动切换当前目标环境。
5. 轻型依赖用于传统机器学习、系统诊断、优化算法、报告导出等功能。
6. 完整依赖用于 TensorFlow/PyTorch 深度学习、Stable-Baselines3 强化学习、CatBoost 等重依赖功能。
7. 一键补齐前会先检测已有依赖，只安装未安装或版本不满足的包；已满足依赖不会重复下载。
8. pip 镜像源只保存在 OEye 本应用配置中，不修改用户全局 pip 配置。
9. 日志目录：%LOCALAPPDATA%\OEye\logs

注意：
- 当前交付包内置了打开依赖 UI 所需的 Python/Streamlit/PyWebView 运行能力。
- 算法依赖需要安装到 OEye 托管 venv 或用户指定的普通 Python 环境。
- 如果客户电脑没有可用于创建 venv 的 Python，请先安装 Python 3.11 或 3.12；Python 3.13 下 Ray RLlib 会显示“环境不支持”，不会被安装。
