"""Algorithm usage audit helpers."""

from __future__ import annotations

import pandas as pd


def build_algorithm_audit_report() -> pd.DataFrame:
    rows = [
        {"检查项": "回归模型选择", "状态": "通过", "说明": "时序预测使用回归模型、集成回归模型和深度学习回归模型。"},
        {"检查项": "分类模型选择", "状态": "通过", "说明": "分类任务使用sklearn/XGBoost/LightGBM分类器直接训练，不误用回归ModelTrainer。"},
        {"检查项": "时序验证", "状态": "通过", "说明": "时序AutoML支持TimeSeriesSplit；普通训练默认按时间顺序划分。"},
        {"检查项": "预处理数据泄漏", "状态": "通过", "说明": "缩放、填补和特征选择只在训练集fit，验证/测试/预测数据只transform。"},
        {"检查项": "分类新数据预测", "状态": "通过", "说明": "优先复用训练时保存的scaler和label_encoder，避免对新数据重新fit。"},
        {"检查项": "算法适用性提示", "状态": "通过", "说明": "帮助文档补充不同算法适用场景、注意事项和评估原则。"},
    ]
    return pd.DataFrame(rows)


def recommended_algorithm_matrix() -> pd.DataFrame:
    rows = [
        {"任务": "表格回归", "优先算法": "Random Forest / LightGBM / XGBoost / Ridge", "注意事项": "SVR/KNN需缩放；树模型可解释性较好。"},
        {"任务": "时序预测", "优先算法": "LSTM / GRU / TCN / Transformer / Gradient Boosting", "注意事项": "按时间划分，避免未来信息泄漏。"},
        {"任务": "故障分类", "优先算法": "Random Forest / XGBoost / LightGBM / SVM", "注意事项": "类别不平衡时关注F1、召回率和PR-AUC。"},
        {"任务": "异常检测", "优先算法": "Isolation Forest / LOF / One-Class SVM / PCA-SPE", "注意事项": "结合工况和传感器质量解释结果。"},
        {"任务": "智能控制", "优先算法": "PPO / SAC / TD3 / DQN / PID基线", "注意事项": "先和PID/MPC基线比较，再扩大训练。"},
    ]
    return pd.DataFrame(rows)


def algorithm_implementation_matrix() -> pd.DataFrame:
    """Describe which advertised algorithm families are backed by real code paths."""
    rows = [
        {
            "功能页": "时序预测",
            "算法/模块": "Linear/Ridge/Lasso/SVR/KNN/RandomForest/GradientBoosting/XGBoost/LightGBM",
            "实现状态": "已实施",
            "实现说明": "通过 sklearn、XGBoost、LightGBM 训练回归器，训练、评估、保存和预测流程可执行。",
            "准确性注意": "时序任务按时间顺序划分；SVR/KNN/线性模型建议启用缩放。",
        },
        {
            "功能页": "时序预测",
            "算法/模块": "LSTM/GRU/TCN/1D-CNN/Transformer",
            "实现状态": "部分主流程实施",
            "实现说明": "ANN/LSTM/GRU/CNN/Transformer 在主训练下拉可真实训练；TCN/Informer/Autoformer 为高级模型工厂入口，未放入主训练下拉。",
            "准确性注意": "未进入主训练下拉的高级模型不能宣传为一键可训练；小样本应与树模型和线性基线对比。",
        },
        {
            "功能页": "分类任务",
            "算法/模块": "LogisticRegression/RandomForest/GradientBoosting/SVM/KNN/XGBoost/LightGBM/GaussianNB",
            "实现状态": "已实施",
            "实现说明": "分类页和批量分类使用分类器直接训练，没有误用回归训练器。",
            "准确性注意": "类别不平衡时优先看 F1、召回率和混淆矩阵，概率输出需检查校准质量。",
        },
        {
            "功能页": "系统状态诊断",
            "算法/模块": "IsolationForest/OneClassSVM/LOF/PCA-T2-SPE/AutoEncoder入口",
            "实现状态": "已实施/轻量实现",
            "实现说明": "前四类为真实无监督诊断流程；AutoEncoder 当前使用轻量 PCA 重构误差接口承载。",
            "准确性注意": "诊断结论应结合工况、传感器量纲和历史案例复核，不应只看异常分。",
        },
        {
            "功能页": "强化学习",
            "算法/模块": "Q-Learning/SARSA/Expected SARSA/DQN/A2C/PPO/DDPG/TD3/SAC/PID/MPC/RLlib PPO",
            "实现状态": "真实接入/环境条件",
            "实现说明": "表格RL使用Q表真实更新；DQN/A2C/PPO/DDPG/TD3/SAC调用Stable-Baselines3真实训练；支持用户数据拟合动力学环境；PID/MPC为基线；RLlib PPO保留真实接口。",
            "准确性注意": "用户数据驱动环境依赖历史状态/动作覆盖范围；Ray RLlib在当前Python 3.13环境无可安装包，不能用其他策略冒充；深度RL需足够训练步数和多随机种子复核。",
        },
    ]
    return pd.DataFrame(rows)


def optimization_algorithm_matrix() -> pd.DataFrame:
    """Audit optimization algorithms exposed in the optimization page."""
    rows = [
        {"算法": "差分进化", "实现状态": "已实施", "实现方式": "scipy.optimize.differential_evolution", "适用场景": "连续变量、多峰全局优化"},
        {"算法": "遗传算法", "实现状态": "已实施", "实现方式": "应用内实数编码 GA：选择、算术交叉、高斯变异、精英保留", "适用场景": "复杂搜索空间的全局搜索"},
        {"算法": "粒子群优化", "实现状态": "已实施", "实现方式": "应用内 PSO：惯性权重、个体最优、群体最优更新", "适用场景": "连续黑箱优化、快速全局搜索"},
        {"算法": "模拟退火", "实现状态": "已实施", "实现方式": "scipy.optimize.dual_annealing", "适用场景": "多峰函数、跳出局部最优"},
        {"算法": "Nelder-Mead/L-BFGS-B/SLSQP/Powell/CG/BFGS/TNC/COBYLA/trust-constr", "实现状态": "已实施", "实现方式": "scipy.optimize.minimize 对应方法", "适用场景": "局部优化、约束优化或梯度近似优化"},
        {"算法": "SHGO/Basin-Hopping", "实现状态": "已实施", "实现方式": "scipy.optimize.shgo / basinhopping", "适用场景": "全局搜索与局部搜索组合"},
        {"算法": "贝叶斯优化", "实现状态": "轻量实现", "实现方式": "GaussianProcessRegressor + Expected Improvement 采集函数", "适用场景": "评估代价较高的低维黑箱函数"},
        {"算法": "CMA-ES", "实现状态": "轻量实现", "实现方式": "应用内 CMA-ES 风格均值、协方差和步长自适应搜索", "适用场景": "病态连续优化的实验入口"},
        {"算法": "灰狼/GWO/蚁群/ACO/混合蛙跳/SFLA/萤火虫/FA/禁忌搜索/TS/人工鱼群/AFSA/免疫遗传/IGA", "实现状态": "简化实现", "实现方式": "应用内元启发式简化版本", "适用场景": "教学、方案初筛和与 SciPy 方法对比"},
        {"算法": "自定义算法", "实现状态": "用户代码执行", "实现方式": "保留 exec 自定义优化函数入口并记录执行日志", "适用场景": "用户自定义搜索策略和工程专用优化器"},
    ]
    return pd.DataFrame(rows)
