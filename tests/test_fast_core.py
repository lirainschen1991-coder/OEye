import numpy as np
import pandas as pd

from src.data.data_preprocessor import DataPreprocessor
from src.models.automl import AutoMLRegressor
from src.models.algorithm_audit import (
    algorithm_implementation_matrix,
    build_algorithm_audit_report,
    optimization_algorithm_matrix,
    recommended_algorithm_matrix,
)
from src.models.deep_learning_extensions import list_deep_learning_algorithms, tensorflow_available
from src.models.enhanced_diagnostics import DiagnosisConfig, EnhancedFaultDiagnosis
from src.models.transfer_learning import DomainAdaptation
from src.models.reinforcement_learning import (
    ControlEnvironmentConfig,
    CustomFunctionControlEnvironment,
    CustomFunctionEnvironmentConfig,
    DataDrivenControlEnvironment,
    DataDrivenEnvironmentConfig,
    HeuristicRLAgent,
    SimpleControlEnvironment,
    TabularRLAgent,
    compare_algorithms,
    list_rl_algorithms,
    rl_backend_availability,
    summarize_evaluation,
    validate_rl_dataset,
)
from src.utils.help_manual import search_help
from src.ui.workbench import build_workbench_summary, next_step_suggestions


class State(dict):
    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value


def test_time_series_split_keeps_order():
    X = pd.DataFrame({"x": np.arange(10)})
    y = pd.Series(np.arange(10))
    preprocessor = DataPreprocessor()

    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X, y, test_size=0.2, val_size=0.25, time_series=True
    )

    assert X_train[-1][0] < X_val[0][0] < X_test[0][0]
    assert list(y_test) == [8, 9]


def test_prepare_train_val_test_fits_scaler_on_train_only():
    X = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5, 6, 1000, 1001, 1002]})
    y = pd.Series(np.arange(10))
    preprocessor = DataPreprocessor()

    prepared = preprocessor.prepare_train_val_test(
        X, y, test_size=0.3, val_size=0.0, time_series=True, scale_method="standard"
    )

    train_values = prepared["X_train"]["x"].to_numpy()
    test_values = prepared["X_test"]["x"].to_numpy()
    assert abs(train_values.mean()) < 1e-9
    assert test_values.min() > 300


def test_automl_regressor_builds_time_series_cv():
    automl = AutoMLRegressor(cv_folds=3, cv_strategy="time_series")
    splitter = automl._build_cv()

    assert splitter.__class__.__name__ == "TimeSeriesSplit"


def test_deep_learning_registry_imports_and_has_complete_rows():
    algorithms = list_deep_learning_algorithms()

    assert len(algorithms) >= 10
    assert all(item.key and item.name and item.task and item.data_type and item.description for item in algorithms)
    assert isinstance(tensorflow_available(), bool)


def test_enhanced_unsupervised_diagnosis_returns_root_causes():
    df = pd.DataFrame(
        {
            "temperature": [20, 21, 19, 22, 80],
            "pressure": [1.0, 1.1, 1.0, 1.2, 5.0],
            "speed": [100, 101, 99, 100, 140],
        }
    )
    diagnosis = EnhancedFaultDiagnosis()

    result = diagnosis.run_unsupervised(df, method="pca_t2_spe")

    assert {"results", "root_causes", "actions"} <= set(result)
    assert len(result["results"]) == len(df)
    assert not result["root_causes"].empty
    assert "health_summary" in result
    assert "feature_drift" in result
    assert 0 <= result["health_summary"]["health_score"] <= 100


def test_diagnosis_advanced_parameters_and_domain_adaptation_are_stable():
    rng = np.random.default_rng(123)
    source = rng.normal(size=(12, 3))
    target = rng.normal(loc=0.2, size=(8, 3))
    diagnosis = EnhancedFaultDiagnosis(DiagnosisConfig(contamination=0.2, lof_neighbors=3, pca_components=2))

    lof_result = diagnosis.run_unsupervised(pd.DataFrame(source, columns=["a", "b", "c"]), method="lof")

    assert len(lof_result["results"]) == len(source)
    for method in ["correlation_alignment", "subspace_alignment", "instance_weighting"]:
        adapter = DomainAdaptation(method=method)
        adapter.fit(source, target)
        transformed = adapter.transform(source, domain="source")
        transformed_values = transformed[0] if isinstance(transformed, tuple) else transformed
        assert transformed_values.shape[0] == source.shape[0]


def test_reinforcement_learning_facade_trains_and_evaluates():
    env = SimpleControlEnvironment(ControlEnvironmentConfig(max_steps=20, initial_state=3.0))
    agent = TabularRLAgent("q_learning")

    history = agent.train(env, episodes=8)
    evaluation = agent.evaluate(env)

    assert len(history) == 8
    assert {"state", "action", "reward", "error"} <= set(evaluation.columns)
    assert set(history["backend"]) == {"tabular"}


def test_reinforcement_learning_catalog_and_comparison_outputs():
    algorithms = list_rl_algorithms()
    config = ControlEnvironmentConfig(max_steps=12, initial_state=2.5, disturbance=0.0)

    history, evaluation = compare_algorithms(config, ["q_learning", "sarsa"], episodes=6, backend="tabular")
    summary = summarize_evaluation(evaluation)

    assert len(algorithms) >= 10
    assert rl_backend_availability()["stable-baselines3"] is True
    assert set(history["algorithm"]) == {"q_learning", "sarsa"}
    assert set(evaluation["algorithm"]) == {"q_learning", "sarsa"}
    assert {"algorithm", "mean_abs_error", "total_reward"} <= set(summary.columns)


def test_data_driven_rl_environment_validates_and_steps():
    df = pd.DataFrame({
        "state": np.linspace(0.0, 3.0, 40),
        "action": np.sin(np.linspace(0.0, 2.0, 40)),
        "reward": -np.abs(np.linspace(0.0, 3.0, 40)),
    })
    config = DataDrivenEnvironmentConfig(
        data=df,
        state_columns=["state"],
        action_column="action",
        reward_column="reward",
        max_steps=5,
        seed=7,
        n_estimators=20,
    )

    validation = validate_rl_dataset(config)
    env = DataDrivenControlEnvironment(config)
    obs = env.reset()
    next_obs, reward, done, info = env.step(0.1)

    assert validation["ok"] is True
    assert obs.shape == (1,)
    assert next_obs.shape == (1,)
    assert isinstance(reward, float)
    assert done is False
    assert "state_norm" in info


def test_builtin_inverted_pendulum_environment_steps():
    env = SimpleControlEnvironment(
        ControlEnvironmentConfig(scenario="inverted_pendulum", initial_state=0.2, action_limit=2.0, max_steps=10)
    )

    obs = env.reset()
    next_obs, reward, done, info = env.step(-0.1)

    assert obs.shape == (2,)
    assert next_obs.shape == (2,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "error" in info


def test_custom_function_rl_environment_steps():
    code = """
import numpy as np

def reset(seed):
    return np.array([0.1, 0.0])

def step(obs, action, step_index):
    x, v = obs
    v = v + float(action) * 0.1
    x = x + v * 0.1
    reward = -(x*x + 0.01*action*action)
    return np.array([x, v]), float(reward), False, {"error": float(x)}
"""
    env = CustomFunctionControlEnvironment(
        CustomFunctionEnvironmentConfig(code=code, observation_dim=2, action_limit=1.0, max_steps=5)
    )

    obs = env.reset()
    next_obs, reward, done, info = env.step(0.2)

    assert obs.shape == (2,)
    assert next_obs.shape == (2,)
    assert isinstance(reward, float)
    assert done is False
    assert "error" in info


def test_workbench_summary_and_suggestions():
    state = State(df=pd.DataFrame({"a": [1, 2], "b": [3, 4]}), model_trained=False, error_log=[], run_history=[])

    summary = build_workbench_summary(state)
    suggestions = next_step_suggestions(state)

    assert summary["data_status"] == "已上传"
    assert any("选择目标列" in item for item in suggestions)


def test_help_search_returns_clickable_summary_shape():
    results = search_help("强化学习")

    assert results
    key, title, icon, summary, score = results[0]
    assert key and title and icon and summary
    assert isinstance(score, int)


def test_algorithm_audit_reports_are_populated():
    audit = build_algorithm_audit_report()
    matrix = recommended_algorithm_matrix()
    implementation = algorithm_implementation_matrix()
    optimization = optimization_algorithm_matrix()

    assert not audit.empty
    assert not matrix.empty
    assert not implementation.empty
    assert not optimization.empty
    assert set(audit["状态"]) == {"通过"}
    assert "真实接入/环境条件" in set(implementation["实现状态"])
    assert {"已实施", "轻量实现", "简化实现"} <= set(optimization["实现状态"])


def test_help_search_covers_data_and_optimization_principles():
    data_results = search_help("数据处理")
    opt_results = search_help("优化算法")

    assert any(key == "data_processing_principles" for key, *_ in data_results)
    assert any(key == "optimization_algorithms" for key, *_ in opt_results)
