"""Reinforcement-learning utilities for intelligent control workflows.

The module separates true trainable backends from engineering baselines. Deep RL
algorithms are only marked trainable when their backend dependencies are
available; otherwise callers receive an explicit ImportError instead of a fake
calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


@dataclass
class ControlEnvironmentConfig:
    target_value: float = 0.0
    initial_state: float = 5.0
    disturbance: float = 0.05
    action_limit: float = 1.0
    max_steps: int = 60
    seed: int = 42
    scenario: str = "position_control"
    action_penalty: float = 0.05
    stability_penalty: float = 0.01
    done_tolerance: float = 0.02


@dataclass
class DataDrivenEnvironmentConfig:
    data: pd.DataFrame
    state_columns: list[str]
    action_column: str
    reward_column: str | None = None
    next_state_columns: list[str] | None = None
    terminal_column: str | None = None
    max_steps: int = 60
    seed: int = 42
    action_limit: float | None = None
    n_estimators: int = 100


@dataclass
class CustomFunctionEnvironmentConfig:
    code: str
    observation_dim: int = 2
    action_limit: float = 1.0
    max_steps: int = 60
    seed: int = 42


@dataclass(frozen=True)
class RLAlgorithmSpec:
    key: str
    name: str
    family: str
    action_space: str
    backend: str
    implementation_status: str
    description: str


RL_ALGORITHMS = [
    RLAlgorithmSpec("q_learning", "Q-Learning", "Value-based", "离散", "tabular", "已实施", "表格型值迭代，使用离散状态/动作Q表训练。"),
    RLAlgorithmSpec("sarsa", "SARSA", "Value-based", "离散", "tabular", "已实施", "在线策略值学习，使用epsilon-greedy行为策略训练。"),
    RLAlgorithmSpec("expected_sarsa", "Expected SARSA", "Value-based", "离散", "tabular", "已实施", "使用下一状态动作价值期望更新Q表。"),
    RLAlgorithmSpec("dqn", "DQN", "Deep value-based", "离散", "stable-baselines3", "依赖可用时已实施", "调用Stable-Baselines3 DQN真实训练，适合离散动作控制。"),
    RLAlgorithmSpec("a2c", "A2C", "Actor-Critic", "连续", "stable-baselines3", "依赖可用时已实施", "调用Stable-Baselines3 A2C真实训练。"),
    RLAlgorithmSpec("ppo", "PPO", "Actor-Critic", "连续", "stable-baselines3", "依赖可用时已实施", "调用Stable-Baselines3 PPO真实训练，鲁棒性较好。"),
    RLAlgorithmSpec("ddpg", "DDPG", "Actor-Critic", "连续", "stable-baselines3", "依赖可用时已实施", "调用Stable-Baselines3 DDPG真实训练，适合连续控制。"),
    RLAlgorithmSpec("td3", "TD3", "Actor-Critic", "连续", "stable-baselines3", "依赖可用时已实施", "调用Stable-Baselines3 TD3真实训练，降低DDPG过估计。"),
    RLAlgorithmSpec("sac", "SAC", "Actor-Critic", "连续", "stable-baselines3", "依赖可用时已实施", "调用Stable-Baselines3 SAC真实训练，探索能力强。"),
    RLAlgorithmSpec("ppo_rllib", "PPO (Ray RLlib)", "Distributed RL", "连续", "ray-rllib", "依赖可用时已实施", "调用Ray RLlib PPO训练入口，适合后续分布式扩展。"),
    RLAlgorithmSpec("pid_baseline", "PID Baseline", "Control baseline", "连续", "baseline", "已实施", "经典比例控制基线，用于和RL策略对比。"),
    RLAlgorithmSpec("mpc_baseline", "MPC Baseline", "Control baseline", "连续", "baseline", "已实施", "滚动预测控制风格基线，用于约束控制对比。"),
]


def list_rl_algorithms() -> list[RLAlgorithmSpec]:
    return RL_ALGORITHMS


def rl_backend_availability() -> dict[str, bool]:
    return {
        "tabular": True,
        "baseline": True,
        "stable-baselines3": bool(importlib.util.find_spec("stable_baselines3") and importlib.util.find_spec("gymnasium") and importlib.util.find_spec("torch")),
        "ray-rllib": bool(importlib.util.find_spec("ray") and importlib.util.find_spec("gymnasium") and importlib.util.find_spec("torch")),
    }


def list_trainable_rl_algorithms(backend: str | None = None) -> list[RLAlgorithmSpec]:
    availability = rl_backend_availability()
    return [
        item for item in RL_ALGORITHMS
        if (backend is None or item.backend == backend) and availability.get(item.backend, False)
    ]


def validate_rl_dataset(config: DataDrivenEnvironmentConfig) -> dict[str, object]:
    df = config.data
    required = list(config.state_columns) + [config.action_column]
    if config.reward_column:
        required.append(config.reward_column)
    if config.next_state_columns:
        required.extend(config.next_state_columns)
    if config.terminal_column:
        required.append(config.terminal_column)
    missing = [col for col in required if col not in df.columns]
    numeric_missing = [col for col in required if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
    usable_rows = int(df[required].dropna().shape[0]) if not missing else 0
    issues = []
    if missing:
        issues.append(f"缺少列: {', '.join(missing)}")
    if numeric_missing:
        issues.append(f"非数值列不能直接用于环境建模: {', '.join(numeric_missing)}")
    if config.next_state_columns and len(config.next_state_columns) != len(config.state_columns):
        issues.append("下一状态列数量必须和状态列数量一致。")
    if usable_rows < 20:
        issues.append("有效样本少于20行，无法可靠拟合环境动力学。")
    return {
        "ok": not issues,
        "issues": issues,
        "rows": int(len(df)),
        "usable_rows": usable_rows,
        "state_dim": len(config.state_columns),
        "has_reward": bool(config.reward_column),
        "has_explicit_next_state": bool(config.next_state_columns),
    }


class DataDrivenControlEnvironment:
    """A learned environment fitted from user state/action/reward transition data."""

    def __init__(self, config: DataDrivenEnvironmentConfig):
        validation = validate_rl_dataset(config)
        if not validation["ok"]:
            raise ValueError("强化学习数据集不可用: " + "; ".join(validation["issues"]))
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.steps = 0
        self._fit_models()
        self.state = self.initial_states[0].copy()

    def reset(self) -> np.ndarray:
        idx = int(self.rng.integers(0, len(self.initial_states)))
        self.state = self.initial_states[idx].astype(float).copy()
        self.steps = 0
        return self.state.astype(float)

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict]:
        action = float(np.clip(action, -self.action_limit, self.action_limit))
        features = np.concatenate([self.state, [action]]).reshape(1, -1)
        next_state = np.atleast_1d(np.asarray(self.dynamics_model.predict(features)[0], dtype=float))
        if self.reward_model is not None:
            reward = float(self.reward_model.predict(features)[0])
        else:
            state_change = float(np.linalg.norm(next_state - self.state))
            reward = -(state_change + 0.05 * abs(action))
        self.steps += 1
        self.state = next_state
        done = self.steps >= self.config.max_steps
        state_norm = float(np.linalg.norm(self.state))
        return self.state.copy(), reward, done, {"action": action, "state_norm": state_norm}

    def _fit_models(self) -> None:
        df = self.config.data.copy()
        state_cols = self.config.state_columns
        action_col = self.config.action_column
        if self.config.next_state_columns:
            next_state = df[self.config.next_state_columns].copy()
            current = df[state_cols].copy()
            model_frame = pd.concat([current, df[[action_col]], next_state], axis=1).dropna()
            X = model_frame[state_cols + [action_col]].to_numpy(dtype=float)
            y_next = model_frame[self.config.next_state_columns].to_numpy(dtype=float)
        else:
            shifted = df[state_cols].shift(-1).add_suffix("__next")
            model_frame = pd.concat([df[state_cols + [action_col]], shifted], axis=1).dropna()
            X = model_frame[state_cols + [action_col]].to_numpy(dtype=float)
            y_next = model_frame[[f"{col}__next" for col in state_cols]].to_numpy(dtype=float)
        self.dynamics_model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            random_state=self.config.seed,
            min_samples_leaf=2,
        )
        y_fit = y_next.ravel() if y_next.ndim == 2 and y_next.shape[1] == 1 else y_next
        self.dynamics_model.fit(X, y_fit)
        if self.config.reward_column:
            reward_frame = df[state_cols + [action_col, self.config.reward_column]].dropna()
            self.reward_model = RandomForestRegressor(
                n_estimators=max(20, self.config.n_estimators // 2),
                random_state=self.config.seed + 17,
                min_samples_leaf=2,
            )
            self.reward_model.fit(
                reward_frame[state_cols + [action_col]].to_numpy(dtype=float),
                reward_frame[self.config.reward_column].to_numpy(dtype=float),
            )
        else:
            self.reward_model = None
        self.initial_states = df[state_cols].dropna().to_numpy(dtype=float)
        observed_action = df[action_col].dropna().astype(float)
        inferred_limit = float(max(abs(observed_action.min()), abs(observed_action.max()), 1e-6))
        self.action_limit = float(self.config.action_limit or inferred_limit)

    @property
    def state_dim(self) -> int:
        return len(self.config.state_columns)


class CustomFunctionControlEnvironment:
    """Gym-compatible environment backed by user-provided reset/step functions."""

    def __init__(self, config: CustomFunctionEnvironmentConfig):
        self.config = config
        self.steps = 0
        self.namespace = {
            "np": np,
            "pd": pd,
        }
        exec(config.code, self.namespace)
        if "reset" not in self.namespace or "step" not in self.namespace:
            raise ValueError("自定义强化学习环境必须定义 reset(seed) 和 step(obs, action, step_index)。")
        self.reset_fn = self.namespace["reset"]
        self.step_fn = self.namespace["step"]
        self.obs = np.zeros(config.observation_dim, dtype=float)

    def reset(self) -> np.ndarray:
        self.steps = 0
        obs = self.reset_fn(self.config.seed)
        self.obs = self._coerce_obs(obs)
        return self.obs.copy()

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict]:
        action = float(np.clip(action, -self.config.action_limit, self.config.action_limit))
        result = self.step_fn(self.obs.copy(), action, self.steps)
        if not isinstance(result, tuple) or len(result) not in {3, 4}:
            raise ValueError("step函数必须返回 (next_obs, reward, done) 或 (next_obs, reward, done, info)。")
        if len(result) == 3:
            next_obs, reward, done = result
            info = {}
        else:
            next_obs, reward, done, info = result
        self.steps += 1
        self.obs = self._coerce_obs(next_obs)
        done = bool(done) or self.steps >= self.config.max_steps
        info = dict(info or {})
        info.setdefault("action", action)
        info.setdefault("state_norm", float(np.linalg.norm(self.obs)))
        return self.obs.copy(), float(reward), done, info

    def _coerce_obs(self, obs) -> np.ndarray:
        arr = np.asarray(obs, dtype=float).ravel()
        if arr.shape[0] != self.config.observation_dim:
            raise ValueError(f"观测维度应为 {self.config.observation_dim}，实际为 {arr.shape[0]}。")
        return arr


class SimpleControlEnvironment:
    """A compact one-dimensional control environment for first-version RL workflows."""

    def __init__(self, config: ControlEnvironmentConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.state = self._initial_state()
        self.steps = 0

    def reset(self) -> np.ndarray:
        self.state = self._initial_state()
        self.steps = 0
        return np.asarray(self.state, dtype=float).ravel()

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict]:
        action = float(np.clip(action, -self.config.action_limit, self.config.action_limit))
        noise = self.rng.normal(0.0, self.config.disturbance)
        if self.config.scenario == "inverted_pendulum":
            theta, theta_dot = np.asarray(self.state, dtype=float).ravel()
            gravity = 9.81
            length = 1.0
            mass = 1.0
            damping = 0.08
            dt = 0.05
            torque = action
            theta_ddot = (gravity / length) * np.sin(theta) + torque / (mass * length * length) - damping * theta_dot + noise
            theta_dot = float(np.clip(theta_dot + theta_ddot * dt, -8.0, 8.0))
            theta = float(((theta + theta_dot * dt + np.pi) % (2 * np.pi)) - np.pi)
            self.state = np.array([theta, theta_dot], dtype=float)
            self.steps += 1
            reward = -(theta * theta + 0.1 * theta_dot * theta_dot + self.config.action_penalty * action * action)
            done = self.steps >= self.config.max_steps or abs(theta) > np.pi / 2
            return self.state.copy(), float(reward), done, {"error": float(theta), "action": action, "state_norm": float(np.linalg.norm(self.state))}
        if self.config.scenario == "thermal_inertia":
            self.state = 0.94 * self.state + 0.75 * action + noise
        elif self.config.scenario == "vibration_damping":
            self.state = 0.82 * self.state + 1.15 * action + 0.08 * np.sin(self.steps / 3) + noise
        elif self.config.scenario == "energy_dispatch":
            self.state = self.state + 0.55 * action - 0.05 * (self.state - self.config.target_value) + noise
        else:
            self.state = self.state + action + noise
        self.steps += 1
        error = self.state - self.config.target_value
        reward = -(
            abs(error)
            + self.config.action_penalty * abs(action)
            + self.config.stability_penalty * error * error
        )
        done = self.steps >= self.config.max_steps or abs(error) < self.config.done_tolerance
        return np.array([self.state], dtype=float), float(reward), done, {"error": float(error), "action": action}

    def observation_dim(self) -> int:
        return 2 if self.config.scenario == "inverted_pendulum" else 1

    def _initial_state(self):
        if self.config.scenario == "inverted_pendulum":
            return np.array([float(self.config.initial_state), 0.0], dtype=float)
        return float(self.config.initial_state)


class TabularRLAgent:
    """True tabular Q-learning/SARSA style controller on discretized states/actions."""

    def __init__(self, algorithm: str = "q_learning", seed: int = 42, n_state_bins: int = 31, n_actions: int = 7):
        self.algorithm = algorithm.lower()
        self.rng = np.random.default_rng(seed)
        self.n_state_bins = n_state_bins
        self.actions = np.linspace(-1.0, 1.0, n_actions)
        self.q_table = np.zeros((n_state_bins, n_actions), dtype=float)
        self.state_min = -10.0
        self.state_max = 10.0

    def train(self, env: SimpleControlEnvironment, episodes: int = 30, learning_rate: float = 0.05,
              exploration: float = 0.2, gamma: float = 0.95) -> pd.DataFrame:
        self.actions = np.linspace(-env.config.action_limit, env.config.action_limit, len(self.actions))
        self.state_min = min(env.config.target_value, env.config.initial_state) - 2 * env.config.action_limit - 5
        self.state_max = max(env.config.target_value, env.config.initial_state) + 2 * env.config.action_limit + 5
        records = []
        best_reward = -np.inf

        for episode in range(1, episodes + 1):
            state = env.reset()
            state_idx = self._state_to_bin(state[0])
            action_idx = self._choose_action(state_idx, exploration, episode)
            total_reward = 0.0
            total_abs_error = 0.0
            total_abs_action = 0.0

            for _ in range(env.config.max_steps):
                next_state, reward, done, info = env.step(self.actions[action_idx])
                next_state_idx = self._state_to_bin(next_state[0])
                next_action_idx = self._choose_action(next_state_idx, exploration, episode)
                if self.algorithm == "sarsa":
                    target = reward + gamma * self.q_table[next_state_idx, next_action_idx] * (not done)
                elif self.algorithm == "expected_sarsa":
                    probs = np.full(len(self.actions), exploration / len(self.actions))
                    probs[int(np.argmax(self.q_table[next_state_idx]))] += 1 - exploration
                    target = reward + gamma * float(probs @ self.q_table[next_state_idx]) * (not done)
                else:
                    target = reward + gamma * float(np.max(self.q_table[next_state_idx])) * (not done)
                self.q_table[state_idx, action_idx] += learning_rate * (target - self.q_table[state_idx, action_idx])
                total_reward += reward
                total_abs_error += abs(info["error"])
                total_abs_action += abs(info["action"])
                state_idx, action_idx = next_state_idx, next_action_idx
                if done:
                    break

            best_reward = max(best_reward, total_reward)
            records.append({
                "episode": episode,
                "algorithm": self.algorithm,
                "backend": "tabular",
                "total_reward": float(total_reward),
                "best_reward": float(best_reward),
                "mean_abs_error": float(total_abs_error / max(env.steps, 1)),
                "mean_abs_action": float(total_abs_action / max(env.steps, 1)),
                "steps": int(env.steps),
            })

        return pd.DataFrame(records)

    def evaluate(self, env: SimpleControlEnvironment) -> pd.DataFrame:
        state = env.reset()
        rows = []
        for step in range(env.config.max_steps):
            state_idx = self._state_to_bin(state[0])
            action = float(self.actions[int(np.argmax(self.q_table[state_idx]))])
            next_state, reward, done, info = env.step(action)
            rows.append({
                "step": step,
                "state": float(state[0]),
                "action": float(info["action"]),
                "reward": float(reward),
                "error": float(info["error"]),
                "abs_error": float(abs(info["error"])),
                "cumulative_reward": float(sum(row["reward"] for row in rows) + reward),
            })
            state = next_state
            if done:
                break
        return pd.DataFrame(rows)

    def _state_to_bin(self, state: float) -> int:
        clipped = np.clip(state, self.state_min, self.state_max)
        ratio = (clipped - self.state_min) / max(self.state_max - self.state_min, 1e-12)
        return int(np.clip(round(ratio * (self.n_state_bins - 1)), 0, self.n_state_bins - 1))

    def _choose_action(self, state_idx: int, exploration: float, episode: int) -> int:
        epsilon = exploration / np.sqrt(max(episode, 1))
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, len(self.actions)))
        return int(np.argmax(self.q_table[state_idx]))


def _make_gymnasium_env(config: ControlEnvironmentConfig, discrete: bool):
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except Exception as exc:
        raise ImportError("缺少 gymnasium，无法运行真实神经网络强化学习后端。请安装 requirements.txt 中的 gymnasium。") from exc

    class ControlGymEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, env_config: ControlEnvironmentConfig):
            super().__init__()
            self.core = SimpleControlEnvironment(env_config)
            self.discrete = discrete
            obs_dim = self.core.observation_dim()
            self.observation_space = spaces.Box(
                low=np.full(obs_dim, -np.inf, dtype=np.float32),
                high=np.full(obs_dim, np.inf, dtype=np.float32),
                dtype=np.float32,
            )
            if discrete:
                self.actions = np.linspace(-env_config.action_limit, env_config.action_limit, 5, dtype=np.float32)
                self.action_space = spaces.Discrete(len(self.actions))
            else:
                self.action_space = spaces.Box(low=np.array([-env_config.action_limit], dtype=np.float32), high=np.array([env_config.action_limit], dtype=np.float32), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self.core.rng = np.random.default_rng(seed)
            obs = self.core.reset().astype(np.float32)
            return obs, {}

        def step(self, action):
            if self.discrete:
                action_value = float(self.actions[int(action)])
            else:
                action_value = float(np.asarray(action).ravel()[0])
            obs, reward, done, info = self.core.step(action_value)
            return obs.astype(np.float32), reward, done, False, info

    return ControlGymEnv(config)


def _make_gymnasium_data_env(config: DataDrivenEnvironmentConfig):
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except Exception as exc:
        raise ImportError("缺少 gymnasium，无法运行用户数据驱动强化学习环境。") from exc

    class DataDrivenGymEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, env_config: DataDrivenEnvironmentConfig):
            super().__init__()
            self.core = DataDrivenControlEnvironment(env_config)
            self.observation_space = spaces.Box(
                low=np.full(self.core.state_dim, -np.inf, dtype=np.float32),
                high=np.full(self.core.state_dim, np.inf, dtype=np.float32),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=np.array([-self.core.action_limit], dtype=np.float32),
                high=np.array([self.core.action_limit], dtype=np.float32),
                dtype=np.float32,
            )

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self.core.rng = np.random.default_rng(seed)
            obs = self.core.reset().astype(np.float32)
            return obs, {}

        def step(self, action):
            action_value = float(np.asarray(action).ravel()[0])
            obs, reward, done, info = self.core.step(action_value)
            return obs.astype(np.float32), reward, done, False, info

    return DataDrivenGymEnv(config)


def _make_gymnasium_custom_env(config: CustomFunctionEnvironmentConfig):
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except Exception as exc:
        raise ImportError("缺少 gymnasium，无法运行自定义强化学习环境。") from exc

    class CustomFunctionGymEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, env_config: CustomFunctionEnvironmentConfig):
            super().__init__()
            self.core = CustomFunctionControlEnvironment(env_config)
            self.observation_space = spaces.Box(
                low=np.full(env_config.observation_dim, -np.inf, dtype=np.float32),
                high=np.full(env_config.observation_dim, np.inf, dtype=np.float32),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=np.array([-env_config.action_limit], dtype=np.float32),
                high=np.array([env_config.action_limit], dtype=np.float32),
                dtype=np.float32,
            )

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self.core.config.seed = seed
            obs = self.core.reset().astype(np.float32)
            return obs, {}

        def step(self, action):
            action_value = float(np.asarray(action).ravel()[0])
            obs, reward, done, info = self.core.step(action_value)
            return obs.astype(np.float32), reward, done, False, info

    return CustomFunctionGymEnv(config)


class HeuristicRLAgent:
    """First-version RL agent facade for DQN/PPO/DDPG/SAC entries.

    The implementation uses a deterministic proportional-control policy search so
    the page works without extra RL dependencies. The algorithm name is preserved
    in results and can later be backed by stable-baselines style trainers.
    """

    def __init__(self, algorithm: str = "ppo", seed: int = 42):
        self.algorithm = algorithm.lower()
        self.rng = np.random.default_rng(seed)
        self.best_gain = 0.2

    def train(self, env: SimpleControlEnvironment, episodes: int = 30, learning_rate: float = 0.05,
              exploration: float = 0.2) -> pd.DataFrame:
        records = []
        candidate_gains = self._candidate_gains(episodes, learning_rate, exploration)
        best_return = -np.inf

        for episode, gain in enumerate(candidate_gains[:episodes], start=1):
            state = env.reset()
            total_reward = 0.0
            total_abs_error = 0.0
            total_abs_action = 0.0
            for _ in range(env.config.max_steps):
                error = state[0] - env.config.target_value
                action = self._policy_action(error, gain, env.config.action_limit, exploration, episode)
                state, reward, done, _ = env.step(action)
                total_reward += reward
                total_abs_error += abs(error)
                total_abs_action += abs(action)
                if done:
                    break

            if total_reward > best_return:
                best_return = total_reward
                self.best_gain = float(gain)

            records.append(
                {
                    "episode": episode,
                    "algorithm": self.algorithm,
                    "gain": float(gain),
                    "total_reward": float(total_reward),
                    "best_reward": float(best_return),
                    "mean_abs_error": float(total_abs_error / max(env.steps, 1)),
                    "mean_abs_action": float(total_abs_action / max(env.steps, 1)),
                    "steps": int(env.steps),
                }
            )

        return pd.DataFrame(records)

    def evaluate(self, env: SimpleControlEnvironment) -> pd.DataFrame:
        state = env.reset()
        rows = []
        for step in range(env.config.max_steps):
            error = state[0] - env.config.target_value
            action = -self.best_gain * error
            next_state, reward, done, info = env.step(action)
            rows.append(
                {
                    "step": step,
                    "state": float(state[0]),
                    "action": float(info["action"]),
                    "reward": float(reward),
                    "error": float(info["error"]),
                    "abs_error": float(abs(info["error"])),
                    "cumulative_reward": float(sum(row["reward"] for row in rows) + reward),
                }
            )
            state = next_state
            if done:
                break
        return pd.DataFrame(rows)

    def _candidate_gains(self, episodes: int, learning_rate: float, exploration: float) -> np.ndarray:
        base = np.linspace(0.05, 1.25, max(episodes, 2))
        algo = self.algorithm
        if algo in {"q_learning", "dqn", "double_dqn", "dueling_dqn"}:
            return np.round(base / max(learning_rate * 8, 0.2), 2)
        if algo in {"sarsa", "expected_sarsa"}:
            return np.linspace(0.05, 0.95, max(episodes, 2))
        if algo in {"ddpg", "td3"}:
            return np.linspace(0.15, 1.5, max(episodes, 2))
        if algo == "sac":
            return np.linspace(0.1, 1.35 + exploration, max(episodes, 2))
        if algo == "pid_baseline":
            return np.linspace(0.25, 0.75, max(episodes, 2))
        if algo == "mpc_baseline":
            return np.linspace(0.35, 1.05, max(episodes, 2))
        return base

    def _policy_action(self, error: float, gain: float, action_limit: float, exploration: float, episode: int) -> float:
        noise_scale = exploration / np.sqrt(max(episode, 1))
        if self.algorithm in {"q_learning", "sarsa", "expected_sarsa", "dqn", "double_dqn", "dueling_dqn"}:
            discrete_actions = np.array([-action_limit, -0.5 * action_limit, 0.0, 0.5 * action_limit, action_limit])
            target_action = -gain * error + self.rng.normal(0.0, noise_scale)
            return float(discrete_actions[np.argmin(np.abs(discrete_actions - target_action))])
        if self.algorithm == "mpc_baseline":
            return float(np.clip(-gain * error - 0.15 * np.sign(error) * min(abs(error), action_limit), -action_limit, action_limit))
        if self.algorithm == "pid_baseline":
            return float(np.clip(-gain * error, -action_limit, action_limit))
        return float(np.clip(-gain * error + self.rng.normal(0.0, noise_scale), -action_limit, action_limit))


def compare_algorithms(config: ControlEnvironmentConfig, algorithms: list[str], episodes: int,
                       learning_rate: float = 0.05, exploration: float = 0.2,
                       backend: str = "auto") -> tuple[pd.DataFrame, pd.DataFrame]:
    histories = []
    evaluations = []
    specs = {item.key: item for item in RL_ALGORITHMS}
    for index, algorithm in enumerate(algorithms):
        env = SimpleControlEnvironment(ControlEnvironmentConfig(**{**config.__dict__, "seed": config.seed + index}))
        spec = specs.get(algorithm)
        selected_backend = backend if backend != "auto" else (spec.backend if spec else "baseline")
        if selected_backend == "tabular":
            agent = TabularRLAgent(algorithm, seed=config.seed + index)
            history = agent.train(env, episodes=episodes, learning_rate=learning_rate, exploration=exploration)
            evaluation = agent.evaluate(SimpleControlEnvironment(ControlEnvironmentConfig(**{**config.__dict__, "seed": config.seed + 100 + index})))
        elif selected_backend == "stable-baselines3":
            history, evaluation = train_stable_baselines3_policy(
                config,
                algorithm,
                episodes=episodes,
                learning_rate=learning_rate,
                seed=config.seed + index,
            )
        elif selected_backend == "ray-rllib":
            history, evaluation = train_rllib_policy(
                config,
                algorithm,
                episodes=episodes,
                learning_rate=learning_rate,
                seed=config.seed + index,
            )
        elif selected_backend == "baseline":
            agent = HeuristicRLAgent(algorithm, seed=config.seed + index)
            history = agent.train(env, episodes=episodes, learning_rate=learning_rate, exploration=exploration)
            evaluation = agent.evaluate(SimpleControlEnvironment(ControlEnvironmentConfig(**{**config.__dict__, "seed": config.seed + 100 + index})))
        else:
            raise ValueError(f"不支持的强化学习后端: {selected_backend}")
        history["backend"] = selected_backend
        evaluation["algorithm"] = algorithm
        evaluation["backend"] = selected_backend
        histories.append(history)
        evaluations.append(evaluation)
    return pd.concat(histories, ignore_index=True), pd.concat(evaluations, ignore_index=True)


def train_stable_baselines3_policy(config: ControlEnvironmentConfig, algorithm: str, episodes: int,
                                   learning_rate: float = 0.001, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not rl_backend_availability()["stable-baselines3"]:
        raise ImportError("Stable-Baselines3真实训练后端不可用：请安装 torch、gymnasium、stable-baselines3。")

    from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback

    algorithm_key = algorithm.lower()
    model_map = {
        "dqn": DQN,
        "a2c": A2C,
        "ppo": PPO,
        "ddpg": DDPG,
        "td3": TD3,
        "sac": SAC,
    }
    if algorithm_key not in model_map:
        raise ValueError(f"Stable-Baselines3后端不支持算法: {algorithm}")
    discrete = algorithm_key == "dqn"
    env = _make_gymnasium_env(config, discrete=discrete)
    eval_env = _make_gymnasium_env(ControlEnvironmentConfig(**{**config.__dict__, "seed": seed + 100}), discrete=discrete)
    total_timesteps = max(episodes * config.max_steps, config.max_steps)

    class RewardCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_rewards = []
            self.current_reward = 0.0

        def _on_step(self) -> bool:
            self.current_reward += float(np.mean(self.locals.get("rewards", [0.0])))
            dones = self.locals.get("dones", [])
            if len(dones) and bool(dones[0]):
                self.episode_rewards.append(self.current_reward)
                self.current_reward = 0.0
            return True

    callback = RewardCallback()
    model = model_map[algorithm_key]("MlpPolicy", env, learning_rate=learning_rate, seed=seed, verbose=0)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    rewards = callback.episode_rewards or [callback.current_reward]
    best = -np.inf
    history_rows = []
    for idx, reward in enumerate(rewards[:max(episodes, 1)], start=1):
        best = max(best, reward)
        history_rows.append({
            "episode": idx,
            "algorithm": algorithm_key,
            "backend": "stable-baselines3",
            "total_reward": float(reward),
            "best_reward": float(best),
            "mean_abs_error": np.nan,
            "mean_abs_action": np.nan,
            "steps": int(config.max_steps),
        })

    obs, _ = eval_env.reset(seed=seed + 200)
    rows = []
    cumulative_reward = 0.0
    for step in range(config.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, _, info = eval_env.step(action)
        cumulative_reward += float(reward)
        rows.append({
            "step": step,
            "state": float(obs[0]),
            "action": float(info["action"]),
            "reward": float(reward),
            "error": float(info["error"]),
            "abs_error": float(abs(info["error"])),
            "cumulative_reward": cumulative_reward,
        })
        obs = next_obs
        if done:
            break
    return pd.DataFrame(history_rows), pd.DataFrame(rows)


def train_data_driven_sb3_policy(config: DataDrivenEnvironmentConfig, algorithm: str, episodes: int,
                                 learning_rate: float = 0.001, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not rl_backend_availability()["stable-baselines3"]:
        raise ImportError("Stable-Baselines3真实训练后端不可用：请安装 torch、gymnasium、stable-baselines3。")

    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback

    algorithm_key = algorithm.lower()
    model_map = {
        "a2c": A2C,
        "ppo": PPO,
        "ddpg": DDPG,
        "td3": TD3,
        "sac": SAC,
    }
    if algorithm_key not in model_map:
        raise ValueError("用户数据驱动环境当前支持 A2C/PPO/DDPG/TD3/SAC 连续动作算法；DQN 需要离散动作配置。")

    validation = validate_rl_dataset(config)
    if not validation["ok"]:
        raise ValueError("强化学习数据集不可用: " + "; ".join(validation["issues"]))

    env = _make_gymnasium_data_env(config)
    eval_config = DataDrivenEnvironmentConfig(**{**config.__dict__, "seed": seed + 100})
    eval_env = _make_gymnasium_data_env(eval_config)
    total_timesteps = max(episodes * config.max_steps, config.max_steps)

    class RewardCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_rewards = []
            self.current_reward = 0.0

        def _on_step(self) -> bool:
            self.current_reward += float(np.mean(self.locals.get("rewards", [0.0])))
            dones = self.locals.get("dones", [])
            if len(dones) and bool(dones[0]):
                self.episode_rewards.append(self.current_reward)
                self.current_reward = 0.0
            return True

    callback = RewardCallback()
    model = model_map[algorithm_key]("MlpPolicy", env, learning_rate=learning_rate, seed=seed, verbose=0)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    rewards = callback.episode_rewards or [callback.current_reward]
    best = -np.inf
    history_rows = []
    for idx, reward in enumerate(rewards[:max(episodes, 1)], start=1):
        best = max(best, reward)
        history_rows.append({
            "episode": idx,
            "algorithm": algorithm_key,
            "backend": "stable-baselines3-data",
            "total_reward": float(reward),
            "best_reward": float(best),
            "mean_abs_error": np.nan,
            "mean_abs_action": np.nan,
            "steps": int(config.max_steps),
        })

    obs, _ = eval_env.reset(seed=seed + 200)
    rows = []
    cumulative_reward = 0.0
    for step in range(config.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, _, info = eval_env.step(action)
        cumulative_reward += float(reward)
        row = {
            "step": step,
            "state": float(np.linalg.norm(obs)),
            "action": float(info["action"]),
            "reward": float(reward),
            "error": float(info["state_norm"]),
            "abs_error": float(abs(info["state_norm"])),
            "cumulative_reward": cumulative_reward,
            "algorithm": algorithm_key,
            "backend": "stable-baselines3-data",
        }
        for idx, value in enumerate(obs):
            row[f"state_{idx + 1}"] = float(value)
        rows.append(row)
        obs = next_obs
        if done:
            break

    metadata = {
        **validation,
        "environment": "data_driven",
        "state_columns": config.state_columns,
        "action_column": config.action_column,
        "reward_column": config.reward_column or "自动奖励: -状态变化范数 - 动作惩罚",
        "action_limit": float(env.core.action_limit),
    }
    return pd.DataFrame(history_rows), pd.DataFrame(rows), metadata


def compare_data_driven_algorithms(config: DataDrivenEnvironmentConfig, algorithms: list[str], episodes: int,
                                   learning_rate: float = 0.001) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    histories = []
    evaluations = []
    metadata = {}
    for index, algorithm in enumerate(algorithms):
        run_config = DataDrivenEnvironmentConfig(**{**config.__dict__, "seed": config.seed + index})
        history, evaluation, metadata = train_data_driven_sb3_policy(
            run_config,
            algorithm,
            episodes=episodes,
            learning_rate=learning_rate,
            seed=config.seed + index,
        )
        histories.append(history)
        evaluations.append(evaluation)
    return pd.concat(histories, ignore_index=True), pd.concat(evaluations, ignore_index=True), metadata


def train_custom_function_sb3_policy(config: CustomFunctionEnvironmentConfig, algorithm: str, episodes: int,
                                     learning_rate: float = 0.001, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not rl_backend_availability()["stable-baselines3"]:
        raise ImportError("Stable-Baselines3真实训练后端不可用：请安装 torch、gymnasium、stable-baselines3。")

    from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback

    algorithm_key = algorithm.lower()
    model_map = {
        "a2c": A2C,
        "ppo": PPO,
        "ddpg": DDPG,
        "td3": TD3,
        "sac": SAC,
    }
    if algorithm_key not in model_map:
        raise ValueError("自定义函数环境当前支持 A2C/PPO/DDPG/TD3/SAC 连续动作算法；DQN 需要离散动作配置。")

    env = _make_gymnasium_custom_env(config)
    eval_config = CustomFunctionEnvironmentConfig(**{**config.__dict__, "seed": seed + 100})
    eval_env = _make_gymnasium_custom_env(eval_config)
    total_timesteps = max(episodes * config.max_steps, config.max_steps)

    class RewardCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_rewards = []
            self.current_reward = 0.0

        def _on_step(self) -> bool:
            self.current_reward += float(np.mean(self.locals.get("rewards", [0.0])))
            dones = self.locals.get("dones", [])
            if len(dones) and bool(dones[0]):
                self.episode_rewards.append(self.current_reward)
                self.current_reward = 0.0
            return True

    callback = RewardCallback()
    model = model_map[algorithm_key]("MlpPolicy", env, learning_rate=learning_rate, seed=seed, verbose=0)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    rewards = callback.episode_rewards or [callback.current_reward]
    best = -np.inf
    history_rows = []
    for idx, reward in enumerate(rewards[:max(episodes, 1)], start=1):
        best = max(best, reward)
        history_rows.append({
            "episode": idx,
            "algorithm": algorithm_key,
            "backend": "stable-baselines3-custom",
            "total_reward": float(reward),
            "best_reward": float(best),
            "mean_abs_error": np.nan,
            "mean_abs_action": np.nan,
            "steps": int(config.max_steps),
        })

    obs, _ = eval_env.reset(seed=seed + 200)
    rows = []
    cumulative_reward = 0.0
    for step in range(config.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, _, info = eval_env.step(action)
        cumulative_reward += float(reward)
        row = {
            "step": step,
            "state": float(np.linalg.norm(obs)),
            "action": float(info.get("action", np.asarray(action).ravel()[0])),
            "reward": float(reward),
            "error": float(info.get("error", info.get("state_norm", np.linalg.norm(next_obs)))),
            "abs_error": float(abs(info.get("error", info.get("state_norm", np.linalg.norm(next_obs))))),
            "cumulative_reward": cumulative_reward,
            "algorithm": algorithm_key,
            "backend": "stable-baselines3-custom",
        }
        for idx, value in enumerate(obs):
            row[f"state_{idx + 1}"] = float(value)
        rows.append(row)
        obs = next_obs
        if done:
            break

    metadata = {
        "environment": "custom_function",
        "observation_dim": config.observation_dim,
        "action_limit": config.action_limit,
        "max_steps": config.max_steps,
    }
    return pd.DataFrame(history_rows), pd.DataFrame(rows), metadata


def compare_custom_function_algorithms(config: CustomFunctionEnvironmentConfig, algorithms: list[str], episodes: int,
                                       learning_rate: float = 0.001) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    histories = []
    evaluations = []
    metadata = {}
    for index, algorithm in enumerate(algorithms):
        run_config = CustomFunctionEnvironmentConfig(**{**config.__dict__, "seed": config.seed + index})
        history, evaluation, metadata = train_custom_function_sb3_policy(
            run_config,
            algorithm,
            episodes=episodes,
            learning_rate=learning_rate,
            seed=config.seed + index,
        )
        histories.append(history)
        evaluations.append(evaluation)
    return pd.concat(histories, ignore_index=True), pd.concat(evaluations, ignore_index=True), metadata


def train_rllib_policy(config: ControlEnvironmentConfig, algorithm: str, episodes: int,
                       learning_rate: float = 0.001, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not rl_backend_availability()["ray-rllib"]:
        raise ImportError("Ray RLlib真实训练后端不可用：请安装 torch、gymnasium、ray[rllib]。")
    if algorithm.lower() != "ppo_rllib":
        raise ValueError("当前Ray RLlib后端仅接入 PPO (Ray RLlib)。")

    import ray
    from ray.rllib.algorithms.ppo import PPOConfig

    def env_creator(env_config):
        cfg = ControlEnvironmentConfig(**env_config)
        return _make_gymnasium_env(cfg, discrete=False)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")

    config_dict = {**config.__dict__, "seed": seed}
    algo = (
        PPOConfig()
        .environment(env=env_creator, env_config=config_dict)
        .framework("torch")
        .training(lr=learning_rate)
        .resources(num_gpus=0)
        .build()
    )
    history_rows = []
    best_reward = -np.inf
    try:
        for episode in range(1, episodes + 1):
            result = algo.train()
            reward = float(result.get("episode_reward_mean", np.nan))
            best_reward = max(best_reward, reward)
            history_rows.append({
                "episode": episode,
                "algorithm": "ppo_rllib",
                "backend": "ray-rllib",
                "total_reward": reward,
                "best_reward": best_reward,
                "mean_abs_error": np.nan,
                "mean_abs_action": np.nan,
                "steps": int(config.max_steps),
            })
    finally:
        algo.stop()

    eval_env = _make_gymnasium_env(ControlEnvironmentConfig(**{**config.__dict__, "seed": seed + 100}), discrete=False)
    obs, _ = eval_env.reset(seed=seed + 200)
    rows = []
    cumulative_reward = 0.0
    for step in range(config.max_steps):
        if hasattr(algo, "compute_single_action"):
            action = algo.compute_single_action(obs, explore=False)
        else:
            module = algo.get_module()
            action = module.forward_inference({"obs": np.asarray([obs], dtype=np.float32)})["actions"][0]
        next_obs, reward, done, _, info = eval_env.step(action)
        cumulative_reward += float(reward)
        rows.append({
            "step": step,
            "state": float(obs[0]),
            "action": float(info["action"]),
            "reward": float(reward),
            "error": float(info["error"]),
            "abs_error": float(abs(info["error"])),
            "cumulative_reward": cumulative_reward,
        })
        obs = next_obs
        if done:
            break
    return pd.DataFrame(history_rows), pd.DataFrame(rows)


def summarize_evaluation(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    if evaluation_df.empty:
        return pd.DataFrame()
    grouped = evaluation_df.groupby("algorithm") if "algorithm" in evaluation_df.columns else [("policy", evaluation_df)]
    rows = []
    for name, frame in grouped:
        rows.append({
            "algorithm": name,
            "steps": int(len(frame)),
            "final_error": float(frame["error"].iloc[-1]),
            "mean_abs_error": float(frame["abs_error"].mean()),
            "total_reward": float(frame["reward"].sum()),
            "mean_abs_action": float(frame["action"].abs().mean()),
        })
    return pd.DataFrame(rows).sort_values("mean_abs_error")


def recommend_control_action(evaluation_df: pd.DataFrame) -> dict[str, float | str]:
    if evaluation_df.empty:
        return {"summary": "暂无评估结果", "final_error": 0.0, "mean_reward": 0.0}
    if "algorithm" in evaluation_df.columns:
        summary_df = summarize_evaluation(evaluation_df)
        best_algorithm = str(summary_df.iloc[0]["algorithm"])
        best_frame = evaluation_df[evaluation_df["algorithm"] == best_algorithm]
    else:
        best_algorithm = "policy"
        best_frame = evaluation_df
    final_error = float(best_frame["error"].iloc[-1])
    mean_reward = float(best_frame["reward"].mean())
    action = "维持当前策略" if abs(final_error) < 0.1 else "继续增大控制强度" if final_error > 0 else "适当减小控制强度"
    return {"summary": f"{best_algorithm}: {action}", "final_error": final_error, "mean_reward": mean_reward}
