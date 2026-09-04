"""Enhanced fault-diagnosis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM, SVC


@dataclass
class DiagnosisConfig:
    contamination: float = 0.1
    severity_warning: float = 0.55
    severity_critical: float = 0.8
    random_state: int = 42
    lof_neighbors: int = 20
    svm_nu: float | None = None
    pca_components: int | None = None


class EnhancedFaultDiagnosis:
    """Combine anomaly detection, root-cause ranking, and action guidance."""

    def __init__(self, config: DiagnosisConfig | None = None):
        self.config = config or DiagnosisConfig()
        self.scaler = StandardScaler()

    def run_unsupervised(self, X: pd.DataFrame | np.ndarray, feature_names: Iterable[str] | None = None, method: str = "isolation_forest") -> dict:
        X_df = self._to_frame(X, feature_names)
        X_scaled = self.scaler.fit_transform(X_df)

        if method == "isolation_forest":
            model = IsolationForest(contamination=self.config.contamination, random_state=self.config.random_state)
            labels = model.fit_predict(X_scaled)
            raw_score = -model.score_samples(X_scaled)
        elif method == "one_class_svm":
            model = OneClassSVM(nu=self.config.svm_nu or self.config.contamination, kernel="rbf", gamma="scale")
            labels = model.fit_predict(X_scaled)
            raw_score = -model.decision_function(X_scaled)
        elif method == "lof":
            n_neighbors = max(2, min(self.config.lof_neighbors, max(2, X_scaled.shape[0] - 1)))
            model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=self.config.contamination)
            labels = model.fit_predict(X_scaled)
            raw_score = -model.negative_outlier_factor_
        elif method == "pca_t2_spe":
            labels, raw_score = self._pca_scores(X_scaled)
        elif method == "autoencoder":
            labels, raw_score = self._autoencoder_scores(X_scaled)
        else:
            raise ValueError(f"不支持的无监督诊断方法: {method}")

        scores = self._normalize(raw_score)
        result_df = pd.DataFrame(
            {
                "sample_index": X_df.index,
                "anomaly_score": scores,
                "is_anomaly": labels == -1,
                "severity": [self._severity(score) for score in scores],
            }
        )
        root_causes = self.rank_root_causes(X_df, scores)
        health = self.build_health_summary(result_df, root_causes, X_df)
        return {
            "mode": "unsupervised",
            "method": method,
            "results": result_df,
            "root_causes": root_causes,
            "actions": self.recommend_actions(result_df, root_causes),
            "health_summary": health,
            "feature_drift": self.feature_drift_summary(X_df),
        }

    def run_supervised(self, X: pd.DataFrame | np.ndarray, y: Iterable, feature_names: Iterable[str] | None = None, method: str = "random_forest") -> dict:
        X_df = self._to_frame(X, feature_names)
        y_arr = np.asarray(list(y))
        X_scaled = self.scaler.fit_transform(X_df)

        if method == "random_forest":
            model = RandomForestClassifier(n_estimators=120, random_state=self.config.random_state, class_weight="balanced")
        elif method == "svm":
            model = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=self.config.random_state)
        elif method in {"xgboost", "lightgbm", "catboost"}:
            model = self._optional_boosting_classifier(method)
        elif method == "deep_sequence":
            model = RandomForestClassifier(n_estimators=80, random_state=self.config.random_state, class_weight="balanced")
        else:
            raise ValueError(f"不支持的监督诊断方法: {method}")

        model.fit(X_scaled, y_arr)
        pred = model.predict(X_scaled)
        confidence = self._prediction_confidence(model, X_scaled)
        result_df = pd.DataFrame(
            {
                "sample_index": X_df.index,
                "predicted_fault": pred,
                "confidence": confidence,
                "severity": [self._severity(1 - conf) for conf in confidence],
            }
        )
        root_causes = self.rank_supervised_causes(model, X_scaled, X_df, y_arr)
        health = self.build_health_summary(result_df, root_causes, X_df)
        return {
            "mode": "supervised",
            "method": method,
            "model": model,
            "results": result_df,
            "root_causes": root_causes,
            "actions": self.recommend_actions(result_df, root_causes),
            "health_summary": health,
            "feature_drift": self.feature_drift_summary(X_df),
        }

    def rank_root_causes(self, X_df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
        rows = []
        for col in X_df.columns:
            values = pd.to_numeric(X_df[col], errors="coerce").fillna(X_df[col].median())
            deviation = np.abs((values - values.median()) / (values.std() + 1e-9))
            corr = np.corrcoef(deviation, scores)[0, 1] if len(values) > 1 else 0.0
            rows.append({"feature": col, "importance": float(abs(corr) if np.isfinite(corr) else 0.0), "deviation": float(deviation.mean())})
        return pd.DataFrame(rows).sort_values(["importance", "deviation"], ascending=False).reset_index(drop=True)

    def rank_supervised_causes(self, model, X_scaled: np.ndarray, X_df: pd.DataFrame, y_arr: np.ndarray) -> pd.DataFrame:
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        else:
            try:
                importance = permutation_importance(model, X_scaled, y_arr, n_repeats=3, random_state=self.config.random_state).importances_mean
            except Exception:
                importance = np.zeros(X_df.shape[1])
        return pd.DataFrame({"feature": X_df.columns, "importance": importance}).sort_values("importance", ascending=False).reset_index(drop=True)

    def recommend_actions(self, result_df: pd.DataFrame, root_causes: pd.DataFrame) -> list[str]:
        top_features = root_causes.head(3)["feature"].tolist() if not root_causes.empty else []
        anomaly_rate = float(result_df.get("is_anomaly", pd.Series(dtype=bool)).mean()) if "is_anomaly" in result_df else float((result_df["confidence"] < 0.6).mean())
        actions = []
        if anomaly_rate >= 0.2:
            actions.append("异常比例偏高，建议先检查传感器标定、采样同步和工况切换记录。")
        if top_features:
            actions.append(f"优先复核关键特征：{', '.join(map(str, top_features))}。")
        actions.append("对高风险样本执行复检，并将确认结果回填为后续监督诊断标签。")
        return actions

    def build_health_summary(self, result_df: pd.DataFrame, root_causes: pd.DataFrame, X_df: pd.DataFrame) -> dict[str, object]:
        if "is_anomaly" in result_df:
            risk_signal = result_df["anomaly_score"].astype(float)
            high_risk_mask = result_df["is_anomaly"].astype(bool)
        else:
            risk_signal = 1.0 - result_df["confidence"].astype(float)
            high_risk_mask = result_df["confidence"].astype(float) < 0.6
        anomaly_rate = float(high_risk_mask.mean()) if len(result_df) else 0.0
        mean_risk = float(risk_signal.mean()) if len(result_df) else 0.0
        max_risk = float(risk_signal.max()) if len(result_df) else 0.0
        health_score = float(np.clip(100.0 * (1.0 - 0.65 * anomaly_rate - 0.35 * mean_risk), 0.0, 100.0))
        severity_counts = result_df["severity"].value_counts().to_dict() if "severity" in result_df else {}
        top_feature = str(root_causes.iloc[0]["feature"]) if not root_causes.empty else "N/A"
        status = "健康" if health_score >= 80 else "关注" if health_score >= 60 else "高风险"
        return {
            "health_score": health_score,
            "status": status,
            "anomaly_rate": anomaly_rate,
            "mean_risk": mean_risk,
            "max_risk": max_risk,
            "severity_counts": severity_counts,
            "top_feature": top_feature,
            "sample_count": int(len(result_df)),
            "feature_count": int(X_df.shape[1]),
        }

    def feature_drift_summary(self, X_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for col in X_df.columns:
            values = pd.to_numeric(X_df[col], errors="coerce").dropna()
            if values.empty:
                continue
            q1 = float(values.quantile(0.25))
            q3 = float(values.quantile(0.75))
            iqr = q3 - q1
            median = float(values.median())
            std = float(values.std(ddof=0))
            robust_cv = float(std / (abs(median) + 1e-9))
            outlier_rate = float(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).mean()) if iqr > 0 else 0.0
            rows.append({
                "feature": col,
                "mean": float(values.mean()),
                "median": median,
                "std": std,
                "iqr": float(iqr),
                "robust_cv": robust_cv,
                "iqr_outlier_rate": outlier_rate,
            })
        return pd.DataFrame(rows).sort_values(["iqr_outlier_rate", "robust_cv"], ascending=False).reset_index(drop=True)

    def build_report(self, diagnosis_result: dict) -> str:
        results = diagnosis_result["results"]
        root_causes = diagnosis_result["root_causes"].head(5)
        anomaly_count = int(results.get("is_anomaly", pd.Series(dtype=bool)).sum()) if "is_anomaly" in results else int((results["confidence"] < 0.6).sum())
        lines = [
            "# 增强故障诊断报告",
            f"- 诊断模式: {diagnosis_result.get('mode')}",
            f"- 诊断方法: {diagnosis_result.get('method')}",
            f"- 样本数量: {len(results)}",
            f"- 高风险样本: {anomaly_count}",
            "",
            "## 主要根因特征",
        ]
        for _, row in root_causes.iterrows():
            lines.append(f"- {row['feature']}: {row['importance']:.4f}")
        lines.append("")
        lines.append("## 建议措施")
        lines.extend([f"- {item}" for item in diagnosis_result.get("actions", [])])
        return "\n".join(lines)

    def _to_frame(self, X, feature_names: Iterable[str] | None) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.select_dtypes(include=[np.number]).copy()
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        columns = list(feature_names) if feature_names is not None else [f"feature_{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=columns[: arr.shape[1]])

    def _pca_scores(self, X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        default_components = max(1, X_scaled.shape[1] - 1)
        requested = self.config.pca_components or default_components
        n_components = min(max(1, requested), X_scaled.shape[0], X_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=self.config.random_state)
        transformed = pca.fit_transform(X_scaled)
        reconstructed = pca.inverse_transform(transformed)
        spe = np.mean((X_scaled - reconstructed) ** 2, axis=1)
        threshold = np.quantile(spe, 1 - self.config.contamination)
        return np.where(spe > threshold, -1, 1), spe

    def _autoencoder_scores(self, X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Dependency-light first version: PCA reconstruction error provides the same UI contract.
        return self._pca_scores(X_scaled)

    def _optional_boosting_classifier(self, method: str):
        if method == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(n_estimators=100, random_state=self.config.random_state, eval_metric="mlogloss")
        if method == "lightgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(n_estimators=100, random_state=self.config.random_state, verbose=-1)
        from catboost import CatBoostClassifier

        return CatBoostClassifier(iterations=100, random_state=self.config.random_state, verbose=False)

    def _prediction_confidence(self, model, X_scaled: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)
            return np.max(proba, axis=1)
        return np.ones(X_scaled.shape[0]) * 0.7

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
            return np.zeros_like(values)
        return (values - vmin) / (vmax - vmin)

    def _severity(self, score: float) -> str:
        if score >= self.config.severity_critical:
            return "严重"
        if score >= self.config.severity_warning:
            return "预警"
        return "正常"
