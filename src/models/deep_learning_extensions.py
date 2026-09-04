"""Deep learning model registry used by the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeepLearningAlgorithm:
    key: str
    name: str
    task: str
    data_type: str
    description: str


REGRESSION_ALGORITHMS = [
    DeepLearningAlgorithm("ann", "ANN / MLP 回归", "regression", "tabular", "适合中小规模表格回归基线。"),
    DeepLearningAlgorithm("lstm", "LSTM 时序回归", "regression", "sequence", "适合存在长期依赖的时序预测。"),
    DeepLearningAlgorithm("gru", "GRU 时序回归", "regression", "sequence", "比 LSTM 更轻量，适合快速时序实验。"),
    DeepLearningAlgorithm("cnn", "1D-CNN 时序回归", "regression", "sequence", "主流程真实可选，提取局部波形、振动和传感器模式。"),
    DeepLearningAlgorithm("transformer", "Transformer 时序回归", "regression", "sequence", "主流程真实可选，使用注意力机制建模多变量时序关系。"),
    DeepLearningAlgorithm("mlp", "MLP 回归", "regression", "tabular", "主流程真实可选，适合表格非线性回归。"),
]

CLASSIFICATION_ALGORITHMS = [
    DeepLearningAlgorithm("ANN", "ANN 分类", "classification", "tabular", "分类页主流程真实可选，适合表格故障状态识别。"),
    DeepLearningAlgorithm("LSTM", "LSTM 分类", "classification", "sequence", "分类页主流程真实可选，适合序列特征分类。"),
    DeepLearningAlgorithm("CNN (1D)", "1D-CNN 分类", "classification", "sequence", "分类页主流程真实可选，适合波形和传感器片段识别。"),
    DeepLearningAlgorithm("CNN (2D)", "2D-CNN 分类", "classification", "image", "分类页主流程真实可选，适合谱图/二维特征图分类。"),
]


def list_deep_learning_algorithms(task: str | None = None) -> list[DeepLearningAlgorithm]:
    algorithms = REGRESSION_ALGORITHMS + CLASSIFICATION_ALGORITHMS
    if task is None:
        return algorithms
    return [item for item in algorithms if item.task == task]


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:
        return False


def create_regression_model(kind: str, input_shape: tuple[int, ...] | None = None, input_dim: int | None = None, **kwargs: Any):
    """Create a compact Keras regression model for advanced UI options."""
    try:
        from tensorflow.keras import Input, Model, Sequential
        from tensorflow.keras.layers import (
            LSTM,
            GRU,
            Conv1D,
            Dense,
            Dropout,
            Flatten,
            GlobalAveragePooling1D,
            LayerNormalization,
            MultiHeadAttention,
        )
        from tensorflow.keras.optimizers import Adam
    except Exception as exc:
        raise ImportError("TensorFlow 未安装，无法创建深度学习模型。") from exc

    learning_rate = kwargs.get("learning_rate", 0.001)
    dropout = kwargs.get("dropout", 0.1)
    units = kwargs.get("units", 64)

    if input_shape is None:
        if input_dim is None:
            raise ValueError("input_shape 或 input_dim 至少需要提供一个。")
        input_shape = (input_dim,)

    if kind in {"ann", "mlp"}:
        model = Sequential(
            [
                Dense(units, activation="relu", input_shape=(input_dim or input_shape[-1],)),
                Dropout(dropout),
                Dense(max(units // 2, 8), activation="relu"),
                Dense(1),
            ]
        )
    elif kind == "lstm":
        model = Sequential([LSTM(units, input_shape=input_shape), Dropout(dropout), Dense(1)])
    elif kind == "gru":
        model = Sequential([GRU(units, input_shape=input_shape), Dropout(dropout), Dense(1)])
    elif kind in {"cnn", "cnn1d", "tcn", "informer", "autoformer"}:
        dilation = 2 if kind == "tcn" else 1
        model = Sequential(
            [
                Conv1D(units, 3, padding="causal" if kind == "tcn" else "same", dilation_rate=dilation, activation="relu", input_shape=input_shape),
                Dropout(dropout),
                Conv1D(max(units // 2, 8), 3, padding="same", activation="relu"),
                GlobalAveragePooling1D(),
                Dense(1),
            ]
        )
    elif kind == "transformer":
        inputs = Input(shape=input_shape)
        attn = MultiHeadAttention(num_heads=kwargs.get("num_heads", 2), key_dim=max(units // 4, 8))(inputs, inputs)
        x = LayerNormalization()(inputs + attn)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(dropout)(x)
        outputs = Dense(1)(x)
        model = Model(inputs, outputs)
    else:
        raise ValueError(f"不支持的深度学习回归模型: {kind}")

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae", "mse"])
    return model


def create_classification_model(kind: str, input_shape: tuple[int, ...], n_classes: int, **kwargs: Any):
    """Create a compact Keras classifier for tabular/sequence diagnostics."""
    try:
        from tensorflow.keras import Input, Model, Sequential
        from tensorflow.keras.layers import LSTM, Conv1D, Dense, Dropout, GlobalAveragePooling1D, LayerNormalization, MultiHeadAttention
        from tensorflow.keras.optimizers import Adam
    except Exception as exc:
        raise ImportError("TensorFlow 未安装，无法创建深度学习分类模型。") from exc

    units = kwargs.get("units", 64)
    dropout = kwargs.get("dropout", 0.1)
    learning_rate = kwargs.get("learning_rate", 0.001)

    if kind in {"mlp", "ANN"}:
        flat_dim = 1
        for dim in input_shape:
            flat_dim *= dim
        model = Sequential([Dense(units, activation="relu", input_shape=(flat_dim,)), Dropout(dropout), Dense(n_classes, activation="softmax")])
    elif kind in {"cnn1d_classifier", "CNN (1D)"}:
        model = Sequential([Conv1D(units, 3, padding="same", activation="relu", input_shape=input_shape), GlobalAveragePooling1D(), Dense(n_classes, activation="softmax")])
    elif kind == "LSTM":
        model = Sequential([LSTM(units, input_shape=input_shape), Dropout(dropout), Dense(n_classes, activation="softmax")])
    elif kind == "CNN (2D)":
        from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D

        model = Sequential([Conv2D(units, 3, padding="same", activation="relu", input_shape=input_shape), GlobalAveragePooling2D(), Dense(n_classes, activation="softmax")])
    elif kind == "cnn_lstm_classifier":
        model = Sequential([Conv1D(units, 3, padding="same", activation="relu", input_shape=input_shape), LSTM(max(units // 2, 8)), Dense(n_classes, activation="softmax")])
    elif kind == "transformer_classifier":
        inputs = Input(shape=input_shape)
        attn = MultiHeadAttention(num_heads=kwargs.get("num_heads", 2), key_dim=max(units // 4, 8))(inputs, inputs)
        x = LayerNormalization()(inputs + attn)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(dropout)(x)
        outputs = Dense(n_classes, activation="softmax")(x)
        model = Model(inputs, outputs)
    else:
        raise ValueError(f"不支持的深度学习分类模型: {kind}")

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
