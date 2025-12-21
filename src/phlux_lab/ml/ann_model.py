from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers  # type: ignore

@dataclass
class OutputHeadSpec:
    """
    Defines one output head for a multi-task model.

    - name: key used for outputs dict
    - units: number of outputs for this head (usually 1)
    - head_type: "regression" or "binary_classification"
    - loss: Keras loss string
    - metrics: list of Keras metrics strings
    - activation: output activation ("linear" for regression, "sigmoid" for probability)
    """
    name: str
    units: int
    head_type: str
    loss: str
    metrics: List[str]
    activation: str


class ANNModel:
    """
    Small wrapper for building a configurable feed-forward ANN.

    Backward compatible:
      - If output_heads is None -> single linear output with MSE (original behavior)

    Multi-task (2-head) mode:
      - Shared trunk + separate heads
      - Compile with per-head losses/metrics
    """

    def __init__(
        self,
        x_shape: tuple[int, ...],
        nodes_output: int,
        nodes_layers: list[int],
        act_func: str = "tanh",
        learning_rate: float = 1e-3,
        output_heads: Optional[List[OutputHeadSpec]] = None,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.x_shape = x_shape
        self.nodes_output = nodes_output
        self.nodes_layers = nodes_layers
        self.act_func = act_func
        self.learning_rate = learning_rate
        self.output_heads = output_heads
        self.loss_weights = loss_weights

        self.ann_model = self._build_model()

    # ----------------------------------------------------------------------
    def _build_model(self) -> tf.keras.Model:
        inputs = layers.Input(shape=self.x_shape)

        x = inputs
        for n in self.nodes_layers:
            x = layers.Dense(n, activation=self.act_func)(x)

        optimizer = optimizers.Adam(learning_rate=self.learning_rate)

        # ------------------------------------------------------------
        # Backward compatible: single-head regression
        # ------------------------------------------------------------
        if not self.output_heads:
            outputs = layers.Dense(self.nodes_output, activation="linear")(x)
            model = models.Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=optimizer,
                loss="mse",
                metrics=["mean_squared_error"],
            )
            return model

        # ------------------------------------------------------------
        # Multi-head model
        # ------------------------------------------------------------
        outputs_dict: Dict[str, tf.Tensor] = {}
        losses: Dict[str, str] = {}
        metrics: Dict[str, List[str]] = {}

        for head in self.output_heads:
            out = layers.Dense(head.units, activation=head.activation, name=head.name)(x)
            outputs_dict[head.name] = out
            losses[head.name] = head.loss
            metrics[head.name] = head.metrics

        model = models.Model(inputs=inputs, outputs=outputs_dict)

        model.compile(
            optimizer=optimizer,
            loss=losses,
            metrics=metrics,
            loss_weights=self.loss_weights,
        )
        return model

    # ----------------------------------------------------------------------
    def summary(self):
        return self.ann_model.summary()
