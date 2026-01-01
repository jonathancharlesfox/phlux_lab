from __future__ import absolute_import, division, print_function

import logging
import os
from typing import List, NoReturn, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, History, CSVLogger  # type: ignore

from .ann_model import ANNModel, OutputHeadSpec
from ..utils.preprocessor import Preprocessor

LAB_ROOT = Path(__file__).resolve().parents[2]   # .../phlux_lab
DEFAULT_LOG_DIR = str(LAB_ROOT / "logs")
DEFAULT_MODEL_DIR = str(LAB_ROOT / "models")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VFMModel(object):
    """
    Wrapper class with methods to train, predict, and save the VFM ANN model.

    Backward compatible:
      - default single-head regression

    Multi-task option (2-head):
      - Head 1: flow regression
      - Head 2: hydraulic wear regression (continuous)
        (and a derived monotonic "probability of wear" computed in predict())
    """

    def __init__(
        self,
        preprocessor: Preprocessor,
        nodes_layers: List[int],
        act_func: str = "tanh",
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 100,
        earlystop: bool = True,
        num_folds: int = 5,
        val_split: float = 0.001,
        log_dir: str = DEFAULT_LOG_DIR,
        model_dir: str = DEFAULT_MODEL_DIR,
        multitask_cfg: Optional[Dict[str, Any]] = None,
    ) -> NoReturn:

        self.preprocessor = preprocessor
        self.epochs = epochs
        self.batch_size = batch_size
        self.earlystop = earlystop
        self.num_folds = num_folds
        self.val_split = val_split
        self.log_dir = log_dir
        self.model_dir = model_dir

        self.multitask_cfg = multitask_cfg or {}
        self.multitask_enabled = bool(self.multitask_cfg.get("enabled", False))

        # Ensure output dirs exist
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # Build underlying ANN
        self.model = self._build_ann(
            nodes_layers=nodes_layers,
            act_func=act_func,
            learning_rate=learning_rate,
        )

        self.history_per_fold: List[History] = []
        self.callbacks = self._generate_callbacks()

    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        preprocessor: Preprocessor,
        model_cfg: Dict[str, Any],
        log_dir: str = DEFAULT_LOG_DIR,
        model_dir: str = DEFAULT_MODEL_DIR
    ) -> "VFMModel":
        nodes_layers = model_cfg.get("nodes_layers", [64, 64, 64])
        act_func = model_cfg.get("activation", "tanh")
        learning_rate = float(model_cfg.get("learning_rate", 0.001))
        epochs = int(model_cfg.get("epochs", 100))
        batch_size = int(model_cfg.get("batch_size", 100))
        earlystop = bool(model_cfg.get("earlystop", True))
        num_folds = int(model_cfg.get("num_folds", 5))
        val_split = float(model_cfg.get("val_split", 0.001))

        multitask_cfg = model_cfg.get("multitask", {}) or {}

        return cls(
            preprocessor=preprocessor,
            nodes_layers=nodes_layers,
            act_func=act_func,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            earlystop=earlystop,
            num_folds=num_folds,
            val_split=val_split,
            log_dir=log_dir,
            model_dir=model_dir,
            multitask_cfg=multitask_cfg,
        )

    # ------------------------------------------------------------------
    def _generate_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        csv_path = os.path.join(self.log_dir, "training_log.csv")
        csv_logger = CSVLogger(csv_path, append=True)
        callbacks: List[tf.keras.callbacks.Callback] = [History(), csv_logger]

        if self.earlystop:
            # Monitor depends on single-head vs multi-head
            if self.multitask_enabled:
                monitor = "val_flow_mean_squared_error"
            else:
                monitor = "val_mean_squared_error"

            callbacks.append(
                EarlyStopping(
                    monitor=monitor,
                    mode="min",
                    patience=5,
                    verbose=1,
                    min_delta=1e-4,          # ignore tiny changes
                    restore_best_weights=True,
                )
            )
        return callbacks

    # ------------------------------------------------------------------
    def _build_ann(self, nodes_layers: List[int], act_func: str, learning_rate: float) -> ANNModel:
        x_shape = (self.preprocessor.x_input_scaled.shape[1],)

        # Default: single-head regression (backward compatible)
        if not self.multitask_enabled:
            return ANNModel(
                x_shape=x_shape,
                nodes_output=self.preprocessor.y_output_scaled.shape[1],
                nodes_layers=nodes_layers,
                act_func=act_func,
                learning_rate=learning_rate,
            )

        # Multi-task: expect exactly 2 targets in y:
        #   col0 = flow (regression)
        #   col1 = hydraulic wear fraction (regression)
        y_dim = int(self.preprocessor.y_output_scaled.shape[1])
        if y_dim != 2:
            raise ValueError(
                f"Multitask enabled but y_output_scaled has {y_dim} columns. "
                f"Expected exactly 2 targets: [flow, hydraulic_wear__frac]."
            )

        heads = [
            OutputHeadSpec(
                name="flow",
                units=1,
                head_type="regression",
                activation="linear",
                loss="mse",
                metrics=["mean_squared_error"],
            ),
            OutputHeadSpec(
                name="hydraulic_wear",
                units=1,
                head_type="regression",
                activation="linear",
                loss="mse",
                metrics=["mean_squared_error"],
            ),
        ]

        # Loss weights
        wear_loss_weight = float(
            (self.multitask_cfg.get("hydraulic_wear_head", {}) or {}).get("loss_weight", 0.2)
        )
        # Backward compatibility: older key "degradation_head.loss_weight"
        if "hydraulic_wear_head" not in (self.multitask_cfg or {}) and "degradation_head" in (self.multitask_cfg or {}):
            wear_loss_weight = float(
                (self.multitask_cfg.get("degradation_head", {}) or {}).get("loss_weight", wear_loss_weight)
            )

        loss_weights = {"flow": 1.0, "hydraulic_wear": wear_loss_weight}

        return ANNModel(
            x_shape=x_shape,
            nodes_output=2,  # not used in multi-head mode
            nodes_layers=nodes_layers,
            act_func=act_func,
            learning_rate=learning_rate,
            output_heads=heads,
            loss_weights=loss_weights,
        )

    # ------------------------------------------------------------------
    def _build_multitask_targets_and_weights(
        self, y_scaled: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Build regression targets + weights for multi-task training.

        Expects y_scaled with shape (n, 2):
          - y_scaled[:, 0] = flow target (scaled)
          - y_scaled[:, 1] = hydraulic_wear__frac target (scaled)

        Returns:
          y_dict: {"flow": (n,1), "hydraulic_wear": (n,1)}
          w_dict: {"flow": ones, "hydraulic_wear": ones}
        """
        y_flow_scaled = y_scaled[:, 0:1].astype(np.float32)
        y_wear_scaled = y_scaled[:, 1:2].astype(np.float32)

        y_dict = {
            "flow": y_flow_scaled,
            "hydraulic_wear": y_wear_scaled,
        }

        w_dict = {
            "flow": np.ones_like(y_flow_scaled, dtype=np.float32),
            "hydraulic_wear": np.ones_like(y_wear_scaled, dtype=np.float32),
        }

        return y_dict, w_dict

    # ------------------------------------------------------------------
    def train_model_kfold(self) -> List[History]:
        kfold = KFold(n_splits=self.num_folds, shuffle=True)

        fold_count = 1
        for train_idx, val_idx in kfold.split(self.preprocessor.x_input_scaled, self.preprocessor.y_output_scaled):
            logger.info(f"Training fold: {fold_count}/{self.num_folds}")

            x_train = self.preprocessor.x_input_scaled[train_idx]
            x_val = self.preprocessor.x_input_scaled[val_idx]

            y_train_scaled = self.preprocessor.y_output_scaled[train_idx]
            y_val_scaled = self.preprocessor.y_output_scaled[val_idx]

            if not self.multitask_enabled:
                history = self.model.ann_model.fit(
                    x_train,
                    y_train_scaled,
                    validation_data=(x_val, y_val_scaled),
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    shuffle=True,
                    callbacks=self.callbacks,
                    verbose=1,
                )
                self.history_per_fold.append(history)
                fold_count += 1
                continue

            y_train_dict, w_train_dict = self._build_multitask_targets_and_weights(y_train_scaled)
            y_val_dict, w_val_dict = self._build_multitask_targets_and_weights(y_val_scaled)

            history = self.model.ann_model.fit(
                x_train,
                y_train_dict,
                sample_weight=w_train_dict,
                validation_data=(x_val, y_val_dict, w_val_dict),
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=True,
                callbacks=self.callbacks,
                verbose=1,
            )
            self.history_per_fold.append(history)
            fold_count += 1

        return self.history_per_fold

    # ------------------------------------------------------------------
    def train_model(self) -> History:
        if not self.multitask_enabled:
            history = self.model.ann_model.fit(
                self.preprocessor.x_input_scaled,
                self.preprocessor.y_output_scaled,
                validation_split=self.val_split,
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=True,
                callbacks=self.callbacks,
                verbose=1,
            )
            self.history_per_fold.append(history)
            return history

        y_dict, w_dict = self._build_multitask_targets_and_weights(self.preprocessor.y_output_scaled)

        history = self.model.ann_model.fit(
            self.preprocessor.x_input_scaled,
            y_dict,
            sample_weight=w_dict,
            validation_split=self.val_split,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            callbacks=self.callbacks,
            verbose=1,
        )
        self.history_per_fold.append(history)
        return history

    # ------------------------------------------------------------------
    def _wear_to_probability_pct(self, wear_frac: np.ndarray) -> np.ndarray:
        """Convert predicted wear fraction to a monotonic 'probability of wear' in [0, 100].

        This is intentionally a post-processing step so the model can learn a smooth
        physical quantity (wear) while you still get an interpretable probability.

        Config (multitask_cfg):
          hydraulic_wear_head:
            probability:
              method: "linear" | "logistic"
              wear_lo: 0.0
              wear_hi: 0.05
              # logistic only:
              mid: 0.02
              sharpness: 120
        """
        wear_cfg = (self.multitask_cfg.get("hydraulic_wear_head", {}) or {}).get("probability", {}) or {}

        method = str(wear_cfg.get("method", "linear")).lower().strip()

        if method == "logistic":
            mid = float(wear_cfg.get("mid", 0.02))
            sharpness = float(wear_cfg.get("sharpness", 120.0))
            p = 1.0 / (1.0 + np.exp(-sharpness * (wear_frac - mid)))
            return np.clip(p, 0.0, 1.0) * 100.0

        # default: linear ramp
        wear_lo = float(wear_cfg.get("wear_lo", 0.0))

        # If wear_hi isn't set, fall back to training max in original units.
        wear_hi = wear_cfg.get("wear_hi", None)
        if wear_hi is None:
            try:
                y_train_orig = self.preprocessor.inverse_transform_y(self.preprocessor.y_output_scaled)
                wear_hi = float(np.nanmax(y_train_orig[:, 1]))
            except Exception:
                wear_hi = 0.05
        wear_hi = float(wear_hi)

        denom = max(wear_hi - wear_lo, 1e-12)
        p = (wear_frac - wear_lo) / denom
        return np.clip(p, 0.0, 1.0) * 100.0

    # ------------------------------------------------------------------
    def predict(self, x_raw) -> Dict[str, Any]:
        """Predict given raw feature matrix.

        Returns:
          - single-head: {"y": <np.ndarray in original units>}
          - multi-head: {
                "flow": <np.ndarray in original units>,
                "hydraulic_wear__frac": <np.ndarray in original units>,
                "hydraulic_wear_probability_pct": <np.ndarray in [0,100]>,
            }
        """
        x_scaled = self.preprocessor.transform_X(x_raw)

        y_out = self.model.ann_model.predict(x_scaled)

        if not self.multitask_enabled:
            y_pred = self.preprocessor.inverse_transform_y(y_out)
            return {"y": y_pred}

        # Keras multi-output returns a dict when the model was built with named heads
        flow_scaled = y_out["flow"]
        wear_scaled = y_out["hydraulic_wear"]

        # Inverse-transform both outputs back to original units using the existing y scaler.
        n = int(flow_scaled.shape[0])
        y_scaled_full = np.zeros((n, 2), dtype=np.float32)
        y_scaled_full[:, 0:1] = flow_scaled.astype(np.float32)
        y_scaled_full[:, 1:2] = wear_scaled.astype(np.float32)

        y_pred_full = self.preprocessor.inverse_transform_y(y_scaled_full)

        flow_pred = y_pred_full[:, 0:1]
        wear_pred = y_pred_full[:, 1:2]

        wear_prob_pct = self._wear_to_probability_pct(wear_pred)

        return {
            "flow": flow_pred,
            "hydraulic_wear__frac": wear_pred,
            "hydraulic_wear_probability_pct": wear_prob_pct,
        }

    # ------------------------------------------------------------------
    def save_model(self, filename: Optional[str] = None) -> str:
        if filename is None:
            filename = "vfm_model.h5"
        filepath = os.path.join(self.model_dir, filename)
        logger.info(f"Saving VFM ANN model to: {filepath}")
        self.model.ann_model.save(filepath)
        return filepath
