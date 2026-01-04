from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

# Silence TF chatter
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # 2=hide INFO, 3=hide INFO+WARNING
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, History, ReduceLROnPlateau  # type: ignore

from .ann_model import ANNModel
from ..utils.preprocessor import Preprocessor

LAB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # .../phlux_lab
DEFAULT_LOG_DIR = os.path.join(LAB_ROOT, "logs")
DEFAULT_MODEL_DIR = os.path.join(LAB_ROOT, "models")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VFMModel:
    """
    Wrapper class to train/predict/save a single-head regression ANN (flow-only).

    Notes
    - This class expects the Preprocessor to already provide scaled X/y arrays:
        - preprocessor.x_input_scaled
        - preprocessor.y_output_scaled
    - `predict(X_scaled)` returns **scaled** y (model output), so you can call
      `preprocessor.inverse_transform_y(y_scaled)` to get user/original units.
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
        earlystop_patience: int = 5,
        lr_scheduler_cfg: Optional[Dict[str, Any]] = None,
        num_folds: int = 1,
        val_split: float = 0.05,
        log_dir: str = DEFAULT_LOG_DIR,
        model_dir: str = DEFAULT_MODEL_DIR,
        flow_loss_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.preprocessor = preprocessor

        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.earlystop = bool(earlystop)
        self.earlystop_patience = int(earlystop_patience)
        self.num_folds = int(num_folds)
        self.val_split = float(val_split)

        self.flow_loss_cfg = flow_loss_cfg or {}
        self.lr_scheduler_cfg = lr_scheduler_cfg or {}

        self.log_dir = str(log_dir)
        self.model_dir = str(model_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.model = self._build_ann(
            nodes_layers=nodes_layers,
            act_func=act_func,
            learning_rate=float(learning_rate),
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
        model_dir: str = DEFAULT_MODEL_DIR,
    ) -> "VFMModel":
        nodes_layers = list(model_cfg.get("nodes_layers", [64, 64, 64]) or [64, 64, 64])
        act_func = str(model_cfg.get("activation", model_cfg.get("act_func", "tanh")))
        learning_rate = float(model_cfg.get("learning_rate", 0.001))
        epochs = int(model_cfg.get("epochs", 100))
        batch_size = int(model_cfg.get("batch_size", 100))
        earlystop = bool(model_cfg.get("earlystop", True))
        earlystop_patience = int(model_cfg.get("earlystop_patience", 5))
        num_folds = int(model_cfg.get("num_folds", 1))
        val_split = float(model_cfg.get("val_split", 0.05))

        flow_loss_cfg = dict(model_cfg.get("flow_loss", {}) or {})
        lr_scheduler_cfg = dict(model_cfg.get("lr_scheduler", {}) or {})

        return cls(
            preprocessor=preprocessor,
            nodes_layers=nodes_layers,
            act_func=act_func,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            earlystop=earlystop,
            earlystop_patience=earlystop_patience,
            lr_scheduler_cfg=lr_scheduler_cfg,
            num_folds=num_folds,
            val_split=val_split,
            log_dir=log_dir,
            model_dir=model_dir,
            flow_loss_cfg=flow_loss_cfg,
        )

    # ------------------------------------------------------------------
    def _generate_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        csv_path = os.path.join(self.log_dir, "training_log.csv")
        callbacks: List[tf.keras.callbacks.Callback] = [History(), CSVLogger(csv_path, append=True)]

        if self.earlystop:
            # Keras aliases: "mse" metric name differs by compile; monitor val_loss is most robust
            monitor = str(self.flow_loss_cfg.get("monitor", "val_loss"))
            callbacks.append(
                EarlyStopping(
                    monitor=monitor,
                    mode="min",
                    patience=self.earlystop_patience,
                    verbose=1,
                    min_delta=float(self.flow_loss_cfg.get("min_delta", 1e-4)),
                    restore_best_weights=True,
                )
            )

        lr_cfg = self.lr_scheduler_cfg or {}
        lr_type = str(lr_cfg.get("type", "")).lower().strip()
        if lr_type in ("reduce_on_plateau", "reduce_lr_on_plateau", "reduceonplateau"):
            callbacks.append(
                ReduceLROnPlateau(
                    monitor=str(lr_cfg.get("monitor", "val_loss")),
                    factor=float(lr_cfg.get("factor", 0.5)),
                    patience=int(lr_cfg.get("patience", 12)),
                    min_lr=float(lr_cfg.get("min_lr", 1e-5)),
                    cooldown=int(lr_cfg.get("cooldown", 0)),
                    verbose=int(lr_cfg.get("verbose", 1)),
                )
            )

        return callbacks

    # ------------------------------------------------------------------
    def _build_ann(self, nodes_layers: List[int], act_func: str, learning_rate: float) -> ANNModel:
        x_shape = (int(self.preprocessor.x_input_scaled.shape[1]),)

        y_dim = int(self.preprocessor.y_output_scaled.shape[1])
        if y_dim != 1:
            raise ValueError(
                "Flow-only model expects exactly 1 target column in preprocessor.y_output_scaled. "
                f"Got y_dim={y_dim}. Ensure stage targets = ['q_liquid']."
            )

        return ANNModel(
            x_shape=x_shape,
            nodes_output=1,
            nodes_layers=nodes_layers,
            act_func=act_func,
            learning_rate=learning_rate,
        )

    # ------------------------------------------------------------------
    # Training APIs
    # ------------------------------------------------------------------
    def train_model_kfold(self) -> List[History]:
        if self.num_folds < 2:
            raise ValueError("num_folds must be >= 2 for k-fold training")

        kfold = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        X = self.preprocessor.x_input_scaled
        y = self.preprocessor.y_output_scaled

        self.history_per_fold = []
        for fold_i, (train_idx, val_idx) in enumerate(kfold.split(X, y), start=1):
            logger.info("Training fold %s/%s", fold_i, self.num_folds)

            history = self.model.ann_model.fit(
                X[train_idx],
                y[train_idx],
                validation_data=(X[val_idx], y[val_idx]),
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=True,
                callbacks=self.callbacks,
                verbose=1,
            )
            self.history_per_fold.append(history)

        return self.history_per_fold

    def train_model(self) -> History:
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

    # A thin shim so train_pipeline can call vfm.fit(...)
    def fit(self, X_scaled: np.ndarray, y_scaled: np.ndarray, model_cfg: Optional[Dict[str, Any]] = None) -> History:
        # Allow per-call overrides (useful if pipeline passes model_cfg)
        if model_cfg:
            self.epochs = int(model_cfg.get("epochs", self.epochs))
            self.batch_size = int(model_cfg.get("batch_size", self.batch_size))
            self.val_split = float(model_cfg.get("val_split", self.val_split))
        history = self.model.ann_model.fit(
            X_scaled,
            y_scaled,
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
    # Prediction APIs
    # ------------------------------------------------------------------
    def predict(self, X_scaled: Union[np.ndarray, list], *, verbose: int = 0) -> np.ndarray:
        X = np.asarray(X_scaled, dtype="float32")
        return self.model.ann_model.predict(X, verbose=verbose)

    def predict_in_units(self, x_raw: Any) -> np.ndarray:
        """Predict from raw user-unit features (DataFrame or array), returning y in original units."""
        x_scaled = self.preprocessor.transform_X(x_raw)
        y_scaled = self.predict(x_scaled, verbose=0)
        return self.preprocessor.inverse_transform_y(y_scaled)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save_model(self, filename: Optional[str] = None) -> str:
        if filename is None:
            filename = "vfm_model.keras"
        filepath = os.path.join(self.model_dir, filename)
        logger.info("Saving VFM ANN model to: %s", filepath)
        self.model.ann_model.save(filepath)
        return filepath

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        logger.info("Saving VFM ANN model to: %s", path)
        self.model.ann_model.save(path)
        return path

    @classmethod
    def load(cls, model_path: str, preprocessor: Preprocessor, model_cfg: Dict[str, Any]) -> "VFMModel":
        inst = cls.from_config(preprocessor=preprocessor, model_cfg=model_cfg)
        inst.model.ann_model = tf.keras.models.load_model(model_path)
        return inst
