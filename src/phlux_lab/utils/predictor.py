from __future__ import annotations

from typing import Any, Dict, List, Union, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from phlux_lab.utils.preprocessor import Preprocessor  # type: ignore


Number = Union[int, float]
ValueOrTuple = Union[Number, Tuple[Number, str]]


class VfmPredictor:
    """Lightweight predictor wrapper for a single trained stage artifact.

    Key behavior:
      - Accepts input DataFrames/dicts that may include *extra* columns.
      - Does NOT drop extra columns in predict_one(); this is important so the
        Preprocessor can compute derived features from source columns that are
        not direct model inputs (e.g., computing `specific_power` from `pump_power`
        and `q_liquid_pred`).
    """

    def __init__(self, model: tf.keras.Model, preprocessor: Preprocessor):
        self.model = model
        self.pp = preprocessor
        self.input_cols = list(self.pp.get_input_feature_names())
        self.target_cols = list(self.pp.get_target_feature_names())

    # -----------------------------
    # Loading helpers
    # -----------------------------
    @classmethod
    def from_paths(cls, model_path: str | Path, preprocessor_path: str | Path) -> "VfmPredictor":
        model = tf.keras.models.load_model(str(model_path))
        preprocessor = Preprocessor.load(str(preprocessor_path))
        return cls(model=model, preprocessor=preprocessor)

    @classmethod
    def from_dir(
        cls,
        stage_dir: str | Path,
        *,
        model_filename: Optional[str] = None,
        preprocessor_filename: Optional[str] = None,
    ) -> "VfmPredictor":
        """Load artifacts from a stage directory.

        Defaults:
          - model: first *.keras found (or model_filename if provided)
          - preprocessor: first *.joblib found (or preprocessor_filename if provided)
        """
        d = Path(stage_dir)
        if model_filename is None:
            models = sorted(d.glob("*.keras"))
            if not models:
                raise FileNotFoundError(f"No .keras model found in: {d}")
            model_path = models[0]
        else:
            model_path = d / model_filename

        if preprocessor_filename is None:
            pps = sorted(d.glob("*.joblib"))
            if not pps:
                raise FileNotFoundError(f"No .joblib preprocessor found in: {d}")
            pp_path = pps[0]
        else:
            pp_path = d / preprocessor_filename

        return cls.from_paths(model_path, pp_path)

    # -----------------------------
    # Input utilities
    # -----------------------------
    @staticmethod
    def normalize_feature_inputs(raw: Dict[str, ValueOrTuple]) -> Dict[str, float]:
        """Convert {k: value or (value, unit)} -> {k: float(value)}.

        Unit strings are accepted but not interpreted here. Unit conversion
        is handled inside the Preprocessor using the YAML config it was built from.
        """
        out: Dict[str, float] = {}
        for k, v in raw.items():
            if isinstance(v, tuple) and len(v) == 2:
                out[k] = float(v[0])
            else:
                out[k] = float(v)
        return out

    @staticmethod
    def build_input_dataframe(raw: Dict[str, ValueOrTuple]) -> pd.DataFrame:
        """Create a one-row DataFrame from manual inputs."""
        feats = VfmPredictor.normalize_feature_inputs(raw)
        return pd.DataFrame([feats])

    # -----------------------------
    # Prediction methods
    # -----------------------------
    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict for a DataFrame. df may include extra columns."""
        X = self.pp.transform_X(df)
        y_scaled = self.model.predict(X, verbose=0)
        y = self.pp.inverse_transform_y(y_scaled)

        # Ensure 2D
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        return pd.DataFrame(y_arr, columns=self.target_cols)

    def predict_one(self, features: Dict[str, float]) -> Dict[str, float]:
        """Predict from a simple {feature: value} mapping (assumed user units).

        IMPORTANT: We keep *all* provided features (do not subset to input_cols)
        so derived features can be computed from source columns.
        """
        df = pd.DataFrame([features])
        preds = self.predict_dataframe(df)
        return {c: float(preds.iloc[0][c]) for c in preds.columns}

    def predict_batch(self, rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
        df = pd.DataFrame(rows)
        preds = self.predict_dataframe(df)
        return [{c: float(preds.iloc[i][c]) for c in preds.columns} for i in range(len(preds))]
