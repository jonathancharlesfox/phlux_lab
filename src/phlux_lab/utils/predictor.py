from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from phlux_lab.utils.preprocessor import Preprocessor
from phlux_lab.utils.units import convert_user_to_internal_SI, convert_internal_back_to_user


@dataclass
class PredictorArtifacts:
    model_path: str
    preprocessor_path: str


def _internal_name(col: str) -> str:
    return col.split("__", 1)[0] if "__" in col else col


def _coerce_input_value(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, dict):
        # handled elsewhere
        return None
    return float(v)


class VfmPredictor:
    """
    Predictor that:
      - Loads keras model
      - Loads preprocessor artifact and rebuilds Preprocessor via Preprocessor.from_artifact()
      - Builds a 1-row DataFrame so Preprocessor can compute derived features and convert units.

    Supports:
      - single-head regression models (array output)
      - multi-target regression (array output with multiple columns)
      - legacy dict-output models
    """

    def __init__(self, model: tf.keras.Model, preprocessor: Preprocessor):
        self.model = model
        self.pp = preprocessor

        # Unitless names (new) OR legacy names (supported).
        self.input_cols: List[str] = list(self.pp.get_input_feature_names())
        self.target_cols: List[str] = list(self.pp.get_target_feature_names())

        # Unit mapping recorded in preprocessor artifact (schema units)
        self.schema_units_by_col: Dict[str, str] = getattr(self.pp, "_schema_units_by_col", {}) or {}
        self.unit_policy: Dict[str, Any] = getattr(self.pp, "unit_policy", {}) or {}

        self.derived_names = set()
        if getattr(self.pp, "derived_cfg", None):
            self.derived_names = {
                d.get("name")
                for d in self.pp.derived_cfg
                if isinstance(d, dict) and d.get("name")
    }

    # --------------------------------------------------------
    @classmethod
    def from_paths(
        cls,
        model_path: str,
        preprocessor_path: Optional[str] = None,
    ) -> "VfmPredictor":
        mp = Path(model_path)

        if preprocessor_path is None:
            preprocessor_path = str(mp.parent / "preprocessor.joblib")

        model = tf.keras.models.load_model(str(mp))

        pp_obj = joblib.load(preprocessor_path)

        # Unwrap common nesting
        if isinstance(pp_obj, dict):
            if "preprocessor" in pp_obj:
                pp_obj = pp_obj["preprocessor"]
            elif "pp" in pp_obj:
                pp_obj = pp_obj["pp"]

        if isinstance(pp_obj, dict):
            pp = Preprocessor.from_artifact(pp_obj)
        elif isinstance(pp_obj, Preprocessor):
            pp = pp_obj
        else:
            if hasattr(pp_obj, "transform_X") and hasattr(pp_obj, "inverse_transform_y"):
                pp = pp_obj  # type: ignore[assignment]
            else:
                raise TypeError(
                    f"Unsupported preprocessor artifact type: {type(pp_obj)}. "
                    "Expected dict (artifact) or Preprocessor instance."
                )

        return cls(model=model, preprocessor=pp)

    # --------------------------------------------------------
    def debug_print_model_inputs(self, feature_inputs: Dict[str, Any]) -> None:
        """
        Print what the model expects vs what the user provided.
        Supports values as float or as {"value": x, "unit": "..."}.
        """
        print("\n--- Inputs expected by model (column -> provided) ---")
        for col in self.input_cols:
            key = _internal_name(col)
            provided = None

            if col in feature_inputs:
                provided = feature_inputs[col]
            elif key in feature_inputs:
                provided = feature_inputs[key]

            if isinstance(provided, dict):
                v = provided.get("value", "<MISSING>")
                u = provided.get("unit", "?")
                provided_str = f"{v} [{u}]"
            elif provided is None:
                if key in self.derived_names:
                    provided_str = "<DERIVED FEATURE, COMPUTED BY PREPROCESSOR>"
                else:
                    provided_str = "<MISSING>"
            else:
                provided_str = str(provided)

            print(f"{col:35s} = {provided_str}")

    # --------------------------------------------------------
    def _convert_user_unit_to_schema_unit(self, value: float, from_unit: str, schema_unit: str) -> float:
        """
        Convert value in 'from_unit' -> canonical SI -> 'schema_unit'.
        This ensures Preprocessor.transform_X(df) receives values in schema user units,
        and can convert to canonical exactly once (no double conversion).
        """
        si = float(convert_user_to_internal_SI(float(value), str(from_unit)))
        out = float(convert_internal_back_to_user(float(si), str(schema_unit)))
        return out

    # --------------------------------------------------------
    def _normalize_feature_inputs_to_schema_units(
        self,
        feature_inputs: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Convert the user's (value, unit) inputs into a simple {col: value_in_schema_unit}
        dict suitable for building a DataFrame.

        Rules:
          - If user provides a raw float: assumed already in schema unit for that column.
          - If user provides {"value": v, "unit": u}:
              convert u -> schema_unit(col) if schema unit exists; else assume u is already acceptable.
          - Accept either unitless keys or legacy __unit keys.
        """
        out: Dict[str, float] = {}

        # Build a list of schema-expected columns (unitless internal names)
        # Use the model's expected input cols if possible.
        expected_cols = [c for c in self.input_cols]

        # First: parse user provided fields
        for k, v in feature_inputs.items():
            key = str(k).strip()
            if v is None:
                continue

            if isinstance(v, dict):
                if "value" not in v:
                    continue
                raw_val = float(v["value"])
                from_unit = str(v.get("unit", "")).strip()

                # Determine schema unit using unitless internal name
                internal = _internal_name(key)
                schema_unit = self.schema_units_by_col.get(internal)

                if schema_unit and from_unit and from_unit != schema_unit:
                    out[internal] = self._convert_user_unit_to_schema_unit(raw_val, from_unit, schema_unit)
                else:
                    # If schema unit is unknown, or units match, keep as-is (assumed compatible)
                    out[internal] = raw_val
            else:
                # Raw float: store under unitless internal name
                raw_val = float(v)
                out[_internal_name(key)] = raw_val

        # Second: map unitless internal keys to the exact column names the model expects (if any legacy names exist)
        # This keeps compatibility with any artifacts that still carry unit-suffixed feature names.
        mapped: Dict[str, float] = {}
        for col in expected_cols:
            internal = _internal_name(col)
            if internal in out:
                mapped[col] = float(out[internal])

        # Also allow any extra user-provided keys that might be required derived columns
        # (if the model expects them and they were provided)
        for k, v in out.items():
            if k in expected_cols:
                mapped[k] = v

        return mapped

    # --------------------------------------------------------
    def _build_input_dataframe(self, feature_inputs: Dict[str, Any]) -> pd.DataFrame:
        """
        Build a 1-row DataFrame in schema user units.
        Preprocessor.transform_X(df) will:
          - convert to canonical SI (if enabled in artifact)
          - compute derived features if needed
          - scale
        """
        normalized = self._normalize_feature_inputs_to_schema_units(feature_inputs)

        # Create DataFrame with provided cols only; Preprocessor will validate required columns exist
        return pd.DataFrame([normalized], dtype="float32")

    # --------------------------------------------------------
    def predict(
        self,
        feature_inputs: Dict[str, Any],
        wear_probability_cfg: Optional[Dict[str, Any]] = None,
        return_units: str = "canonical",   # default is safe
    ) -> Dict[str, float]:
        df_in = self._build_input_dataframe(feature_inputs)

        # Derived features and canonical conversion happen inside transform_X(df)
        X_scaled = self.pp.transform_X(df_in)
        y_out = self.model.predict(X_scaled, verbose=0)

        # ----------------------------------------------------
        # Legacy dict-output model
        # ----------------------------------------------------
        if isinstance(y_out, dict):
            # Best-effort: flatten dict entries
            out_vals: Dict[str, float] = {}
            for k, arr in y_out.items():
                a = np.asarray(arr, dtype="float32")
                out_vals[str(k)] = float(a.reshape(-1)[0])
            return out_vals

        # ----------------------------------------------------
        # Array output (most common)
        # ----------------------------------------------------
        y_scaled = np.asarray(y_out, dtype="float32")
        y = self.pp.inverse_transform_y(y_scaled)

        if y.ndim == 1:
            y = y.reshape(1, -1)

        pred: Dict[str, float] = {
            _internal_name(self.target_cols[i]): float(y[0, i])
            for i in range(min(len(self.target_cols), y.shape[1]))
        }

        # Build prediction dict in CANONICAL units (SI)
        pred: Dict[str, float] = {
            _internal_name(self.target_cols[i]): float(y[0, i])
            for i in range(min(len(self.target_cols), y.shape[1]))
        }

        # ----------------------------------------------------
        # Convert outputs if requested
        # ----------------------------------------------------
        return_units = str(return_units).lower().strip()
        if return_units == "user":
            for k, v in list(pred.items()):
                schema_unit = self.schema_units_by_col.get(k)
                if schema_unit:
                    pred[k] = convert_internal_back_to_user(v, schema_unit)
        elif return_units != "canonical":
            raise ValueError("return_units must be 'canonical' or 'user'")


        # Optional: compute wear probability if hydraulic_wear exists
        if wear_probability_cfg and "hydraulic_wear" in pred:
            method = str(wear_probability_cfg.get("method", "linear")).strip().lower()
            wear_lo = float(wear_probability_cfg.get("wear_lo", 0.0))
            wear_hi = float(wear_probability_cfg.get("wear_hi", 1.0))

            wear = float(pred["hydraulic_wear"])
            if method == "linear":
                # map wear_lo..wear_hi -> 0..100 (clamped)
                denom = max(wear_hi - wear_lo, 1e-12)
                pct = (wear - wear_lo) / denom
                pred["degradation_probability_pct"] = float(np.clip(pct, 0.0, 1.0) * 100.0)

        return pred
