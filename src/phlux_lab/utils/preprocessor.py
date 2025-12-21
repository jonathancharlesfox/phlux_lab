from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from phlux_lab.utils.units import convert_user_to_internal_SI


class Preprocessor:
    """
    Preprocessing for VFM models with:
      - Optional user-units -> canonical (SI) conversion using schema sidecar YAML
      - Derived feature computation (data_cfg["derived_features"])
      - Optional train-time noise augmentation (additive / relative / hybrid)
      - Scaling for X and y

    Canonical approach:
      - CSV has unitless column names (e.g., suction_pressure)
      - CSV values are in USER units (as exported)
      - schema.yaml stores the unit for each column
      - Preprocessor converts to CANONICAL internal SI BEFORE scaling/training
    """

    # ------------------------------
    # Init / Load (training)
    # ------------------------------
    def __init__(self, data_cfg: Dict[str, Any]) -> None:
        # Required config
        self.csv_path: str = data_cfg["csv_path"]
        self.input_features: List[str] = list(data_cfg["input_features"])
        self.target_features: List[str] = list(data_cfg["targets"])

        # Optional: schema + unit policy
        self.schema_path: str = str(data_cfg.get("schema_path", "") or "")
        self.unit_policy: Dict[str, Any] = dict(data_cfg.get("unit_policy", {}) or {})
        self.convert_inputs_to_canonical: bool = bool(self.unit_policy.get("convert_inputs_to_canonical", False))
        self.convert_targets_to_canonical: bool = bool(self.unit_policy.get("convert_targets_to_canonical", False))
        self.canonical_internal_system: str = str(self.unit_policy.get("canonical_internal_system", "SI"))

        # Optional derived features config
        self.derived_cfg: List[Dict[str, Any]] = list(data_cfg.get("derived_features", []) or [])

        # Optional noise config
        self.noise_cfg: Dict[str, Any] = dict(data_cfg.get("noise", {}) or {})
        self.noise_enabled: bool = bool(self.noise_cfg.get("enabled", False))
        self.noise_apply_to: str = str(self.noise_cfg.get("apply_to", "train_only")).strip().lower()
        self.noise_seed: int = int(self.noise_cfg.get("random_seed", 42))
        self.noise_features: Dict[str, Any] = dict(self.noise_cfg.get("features", {}) or {})
        self.default_noise_mode: str = str(self.noise_cfg.get("default_mode", "additive")).strip().lower()
        self._noise_rng = np.random.default_rng(self.noise_seed)

        # Schema mapping: column_name -> user_unit (as stored in schema)
        self._schema_units_by_col: Dict[str, str] = {}
        if self.schema_path:
            self._schema_units_by_col = self._load_schema_units(self.schema_path)

        # Derived bookkeeping (also used post-noise)
        self._derived_clip_by_col: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        self._derived_units_by_col: Dict[str, str] = {}  # user-provided unit label (metadata only)

        # Load CSV (user-units)
        self.df: pd.DataFrame = pd.read_csv(self.csv_path)

        # Validate base columns exist (before conversion/derived)
        missing_inputs = [c for c in self.input_features if c not in self.df.columns]
        missing_targets = [c for c in self.target_features if c not in self.df.columns]
        if missing_inputs or missing_targets:
            raise ValueError(
                "Missing columns in CSV.\n"
                f"  Missing inputs: {missing_inputs}\n"
                f"  Missing targets: {missing_targets}\n"
                f"  Available columns: {list(self.df.columns)}"
            )

        # 1) Convert USER -> CANONICAL (SI) before any derived features or scaling
        #    (Derived features should be computed in canonical space for correctness.)
        if self._should_convert_inputs_now():
            self._convert_columns_user_to_canonical_inplace(self.df, self.input_features)

        if self._should_convert_targets_now():
            self._convert_columns_user_to_canonical_inplace(self.df, self.target_features)

        # 2) Compute derived features in canonical space (adds unitless columns into df)
        if self.derived_cfg:
            self._add_derived_features_inplace(self.df)

        # Re-validate after derived (derived might append to inputs)
        missing_inputs2 = [c for c in self.input_features if c not in self.df.columns]
        missing_targets2 = [c for c in self.target_features if c not in self.df.columns]
        if missing_inputs2 or missing_targets2:
            raise ValueError(
                "Missing columns after preprocessing.\n"
                f"  Missing inputs: {missing_inputs2}\n"
                f"  Missing targets: {missing_targets2}\n"
                f"  Available columns: {list(self.df.columns)}"
            )

        # Extract raw arrays (canonical SI if conversion enabled)
        X_raw = self.df[self.input_features].to_numpy(dtype="float32")
        y_raw = self.df[self.target_features].to_numpy(dtype="float32")

        # 3) Apply noise augmentation (train stage) BEFORE scaling
        X_aug = self._maybe_apply_noise_to_X(
            X_raw,
            feature_columns=self.input_features,
            stage="fit",
        )

        # 4) Fit scalers
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        self.x_input_scaled = self.scaler_X.fit_transform(X_aug)
        self.y_output_scaled = self.scaler_y.fit_transform(y_raw)

    # ------------------------------
    # Schema helpers
    # ------------------------------
    @staticmethod
    def _load_schema_units(schema_path: str) -> Dict[str, str]:
        import yaml

        with open(schema_path, "r", encoding="utf-8") as f:
            schema = yaml.safe_load(f) or {}

        cols = schema.get("columns", []) or []
        units_by_col: Dict[str, str] = {}
        for c in cols:
            name = str(c.get("name", "")).strip()
            unit = str(c.get("user_unit", "")).strip()
            if name:
                units_by_col[name] = unit
        return units_by_col

    def _should_convert_inputs_now(self) -> bool:
        # Only convert if explicitly enabled AND we have schema units for the columns.
        if not self.convert_inputs_to_canonical:
            return False
        if not self._schema_units_by_col:
            return False
        return True

    def _should_convert_targets_now(self) -> bool:
        if not self.convert_targets_to_canonical:
            return False
        if not self._schema_units_by_col:
            return False
        return True

    def _convert_columns_user_to_canonical_inplace(self, df: pd.DataFrame, cols: List[str]) -> None:
        """
        Convert the provided df columns from USER units -> CANONICAL internal SI basis in-place.

        Uses schema.yaml mapping:
          col_name -> user_unit

        Conversion function signature is:
          convert_user_to_internal_SI(value, unit)
        """
        for col in cols:
            if col not in df.columns:
                continue

            user_unit = self._schema_units_by_col.get(col)
            if not user_unit:
                # If schema is missing a unit for this column, we skip to avoid silently guessing.
                # Better to fail early than train on mixed units.
                raise KeyError(
                    f"Schema '{self.schema_path}' is missing user_unit for column '{col}'. "
                    f"Either add it to schema, or disable convert_*_to_canonical."
                )

            # Convert elementwise; keep NaNs
            df[col] = df[col].apply(lambda v: convert_user_to_internal_SI(float(v), user_unit) if pd.notna(v) else v)

    # ------------------------------
    # Backward-compatible column helpers
    # ------------------------------
    @staticmethod
    def _base_name(col: str) -> str:
        return col.split("__", 1)[0] if "__" in col else col

    @staticmethod
    def _is_full_col_name(s: str) -> bool:
        return "__" in s

    def _resolve_input_token_to_col(self, token: str, df: pd.DataFrame) -> str:
        """
        Resolve a derived input token to an actual column name in df.

        Supports:
          - New unitless: "discharge_pressure"
          - Old unit-suffixed: "discharge_pressure__bar" (backward compatibility)
        """
        token = str(token).strip()
        if not token:
            raise ValueError("Empty input token in derived feature definition.")

        # If token is a full unit-suffixed name, accept if present
        if self._is_full_col_name(token) and token in df.columns:
            return token

        # If token is unitless and present, use it
        if token in df.columns:
            return token

        # Backward-compat: if token is unitless but df has discharge_pressure__<unit>
        base = self._base_name(token)
        matches = [c for c in df.columns if c.startswith(base + "__")]
        if matches:
            return matches[0]

        raise KeyError(
            f"Could not resolve derived input token '{token}' to a dataframe column. "
            f"Available columns: {list(df.columns)}"
        )

    # ------------------------------
    # Derived features
    # ------------------------------
    def _add_derived_features_inplace(self, df: pd.DataFrame) -> None:
        """
        Adds derived columns to df in-place, updates self.input_features,
        and stores derived clip bounds so they can be re-applied after noise.

        NOTE (new canonical approach):
          - Derived output column name is unitless: out_col = name
          - 'unit' in the YAML is treated as metadata only (for documentation)
          - Computation is done in whatever units are currently in df.
            With convert_inputs_to_canonical=True, df is canonical SI at this point.
        """
        for d in self.derived_cfg:
            name = str(d.get("name", "")).strip()
            unit = str(d.get("unit", "")).strip()  # metadata only
            op = str(d.get("op", "")).strip().lower()
            inputs = d.get("inputs", [])

            if not name:
                raise ValueError(f"Derived feature must include 'name'. Got: {d}")
            if not op:
                raise ValueError(f"Derived feature '{name}' missing 'op'. Got: {d}")
            if not isinstance(inputs, list) or len(inputs) == 0:
                raise ValueError(f"Derived feature '{name}' must define non-empty 'inputs' list. Got: {d}")

            out_col = name

            # If already present, don't recompute (allows pre-materialized CSVs)
            if out_col in df.columns:
                if out_col not in self.input_features and out_col not in self.target_features:
                    self.input_features.append(out_col)
                if unit:
                    self._derived_units_by_col[out_col] = unit
                continue

            # Resolve input columns (accept unitless or unit-suffixed)
            in_cols = [self._resolve_input_token_to_col(tok, df) for tok in inputs]

            # Extract arrays
            arrs = [df[c].to_numpy(dtype="float32") for c in in_cols]

            # Compute operation
            out = self._apply_op(op, arrs, context=f"derived_features:{out_col}")

            # Optional post-op
            post_op = d.get("post_op")
            if post_op is not None:
                out = self._apply_post_op(str(post_op).strip().lower(), out, context=f"derived_features:{out_col}")

            # Optional clip on derived feature
            clip_cfg = d.get("clip")
            lo, hi = None, None
            if isinstance(clip_cfg, dict):
                lo = clip_cfg.get("min", None)
                hi = clip_cfg.get("max", None)
                if lo is not None:
                    lo = float(lo)
                if hi is not None:
                    hi = float(hi)
                out = np.clip(
                    out,
                    lo if lo is not None else -np.inf,
                    hi if hi is not None else np.inf,
                ).astype("float32")

            # Save in df
            df[out_col] = out

            # Remember clip bounds so we can re-apply after noise too
            if lo is not None or hi is not None:
                self._derived_clip_by_col[out_col] = (lo, hi)

            # Save unit metadata
            if unit:
                self._derived_units_by_col[out_col] = unit

            # Append to inputs if not already present and not a target
            if out_col not in self.input_features and out_col not in self.target_features:
                self.input_features.append(out_col)

    def _apply_op(self, op: str, arrs: List[np.ndarray], context: str) -> np.ndarray:
        op = op.lower().strip()

        unary_ops = {"abs", "log", "log10", "sqrt", "square"}
        if op in unary_ops:
            if len(arrs) != 1:
                raise ValueError(f"{context}: op '{op}' requires exactly 1 input, got {len(arrs)}")
            return self._apply_post_op(op, arrs[0], context=context)

        if op in {"add", "subtract", "multiply", "divide"}:
            if len(arrs) != 2:
                raise ValueError(f"{context}: op '{op}' requires exactly 2 inputs, got {len(arrs)}")
            a, b = arrs[0], arrs[1]
            if op == "add":
                return (a + b).astype("float32")
            if op == "subtract":
                return (a - b).astype("float32")
            if op == "multiply":
                return (a * b).astype("float32")
            if op == "divide":
                eps = 1e-12
                return (a / (b + eps)).astype("float32")

        if op in {"sum", "mean", "min", "max"}:
            stack = np.vstack([x.reshape(1, -1) for x in arrs]).astype("float32")
            if op == "sum":
                return np.sum(stack, axis=0).astype("float32")
            if op == "mean":
                return np.mean(stack, axis=0).astype("float32")
            if op == "min":
                return np.min(stack, axis=0).astype("float32")
            if op == "max":
                return np.max(stack, axis=0).astype("float32")

        raise ValueError(
            f"{context}: Unknown op '{op}'. Supported: add/subtract/multiply/divide/sum/mean/min/max "
            f"(+ unary: abs/log/log10/sqrt/square)."
        )

    def _apply_post_op(self, post_op: str, x: np.ndarray, context: str) -> np.ndarray:
        post_op = post_op.lower().strip()

        if post_op == "abs":
            return np.abs(x).astype("float32")
        if post_op == "square":
            return (x * x).astype("float32")
        if post_op == "sqrt":
            return np.sqrt(np.maximum(x, 0.0)).astype("float32")
        if post_op == "log":
            eps = 1e-12
            return np.log(np.maximum(x, eps)).astype("float32")
        if post_op == "log10":
            eps = 1e-12
            return np.log10(np.maximum(x, eps)).astype("float32")

        raise ValueError(f"{context}: Unknown post_op '{post_op}'. Supported: abs/square/sqrt/log/log10")

    # ------------------------------
    # Noise support (absolute/relative/hybrid + clip)
    # ------------------------------
    def _get_feature_noise_cfg(self, col: str) -> Optional[Dict[str, Any]]:
        base = self._base_name(col)
        cfg = self.noise_features.get(col) or self.noise_features.get(base)
        if cfg is None:
            return None

        if isinstance(cfg, (int, float)):
            return {"mode": "additive", "abs_std": float(cfg)}

        if isinstance(cfg, dict):
            return dict(cfg)

        return None

    def _compute_sigma(self, x: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
        mode = str(cfg.get("mode", self.default_noise_mode)).strip().lower()

        if mode == "additive":
            return np.full_like(x, float(cfg.get("abs_std", 0.0)), dtype="float32")

        if mode == "relative":
            rel = float(cfg.get("rel_std", 0.0))
            return (np.abs(x) * rel).astype("float32")

        if mode == "hybrid":
            rel = float(cfg.get("rel_std", 0.0))
            abs_min = float(cfg.get("abs_min_std", 0.0))
            sig = np.abs(x) * rel
            return np.maximum(sig, abs_min).astype("float32")

        raise ValueError(f"Unknown noise mode '{mode}'. Supported: additive/relative/hybrid")

    def _apply_clip_if_configured(self, x: np.ndarray, clip_cfg: Any) -> np.ndarray:
        if isinstance(clip_cfg, dict):
            lo = clip_cfg.get("min", None)
            hi = clip_cfg.get("max", None)
            lo_f = float(lo) if lo is not None else None
            hi_f = float(hi) if hi is not None else None
            return np.clip(
                x,
                lo_f if lo_f is not None else -np.inf,
                hi_f if hi_f is not None else np.inf,
            ).astype("float32")
        return x

    def _apply_derived_clips_post_noise(self, X: np.ndarray, feature_columns: List[str]) -> np.ndarray:
        if not self._derived_clip_by_col:
            return X
        Xn = X
        col_index = {c: i for i, c in enumerate(feature_columns)}
        for col, (lo, hi) in self._derived_clip_by_col.items():
            if col not in col_index:
                continue
            j = col_index[col]
            Xn[:, j] = np.clip(
                Xn[:, j],
                lo if lo is not None else -np.inf,
                hi if hi is not None else np.inf,
            ).astype("float32")
        return Xn

    def _maybe_apply_noise_to_X(
        self,
        X: np.ndarray,
        feature_columns: List[str],
        stage: str,
        enable_override: Optional[bool] = None,
    ) -> np.ndarray:
        enabled = bool(enable_override) if enable_override is not None else self.noise_enabled
        if not enabled or self.noise_apply_to == "none":
            return X
        if self.noise_apply_to == "train_only" and stage != "fit":
            return X

        Xn = X.copy()

        for j, col in enumerate(feature_columns):
            cfg = self._get_feature_noise_cfg(col)
            if not cfg:
                continue

            sigma = self._compute_sigma(Xn[:, j], cfg)
            if not np.any(sigma > 0):
                Xn[:, j] = self._apply_clip_if_configured(Xn[:, j], cfg.get("clip"))
                continue

            noise = self._noise_rng.normal(loc=0.0, scale=sigma, size=Xn[:, j].shape).astype("float32")
            Xn[:, j] = (Xn[:, j] + noise).astype("float32")
            Xn[:, j] = self._apply_clip_if_configured(Xn[:, j], cfg.get("clip"))

        # Re-apply derived bounds after noise (e.g., valve_dp >= 0)
        Xn = self._apply_derived_clips_post_noise(Xn, feature_columns)
        return Xn

    # ------------------------------
    # Public transforms
    # ------------------------------
    def transform_X(
        self,
        x_raw: Union[pd.DataFrame, np.ndarray, List[List[float]]],
        *,
        apply_noise: bool = False,
    ) -> np.ndarray:
        """
        Transform raw features into scaled features.

        Behavior (recommended):
          - If x_raw is a DataFrame, we assume it's USER units and convert to canonical SI (if enabled),
            then compute derived features, then scale.
          - If x_raw is NOT a DataFrame, we assume it's already in the same numeric space as the artifact
            expects (canonical SI if conversion was enabled during training).
        """
        if isinstance(x_raw, pd.DataFrame):
            df = x_raw.copy()

            # Convert user -> canonical for the base inputs (only those used by the model)
            if self._should_convert_inputs_now():
                self._convert_columns_user_to_canonical_inplace(df, [c for c in self.input_features if c in df.columns])

            # Compute derived columns if missing
            if self.derived_cfg:
                self._add_derived_features_inplace(df)

            missing = [c for c in self.input_features if c not in df.columns]
            if missing:
                raise KeyError(
                    "Input DataFrame is missing required columns for transform_X (must match artifact inputs):\n"
                    + "\n".join([f"  - {c}" for c in missing])
                )

            X = df[self.input_features].to_numpy(dtype="float32")
        else:
            X = np.asarray(x_raw, dtype="float32")
            if X.ndim != 2:
                raise ValueError(f"transform_X expects a 2D array; got shape {X.shape}")
            if X.shape[1] != len(self.input_features):
                raise ValueError(
                    f"transform_X received array with {X.shape[1]} columns, but artifact expects "
                    f"{len(self.input_features)} columns: {self.input_features}"
                )

        if apply_noise:
            X = self._maybe_apply_noise_to_X(X, self.input_features, stage="transform", enable_override=True)

        return self.scaler_X.transform(X)

    def inverse_transform_y(self, y_scaled: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        y_scaled_arr = np.asarray(y_scaled, dtype="float32")
        return self.scaler_y.inverse_transform(y_scaled_arr)

    # ------------------------------
    # Accessors
    # ------------------------------
    def get_input_feature_names(self) -> List[str]:
        return list(self.input_features)

    def get_target_feature_names(self) -> List[str]:
        return list(self.target_features)

    # ------------------------------
    # Artifact helpers
    # ------------------------------
    def to_artifact(self) -> Dict[str, Any]:
        """
        Store fitted scalers, feature names, derived feature specs, AND unit conversion metadata.
        """
        return {
            "input_features": list(self.input_features),
            "target_features": list(self.target_features),
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "derived_features": list(self.derived_cfg),
            "derived_clip_by_col": dict(self._derived_clip_by_col),
            "derived_units_by_col": dict(self._derived_units_by_col),

            # unit conversion metadata for runtime
            "schema_units_by_col": dict(self._schema_units_by_col),
            "unit_policy": dict(self.unit_policy),
            "canonical_internal_system": self.canonical_internal_system,
        }

    def save_artifact(self, path: str) -> str:
        try:
            import joblib  # type: ignore
        except ImportError as e:
            raise RuntimeError("joblib is required to save artifacts. Install with: pip install joblib") from e

        import os
        abs_path = os.path.abspath(path)
        joblib.dump(self.to_artifact(), abs_path)
        return abs_path

    @classmethod
    def from_artifact(cls, artifact: Dict[str, Any]) -> "Preprocessor":
        required = {"input_features", "target_features", "scaler_X", "scaler_y"}
        missing = required.difference(artifact.keys())
        if missing:
            raise ValueError(f"Artifact missing keys: {sorted(missing)}")

        self = cls.__new__(cls)
        self.csv_path = ""
        self.df = pd.DataFrame()

        self.input_features = list(artifact["input_features"])
        self.target_features = list(artifact["target_features"])
        self.scaler_X = artifact["scaler_X"]
        self.scaler_y = artifact["scaler_y"]

        # Derived specs
        self.derived_cfg = list(artifact.get("derived_features", []) or [])
        self._derived_clip_by_col = dict(artifact.get("derived_clip_by_col", {}) or {})
        self._derived_units_by_col = dict(artifact.get("derived_units_by_col", {}) or {})

        # Unit conversion metadata
        self.unit_policy = dict(artifact.get("unit_policy", {}) or {})
        self.canonical_internal_system = str(artifact.get("canonical_internal_system", "SI"))
        self.convert_inputs_to_canonical = bool(self.unit_policy.get("convert_inputs_to_canonical", False))
        self.convert_targets_to_canonical = bool(self.unit_policy.get("convert_targets_to_canonical", False))
        self.schema_path = ""  # not needed at runtime; artifact contains mapping
        self._schema_units_by_col = dict(artifact.get("schema_units_by_col", {}) or {})

        # Disable noise for inference/eval
        self.noise_cfg = {}
        self.noise_enabled = False
        self.noise_apply_to = "none"
        self.noise_seed = 0
        self.noise_features = {}
        self.default_noise_mode = "additive"
        self._noise_rng = np.random.default_rng(0)

        self.x_input_scaled = np.empty((0, len(self.input_features)), dtype="float32")
        self.y_output_scaled = np.empty((0, len(self.target_features)), dtype="float32")

        return self

    @classmethod
    def load_artifact(cls, path: str) -> "Preprocessor":
        try:
            import joblib  # type: ignore
        except ImportError as e:
            raise RuntimeError("joblib is required to load artifacts. Install with: pip install joblib") from e

        artifact = joblib.load(path)
        if not isinstance(artifact, dict):
            raise ValueError(f"Loaded artifact is not a dict: {type(artifact)}")
        return cls.from_artifact(artifact)
