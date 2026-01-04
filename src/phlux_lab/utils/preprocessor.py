from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from phlux_lab.utils.units import convert_user_to_internal_SI


def _resolve_repo_path(p: str) -> str:
    """
    Resolve repo-relative paths robustly for this project structure:
      F:\\phlux\\phlux_lab\\src\\phlux_lab\\utils\\preprocessor.py

    If YAML provides paths like 'phlux_lab/data/...', we resolve relative to repo root (F:\\phlux).
    Otherwise, we try:
      - as-is (cwd relative)
      - repo root join
      - lab root join
    """
    raw = (p or "").strip()
    if not raw:
        return raw

    pp = Path(raw)
    if pp.is_absolute():
        return str(pp)

    # Infer roots from this file location
    this_file = Path(__file__).resolve()
    # .../phlux_lab/src/phlux_lab/utils/preprocessor.py
    repo_root = this_file.parents[4]  # F:\phlux
    lab_root = this_file.parents[3]  # F:\phlux\phlux_lab

    # normalize slashes for Windows
    norm = raw.replace("\\", "/")

    # If YAML already includes 'phlux_lab/...', treat as repo-relative
    if norm.startswith("phlux_lab/"):
        cand = (repo_root / norm).resolve()
        return str(cand)

    # Try cwd-relative first
    cand1 = (Path.cwd() / raw).resolve()
    if cand1.exists():
        return str(cand1)

    # Try repo-root relative
    cand2 = (repo_root / raw).resolve()
    if cand2.exists():
        return str(cand2)

    # Try lab-root relative (sometimes helpful)
    cand3 = (lab_root / raw).resolve()
    return str(cand3)


class Preprocessor:
    """
    Preprocessing for VFM models with:
      - Optional user-units -> canonical (SI) conversion using schema sidecar YAML
      - Derived feature computation (data_cfg["derived_features"])
      - Optional train-time noise augmentation (additive / relative / hybrid)
      - Scaling for X and y

    This module is used from the *package* path:
      phlux_lab/src/phlux_lab/utils/preprocessor.py

    New in stacked workflow:
      - `Preprocessor.from_config()` adapts your training_config.yaml stage blocks into this class.
      - Supports BOTH stage layouts:
          A) legacy/nested:
              models.<stage>.data.train_dataset, input_features, target_features, derived_features, ...
              models.<stage>.model.<hyperparams...>
          B) flat:
              models.<stage>.train_dataset, inputs, targets, stack_inputs, ...
    """

    # ------------------------------
    # Target transform defaults (log1p for flow-like targets)
    # ------------------------------
    ENABLE_FLOW_LOG1P_DEFAULT: bool = True
    _FLOW_INCLUDE_TOKENS = ("flow", "q_liquid", "q_", "rate", "vol_flow", "volume_flow")
    _FLOW_EXCLUDE_TOKENS = ("wear", "prob", "probability", "pct", "percent", "class", "label", "delta_")

    @classmethod
    def from_config(
        cls,
        cfg_one: Dict[str, Any],
        *,
        stage_name: Optional[str] = None,
        dataset: str = "train",
        global_preprocessing: Optional[Dict[str, Any]] = None,
    ) -> "Preprocessor":
        """
        Adapter from training_config.yaml stage block to this Preprocessor.

        Supports:
          - cfg_one["data"]["train_dataset"] (legacy)
          - cfg_one["train_dataset"]         (flat)

        Inputs/targets:
          - legacy: data.input_features + data.stack_inputs (optional)
                    data.target_features
          - flat:   inputs + stack_inputs (optional)
                    targets
        """
        ds = dataset.strip().lower()
        if ds not in ("train", "test"):
            raise ValueError("dataset must be 'train' or 'test'")

        data_block = cfg_one.get("data", {}) if isinstance(cfg_one.get("data", {}), dict) else {}

        # ---- dataset paths ----
        csv_path = None
        if data_block:
            csv_path = data_block.get("train_dataset") if ds == "train" else data_block.get("test_dataset")
        if not csv_path:
            csv_path = cfg_one.get("train_dataset") if ds == "train" else cfg_one.get("test_dataset")
        if not csv_path:
            raise KeyError(f"Stage config missing {'train_dataset' if ds=='train' else 'test_dataset'}")

        # ---- inputs / stack inputs ----
        inputs: List[str] = []
        stack_inputs: List[str] = []
        targets: List[str] = []

        if data_block:
            inputs = list(data_block.get("input_features", []) or [])
            stack_inputs = list(data_block.get("stack_inputs", []) or [])
            targets = list(data_block.get("target_features", []) or [])
        else:
            inputs = list(cfg_one.get("inputs", []) or [])
            stack_inputs = list(cfg_one.get("stack_inputs", []) or [])
            targets = list(cfg_one.get("targets", []) or [])

        # Include stacked inputs in the input_features list
        input_features = inputs.copy()
        for c in stack_inputs:
            if c not in input_features:
                input_features.append(c)

        # ---- derived features ----
        derived_features: List[Dict[str, Any]] = []
        if data_block and data_block.get("derived_features") is not None:
            derived_features = [dict(x) for x in (data_block.get("derived_features") or [])]
        elif global_preprocessing and global_preprocessing.get("feature_engineering") is not None:
            derived_features = [dict(x) for x in (global_preprocessing.get("feature_engineering") or [])]

        # ---- noise ----
        noise_block: Dict[str, Any] = {"enabled": False}
        if data_block and data_block.get("noise") is not None:
            noise_block = dict(data_block.get("noise") or {})
        elif global_preprocessing and global_preprocessing.get("noise") is not None:
            noise_block = dict(global_preprocessing.get("noise") or {})

        # Normalize "pretty" noise format: features: {col: [0, sigma]} into {col: {mode,sigma,clip}}
        if isinstance(noise_block, dict):
            features = noise_block.get("features", {}) or {}
            if isinstance(features, dict):
                normed: Dict[str, Any] = {}
                for k, v in features.items():
                    if isinstance(v, (list, tuple)) and len(v) == 2:
                        _lo, hi = float(v[0]), float(v[1])
                        normed[k] = {"mode": noise_block.get("default_mode", "additive"), "sigma": hi, "clip": None}
                    else:
                        normed[k] = v
                noise_block["features"] = normed
            noise_block.setdefault("apply_to", "train_only")
            noise_block.setdefault("random_seed", 42)
            noise_block.setdefault("default_mode", "additive")

        # ---- schema / unit policy ----
        schema_path = ""
        unit_policy: Dict[str, Any] = {}
        if data_block:
            schema_path = str(data_block.get("schema_path", "") or "")
            unit_policy = dict(data_block.get("unit_policy", {}) or {})

        csv_path = _resolve_repo_path(str(csv_path))
        data_cfg: Dict[str, Any] = {
            "stage_name": str(stage_name or cfg_one.get("stage_name", "") or ""),
            "csv_path": str(csv_path),
            "input_features": input_features,
            "targets": targets,
            "schema_path": schema_path,
            "unit_policy": unit_policy,
            "derived_features": derived_features,
            "noise": noise_block,
            "enable_flow_log1p": bool(
                (data_block.get("enable_flow_log1p") if data_block else None)
                if (data_block.get("enable_flow_log1p") if data_block else None) is not None
                else cfg_one.get("enable_flow_log1p", cls.ENABLE_FLOW_LOG1P_DEFAULT)
            ),
        }

        return cls(data_cfg)

    # ------------------------------
    # Init / Load (training)
    # ------------------------------
    def __init__(self, data_cfg: Dict[str, Any]) -> None:
        # Required config
        self.csv_path: str = data_cfg["csv_path"]
        self.input_features: List[str] = list(data_cfg["input_features"])
        self.target_features: List[str] = list(data_cfg["targets"])

        # Stage name (for stage-scoped feature engineering)
        self.stage_name: str = str(data_cfg.get("stage_name", "") or "").strip()

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

        # Target-transform bookkeeping (stored in artifact)
        self.enable_flow_log1p: bool = bool(data_cfg.get("enable_flow_log1p", self.ENABLE_FLOW_LOG1P_DEFAULT))
        self._log1p_target_indices: List[int] = []

        # Load CSV
        self.df: pd.DataFrame = pd.read_csv(self.csv_path)

        # Validate base columns exist (before conversion/derived)
        missing_inputs = [c for c in self.input_features if c not in self.df.columns]
        missing_targets = [c for c in self.target_features if c not in self.df.columns]
        if missing_inputs or missing_targets:
            raise ValueError(
                "Missing columns in CSV.\n"
                f"  Missing inputs: {missing_inputs}\n"
                f"  Missing targets: {missing_targets}\n"
                f"  CSV path: {self.csv_path}\n"
                f"  Available columns: {list(self.df.columns)}"
            )

        # 1) Convert USER -> CANONICAL (SI) before any derived features or scaling
        if self._should_convert_inputs_now():
            self._convert_columns_user_to_canonical_inplace(self.df, self.input_features)

        if self._should_convert_targets_now():
            self._convert_columns_user_to_canonical_inplace(self.df, self.target_features)

        # 2) Derived features
        if self.derived_cfg:
            self._add_derived_features_inplace(self.df)

        # Re-validate after derived
        missing_inputs2 = [c for c in self.input_features if c not in self.df.columns]
        missing_targets2 = [c for c in self.target_features if c not in self.df.columns]
        if missing_inputs2 or missing_targets2:
            raise ValueError(
                "Missing columns after preprocessing.\n"
                f"  Missing inputs: {missing_inputs2}\n"
                f"  Missing targets: {missing_targets2}\n"
                f"  CSV path: {self.csv_path}\n"
                f"  Available columns: {list(self.df.columns)}"
            )

        # Extract arrays
        X_raw = self.df[self.input_features].to_numpy(dtype="float32")
        y_raw = self.df[self.target_features].to_numpy(dtype="float32")

        # 3) Noise on X during fit (if enabled)
        X_aug = self._maybe_apply_noise_to_X(X_raw, feature_columns=self.input_features, stage="fit")

        # 3.5) Target transform
        y_for_scaling = self._transform_targets_for_fit(y_raw)

        # 4) Fit scalers
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.x_input_scaled = self.scaler_X.fit_transform(X_aug)
        self.y_output_scaled = self.scaler_y.fit_transform(y_for_scaling)

    # ------------------------------
    # Target transforms (log1p flow-like targets)
    # ------------------------------
    def _infer_log1p_target_indices(self) -> List[int]:
        idxs: List[int] = []
        if not self.enable_flow_log1p:
            return idxs

        for i, name in enumerate(self.target_features):
            n = str(name).strip().lower()
            if not n:
                continue
            if any(tok in n for tok in self._FLOW_EXCLUDE_TOKENS):
                continue
            if any(tok in n for tok in self._FLOW_INCLUDE_TOKENS):
                idxs.append(i)
        return idxs

    def _transform_targets_for_fit(self, y_raw: np.ndarray) -> np.ndarray:
        if y_raw.ndim != 2:
            raise ValueError(f"Targets must be 2D (n, y_dim). Got shape {y_raw.shape}")

        self._log1p_target_indices = self._infer_log1p_target_indices()
        if not self._log1p_target_indices:
            return y_raw

        y = y_raw.astype("float32", copy=True)
        for j in self._log1p_target_indices:
            col = np.maximum(y[:, j], 0.0).astype("float32")
            y[:, j] = np.log1p(col).astype("float32")
        return y

    def _inverse_target_transforms_after_inverse_scaling(self, y_inv: np.ndarray) -> np.ndarray:
        if y_inv.ndim != 2:
            raise ValueError(f"Inverse targets must be 2D (n, y_dim). Got shape {y_inv.shape}")

        if not getattr(self, "_log1p_target_indices", None):
            return y_inv

        y = y_inv.astype("float32", copy=True)
        for j in self._log1p_target_indices:
            y[:, j] = np.expm1(y[:, j]).astype("float32")
        return y

    # ------------------------------
    # Schema helpers
    # ------------------------------
    @staticmethod
    def _load_schema_units(schema_path: str) -> Dict[str, str]:
        import yaml

        with open(schema_path, "r", encoding="utf-8") as f:
            schema = yaml.safe_load(f) or {}
        units: Dict[str, str] = {}
        for item in schema.get("data", []) or []:
            name = item.get("name")
            unit = item.get("unit")
            if name and unit:
                units[str(name)] = str(unit)
        return units

    def _should_convert_inputs_now(self) -> bool:
        return bool(self.convert_inputs_to_canonical and self.schema_path)

    def _should_convert_targets_now(self) -> bool:
        return bool(self.convert_targets_to_canonical and self.schema_path)

    def _convert_columns_user_to_canonical_inplace(self, df: pd.DataFrame, cols: List[str]) -> None:
        for c in cols:
            if c not in df.columns:
                continue
            user_unit = self._schema_units_by_col.get(c)
            if not user_unit:
                continue
            df[c] = convert_user_to_internal_SI(df[c].to_numpy(dtype="float32"), user_unit)

    # ------------------------------
    # Derived features
    # ------------------------------
    def _add_derived_features_inplace(self, df: pd.DataFrame) -> None:
        """Add derived / engineered features defined in config.

        Supports stage scoping via spec["apply_to"].

        Behavior:
          - If spec has apply_to (list/str) and current stage_name is not included -> skip.
          - If required input columns are missing -> skip (common for *_pred early stages).
          - Only append derived feature to self.input_features if it applies to this stage.
        """
        stage = (self.stage_name or "").strip()
        for spec in self.derived_cfg:
            name = spec.get("name")
            op_type = str(spec.get("type", "")).strip().lower()
            inputs = list(spec.get("inputs", []) or [])
            eps = float(spec.get("eps", 0.0) or 0.0)

            if not name or not op_type or len(inputs) < 2:
                continue

            # ---- stage scoping (optional) ----
            apply_to = spec.get("apply_to", None)
            if apply_to is not None:
                if isinstance(apply_to, str):
                    apply_list = [apply_to]
                elif isinstance(apply_to, (list, tuple)):
                    apply_list = list(apply_to)
                else:
                    apply_list = []
                apply_list = [str(x).strip() for x in apply_list if str(x).strip()]
                if stage and apply_list and stage not in apply_list:
                    continue

            a, b = inputs[0], inputs[1]

            # If required inputs are missing, skip this derived feature.
            if a not in df.columns or b not in df.columns:
                continue

            if op_type == "subtract":
                df[name] = df[a].astype("float32") - df[b].astype("float32")
            elif op_type == "divide":
                denom = df[b].astype("float32")
                if eps != 0.0:
                    denom = denom + float(eps)
                df[name] = df[a].astype("float32") / denom
            else:
                raise ValueError(f"Unsupported derived feature type: {op_type}")

            # Ensure derived features are available as model inputs *for this stage*.
            if name not in self.input_features:
                self.input_features.append(name)

    # ------------------------------
    # Noise helpers
    # ------------------------------
    def _get_feature_noise_cfg(self, feature: str) -> Optional[Dict[str, Any]]:
        cfg = self.noise_features.get(feature)
        if not cfg:
            return None
        if isinstance(cfg, dict):
            return cfg
        # if someone gave a scalar, treat as sigma additive
        try:
            return {"mode": self.default_noise_mode, "sigma": float(cfg), "clip": None}
        except Exception:
            return None

    def _compute_sigma(self, col: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
        mode = str(cfg.get("mode", self.default_noise_mode)).strip().lower()
        sigma = float(cfg.get("sigma", 0.0) or 0.0)

        if sigma <= 0.0:
            return np.zeros_like(col, dtype="float32")

        if mode == "additive":
            return np.full_like(col, sigma, dtype="float32")
        if mode == "relative":
            return (np.abs(col).astype("float32") * sigma).astype("float32")
        if mode == "hybrid":
            base = np.full_like(col, sigma, dtype="float32")
            rel = (np.abs(col).astype("float32") * sigma).astype("float32")
            return (base + rel).astype("float32")

        return np.full_like(col, sigma, dtype="float32")

    @staticmethod
    def _apply_clip_if_configured(
        x: np.ndarray, clip: Optional[Union[List[float], Tuple[float, float]]]
    ) -> np.ndarray:
        if not clip:
            return x.astype("float32")
        lo = float(clip[0]) if clip[0] is not None else -np.inf
        hi = float(clip[1]) if clip[1] is not None else np.inf
        return np.clip(x, lo, hi).astype("float32")

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

        return Xn.astype("float32")

    # ------------------------------
    # Public transforms
    # ------------------------------
    def transform_X(
        self,
        x_raw: Union[pd.DataFrame, np.ndarray, List[List[float]]],
        *,
        apply_noise: bool = False,
    ) -> np.ndarray:
        if isinstance(x_raw, pd.DataFrame):
            df = x_raw.copy()

            if self._should_convert_inputs_now():
                self._convert_columns_user_to_canonical_inplace(df, [c for c in self.input_features if c in df.columns])

            if self.derived_cfg:
                self._add_derived_features_inplace(df)

            missing = [c for c in self.input_features if c not in df.columns]
            if missing:
                raise KeyError(
                    "Input DataFrame is missing required columns for transform_X:\n"
                    + "\n".join([f"  - {c}" for c in missing])
                )
            X = df[self.input_features].to_numpy(dtype="float32")
        else:
            X = np.asarray(x_raw, dtype="float32")
            if X.ndim != 2:
                raise ValueError(f"transform_X expects 2D array; got shape {X.shape}")
            if X.shape[1] != len(self.input_features):
                raise ValueError(
                    f"transform_X received {X.shape[1]} cols, artifact expects {len(self.input_features)}: {self.input_features}"
                )

        if apply_noise:
            X = self._maybe_apply_noise_to_X(X, self.input_features, stage="transform", enable_override=True)

        return self.scaler_X.transform(X)

    def inverse_transform_y(self, y_scaled: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        y_scaled_arr = np.asarray(y_scaled, dtype="float32")
        y_inv = self.scaler_y.inverse_transform(y_scaled_arr)
        return self._inverse_target_transforms_after_inverse_scaling(y_inv)

    # ------------------------------
    # Introspection helpers
    # ------------------------------
    def get_input_feature_names(self) -> List[str]:
        return list(self.input_features)

    def get_target_feature_names(self) -> List[str]:
        return list(self.target_features)

    # ------------------------------
    # Artifact helpers (save/load)
    # ------------------------------
    def to_artifact(self) -> Dict[str, Any]:
        return {
            "stage_name": str(getattr(self, "stage_name", "")),
            "input_features": list(self.input_features),
            "target_features": list(self.target_features),
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "derived_features": list(self.derived_cfg),
            "schema_units_by_col": dict(self._schema_units_by_col),
            "unit_policy": dict(self.unit_policy),
            "canonical_internal_system": self.canonical_internal_system,
            "enable_flow_log1p": bool(getattr(self, "enable_flow_log1p", self.ENABLE_FLOW_LOG1P_DEFAULT)),
            "log1p_target_indices": list(getattr(self, "_log1p_target_indices", []) or []),
        }

    def save_artifact(self, path: str) -> str:
        try:
            import joblib  # type: ignore
        except ImportError as e:
            raise RuntimeError("joblib is required. Install with: pip install joblib") from e

        import os

        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        joblib.dump(self.to_artifact(), abs_path)
        return abs_path

    def save(self, path: str) -> str:
        return self.save_artifact(path)

    @classmethod
    def load(cls, path: str) -> "Preprocessor":
        try:
            import joblib  # type: ignore
        except ImportError as e:
            raise RuntimeError("joblib is required. Install with: pip install joblib") from e
        artifact = joblib.load(path)
        return cls.from_artifact(artifact)

    @classmethod
    def from_artifact(cls, artifact: Dict[str, Any]) -> "Preprocessor":
        required = {"input_features", "target_features", "scaler_X", "scaler_y"}
        missing = required.difference(artifact.keys())
        if missing:
            raise ValueError(f"Artifact missing keys: {sorted(missing)}")

        self = cls.__new__(cls)
        self.csv_path = ""
        self.df = pd.DataFrame()

        self.stage_name = str(artifact.get("stage_name", "") or "").strip()

        self.input_features = list(artifact["input_features"])
        self.target_features = list(artifact["target_features"])
        self.scaler_X = artifact["scaler_X"]
        self.scaler_y = artifact["scaler_y"]

        self.derived_cfg = list(artifact.get("derived_features", []) or [])
        self._schema_units_by_col = dict(artifact.get("schema_units_by_col", {}) or {})
        self.unit_policy = dict(artifact.get("unit_policy", {}) or {})
        self.convert_inputs_to_canonical = bool(self.unit_policy.get("convert_inputs_to_canonical", False))
        self.convert_targets_to_canonical = bool(self.unit_policy.get("convert_targets_to_canonical", False))
        self.canonical_internal_system = str(artifact.get("canonical_internal_system", "SI"))

        self.enable_flow_log1p = bool(artifact.get("enable_flow_log1p", cls.ENABLE_FLOW_LOG1P_DEFAULT))
        self._log1p_target_indices = list(artifact.get("log1p_target_indices", []) or [])

        # noise config not needed for inference; disable
        self.noise_cfg = {"enabled": False}
        self.noise_enabled = False
        self.noise_apply_to = "none"
        self.noise_seed = 42
        self.noise_features = {}
        self.default_noise_mode = "additive"
        self._noise_rng = np.random.default_rng(self.noise_seed)

        # schema path not needed at inference (units already canonicalized in training arrays)
        self.schema_path = ""
        self.convert_inputs_to_canonical = False
        self.convert_targets_to_canonical = False

        return self
