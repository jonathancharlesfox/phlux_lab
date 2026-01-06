from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Path resolving helpers
# =============================================================================

def _resolve_repo_path(raw: str) -> str:
    """
    Resolve a repo-relative path string into an absolute path string.
    Assumptions match your project layout:
      repo_root/
        phlux_lab/
          configs/
          data/
          scripts/
          src/
    This file lives in: phlux_lab/src/phlux_lab/utils/preprocessor.py
    """
    p = Path(str(raw)).expanduser()

    # If already absolute, just return
    if p.is_absolute():
        return str(p)

    # repo_root = .../phlux (two levels up from phlux_lab)
    this_file = Path(__file__).resolve()
    lab_root = this_file.parents[3]  # .../phlux/phlux_lab
    repo_root = lab_root.parent      # .../phlux

    s = str(raw).replace("\\", "/")

    # If the user gave an explicit phlux_lab/... path or src/... path, anchor to repo_root
    if s.startswith("phlux_lab/") or s.startswith("src/") or s.startswith("logs/"):
        return str((repo_root / s).resolve())

    # Otherwise treat as relative to phlux_lab
    return str((lab_root / s).resolve())


# =============================================================================
# Preprocessor
# =============================================================================

class Preprocessor:
    """
    Preprocessor responsibilities:
      - Load CSV
      - Apply derived feature engineering (from YAML)
      - Optionally apply training noise (from YAML)
      - Fit StandardScaler for X and y
      - Provide transform_X and inverse_transform_y for inference usage
      - Save/load as a joblib artifact

    Compatible with your current scripts:
      - Preprocessor.from_config(cfg_one, stage_name=..., dataset="train"/"test", global_preprocessing=...)
      - .x_input_scaled, .y_output_scaled used during training
      - .transform_X(df) used for predictions
      - .inverse_transform_y(y_scaled) used after model prediction
      - .save(path) / .load(path)
    """

    ENABLE_FLOW_LOG1P_DEFAULT: bool = True
    ENABLE_WEAR_LOG1P_DEFAULT: bool = False  # NEW: log1p for wear target

    _FLOW_INCLUDE_TOKENS = ("flow", "q_liquid", "q_", "rate", "vol_flow", "volume_flow")
    _FLOW_EXCLUDE_TOKENS = ("wear", "prob", "probability", "pct", "percent", "class", "label", "delta_")

    # -------------------------------------------------------------------------
    # Construction helpers
    # -------------------------------------------------------------------------

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
        Build a Preprocessor from a per-stage config snippet (cfg_one), plus
        optional global preprocessing block (from master YAML).

        Expected cfg_one shape (as your pipeline uses):
        cfg_one = {
          "data": {
            "train_dataset": "...csv",
            "test_dataset": "...csv",
            "inputs": [...],
            "stack_inputs": [...],      # optional
            "targets": [...],
          }
        }

        global_preprocessing expected shape (from training_config.yaml):
        preprocessing:
          noise: { enabled: true, features: {...}, seed: 42? }
          feature_engineering: [ {name, type, inputs, apply_to?}, ... ]
        """
        ds = str(dataset).lower().strip()
        if ds not in ("train", "test"):
            raise ValueError("dataset must be 'train' or 'test'")

        stage_name = str(stage_name or cfg_one.get("stage_name") or cfg_one.get("stage") or "").strip()

        data_block = cfg_one.get("data", {}) if isinstance(cfg_one.get("data"), dict) else {}

        # Choose dataset path
        csv_path = data_block.get("train_dataset") if ds == "train" else data_block.get("test_dataset")
        if not csv_path:
            # fallback keys if user passed alternative schema
            csv_path = cfg_one.get("train_dataset") if ds == "train" else cfg_one.get("test_dataset")
        if not csv_path:
            raise ValueError(f"[{stage_name or 'unknown'}] Missing {ds}_dataset path in config")

        raw_inputs = list(
            data_block.get("inputs")
            or data_block.get("input_features")
            or cfg_one.get("inputs")
            or cfg_one.get("input_features")
            or []
        )
        stack_inputs = list(
            data_block.get("stack_inputs")
            or cfg_one.get("stack_inputs")
            or []
        )
        targets = list(
            data_block.get("targets")
            or data_block.get("target_features")
            or cfg_one.get("targets")
            or []
        )

        if not raw_inputs:
            raise ValueError(f"[{stage_name}] No inputs defined")
        if not targets:
            raise ValueError(f"[{stage_name}] No targets defined")

        # Ordered union: raw → stacked (dedupe)
        input_features: List[str] = []
        for c in raw_inputs + stack_inputs:
            if c not in input_features:
                input_features.append(c)

        # Derived features (stage overrides first, then global preprocessing)
        derived_features: List[Dict[str, Any]] = []
        if isinstance(data_block.get("derived_features"), list):
            derived_features = list(data_block["derived_features"])
        elif global_preprocessing and isinstance(global_preprocessing.get("feature_engineering"), list):
            derived_features = list(global_preprocessing["feature_engineering"])

        # Noise (stage overrides first, then global)
        noise_block: Dict[str, Any] = {"enabled": False}
        if isinstance(data_block.get("noise"), dict):
            noise_block = dict(data_block["noise"])
        elif global_preprocessing and isinstance(global_preprocessing.get("noise"), dict):
            noise_block = dict(global_preprocessing["noise"])


        # Sample weighting (stage overrides first, then global)
        sample_weighting_block: Dict[str, Any] = {"enabled": False}
        if isinstance(data_block.get("sample_weighting"), dict):
            sample_weighting_block = dict(data_block["sample_weighting"])
        elif global_preprocessing and isinstance(global_preprocessing.get("sample_weighting"), dict):
            sample_weighting_block = dict(global_preprocessing["sample_weighting"])

        schema_path = data_block.get("schema_path", "")
        unit_policy = dict(data_block.get("unit_policy", {}) or {})

        # Optional seed: prefer noise.seed, else cfg_one/global keys if present
        seed = None
        if isinstance(noise_block, dict) and "seed" in noise_block:
            try:
                seed = int(noise_block["seed"])
            except Exception:
                seed = None
        if seed is None:
            # if user put random_seed in cfg_one
            for k in ("random_seed", "seed"):
                if k in cfg_one:
                    try:
                        seed = int(cfg_one[k])
                        break
                    except Exception:
                        pass

        return cls(
            {
                "stage_name": stage_name,
                "dataset": ds,
                "csv_path": _resolve_repo_path(str(csv_path)),
                "raw_inputs": raw_inputs,
                "stack_inputs": stack_inputs,
                "input_features": input_features,
                "targets": targets,
                "schema_path": schema_path,
                "unit_policy": unit_policy,
                "derived_features": derived_features,
                "noise": noise_block,
                "noise_seed": seed,
                "sample_weighting": sample_weighting_block,
                "enable_flow_log1p": cfg_one.get("enable_flow_log1p", cls.ENABLE_FLOW_LOG1P_DEFAULT),
                # NEW: wear target log1p (stage-specific; we gate by stage_name later too)
                "enable_wear_log1p": cfg_one.get("enable_wear_log1p", cls.ENABLE_WEAR_LOG1P_DEFAULT),
            }
        )

    def __init__(self, data_cfg: Dict[str, Any]) -> None:
        self.stage_name: str = str(data_cfg.get("stage_name") or "")
        self.dataset: str = str(data_cfg.get("dataset") or "train").lower().strip()

        self.csv_path: str = str(data_cfg["csv_path"])

        self.raw_inputs: List[str] = list(data_cfg.get("raw_inputs", []))
        self.stack_inputs: List[str] = list(data_cfg.get("stack_inputs", []))

        self.input_features: List[str] = list(data_cfg["input_features"])
        self.target_features: List[str] = list(data_cfg["targets"])

        self.schema_path: str = str(data_cfg.get("schema_path") or "")
        self.unit_policy: Dict[str, Any] = dict(data_cfg.get("unit_policy") or {})
        self.convert_inputs_to_canonical: bool = bool(self.unit_policy.get("convert_inputs_to_canonical", False))
        self.convert_targets_to_canonical: bool = bool(self.unit_policy.get("convert_targets_to_canonical", False))

        self.derived_cfg: List[Dict[str, Any]] = list(data_cfg.get("derived_features", []))
        self.noise_cfg: Dict[str, Any] = dict(data_cfg.get("noise", {}) or {})
        self.noise_enabled: bool = bool(self.noise_cfg.get("enabled", False))


        # Sample weighting config (training only)
        self.sample_weight_cfg: Dict[str, Any] = dict(data_cfg.get("sample_weighting", {}) or {})
        self.sample_weight_enabled: bool = bool(self.sample_weight_cfg.get("enabled", False))
        self.sample_weight_scheme: str = str(self.sample_weight_cfg.get("scheme", "")).strip().lower()
        self.sample_weights: Optional[np.ndarray] = None
        self.sample_weight_target_used: Optional[str] = None

        self.noise_seed: Optional[int] = None
        if "noise_seed" in data_cfg:
            try:
                self.noise_seed = int(data_cfg["noise_seed"]) if data_cfg["noise_seed"] is not None else None
            except Exception:
                self.noise_seed = None

        self.enable_flow_log1p: bool = bool(data_cfg.get("enable_flow_log1p", self.ENABLE_FLOW_LOG1P_DEFAULT))
        # NEW
        self.enable_wear_log1p: bool = bool(data_cfg.get("enable_wear_log1p", self.ENABLE_WEAR_LOG1P_DEFAULT))

        # Load CSV
        self.df: pd.DataFrame = pd.read_csv(self.csv_path)

        # Apply derived features BEFORE validation (because they may be requested as inputs)
        self._apply_derived_features_inplace()

        # Apply noise AFTER derived features but BEFORE scaling (training only)
        self._apply_noise_inplace()

        # Validate required columns exist after derivation/noise
        missing = [c for c in (self.input_features + self.target_features) if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"[{self.stage_name}] Missing columns in CSV after preprocessing:\n"
                + "\n".join(f"  - {c}" for c in missing)
            )

        X_raw = self.df[self.input_features].to_numpy(dtype="float32")
        y_raw = self.df[self.target_features].to_numpy(dtype="float32")

        # ---- WEAR target transform (log1p) ----
        # IMPORTANT: must happen BEFORE computing sample weights
        if self.stage_name == "wear" and self.enable_wear_log1p:
            if np.any(y_raw < 0):
                raise ValueError("[wear] hydraulic_wear contains negative values; cannot apply log1p.")
            y_raw = np.log1p(y_raw)

        # Compute optional per-sample weights (training-time)
        # NOTE: weights are now computed in the SAME space as training y
        self.sample_weights = self._compute_sample_weights(y_raw)



        # NOTE: unit conversion hooks exist but are intentionally OFF unless you implement conversions.
        # Keeping the flags for compatibility with your earlier philosophy.
        # If you later wire unit conversion, do it here before scaling.

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        self.x_input_scaled = self.scaler_X.fit_transform(X_raw)
        self.y_output_scaled = self.scaler_y.fit_transform(y_raw)

    # -------------------------------------------------------------------------
    # Derived features
    # -------------------------------------------------------------------------

    def _apply_derived_features_inplace(self) -> None:
        """
        Implements your YAML feature_engineering list.

        Supported:
          - type: "subtract" -> out = a - b
          - type: "divide"   -> out = a / (b + eps)

        Honors:
          - apply_to: [flow, wear, flow_correction] (if present)
        """
        if not self.derived_cfg:
            return

        for spec in self.derived_cfg:
            if not isinstance(spec, dict):
                continue

            apply_to = spec.get("apply_to")
            if isinstance(apply_to, list) and self.stage_name:
                if self.stage_name not in apply_to:
                    continue

            name = str(spec.get("name") or "").strip()
            ftype = str(spec.get("type") or "").strip().lower()
            inputs = spec.get("inputs") or []

            if not name or not ftype or not isinstance(inputs, list) or len(inputs) < 2:
                continue

            a = str(inputs[0])
            b = str(inputs[1])

            # If required columns don't exist, we skip (final validation will catch if the derived feature is required)
            if a not in self.df.columns or b not in self.df.columns:
                continue

            if ftype == "subtract":
                self.df[name] = self.df[a] - self.df[b]
            elif ftype == "divide":
                eps = float(spec.get("eps", 1e-6))
                self.df[name] = self.df[a] / (self.df[b] + eps)
            else:
                # Unknown type => ignore
                continue

    # -------------------------------------------------------------------------
    # Noise injection
    # -------------------------------------------------------------------------

    def _stable_noise_seed(self) -> int:
        """
        Deterministic fallback seed if user didn't provide one.
        """
        s = f"{self.stage_name}|{self.csv_path}"
        # stable across runs in same python version
        return int(abs(hash(s)) % (2**32))

    def _apply_noise_inplace(self) -> None:
        """
        Applies multiplicative Gaussian noise to selected input features:
          x_noisy = x * (1 + N(0, sigma))

        Only applies for dataset == "train" and if noise.enabled == True.

        YAML example:
          noise:
            enabled: true
            features:
              suction_pressure: 0.05
              discharge_pressure: 0.05
        """
        if self.dataset != "train":
            return
        if not self.noise_enabled:
            return

        feats = self.noise_cfg.get("features", {})
        if not isinstance(feats, dict) or not feats:
            return

        seed = self.noise_seed if self.noise_seed is not None else self._stable_noise_seed()
        rng = np.random.default_rng(seed)

        for col, sigma in feats.items():
            if col not in self.df.columns:
                continue
            try:
                s = float(sigma)
            except Exception:
                continue
            if s <= 0:
                continue

            x = self.df[col].to_numpy(dtype=float)
            n = rng.normal(loc=0.0, scale=s, size=x.shape)
            self.df[col] = x * (1.0 + n)



    # -------------------------------------------------------------------------
    # Sample weighting (training-time)
    # -------------------------------------------------------------------------

    def _compute_sample_weights(self, y_raw: np.ndarray) -> Optional[np.ndarray]:
        """Compute per-sample weights for training based on preprocessing.sample_weighting.

        Supports the YAML structure:
          preprocessing:
            sample_weighting:
              enabled: true
              scheme: by_target_bins | by_percentile
              rules:
                <target_name>:
                  bins: [...]
                  weights: [...]

        Notes:
          - Only applies for dataset == "train"
          - If multiple targets exist for a stage, we use the first target that has a rule.
          - Returns a 1D float32 array of length n_samples, or None.
        """
        if self.dataset != "train":
            return None
        if not self.sample_weight_enabled:
            return None

        rules = self.sample_weight_cfg.get("rules", {})
        if not isinstance(rules, dict) or not rules:
            # Backward compatible fallback: legacy schema with 'target/bins/weights'
            legacy_target = self.sample_weight_cfg.get("target")
            if legacy_target and isinstance(legacy_target, str):
                rules = {legacy_target: {
                    "bins": self.sample_weight_cfg.get("bins"),
                    "weights": self.sample_weight_cfg.get("weights"),
                }}
            else:
                return None

        # Choose target to weight (first matching rule)
        tgt_used = None
        rule_used = None
        for t in self.target_features:
            if t in rules and isinstance(rules[t], dict):
                tgt_used = t
                rule_used = rules[t]
                break

        if tgt_used is None or rule_used is None:
            return None

        self.sample_weight_target_used = tgt_used

        # Extract y vector for the chosen target
        # y_raw shape is (n, n_targets)
        try:
            j = self.target_features.index(tgt_used)
        except ValueError:
            return None

        y = np.asarray(y_raw[:, j], dtype=float)

        bins = rule_used.get("bins")
        weights = rule_used.get("weights")
        if not isinstance(bins, list) or not isinstance(weights, list):
            return None
        if len(bins) == 0 or len(weights) == 0:
            return None
        if len(bins) != len(weights):
            raise ValueError(
                f"[{self.stage_name}] sample_weighting rule for '{tgt_used}' requires "
                f"len(bins) == len(weights), got {len(bins)} and {len(weights)}"
            )

        # Determine scheme
        scheme = self.sample_weight_scheme or str(self.sample_weight_cfg.get("scheme", "")).strip().lower()
        if scheme not in ("by_target_bins", "by_percentile"):
            raise ValueError(
                f"[{self.stage_name}] sample_weighting.scheme must be 'by_target_bins' or 'by_percentile', got '{scheme}'"
            )

        # Convert bins to numeric
        try:
            bins_arr = np.asarray([float(b) for b in bins], dtype=float)
            w_arr = np.asarray([float(w) for w in weights], dtype=float)
        except Exception as e:
            raise ValueError(f"[{self.stage_name}] sample_weighting bins/weights must be numeric: {e}")

        if scheme == "by_percentile":
            # bins are percentiles (e.g., [0, 50, 90, 100]) -> convert to value edges
            if np.any((bins_arr < 0) | (bins_arr > 100)):
                raise ValueError(f"[{self.stage_name}] by_percentile bins must be within [0, 100]")
            edges = np.percentile(y, bins_arr)
        else:
            # bins are value edges directly
            edges = bins_arr

        # Assign each sample to an interval [edges[i], edges[i+1]) with i in [0, n-2]
        # With len(edges)==len(weights), we interpret weights[i] as the weight for interval i,
        # except the last weight applies to y >= edges[-1].
        # This matches your comment "bin i -> weight weights[i]" when bins are ascending edges.
        order_ok = np.all(np.diff(edges) >= 0)
        if not order_ok:
            raise ValueError(f"[{self.stage_name}] sample_weighting bins must be ascending")

        idx = np.searchsorted(edges, y, side="right") - 1
        idx = np.clip(idx, 0, len(w_arr) - 1)

        sw = w_arr[idx].astype("float32")
        return sw

    # -------------------------------------------------------------------------
    # Transform / inverse-transform
    # -------------------------------------------------------------------------

    def transform_X(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform a raw dataframe to scaled X in the same feature order.
        Applies derived features (same rules) but does NOT apply noise.
        """
        work = df.copy()

        # Apply derived features for inference path too (but no noise)
        if self.derived_cfg:
            for spec in self.derived_cfg:
                if not isinstance(spec, dict):
                    continue

                apply_to = spec.get("apply_to")
                if isinstance(apply_to, list) and self.stage_name:
                    if self.stage_name not in apply_to:
                        continue

                name = str(spec.get("name") or "").strip()
                ftype = str(spec.get("type") or "").strip().lower()
                inputs = spec.get("inputs") or []
                if not name or not ftype or not isinstance(inputs, list) or len(inputs) < 2:
                    continue

                a = str(inputs[0])
                b = str(inputs[1])
                if a not in work.columns or b not in work.columns:
                    continue

                if ftype == "subtract":
                    work[name] = work[a] - work[b]
                elif ftype == "divide":
                    eps = float(spec.get("eps", 1e-6))
                    work[name] = work[a] / (work[b] + eps)

        missing = [c for c in self.input_features if c not in work.columns]
        if missing:
            raise ValueError(
                f"[{self.stage_name}] transform_X missing columns:\n"
                + "\n".join(f"  - {c}" for c in missing)
            )

        X = work[self.input_features].to_numpy(dtype="float32")
        return self.scaler_X.transform(X)

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled targets back to original target units.
        Applies expm1 for wear stage if enable_wear_log1p is enabled.
        """
        y_scaled = np.asarray(y_scaled)
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)

        y = self.scaler_y.inverse_transform(y_scaled)

        # ---- WEAR inverse transform (expm1) ----
        if self.stage_name == "wear" and getattr(self, "enable_wear_log1p", False):
            y = np.expm1(y)
            y = np.clip(y, 0.0, None)  # safety: never negative

        return y

    # -------------------------------------------------------------------------
    # Introspection helpers
    # -------------------------------------------------------------------------

    def get_input_feature_names(self) -> List[str]:
        return list(self.input_features)

    def get_target_feature_names(self) -> List[str]:
        return list(self.target_features)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_artifact(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "dataset": self.dataset,
            "input_features": self.input_features,
            "target_features": self.target_features,
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "derived_cfg": self.derived_cfg,
            "enable_flow_log1p": self.enable_flow_log1p,
            "enable_wear_log1p": self.enable_wear_log1p,  # NEW
        }

    def save(self, path: str) -> None:
        import joblib
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.to_artifact(), str(out))

    @classmethod
    def load(cls, path: str) -> "Preprocessor":
        import joblib
        return cls.from_artifact(joblib.load(path))

    @classmethod
    def from_artifact(cls, artifact: Dict[str, Any]) -> "Preprocessor":
        self = cls.__new__(cls)
        self.stage_name = artifact.get("stage_name", "")
        self.dataset = artifact.get("dataset", "train")
        self.input_features = artifact["input_features"]
        self.target_features = artifact["target_features"]
        self.scaler_X = artifact["scaler_X"]
        self.scaler_y = artifact["scaler_y"]
        self.derived_cfg = list(artifact.get("derived_cfg", []))
        self.enable_flow_log1p = bool(artifact.get("enable_flow_log1p", cls.ENABLE_FLOW_LOG1P_DEFAULT))
        self.enable_wear_log1p = bool(artifact.get("enable_wear_log1p", cls.ENABLE_WEAR_LOG1P_DEFAULT))  # NEW

        # fields not required post-load but present for completeness
        self.raw_inputs = []
        self.stack_inputs = []
        self.df = pd.DataFrame()
        self.schema_path = ""
        self.unit_policy = {}
        self.convert_inputs_to_canonical = False
        self.convert_targets_to_canonical = False
        self.noise_cfg = {"enabled": False}
        self.noise_enabled = False
        self.noise_seed = None
        self.csv_path = ""


        # sample weighting (not used post-load)
        self.sample_weight_cfg = {"enabled": False}
        self.sample_weight_enabled = False
        self.sample_weight_scheme = ""
        self.sample_weights = None
        self.sample_weight_target_used = None

        return self
