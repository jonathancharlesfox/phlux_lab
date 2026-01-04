from __future__ import annotations

# --- MUST be first (silence TF warnings) ---
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import yaml
import pandas as pd
import numpy as np

from phlux_lab.utils.preprocessor import Preprocessor  # type: ignore
from phlux_lab.ml.vfm_model import VFMModel  # type: ignore


# =============================================================================
# Roots / path resolution
# =============================================================================

HERE = Path(__file__).resolve()
LAB_ROOT = HERE.parents[1]              # .../phlux_lab
REPO_ROOT = LAB_ROOT.parent             # .../phlux
DEFAULT_CONFIG = LAB_ROOT / "configs" / "training_config.yaml"


def _resolve_path(p: str) -> Path:
    """Resolve repo-relative paths like 'phlux_lab/data/...' to absolute paths."""
    p = (p or "").strip()
    if not p:
        return Path(p)
    pp = Path(p)
    if pp.is_absolute():
        return pp
    norm = p.replace("\\", "/")
    # YAML usually uses "phlux_lab/..." => repo-relative
    if norm.startswith("phlux_lab/"):
        return (REPO_ROOT / norm).resolve()
    # fallback: treat as repo-relative
    return (REPO_ROOT / norm).resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_project_block(cfg: Dict[str, Any]) -> Dict[str, Any]:
    blk = cfg.get("project", {})
    return blk if isinstance(blk, dict) else {}


def _get_client_name(cfg: Dict[str, Any]) -> str:
    proj = _get_project_block(cfg)
    # support project.client_name, project.client, legacy top-level client_name
    name = cfg.get("client_name") or proj.get("client_name") or proj.get("client")
    if not name:
        raise KeyError("training_config.yaml must define project.client_name (or project.client)")
    return str(name)


def _stage_cfg(cfg: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
    models = cfg.get("models", {})
    if not isinstance(models, dict) or stage_name not in models:
        raise KeyError(f"training_config.yaml missing models.{stage_name}")
    sc = models[stage_name]
    if not isinstance(sc, dict):
        raise TypeError(f"models.{stage_name} must be a dict")
    return sc


def _stage_data(stage_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Support both:
      - nested: stage_cfg['data']['train_dataset']
      - flat:   stage_cfg['train_dataset']
    """
    d = stage_cfg.get("data", None)
    if isinstance(d, dict):
        return d
    return stage_cfg


def _stage_save(stage_cfg: Dict[str, Any]) -> Dict[str, Any]:
    s = stage_cfg.get("save", None)
    return s if isinstance(s, dict) else {}


def _get_train_test_paths(stage_cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    d = _stage_data(stage_cfg)
    tr = d.get("train_dataset")
    te = d.get("test_dataset")
    if not tr or not te:
        raise KeyError("Stage config must define data.train_dataset and data.test_dataset (or flat train_dataset/test_dataset)")
    return _resolve_path(str(tr)), _resolve_path(str(te))


def _get_inputs_targets(stage_cfg: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    d = _stage_data(stage_cfg)
    inputs = list(d.get("inputs", d.get("input_features", [])) or [])
    stack_inputs = list(d.get("stack_inputs", []) or [])
    targets = list(d.get("targets", d.get("target_features", [])) or [])
    if not inputs:
        raise KeyError("Stage config missing inputs (or input_features)")
    if not targets:
        raise KeyError("Stage config missing targets (or target_features)")
    return inputs, stack_inputs, targets


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Sample-weight helpers (stage 3 wear-weighted correction, etc.)
# =============================================================================

def _get_sample_weight_cfg(stage_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Support sample_weight either at stage root or under stage data block.
    e.g.
      models.flow_correction.sample_weight: {...}
    """
    sw = stage_cfg.get("sample_weight")
    if isinstance(sw, dict):
        return sw
    d = _stage_data(stage_cfg)
    sw2 = d.get("sample_weight")
    return sw2 if isinstance(sw2, dict) else {}


def _compute_linear_sample_weights(
    wear: np.ndarray,
    wear_lo: float,
    wear_hi: float,
    min_weight: float,
) -> np.ndarray:
    """
    Linear ramp:
      wear <= wear_lo -> min_weight
      wear >= wear_hi -> 1.0
      else linear interpolation, then lifted by min_weight.
    """
    wear = np.asarray(wear, dtype="float32")
    if wear_hi <= wear_lo:
        raise ValueError(f"sample_weight wear_hi must be > wear_lo (got {wear_hi} <= {wear_lo})")

    w = (wear - float(wear_lo)) / (float(wear_hi) - float(wear_lo))
    w = np.clip(w, 0.0, 1.0).astype("float32")

    mw = float(min_weight)
    mw = max(0.0, min(1.0, mw))
    w = mw + (1.0 - mw) * w
    return w.astype("float32")


def _fit_with_optional_sample_weight(
    vfm: VFMModel,
    X: np.ndarray,
    y: np.ndarray,
    model_cfg: Dict[str, Any],
    sample_weight: Optional[np.ndarray],
) -> None:
    """
    Try very hard to pass sample_weight without breaking older VFMModel APIs.

    Priority:
      1) If VFMModel.fit accepts sample_weight (or **kwargs), pass it.
      2) Else, if vfm has attribute 'model' with Keras-like .fit, call that.
      3) Else, fall back to vfm.fit without weights (with a warning).
    """
    if sample_weight is None:
        vfm.fit(X, y, model_cfg)
        return

    # 1) Try VFMModel.fit signature
    try:
        sig = inspect.signature(vfm.fit)
        params = sig.parameters
        accepts_sw = ("sample_weight" in params)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if accepts_sw or accepts_kwargs:
            vfm.fit(X, y, model_cfg, sample_weight=sample_weight)
            return
    except Exception:
        # If signature introspection fails, continue to attempt keras model
        pass

    # 2) Try direct Keras model access
    m = getattr(vfm, "model", None)
    if m is not None and hasattr(m, "fit"):
        # Pull training hyperparams from model_cfg if VFMModel usually does.
        # If VFMModel.fit handled these internally, we can't perfectly replicate them,
        # but passing sample_weight is the goal. We'll use common keys if present.
        epochs = int(model_cfg.get("epochs", 100))
        batch_size = int(model_cfg.get("batch_size", 128))
        val_split = float(model_cfg.get("val_split", 0.1))
        verbose = int(model_cfg.get("verbose", 1))

        # If VFMModel usually uses callbacks/earlystop internally, those won't be here.
        # This branch is a last-resort fallback to avoid breaking training entirely.
        m.fit(
            X,
            y,
            sample_weight=sample_weight,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            verbose=verbose,
            shuffle=True,
        )
        return

    # 3) Final fallback
    print("  ⚠ sample_weight requested but VFMModel does not support it; training without weights.")
    vfm.fit(X, y, model_cfg)


# =============================================================================
# Training
# =============================================================================

def _train_stage(global_cfg: Dict[str, Any], client_dir: Path, stage_name: str) -> Tuple[VFMModel, Preprocessor, Path, Path]:
    stage_cfg = _stage_cfg(global_cfg, stage_name)

    # Skip if disabled
    if stage_cfg.get("enabled", True) is False:
        raise RuntimeError(f"Stage '{stage_name}' is disabled in YAML but train_pipeline expects it enabled.")

    train_csv, test_csv = _get_train_test_paths(stage_cfg)
    inputs, stack_inputs, targets = _get_inputs_targets(stage_cfg)

    # Output paths
    stage_dir = client_dir / stage_name
    _ensure_dir(stage_dir)

    save_cfg = _stage_save(stage_cfg)
    model_name = save_cfg.get("model_name") or f"{stage_name}.keras"
    pp_name = save_cfg.get("preprocessor") or f"preprocessor_{stage_name}.joblib"
    model_path = stage_dir / str(model_name)
    pp_path = stage_dir / str(pp_name)

    print(f"\n▶ Training stage: {stage_name}")
    print(f"  - targets: {targets}")
    print(f"  - train_dataset: {train_csv.relative_to(REPO_ROOT) if train_csv.is_absolute() else train_csv}")
    print(f"  - test_dataset:  {test_csv.relative_to(REPO_ROOT) if test_csv.is_absolute() else test_csv}")
    print(f"  - save model:    {model_path}")
    print(f"  - save pp:       {pp_path}")

    # Build all inputs = inputs + any stack_inputs not already present
    all_inputs = list(inputs)
    for c in stack_inputs:
        if c not in all_inputs:
            all_inputs.append(c)

    # Build a stage snapshot in the shape Preprocessor.from_config expects
    cfg_one = {
        "data": {
            "train_dataset": str(train_csv),
            "test_dataset": str(test_csv),

            # legacy keys (what your preprocessor likely expects)
            "input_features": all_inputs,
            "target_features": targets,

            # new keys (keep for clarity / future)
            "inputs": inputs,
            "stack_inputs": stack_inputs,
            "targets": targets,
        }
    }

    # pass global preprocessing through (if defined)
    preprocessing = global_cfg.get("preprocessing", {})
    if isinstance(preprocessing, dict):
        cfg_one["preprocessing"] = preprocessing

    # Train preprocessor on train
    pp = Preprocessor.from_config(cfg_one, stage_name=stage_name, dataset="train")

    # Train model
    model_cfg = stage_cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        model_cfg = {}

    vfm = VFMModel.from_config(
        preprocessor=pp,
        model_cfg=model_cfg,
        log_dir=str(client_dir / stage_name / "logs"),
        model_dir=str(client_dir / stage_name),
    )

    # ----------------------------
    # Optional: compute sample weights (e.g., wear-weighted correction training)
    # ----------------------------
    sample_weight: Optional[np.ndarray] = None
    sw_cfg = _get_sample_weight_cfg(stage_cfg)
    if isinstance(sw_cfg, dict) and sw_cfg.get("enabled", False):
        col = str(sw_cfg.get("column", "") or "").strip()
        if not col:
            raise KeyError(f"models.{stage_name}.sample_weight.column is required when sample_weight.enabled=true")

        wear_lo = float(sw_cfg.get("wear_lo", 0.0))
        wear_hi = float(sw_cfg.get("wear_hi", 1.0))
        min_weight = float(sw_cfg.get("min_weight", 0.0))

        # Read train CSV to get the weighting column. (Weights are training-only; not part of inputs.)
        df_train = pd.read_csv(train_csv)
        if col not in df_train.columns:
            raise KeyError(
                f"models.{stage_name}.sample_weight.column='{col}' not found in {train_csv}. "
                f"Available columns: {list(df_train.columns)[:20]}{'...' if len(df_train.columns)>2020 else ''}"
            )

        wear = df_train[col].to_numpy(dtype="float32")
        sample_weight = _compute_linear_sample_weights(wear, wear_lo=wear_lo, wear_hi=wear_hi, min_weight=min_weight)

        # Make sure alignment matches the preprocessor's X rows
        n_pp = int(pp.x_input_scaled.shape[0])
        n_sw = int(sample_weight.shape[0])
        if n_pp != n_sw:
            print(
                f"  ⚠ sample_weight length mismatch for stage '{stage_name}': "
                f"weights={n_sw}, preprocessor_X={n_pp}. Disabling sample_weight to avoid misalignment."
            )
            sample_weight = None
        else:
            # tiny diagnostics (helps verify your thresholds are doing what you expect)
            pcts = np.percentile(sample_weight, [0, 5, 50, 95, 100]).astype("float32")
            print(
                f"  ✓ sample_weight enabled ({col}); "
                f"wear_lo={wear_lo}, wear_hi={wear_hi}, min_weight={min_weight} "
                f"| w[p0,p5,p50,p95,p100]={pcts.tolist()}"
            )

    # Fit (with optional sample_weight)
    _fit_with_optional_sample_weight(vfm, pp.x_input_scaled, pp.y_output_scaled, model_cfg, sample_weight)

    # Save artifacts
    vfm.save(str(model_path))
    pp.save(str(pp_path))

    # Also evaluate quickly on test to ensure pipeline sanity
    pp_test = Preprocessor.from_config(cfg_one, stage_name=stage_name, dataset="test")
    yhat_scaled = vfm.predict(pp_test.x_input_scaled)
    yhat = pp_test.inverse_transform_y(yhat_scaled)
    print(f"  ✓ {stage_name} predicted shape: {yhat.shape}")

    return vfm, pp, model_path, pp_path


# =============================================================================
# Dataset augmentation for stacking
# =============================================================================

def _predict_to_csv(
    *,
    model: VFMModel,
    pp: Preprocessor,
    csv_in: Path,
    csv_out: Path,
    out_col: str,
) -> None:
    df = pd.read_csv(csv_in)
    Xs = pp.transform_X(df)
    y_scaled = model.predict(Xs)
    y = pp.inverse_transform_y(y_scaled)

    if y.shape[1] != 1:
        raise ValueError(f"Expected single-output prediction for '{out_col}', got shape {y.shape}")

    df[out_col] = y[:, 0].astype(float)
    df.to_csv(csv_out, index=False)


def main() -> None:
    cfg_path = _resolve_path(str(DEFAULT_CONFIG))
    if len(sys.argv) > 1:
        cfg_path = _resolve_path(sys.argv[1])

    cfg = _load_yaml(cfg_path)
    client = _get_client_name(cfg)

    # model root: prefer cfg.paths.model_root if present, else default phlux_lab/models/<client>
    paths_blk = cfg.get("paths", {})
    if not isinstance(paths_blk, dict):
        paths_blk = {}
    model_root_cfg = paths_blk.get("model_root")

    if model_root_cfg:
        base = _resolve_path(str(model_root_cfg))
        # If model_root already ends with the client folder, don't append again
        client_dir = base if base.name.lower() == client.lower() else (base / client)
    else:
        client_dir = LAB_ROOT / "models" / client

    _ensure_dir(client_dir)

    stages = ["flow", "wear", "flow_correction"]

    print(f"\n✅ Client: {client}")
    print(f"✅ Config: {cfg_path}")
    print(f"✅ Output dir: {client_dir}")
    print(f"✅ Stages: {stages}")

    # ----------------------------
    # 1) Train FLOW
    # ----------------------------
    flow_model, flow_pp, flow_model_path, flow_pp_path = _train_stage(cfg, client_dir, "flow")

    # ----------------------------
    # 2) Augment WEAR datasets with q_liquid_pred
    # ----------------------------
    wear_cfg = _stage_cfg(cfg, "wear")
    wear_data = _stage_data(wear_cfg)
    wear_train = _resolve_path(str(wear_data["train_dataset"]))
    wear_test = _resolve_path(str(wear_data["test_dataset"]))

    print("\n▶ Augmenting wear datasets with q_liquid_pred (overwrite)")
    _predict_to_csv(model=flow_model, pp=flow_pp, csv_in=wear_train, csv_out=wear_train, out_col="q_liquid_pred")
    _predict_to_csv(model=flow_model, pp=flow_pp, csv_in=wear_test, csv_out=wear_test, out_col="q_liquid_pred")
    print(f"  ✓ wrote q_liquid_pred to:\n    - {wear_train}\n    - {wear_test}")

    # ----------------------------
    # 3) Train WEAR
    # ----------------------------
    wear_model, wear_pp, wear_model_path, wear_pp_path = _train_stage(cfg, client_dir, "wear")

    # ----------------------------
    # 4) Augment PAIRED datasets with q_liquid_pred and hydraulic_wear_pred
    # ----------------------------
    corr_cfg = _stage_cfg(cfg, "flow_correction")
    corr_data = _stage_data(corr_cfg)
    paired_train = _resolve_path(str(corr_data["train_dataset"]))
    paired_test = _resolve_path(str(corr_data["test_dataset"]))

    print("\n▶ Augmenting paired datasets with q_liquid_pred + hydraulic_wear_pred (overwrite)")

    # q_liquid_pred
    _predict_to_csv(model=flow_model, pp=flow_pp, csv_in=paired_train, csv_out=paired_train, out_col="q_liquid_pred")
    _predict_to_csv(model=flow_model, pp=flow_pp, csv_in=paired_test, csv_out=paired_test, out_col="q_liquid_pred")

    # hydraulic_wear_pred (needs q_liquid_pred present now)
    _predict_to_csv(model=wear_model, pp=wear_pp, csv_in=paired_train, csv_out=paired_train, out_col="hydraulic_wear_pred")
    _predict_to_csv(model=wear_model, pp=wear_pp, csv_in=paired_test, csv_out=paired_test, out_col="hydraulic_wear_pred")

    print(f"  ✓ wrote stacked columns to:\n    - {paired_train}\n    - {paired_test}")

    # ----------------------------
    # 5) Train FLOW CORRECTION
    # ----------------------------
    _train_stage(cfg, client_dir, "flow_correction")

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
