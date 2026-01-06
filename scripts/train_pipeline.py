from __future__ import annotations

# --- MUST be first (silence TF warnings) ---
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
import pandas as pd

from phlux_lab.utils.preprocessor import Preprocessor  # type: ignore
from phlux_lab.ml.vfm_model import VFMModel  # type: ignore

HERE = Path(__file__).resolve().parent
LAB_ROOT = HERE.parent
REPO_ROOT = LAB_ROOT.parent
DEFAULT_CONFIG = LAB_ROOT / "configs" / "training_config.yaml"


# =============================================================================
# Helpers
# =============================================================================

def _resolve_path(p: str | Path) -> Path:
    pp = Path(str(p)).expanduser()
    if pp.is_absolute():
        return pp
    s = str(p).replace("\\", "/")
    if s.startswith("phlux_lab/") or s.startswith("src/") or s.startswith("logs/"):
        return (REPO_ROOT / s).resolve()
    return (LAB_ROOT / s).resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_client_name(cfg: Dict[str, Any]) -> str:
    proj = cfg.get("project", {})
    name = cfg.get("client_name") or proj.get("client_name") or proj.get("client")
    if not name:
        raise KeyError("training_config.yaml must define project.client_name")
    return str(name)


def _stage_cfg(cfg: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
    models = cfg.get("models", {})
    if stage_name not in models:
        raise KeyError(f"training_config.yaml missing models.{stage_name}")
    return models[stage_name]


def _stage_data(stage_cfg: Dict[str, Any]) -> Dict[str, Any]:
    d = stage_cfg.get("data")
    return d if isinstance(d, dict) else stage_cfg


def _get_train_test_paths(stage_cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    d = _stage_data(stage_cfg)
    tr, te = d.get("train_dataset"), d.get("test_dataset")
    if not tr or not te:
        raise KeyError("Stage must define data.train_dataset and data.test_dataset")
    return _resolve_path(tr), _resolve_path(te)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _assert_columns_exist(csv_path: Path, cols: List[str], stage_name: str) -> None:
    if not cols:
        return
    df = pd.read_csv(csv_path, nrows=5)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"[{stage_name}] Missing columns in {csv_path.name}: {missing}")


def _canonical_client_dir(cfg: Dict[str, Any], client: str) -> Path:
    """
    Accepts either:
      paths.model_root = phlux_lab/models              -> uses /<client>
      paths.model_root = phlux_lab/models/ClientA     -> uses as-is
      paths.model_root = phlux_lab/models/{client}    -> formats
    """
    raw = str(cfg.get("paths", {}).get("model_root", LAB_ROOT / "models"))
    raw = raw.replace("{client}", client).replace("{client_name}", client)
    base = _resolve_path(raw)

    # If user already pointed at the client folder, don't append again
    if base.name.lower() == client.lower():
        return base
    return base / client


def _merge_model_cfg(cfg: Dict[str, Any], stage_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge training_defaults into a per-stage config.

    Supported YAML shapes:
      - legacy: models.<stage>.model (architecture + training params)
      - current: models.<stage>.training (training params)
      - both can coexist; precedence is:
            training_defaults  <  stage.model  <  stage.training

    Also inject reduce_lr_on_plateau into lr_scheduler_cfg when enabled globally.
    """
    defaults = dict(cfg.get("training_defaults", {}) or {})
    stage_model = dict(stage_cfg.get("model", {}) or {})       # legacy / optional
    stage_training = dict(stage_cfg.get("training", {}) or {}) # current / optional

    merged: Dict[str, Any] = {}
    merged.update(defaults)
    merged.update(stage_model)
    merged.update(stage_training)

    # -----------------------------
    # YAML → VFMModel key mapping
    # -----------------------------

    # early_stopping can be:
    #   - bool (legacy)
    #   - dict: {enabled, patience, monitor, mode}
    if "early_stopping" in merged:
        es = merged.get("early_stopping")
        if isinstance(es, dict):
            merged["earlystop"] = bool(es.get("enabled", True))
            # patience (prefer explicit earlystop_patience if already set)
            if "earlystop_patience" not in merged:
                merged["earlystop_patience"] = int(es.get("patience", merged.get("patience", 5)))
            # keep monitor/mode for future use (VFMModel may ignore today)
            if "earlystop_monitor" not in merged and "monitor" in es:
                merged["earlystop_monitor"] = es.get("monitor")
            if "earlystop_mode" not in merged and "mode" in es:
                merged["earlystop_mode"] = es.get("mode")
        else:
            # bool / truthy
            if "earlystop" not in merged:
                merged["earlystop"] = bool(es)

    # legacy patience at top-level → earlystop_patience
    if "patience" in merged and "earlystop_patience" not in merged:
        merged["earlystop_patience"] = int(merged["patience"])

    # validation_split → val_split
    if "validation_split" in merged and "val_split" not in merged:
        merged["val_split"] = float(merged["validation_split"])

    # -----------------------------
    # Reduce LR on plateau wiring
    # -----------------------------
    rlr = dict(cfg.get("reduce_lr_on_plateau", {}) or {})
    if bool(rlr.get("enabled", False)):
        # only inject if stage didn't explicitly set lr_scheduler_cfg
        if "lr_scheduler_cfg" not in merged:
            merged["lr_scheduler_cfg"] = {
                "type": "reduce_lr_on_plateau",
                "monitor": rlr.get("monitor", "val_loss"),
                "factor": rlr.get("factor", 0.5),
                "patience": rlr.get("patience", 3),
                "min_lr": rlr.get("min_lr", 1e-5),
                "cooldown": rlr.get("cooldown", 0),
                "verbose": rlr.get("verbose", 1),
            }

    return merged


# =============================================================================
# CSV augmentation helper
# =============================================================================

def _predict_to_csv(model: VFMModel, pp: Preprocessor, csv_in: Path, csv_out: Path, out_col: str) -> None:
    df = pd.read_csv(csv_in)
    X = pp.transform_X(df)
    y_scaled = model.predict(X)
    y = pp.inverse_transform_y(y_scaled)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    df[out_col] = y
    df.to_csv(csv_out, index=False)


# =============================================================================
# Training
# =============================================================================

def _train_stage(
    global_cfg: Dict[str, Any],
    client_dir: Path,
    stage_name: str,
) -> Tuple[VFMModel | None, Preprocessor | None]:

    stage_cfg = _stage_cfg(global_cfg, stage_name)
    if stage_cfg.get("enabled", True) is False:
        print(f"[SKIP]  Skipping stage '{stage_name}' (disabled in config)")
        return None, None

    d = _stage_data(stage_cfg)
    train_csv, test_csv = _get_train_test_paths(stage_cfg)

    inputs = list(d.get("inputs", []))
    stack_inputs = list(d.get("stack_inputs", []))   # semantic only
    targets = list(d.get("targets", []))

    if not inputs:
        raise KeyError(f"[{stage_name}] data.inputs missing")
    if not targets:
        raise KeyError(f"[{stage_name}] data.targets missing")

    # verify required raw columns exist OR can be derived
    # (Preprocessor will derive features before scaling, but we still validate existence
    # for stack_inputs because those must be present in the CSV)
    _assert_columns_exist(train_csv, stack_inputs, stage_name)
    _assert_columns_exist(test_csv, stack_inputs, stage_name)

    stage_dir = client_dir / stage_name
    _ensure_dir(stage_dir)

    save_cfg = stage_cfg.get("save", {})
    model_path = stage_dir / (save_cfg.get("model_name") or f"{stage_name}.keras")
    pp_path = stage_dir / (save_cfg.get("preprocessor") or f"preprocessor_{stage_name}.joblib")

    print(f"\n>> Training stage: {stage_name}")
    print(f"  - train_dataset: {train_csv}")
    print(f"  - test_dataset:  {test_csv}")
    print(f"  - inputs:        {inputs}")
    print(f"  - stack_inputs:  {stack_inputs}")
    print(f"  - targets:       {targets}")
    print(f"  - save model:    {model_path}")
    print(f"  - save pp:       {pp_path}")

    cfg_one = {
        "data": {
            "train_dataset": str(train_csv),
            "test_dataset": str(test_csv),
            "inputs": inputs,
            "stack_inputs": stack_inputs,
            "targets": targets,
            "target_policy": d.get("target_policy", {}) or {},
        },
        # Backward-compatible: allow target_policy at top-level too
        "target_policy": d.get("target_policy", {}) or {},
    }

    global_pre = global_cfg.get("preprocessing", {}) or {}
    pp = Preprocessor.from_config(cfg_one, stage_name=stage_name, dataset="train", global_preprocessing=global_pre)

    model_cfg = _merge_model_cfg(global_cfg, stage_cfg)
    vfm = VFMModel.from_config(
        preprocessor=pp,
        model_cfg=model_cfg,
        log_dir=str(stage_dir / "logs"),
        model_dir=str(stage_dir),
    )
    # Pass sample weights to training if preprocessing computed them
    sw = getattr(pp, "sample_weights", None)

    vfm.fit(
        pp.x_input_scaled,
        pp.y_output_scaled,
        model_cfg,
        sample_weight=sw,
    )

    vfm.save(str(model_path))
    pp.save(str(pp_path))

    print(f"  [SAVED] saved model → {model_path.name}")
    print(f"  [SAVED] saved preprocessor → {pp_path.name}")

    return vfm, pp


def main() -> None:
    # UTF-8 safe console output (Windows cp1252 can choke on emojis)
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    cfg_path = _resolve_path(str(DEFAULT_CONFIG))
    if len(sys.argv) > 1:
        cfg_path = _resolve_path(sys.argv[1])

    cfg = _load_yaml(cfg_path)
    client = _get_client_name(cfg)

    client_dir = _canonical_client_dir(cfg, client)
    _ensure_dir(client_dir)

    def _enabled(stage: str) -> bool:
        try:
            return bool(_stage_cfg(cfg, stage).get("enabled", True))
        except KeyError:
            return False

    print(f"\n[OK] Client: {client}")
    print(f"[OK] Config: {cfg_path}")
    print(f"[OK] Output dir: {client_dir}")

    stages = ["flow", "wear", "flow_correction"]
    enabled_stages = [s for s in stages if _enabled(s)]
    print(f"[OK] Stages (enabled): {enabled_stages}")

    # ------------------------------------------------------------------
    # 1) FLOW
    # ------------------------------------------------------------------
    flow_model = flow_pp = None
    if _enabled("flow"):
        flow_model, flow_pp = _train_stage(cfg, client_dir, "flow")
    else:
        print("[SKIP]  Flow stage disabled (models.flow.enabled=false). Nothing to do.")
        print("\n[OK] Done.")
        return

    # If nothing else is enabled, stop after saving the flow model
    if not _enabled("wear") and not _enabled("flow_correction"):
        print("\n[OK] Done.")
        return

    # ------------------------------------------------------------------
    # 2) WEAR (optional)
    # ------------------------------------------------------------------
    wear_model = wear_pp = None
    if _enabled("wear"):
        # Inject q_liquid_pred into wear datasets ONLY if wear stage is enabled
        wear_data = _stage_data(_stage_cfg(cfg, "wear"))
        wear_train = _resolve_path(wear_data["train_dataset"])
        wear_test = _resolve_path(wear_data["test_dataset"])

        print("\n>> Augmenting wear datasets with q_liquid_pred")
        _predict_to_csv(model=flow_model, pp=flow_pp, csv_in=wear_train, csv_out=wear_train, out_col="q_liquid_pred")
        _predict_to_csv(model=flow_model, pp=flow_pp, csv_in=wear_test, csv_out=wear_test, out_col="q_liquid_pred")

        wear_model, wear_pp = _train_stage(cfg, client_dir, "wear")
    else:
        print("[SKIP]  Wear stage disabled (models.wear.enabled=false). Skipping wear augmentation + training.")

    # If flow_correction is disabled, stop after saving the last enabled model
    if not _enabled("flow_correction"):
        print("\n[OK] Done.")
        return

    # ------------------------------------------------------------------
    # 3) FLOW CORRECTION (optional)
    # ------------------------------------------------------------------
    corr_cfg = _stage_cfg(cfg, "flow_correction")
    corr_data = _stage_data(corr_cfg)

    # Only compute the stacked predictions that the correction stage actually asks for
    corr_stack_inputs = list(corr_data.get("stack_inputs", []) or [])

    paired_train = _resolve_path(corr_data["train_dataset"])
    paired_test = _resolve_path(corr_data["test_dataset"])

    # Dependency validation (fail fast, clear message)
    if "q_liquid_pred" in corr_stack_inputs and (flow_model is None or flow_pp is None):
        raise RuntimeError("flow_correction requires q_liquid_pred but flow stage did not produce it.")
    if "hydraulic_wear_pred" in corr_stack_inputs and (wear_model is None or wear_pp is None):
        raise RuntimeError(
            "flow_correction requires hydraulic_wear_pred but wear stage is disabled or did not produce it. "
            "Either enable wear or remove hydraulic_wear_pred from models.flow_correction.data.stack_inputs."
        )

    print("\n>> Augmenting paired datasets with stacked predictions")
    if "q_liquid_pred" in corr_stack_inputs:
        _predict_to_csv(model=flow_model, pp=flow_pp, csv_in=paired_train, csv_out=paired_train, out_col="q_liquid_pred")
        _predict_to_csv(model=flow_model, pp=flow_pp, csv_in=paired_test, csv_out=paired_test, out_col="q_liquid_pred")

    if "hydraulic_wear_pred" in corr_stack_inputs:
        _predict_to_csv(model=wear_model, pp=wear_pp, csv_in=paired_train, csv_out=paired_train, out_col="hydraulic_wear_pred")
        _predict_to_csv(model=wear_model, pp=wear_pp, csv_in=paired_test, csv_out=paired_test, out_col="hydraulic_wear_pred")

    _train_stage(cfg, client_dir, "flow_correction")

    print("\n[OK] Done.")


if __name__ == "__main__":
    main()
