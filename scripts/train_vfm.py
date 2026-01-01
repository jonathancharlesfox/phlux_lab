from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 2=hide INFO, 3=hide INFO+WARNING
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional

from pathlib import Path
import yaml
import traceback

from phlux_lab.utils.preprocessor import Preprocessor # type: ignore
from phlux_lab.ml.vfm_model import VFMModel # type: ignore

LAB_ROOT = Path(__file__).resolve().parents[1]   # adjust if script depth differs

def _base_name(col: str) -> str:
    """Convert 'suction_pressure__bar' -> 'suction_pressure'."""
    s = str(col).strip()
    return s.split("__", 1)[0] if "__" in s else s


def _infer_schema_path(train_csv: str) -> str:
    """
    Infer schema path from a train CSV:
      ..._train.csv -> ..._schema.yaml
    Falls back to replacing .csv with _schema.yaml.
    """
    p = str(train_csv)
    if p.endswith("_train.csv"):
        return p.replace("_train.csv", "_schema.yaml")
    if p.endswith(".csv"):
        return p[:-4] + "_schema.yaml"
    return p + "_schema.yaml"


def main() -> None:
    # Load training_config.yaml
    with open("phlux_lab/configs/training_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    client_name = cfg.get("client_name") or "Client"
    model_cfg = cfg.get("model", {}) or {}

    # ----------------------------
    # Resolve data config (NEW or OLD format)
    # ----------------------------
    data_cfg: dict = {}

    if isinstance(cfg.get("data"), dict) and cfg["data"]:
        # NEW format: top-level data block
        data_cfg = dict(cfg["data"])
        profile_name = None
    else:
        # OLD format: active_profile + profiles
        profile_name = cfg.get("active_profile")
        profiles = cfg.get("profiles", {}) or {}
        if not profile_name or profile_name not in profiles:
            raise KeyError(
                "Config must either define a top-level 'data:' block (new format), "
                "or define 'active_profile' and 'profiles[active_profile]' (old format)."
            )
        data_cfg = dict(profiles[profile_name])

    # ----------------------------
    # Normalize paths: train/test/schema
    # ----------------------------
    # Accept either train_path or csv_path for training data
    train_path = data_cfg.get("train_path") or data_cfg.get("csv_path")
    test_path = data_cfg.get("test_path")
    schema_path = data_cfg.get("schema_path")

    if not train_path:
        raise KeyError(
            "training_config.yaml must define either:\n"
            "  data.train_path (preferred) OR data.csv_path (legacy)\n"
            "Or in old format:\n"
            "  profiles[active_profile].train_path / csv_path"
        )

    # Infer schema path if not given
    if not schema_path:
        schema_guess = _infer_schema_path(str(train_path))
        if Path(schema_guess).exists():
            schema_path = schema_guess

    # Store normalized keys for Preprocessor (it expects csv_path)
    data_cfg["csv_path"] = str(train_path)
    if test_path:
        data_cfg["test_path"] = str(test_path)
    if schema_path:
        data_cfg["schema_path"] = str(schema_path)

    # ----------------------------
    # Normalize feature names (strip __unit suffixes)
    # ----------------------------
    input_features = data_cfg.get("input_features")
    targets = data_cfg.get("targets")

    if not input_features or not isinstance(input_features, list):
        raise KeyError(
            "Config must define input_features as a list. "
            "Use unitless names (recommended) or legacy __unit names (will be normalized)."
        )
    if not targets or not isinstance(targets, list):
        raise KeyError(
            "Config must define targets as a list. "
            "Use unitless names (recommended) or legacy __unit names (will be normalized)."
        )

    data_cfg["input_features"] = [_base_name(c) for c in input_features]
    data_cfg["targets"] = [_base_name(c) for c in targets]

    # Also normalize derived_features inputs + name (if present)
    if isinstance(data_cfg.get("derived_features"), list):
        norm_derived = []
        for d in data_cfg["derived_features"]:
            if not isinstance(d, dict):
                norm_derived.append(d)
                continue
            d2 = dict(d)
            if "name" in d2:
                d2["name"] = _base_name(d2["name"])
            if "inputs" in d2 and isinstance(d2["inputs"], list):
                d2["inputs"] = [_base_name(x) for x in d2["inputs"]]
            norm_derived.append(d2)
        data_cfg["derived_features"] = norm_derived

    # ----------------------------
    # Canonical unit conversion policy (recommended)
    # ----------------------------
    # If schema is available and user didn't specify, default to converting both inputs and targets.
    unit_policy = data_cfg.get("unit_policy")
    if not isinstance(unit_policy, dict):
        unit_policy = {}

    if schema_path and not unit_policy:
        unit_policy = {
            "canonical_internal_system": "SI",
            "convert_inputs_to_canonical": True,
            "convert_targets_to_canonical": True,
            "units_source": "schema_path",
        }

    if unit_policy:
        data_cfg["unit_policy"] = unit_policy

    # ----------------------------
    # Hydraulic wear head config (map old multitask block if present)
    # ----------------------------
    # New preferred: top-level hydraulic_wear_head (you asked for this)
    hw_head_cfg = cfg.get("hydraulic_wear_head", {}) or {}

    # Old: profiles[...].multitask.hydraulic_wear_head
    if not hw_head_cfg:
        mt = data_cfg.get("multitask", {}) or {}
        if isinstance(mt, dict):
            hw_head_cfg = (mt.get("hydraulic_wear_head", {}) or {})

    if hw_head_cfg:
        data_cfg["hydraulic_wear_head"] = hw_head_cfg

    # ----------------------------
    # Train
    # ----------------------------
    pp = Preprocessor(data_cfg=data_cfg)
    vfm = VFMModel.from_config(preprocessor=pp, model_cfg=model_cfg)

    print("\n===============================")
    print("🚀 TRAINING CONFIGURATION")
    print("===============================\n")

    print(f"Client name: {client_name}")
    if profile_name:
        print(f"Profile:     {profile_name}")
    print(f"Train CSV:   {data_cfg['csv_path']}")
    if test_path:
        print(f"Test CSV:    {test_path}")
    if schema_path:
        print(f"Schema:      {schema_path}")

    if data_cfg.get("unit_policy"):
        print("\nUnit policy:")
        for k, v in data_cfg["unit_policy"].items():
            print(f"  - {k}: {v}")

    print("\nInput Features (post-derived):")
    for f_name in pp.get_input_feature_names():
        print(f"  - {f_name}")

    print("\nTarget(s):")
    for t_name in pp.get_target_feature_names():
        print(f"  - {t_name}")

    print("\nStarting training...\n")

    if int(model_cfg.get("num_folds", 1)) > 1:
        vfm.train_model_kfold()
    else:
        vfm.train_model()

    # Save artifacts under models/<client_name>/
    out_dir = LAB_ROOT / "models" / str(client_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"vfm_{client_name}.keras"
    vfm.save_model(str(model_path))

    artifact_path = pp.save_artifact(str(out_dir / "preprocessor.joblib"))

    print(f"\n✅ Saved model: {model_path}")
    print(f"✅ Saved preprocessor: {artifact_path}")
    print(f"✅ Inputs: {data_cfg['input_features']}")
    print(f"✅ Targets: {data_cfg['targets']}\n")


if __name__ == "__main__":
    main()
