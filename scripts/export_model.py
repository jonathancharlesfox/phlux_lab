from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# =========================
# USER SETTINGS
# =========================
CLIENT_NAME = "ClientA"
VERSION = "v1"

LAB_ROOT = Path(__file__).resolve().parents[1]   # C:\phlux_lab
CONFIG_PATH = LAB_ROOT / "configs" / "training_config.yaml"

MODEL_DIR = LAB_ROOT / "models" / CLIENT_NAME
MODEL_PATH = MODEL_DIR / f"vfm_{CLIENT_NAME}.keras"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"

OUTPUT_ROOT = Path(r"C:\phlux_models")
# =========================


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _split_name_unit(col: str) -> Tuple[str, Optional[str]]:
    """
    "q_liquid__m3/h" -> ("q_liquid", "m3/h")
    "speed__rpm"     -> ("speed", "rpm")
    "sink_pressure"  -> ("sink_pressure", None)
    """
    s = str(col).strip()
    if "__" in s:
        name, unit = s.split("__", 1)
        return name.strip(), (unit.strip() or None)
    return s, None


def _infer_schema_path_from_train_csv(train_csv: str) -> Path:
    p = str(train_csv)
    if p.endswith("_train.csv"):
        return Path(p.replace("_train.csv", "_schema.yaml"))
    if p.endswith(".csv"):
        return Path(p[:-4] + "_schema.yaml")
    return Path(p + "_schema.yaml")


def _resolve_schema_path(cfg: Dict[str, Any]) -> Optional[Path]:
    """
    Best effort:
      1) cfg.data.schema_path
      2) infer from cfg.data.train_path or cfg.data.csv_path
      3) legacy: infer from profiles[active].train_path or csv_path
    """
    data = cfg.get("data")
    if isinstance(data, dict) and data:
        if data.get("schema_path"):
            p = Path(str(data["schema_path"]))
            return p
        train_path = data.get("train_path") or data.get("csv_path")
        if train_path:
            guess = _infer_schema_path_from_train_csv(str(train_path))
            return guess

    # legacy
    profiles = cfg.get("profiles", {}) or {}
    active = cfg.get("active_profile")
    prof = None
    if active and active in profiles:
        prof = profiles.get(active) or {}
    elif isinstance(profiles, dict) and profiles:
        first = next(iter(profiles.keys()))
        prof = profiles.get(first) or {}

    if isinstance(prof, dict) and prof:
        if prof.get("schema_path"):
            return Path(str(prof["schema_path"]))
        train_path = prof.get("train_path") or prof.get("csv_path")
        if train_path:
            return _infer_schema_path_from_train_csv(str(train_path))

    return None


def _load_units_from_dataset_schema(schema_path: Path) -> Dict[str, str]:
    """
    Expects synthgen dataset schema style:
      columns:
        - name: suction_pressure
          user_unit: bar
    Returns {name: user_unit}
    """
    if not schema_path.exists():
        raise FileNotFoundError(f"Dataset schema YAML not found: {schema_path}")

    schema = _load_yaml(schema_path)
    cols = schema.get("columns", []) or []
    out: Dict[str, str] = {}
    for c in cols:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name", "")).strip()
        unit = str(c.get("user_unit", "")).strip()
        if name and unit:
            out[name] = unit
    return out


def _extract_from_data(cfg: Dict[str, Any]) -> Optional[Tuple[List[str], List[str], List[Dict[str, Any]]]]:
    data = cfg.get("data")
    if not isinstance(data, dict) or not data:
        return None
    ins = data.get("input_features", []) or []
    outs = data.get("targets", []) or []
    derived = data.get("derived_features", []) or []
    if not ins or not outs:
        raise KeyError("training_config.yaml data.input_features and data.targets are required.")
    return ins, outs, derived


def _extract_from_legacy_profiles(cfg: Dict[str, Any]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    profiles = cfg.get("profiles", {}) or {}
    if not isinstance(profiles, dict) or not profiles:
        raise KeyError(
            "training_config.yaml must contain either:\n"
            "  - a top-level 'data:' block, OR\n"
            "  - legacy 'profiles:' + 'active_profile'."
        )
    active = cfg.get("active_profile")
    if active and active in profiles:
        prof = profiles[active] or {}
    else:
        first_key = next(iter(profiles.keys()))
        prof = profiles[first_key] or {}

    ins = prof.get("input_features", []) or []
    outs = prof.get("targets", []) or []
    derived = prof.get("derived_features", []) or []
    if not ins or not outs:
        raise KeyError("Legacy profile must define input_features and targets.")
    return ins, outs, derived


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Training config not found: {CONFIG_PATH}")

    cfg = _load_yaml(CONFIG_PATH)

    got = _extract_from_data(cfg)
    if got is None:
        input_features_raw, targets_raw, derived_features = _extract_from_legacy_profiles(cfg)
    else:
        input_features_raw, targets_raw, derived_features = got

    # Resolve dataset schema yaml (for units)
    schema_path = _resolve_schema_path(cfg)
    if schema_path is None:
        raise KeyError(
            "Could not resolve dataset schema path. Add one of these to training_config.yaml:\n"
            "  data:\n"
            "    schema_path: <path to *_schema.yaml>\n"
            "OR ensure data.train_path points to *_train.csv so exporter can infer *_schema.yaml."
        )

    units_by_name = _load_units_from_dataset_schema(Path(schema_path))

    # Build inputs/outputs using base names; units from dataset schema mapping
    inputs_required: List[Dict[str, Any]] = []
    for c in input_features_raw:
        name, _ = _split_name_unit(c)
        inputs_required.append(
            {"name": name, "dtype": "float", "unit": units_by_name.get(name), "required": True}
        )

    outputs: List[Dict[str, Any]] = []
    for c in targets_raw:
        name, _ = _split_name_unit(c)
        outputs.append({"name": name, "dtype": "float", "unit": units_by_name.get(name)})

    derived: List[Dict[str, Any]] = []
    for d in derived_features:
        if not isinstance(d, dict):
            continue
        dname = str(d.get("name", "")).strip()
        if not dname:
            continue
        dname, _ = _split_name_unit(dname)
        deps = d.get("inputs", []) or []
        deps = [_split_name_unit(x)[0] for x in deps]
        derived.append(
            {
                "name": dname,
                "dtype": "float",
                "unit": units_by_name.get(dname) or d.get("unit"),
                "inputs": deps,
            }
        )

    # Output folder
    out_dir = OUTPUT_ROOT / f"{CLIENT_NAME}_{VERSION}"
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(MODEL_PATH, out_dir / "model.keras")
    shutil.copy2(PREPROCESSOR_PATH, out_dir / "preprocessor.joblib")

    schema_json = {
        "schema_version": "1.0",
        "client": CLIENT_NAME,
        "version": VERSION,
        "unit_policy": {
            "model_internal_units": "canonical",
            "default_return_units": "user",
            "client_input_units_source": "schema.json",
        },
        "inputs": {
            "required": inputs_required,
            "derived": derived,
        },
        "outputs": outputs,
    }

    metadata_json = {
        "client": CLIENT_NAME,
        "version": VERSION,
        "exported_utc": _utc_now_iso(),
        "included_files": [
            "model.keras",
            "preprocessor.joblib",
            "schema.json",
            "metadata.json",
        ],
        "source_model": str(MODEL_PATH),
        "source_preprocessor": str(PREPROCESSOR_PATH),
        "source_training_config": str(CONFIG_PATH),
        "source_dataset_schema": str(schema_path),
    }

    (out_dir / "schema.json").write_text(json.dumps(schema_json, indent=2), encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps(metadata_json, indent=2), encoding="utf-8")

    print("✅ Export complete")
    print(f"   Client : {CLIENT_NAME}")
    print(f"   Version: {VERSION}")
    print(f"   Output : {out_dir}")
    print("   Files  : model.keras, preprocessor.joblib, schema.json, metadata.json")


if __name__ == "__main__":
    main()
