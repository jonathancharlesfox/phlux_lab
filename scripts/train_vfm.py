from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from pathlib import Path
from typing import Any, Dict

import yaml

from phlux_lab.ml.vfm_model import VFMModel  # type: ignore
from phlux_lab.utils.preprocessor import Preprocessor  # type: ignore

STAGE_NAME = "flow"

SCRIPT_DIR = Path(__file__).resolve().parent
LAB_ROOT = SCRIPT_DIR.parent
REPO_ROOT = LAB_ROOT.parent
DEFAULT_CONFIG_PATH = LAB_ROOT / "configs" / "training_config.yaml"


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
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError("training_config.yaml must parse into a dict")
    return obj


def main() -> None:
    cfg = _load_yaml(DEFAULT_CONFIG_PATH)
    client_name = str(cfg.get("client_name") or (cfg.get("project", {}) or {}).get("client_name") or "")
    if not client_name:
        raise KeyError("training_config.yaml must define client_name (or project.client_name)")

    stage_cfg = (cfg.get("models", {}) or {}).get(STAGE_NAME)
    if not stage_cfg:
        raise KeyError("training_config.yaml missing models.flow")

    model_block = stage_cfg.get("model", {}) if isinstance(stage_cfg.get("model", {}), dict) else {}

    out_dir = LAB_ROOT / "models" / client_name / STAGE_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    pp = Preprocessor.from_config(stage_cfg, dataset="train")
    vfm = VFMModel.from_config(
        preprocessor=pp,
        model_cfg=model_block,
        log_dir=str(_resolve_path("phlux_lab/logs")),
        model_dir=str(out_dir),
    )

    num_folds = int(model_block.get("num_folds", 1))
    if num_folds and num_folds > 1:
        vfm.train_model_kfold()
    else:
        vfm.train_model()

    artifact_name = str(stage_cfg.get("artifact_name", "vfm_flow"))
    model_filename = f"{artifact_name}.keras" if not artifact_name.endswith(".keras") else artifact_name
    pp_filename = f"preprocessor_{STAGE_NAME}.joblib"

    vfm.save_model(filename=model_filename)
    pp.save_artifact(str(out_dir / pp_filename))

    print(f"✅ Trained {STAGE_NAME} model for {client_name}")
    print(f"   - model: {out_dir / model_filename}")
    print(f"   - pp:    {out_dir / pp_filename}")


if __name__ == "__main__":
    main()
