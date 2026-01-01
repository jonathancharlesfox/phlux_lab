import argparse
import yaml
import sys
import os
from pathlib import Path

# Silence TF chatter (keep if you want; harmless even if TF not used here)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 2=hide INFO, 3=hide INFO+WARNING
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional

# --- Paths ---
HERE = Path(__file__).resolve().parent          # phlux_lab/scripts
LAB_ROOT = HERE.parent                          # phlux_lab
DEFAULT_CONFIG = LAB_ROOT / "configs" / "synthgen_config.yaml"

from phlux_lab.datagen.centrifugal.generator import CentrifugalPumpDatasetGenerator # type: ignore


def main() -> None:
    
    parser = argparse.ArgumentParser(description="Synthetic dataset generator for Phlux VFM.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help=f"Path to YAML config. Default: {DEFAULT_CONFIG}",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)

    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {cfg_path}\n"
            "Update --config argument or DEFAULT_CONFIG in generate_synth.py."
        )

    print(f"Using config: {cfg_path}")

    # Load YAML header to detect equipment type
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    eq_type_raw = cfg.get("equipment_type", "centrifugal")
    eq_type = str(eq_type_raw).strip().lower()
    print(f"Detected equipment_type='{eq_type}'")

    if eq_type == "centrifugal":
        GenClass = CentrifugalPumpDatasetGenerator
    elif eq_type in ("positivedisplacement", "positive_displacement", "pd"):
        raise NotImplementedError("PD pump generator not yet added.")
    elif eq_type == "compressor":
        raise NotImplementedError("Compressor generator not yet added.")
    else:
        raise ValueError(f"Unknown equipment_type: {eq_type}")

    gen = GenClass.from_yaml(str(cfg_path))
    print("  • Generating synthetic data...")
    train_path, test_path = gen.export_csv_and_schema()

    print("\n✔ Synthetic datasets generated:")
    print(f"  Train: {train_path}")
    print(f"  Test : {test_path}\n")


if __name__ == "__main__":
    main()
