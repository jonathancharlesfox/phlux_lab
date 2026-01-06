from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
from typing import Dict, Tuple, Union

from phlux_lab.utils.predictor import VfmPredictor  # type: ignore

# ============================================================
# ONE-OFF STACKED PREDICTION (FLOW -> WEAR -> FLOW_CORRECTION)
#
# ✅ This is for SINGLE-POINT testing with manual inputs.
#
# You ONLY edit:
#   1) BASE_INPUTS  (measured physical variables)
#   2) STAGE2_INPUTS (optional measured vars available after flow exists)
#
# The script automatically:
#   - runs flow -> produces q_liquid_pred
#   - injects q_liquid_pred into wear + correction (if those stages expect it)
#   - runs wear -> produces hydraulic_wear_pred
#   - injects hydraulic_wear_pred into correction (if expected)
#   - runs correction -> produces delta_q_liquid (or your correction target)
#   - prints all three model outputs
# ============================================================

CLIENT = "ClientA"
LAB_ROOT = Path(__file__).resolve().parents[1]  # .../phlux_lab

# --- Stage artifact paths (update filenames if yours differ) ---
FLOW_MODEL = LAB_ROOT / "models" / CLIENT / "flow" / "vfm_flow.keras"
FLOW_PP    = LAB_ROOT / "models" / CLIENT / "flow" / "preprocessor_flow.joblib"

WEAR_MODEL = LAB_ROOT / "models" / CLIENT / "wear" / "vfm_wear.keras"
WEAR_PP    = LAB_ROOT / "models" / CLIENT / "wear" / "preprocessor_wear.joblib"

CORR_MODEL = LAB_ROOT / "models" / CLIENT / "flow_correction" / "vfm_flow_correction.keras"
CORR_PP    = LAB_ROOT / "models" / CLIENT / "flow_correction" / "preprocessor_flow_correction.joblib"


# ------------------------------------------------------------
# USER INPUTS (EDIT THESE)
# ------------------------------------------------------------
ValueOrTuple = Union[float, int, Tuple[float, str]]

# 1️⃣ Base physical inputs (used by ALL stages)
# Put your measured values here. Units are accepted but not interpreted here;
# the Preprocessor handles unit conversion using the config it was built from.
BASE_INPUTS: Dict[str, ValueOrTuple] = {
    "suction_pressure": (2.7, "bar"),
    "discharge_pressure": (7.8, "bar"),
    "speed": (1787, "rpm"),
    "temperature": (16.3, "C"),
    "fluid_density": (996.4, "kg/m3"),
    "fluid_viscosity": (1.09, "cP"),
    "valve_opening": (0.23, "frac"),
    "pump_power": (7.0, "kW"),

    # Optional: if you provide suction+discharge but not delta_pressure,
    # the script computes it as discharge - suction.
    # "delta_pressure": (5.1, "bar"),
}

# 2️⃣ Optional measured inputs that only apply after flow exists (wear/correction)
# Include only what you really have. Example shown:
STAGE2_INPUTS: Dict[str, ValueOrTuple] = {
    # If your wear/correction stages use derived 'specific_power',
    # you usually need pump_power available as a source.
    # Uncomment if you have it:
    # "pump_power": (7.16, "kW"),
}

# 3️⃣ MODEL-GENERATED INPUTS (do not edit)
# q_liquid_pred and hydraulic_wear_pred are injected automatically.


def _compute_delta_pressure_if_missing(base: Dict[str, float]) -> None:
    if "delta_pressure" in base:
        return
    if "discharge_pressure" in base and "suction_pressure" in base:
        base["delta_pressure"] = float(base["discharge_pressure"] - base["suction_pressure"])


def _merge_inputs_for_stage(
    predictor: VfmPredictor,
    base: Dict[str, float],
    stage2: Dict[str, float],
    *,
    q_liquid_pred: float | None = None,
    hydraulic_wear_pred: float | None = None,
) -> Dict[str, float]:
    """
    Build the feature dict to pass to predictor.predict_one().

    IMPORTANT:
      - We include ALL manual inputs (base + stage2), even if they are not
        direct model inputs. The Preprocessor may need extra/source columns
        to compute derived features (e.g. specific_power).
      - We only inject *_pred columns if this stage actually expects them.
    """
    feats = {**base, **stage2}
    needed = set(predictor.input_cols)

    if "q_liquid_pred" in needed and q_liquid_pred is not None:
        feats["q_liquid_pred"] = float(q_liquid_pred)

    if "hydraulic_wear_pred" in needed and hydraulic_wear_pred is not None:
        feats["hydraulic_wear_pred"] = float(hydraulic_wear_pred)

    return feats


def main() -> None:
    # Normalize manual inputs: (value, unit) -> value
    base = VfmPredictor.normalize_feature_inputs(BASE_INPUTS)
    stage2 = VfmPredictor.normalize_feature_inputs(STAGE2_INPUTS)

    _compute_delta_pressure_if_missing(base)

    # Load stage predictors
    flow = VfmPredictor.from_paths(FLOW_MODEL, FLOW_PP)
    wear = VfmPredictor.from_paths(WEAR_MODEL, WEAR_PP)
    corr = VfmPredictor.from_paths(CORR_MODEL, CORR_PP)

    print("\\n[DEBUG] Flow model expected inputs:")
    for c in flow.input_cols:
        print(f"  - {c}")

    # -------------------------
    # Stage 1: FLOW
    # -------------------------
    print("\\n=== STAGE 1: FLOW ===")
    flow_feats = _merge_inputs_for_stage(flow, base, stage2)
    out1 = flow.predict_one(flow_feats)
    for k, v in out1.items():
        print(f"{k}: {v}")

    # Prefer q_liquid if present, otherwise first output
    q_liquid_pred = float(out1.get("q_liquid", next(iter(out1.values()))))

    print("\\n[DEBUG] Wear model expected inputs:")
    for c in wear.input_cols:
        print(f"  - {c}")

    # -------------------------
    # Stage 2: WEAR
    # -------------------------
    print("\\n=== STAGE 2: WEAR ===")
    wear_feats = _merge_inputs_for_stage(wear, base, stage2, q_liquid_pred=q_liquid_pred)
    out2 = wear.predict_one(wear_feats)
    for k, v in out2.items():
        print(f"{k}: {v}")

    hydraulic_wear_pred = float(out2.get("hydraulic_wear", next(iter(out2.values()))))

    print("\\n[DEBUG] Flow-correction model expected inputs:")
    for c in corr.input_cols:
        print(f"  - {c}")

    # -------------------------
    # Stage 3: FLOW CORRECTION
    # -------------------------
    print("\\n=== STAGE 3: FLOW CORRECTION ===")
    corr_feats = _merge_inputs_for_stage(
        corr,
        base,
        stage2,
        q_liquid_pred=q_liquid_pred,
        hydraulic_wear_pred=hydraulic_wear_pred,
    )
    out3 = corr.predict_one(corr_feats)
    for k, v in out3.items():
        print(f"{k}: {v}")

    # -------------------------
    # Summary
    # -------------------------
    print("\\n=== SUMMARY (all stage predictions) ===")
    print(f"q_liquid_pred: {q_liquid_pred}")
    print(f"hydraulic_wear_pred: {hydraulic_wear_pred}")
    for k, v in out3.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
