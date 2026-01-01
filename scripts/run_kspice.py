from __future__ import annotations

import os
import sys
import csv
import random
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import yaml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 2=hide INFO, 3=hide INFO+WARNING
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional: disables oneDNN message + op

# ============================================================
# Paths (repo-anchored)
# ============================================================
# This script should live in: phlux_lab/scripts/
LAB_ROOT = Path(__file__).resolve().parents[1]  # .../phlux_lab
TRAINING_CFG_PATH = LAB_ROOT / "configs" / "training_config.yaml"

# Save outputs under phlux_lab/outputs/
OUTPUT_DIR = LAB_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = str(OUTPUT_DIR / "kspice_vs_ml.csv")

# ============================================================
# K-Spice API Setup
# ============================================================
KSPICE_API_PATH = r"C:\Program Files (x86)\Kongsberg\K-Spice\bin64"
sys.path.append(KSPICE_API_PATH)
try:
    os.add_dll_directory(KSPICE_API_PATH)
except Exception:
    pass

from kspice import Simulator  # type: ignore

# ============================================================
# Project / Timeline
# ============================================================
PROJECT_PATH = r"C:\K-Spice-Projects\Phlux"
TIMELINE = "Engineering"
MDLFILE = "Phlux"
PRMFILE = "Phlux"
VALFILE = "Phlux"

# ============================================================
# Model writes (K-Spice expects internal SI)
# ============================================================
TAG_P_IN = "Streamboundary1:InputPressure"
TAG_T_IN = "Streamboundary1:InputTemperature"
TAG_P_SINK = "Streamboundary2:InputPressure"  # downstream sink boundary pressure (set)

# Optional degradation / wear (plugging)
TAG_HYDRAULIC_WEAR = "PumpAndMotor1:Plugging"

# OPTIONAL: if your model has a writable valve opening tag
TAG_VALVE_OPENING = "ControlValve1:LocalControlSignalIn"

# ============================================================
# Measured tags (K-Spice returns internal SI)
# ============================================================
TAG_P_IN_MEAS = "P_In:MeasuredValue"      # Pa
TAG_T_IN_MEAS = "T_In:MeasuredValue"      # K
TAG_P_OUT_MEAS = "P_Out:MeasuredValue"    # Pa
TAG_FLOW_MEAS = "Flow:MeasuredValue"      # m3/s
TAG_PWR_MEAS = "Pump_Power:MeasuredValue" # W

# NOTE: if you don't actually have a "measured" sink pressure transmitter,
# you can set READ_SINK_MEAS = False below and use the boundary setpoint.
TAG_P_SINK_MEAS = "Streamboundary2:InputPressure"  # Pa (setpoint / optional readback)

# Fluid properties from K-Spice (SI)
TAG_DENSITY_KG_M3 = "PipeFlow1:OutletStream.r"  # kg/m3
TAG_VISCOSITY_PA_S = "PipeFlow1:Viscosity"      # Pa*s

READ_SINK_MEAS = True

# Read list
ALARM_TAGS = [
    TAG_P_IN_MEAS,
    TAG_T_IN_MEAS,
    TAG_P_OUT_MEAS,
    TAG_FLOW_MEAS,
    TAG_PWR_MEAS,
    TAG_DENSITY_KG_M3,
    TAG_VISCOSITY_PA_S,
]
if READ_SINK_MEAS:
    ALARM_TAGS.append(TAG_P_SINK_MEAS)

# ============================================================
# Units helper
# ============================================================
from phlux_lab.utils.units import convert_internal_back_to_user  # type: ignore

# ============================================================
# Default user units for unitless feature names
# (matches what you want to see in CSV + what predict_vfm.py expects)
# ============================================================
DEFAULT_USER_UNITS: Dict[str, str] = {
    "suction_pressure": "bar",
    "discharge_pressure": "bar",
    "sink_pressure": "bar",
    "delta_pressure": "bar",
    "temperature": "degC",
    "speed": "rpm",
    "fluid_density": "kg/m3",
    "fluid_viscosity": "cP",   # we will convert Pa*s -> cP
    "pump_power": "kW",
    "valve_opening": "frac",
    "hydraulic_wear": "frac",
    "q_liquid": "m3/h",
    "valve_dp": "bar",
}

# ============================================================
# Training-config reader (NEW FORMAT)
# ============================================================
def load_training_config() -> Tuple[str, Dict[str, Any], Dict[str, Any], List[str], List[str], Dict[str, Any]]:
    """
    Reads phlux_lab/configs/training_config.yaml (new format).

    Returns:
      client_name, data_cfg, model_cfg, input_cols, target_cols, wear_prob_cfg
    """
    with open(TRAINING_CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    client_name = str(cfg.get("client_name", "")).strip()
    if not client_name:
        raise KeyError("training_config.yaml must define non-empty 'client_name:'")

    model_cfg = cfg.get("model", {}) or {}
    data_cfg = cfg.get("data", {}) or {}

    input_cols = data_cfg.get("input_features")
    target_cols = data_cfg.get("targets")

    if not input_cols or not isinstance(input_cols, list):
        raise KeyError("training_config.yaml must define data.input_features as a non-empty list.")
    if not target_cols or not isinstance(target_cols, list):
        raise KeyError("training_config.yaml must define data.targets as a non-empty list.")

    # Wear probability config (new preferred location in your training script was top-level hydraulic_wear_head)
    wear_prob_cfg: Dict[str, Any] = {}
    hw = cfg.get("hydraulic_wear_head", {}) or {}
    prob = (hw.get("probability", {}) or {}) if isinstance(hw, dict) else {}
    # allow older nested multitask keys too (won't break if absent)
    mt = (model_cfg.get("multitask", {}) or {}) if isinstance(model_cfg, dict) else {}
    prob2 = ((mt.get("hydraulic_wear_head", {}) or {}).get("probability", {}) or {}) if isinstance(mt, dict) else {}

    wear_prob_cfg = dict(prob2) if prob2 else dict(prob)
    # Normalize to predictor interface (expects enabled/method/wear_lo/wear_hi at least)
    if wear_prob_cfg:
        wear_prob_cfg = {
            "enabled": True,
            "method": str(wear_prob_cfg.get("method", "linear")),
            "wear_lo": float(wear_prob_cfg.get("wear_lo", 0.0)),
            "wear_hi": float(wear_prob_cfg.get("wear_hi", wear_prob_cfg.get("wear_lo", 0.0) + 0.01)),
        }

    return client_name, data_cfg, model_cfg, input_cols, target_cols, wear_prob_cfg

# ============================================================
# Feature mapping helpers
# ============================================================
def _col_unit_suffix(col: str) -> str:
    return col.split("__", 1)[1] if "__" in col else ""

def _internal_name(col: str) -> str:
    return col.split("__", 1)[0] if "__" in col else col

def _derived_cols_from_data_cfg(data_cfg: dict) -> set[str]:
    derived = data_cfg.get("derived_features", []) or []
    out = set()
    for d in derived:
        if isinstance(d, dict) and d.get("name"):
            out.add(str(d["name"]))
    return out

def _resolved_unit(col: str) -> str:
    suf = _col_unit_suffix(col)
    if suf:
        return suf
    return DEFAULT_USER_UNITS.get(_internal_name(col), "")

def _user_value_for_col(
    col: str,
    *,
    ks_p_in_pa: float,
    ks_p_out_pa: float,
    ks_p_sink_pa: float,
    ks_t_in_k: float,
    ks_pwr_w: float,
    speed_rpm: float,
    density_kg_m3: float,
    viscosity_pa_s: float,
    valve_opening_frac: float,
    hydraulic_wear_frac: float,
) -> float:
    unit = _resolved_unit(col) or ""
    c = _internal_name(col).lower()

    if "suction" in c and "pressure" in c:
        return float(convert_internal_back_to_user(ks_p_in_pa, unit or "Pa"))

    if ("discharge" in c or "p_out" in c or ("out" in c and "pressure" in c)) and "pressure" in c:
        return float(convert_internal_back_to_user(ks_p_out_pa, unit or "Pa"))

    if "sink" in c and "pressure" in c:
        return float(convert_internal_back_to_user(ks_p_sink_pa, unit or "Pa"))

    if "delta" in c and "pressure" in c:
        dp_pa = ks_p_out_pa - ks_p_in_pa
        return float(convert_internal_back_to_user(dp_pa, unit or "Pa"))

    if "temp" in c or "temperature" in c:
        return float(convert_internal_back_to_user(ks_t_in_k, unit or "K"))

    if "speed" in c:
        return float(convert_internal_back_to_user(speed_rpm, unit or "rpm"))

    if "density" in c:
        return float(convert_internal_back_to_user(density_kg_m3, unit or "kg/m3"))

    if "viscos" in c:
        # K-Spice gives Pa*s; your features are usually in cP
        if unit.lower() in {"cp", "cpoise"}:
            return float(viscosity_pa_s * 1000.0)  # 1 Pa*s = 1000 cP
        return float(convert_internal_back_to_user(viscosity_pa_s, unit or "Pa*s"))

    if "valve" in c and ("opening" in c or "open" in c or "position" in c):
        return float(valve_opening_frac)

    if "wear" in c or "plug" in c or "degrad" in c:
        return float(hydraulic_wear_frac)

    if "pump_power" in c or ("power" in c and "pump" in c):
        return float(convert_internal_back_to_user(ks_pwr_w, unit or "W"))

    if c.startswith("head_proxy"):
        # If you have this as a derived feature, your Preprocessor can compute it.
        # We keep it here in case your model expects it as an explicit input.
        dp_bar = float(convert_internal_back_to_user(ks_p_out_pa - ks_p_in_pa, "bar"))
        dens = float(convert_internal_back_to_user(density_kg_m3, "kg/m3"))
        return dp_bar / max(dens, 1e-12)

    if "valve_dp" in c:
        # dp across valve == discharge - sink (if that's how you defined it)
        dp_pa = ks_p_out_pa - ks_p_sink_pa
        return float(convert_internal_back_to_user(dp_pa, unit or "Pa"))

    raise RuntimeError(f"Unhandled input feature column: '{col}'")

def build_feature_inputs_for_predictor(
    cols: List[str],
    *,
    ks_p_in_pa: float,
    ks_p_out_pa: float,
    ks_p_sink_pa: float,
    ks_t_in_k: float,
    ks_pwr_w: float,
    speed_rpm: float,
    density_kg_m3: float,
    viscosity_pa_s: float,
    valve_opening_frac: float,
    hydraulic_wear_frac: float,
) -> Dict[str, Dict[str, float]]:
    """
    Build predictor-style input dict:
      {"suction_pressure": {"value": 2.5, "unit": "bar"}, ...}
    """
    out: Dict[str, Dict[str, float]] = {}
    for col in cols:
        name = _internal_name(col)
        unit = _resolved_unit(col) or DEFAULT_USER_UNITS.get(name, "")
        val = _user_value_for_col(
            col,
            ks_p_in_pa=ks_p_in_pa,
            ks_p_out_pa=ks_p_out_pa,
            ks_p_sink_pa=ks_p_sink_pa,
            ks_t_in_k=ks_t_in_k,
            ks_pwr_w=ks_pwr_w,
            speed_rpm=speed_rpm,
            density_kg_m3=density_kg_m3,
            viscosity_pa_s=viscosity_pa_s,
            valve_opening_frac=valve_opening_frac,
            hydraulic_wear_frac=hydraulic_wear_frac,
        )
        out[name] = {"value": float(val), "unit": str(unit) if unit else ""}
    return out

def _pick_flow_key(pred: Dict[str, Any], targets: List[str]) -> Optional[str]:
    # Prefer target name if present
    for t in targets:
        if t in pred:
            return t
    # Common fallbacks
    for k in ("q_liquid", "flow"):
        if k in pred:
            return k
    # Last resort: first numeric key not probability
    skip = {"hydraulic_wear_probability_pct", "degradation_probability_pct"}
    for k, v in pred.items():
        if k in skip:
            continue
        if isinstance(v, (int, float)):
            return k
    return None

# ============================================================
# ML Predictor (artifact-based) - same call signature as predict_vfm.py
# ============================================================
USE_ML_MODEL = True
PRINT_ML_DEBUG = False  # set True if you want to print every case inputs

predictor = None
CLIENT_NAME, DATA_CFG, MODEL_CFG, INPUT_COLS, TARGET_COLS, WEAR_PROB_CFG = load_training_config()

MODEL_PATH = LAB_ROOT / "models" / CLIENT_NAME / f"vfm_{CLIENT_NAME}.keras"
PREPROCESSOR_PATH = LAB_ROOT / "models" / CLIENT_NAME / "preprocessor.joblib"

if USE_ML_MODEL:
    try:
        from phlux_lab.utils.predictor import VfmPredictor  # type: ignore
        predictor = VfmPredictor.from_paths(
            model_path=str(MODEL_PATH),
            preprocessor_path=str(PREPROCESSOR_PATH),
        )
        print(f"✅ Loaded ML predictor from: {MODEL_PATH}")
        print(f"✅ Loaded preprocessor from: {PREPROCESSOR_PATH}")
    except Exception as e:
        print(f"⚠️ Could not load ML predictor: {e}")
        USE_ML_MODEL = False

# ============================================================
# Sampling controls (internal SI for K-Spice boundaries)
# ============================================================
N_CASES = 100

P_IN_RANGE_PA = (1.0e5, 3.0e5)
P_SINK_RANGE_PA = (2.0e5, 5.0e5)
T_IN_RANGE_K = (288.15, 298.15)  # 15–25 °C

VALVE_OPENING_RANGE_FRAC = (0.1, 0.75)
HYDRAULIC_WEAR_RANGE = (0.0, 0.1)
DEFAULT_SPEED_RPM = 1800.0

# ============================================================
# K-Spice Boot
# ============================================================
sim = Simulator(PROJECT_PATH)
tl = sim.activate_timeline(TIMELINE)
tl.load(MDLFILE, PRMFILE, VALFILE)
tl.initialize()
tl.set_speed(1000)
APP = tl.applications[0].name

# ============================================================
# Run cases
# ============================================================
rows: List[List[float]] = []

wear_series = np.random.uniform(*HYDRAULIC_WEAR_RANGE, N_CASES)
valve_series = np.random.uniform(*VALVE_OPENING_RANGE_FRAC, N_CASES)

for i in range(N_CASES):
    p_in = random.uniform(*P_IN_RANGE_PA)
    p_sink = random.uniform(*P_SINK_RANGE_PA)
    t_in = random.uniform(*T_IN_RANGE_K)

    hydraulic_wear_frac = float(wear_series[i])
    valve_opening_frac = float(valve_series[i])

    # model writes (SI)
    tl.set_value(APP, TAG_P_IN, float(p_in))
    tl.set_value(APP, TAG_T_IN, float(t_in))
    tl.set_value(APP, TAG_P_SINK, float(p_sink))
    tl.set_value(APP, TAG_HYDRAULIC_WEAR, float(hydraulic_wear_frac))
    if TAG_VALVE_OPENING:
        tl.set_value(APP, TAG_VALVE_OPENING, float(valve_opening_frac))

    tl.run_for(timedelta(seconds=60))

    # robust read
    vals = tl.get_values(APP, ALARM_TAGS)
    out: Dict[str, float] = {}
    for tag, val in zip(ALARM_TAGS, vals):
        if val is None:
            raise RuntimeError(
                f"K-Spice returned None for tag '{tag}'. "
                f"Check the tag exists in application '{APP}' and is available after run_for()."
            )
        out[tag] = float(val)

    # measured (SI)
    ks_p_in_pa = out[TAG_P_IN_MEAS]
    ks_t_in_k = out[TAG_T_IN_MEAS]
    ks_p_out_pa = out[TAG_P_OUT_MEAS]
    ks_flow_m3s = out[TAG_FLOW_MEAS]
    ks_pwr_w = out[TAG_PWR_MEAS]

    # sink pressure
    ks_p_sink_pa = out[TAG_P_SINK_MEAS] if READ_SINK_MEAS else float(p_sink)
    speed_rpm = float(DEFAULT_SPEED_RPM)

    # fluid props (SI from K-Spice)
    density_kg_m3 = out[TAG_DENSITY_KG_M3]
    viscosity_pa_s = out[TAG_VISCOSITY_PA_S]

    # CSV conversions (user units)
    ks_p_in_bar = float(convert_internal_back_to_user(ks_p_in_pa, "bar"))
    ks_p_out_bar = float(convert_internal_back_to_user(ks_p_out_pa, "bar"))
    ks_p_sink_bar = float(convert_internal_back_to_user(ks_p_sink_pa, "bar"))
    ks_dp_bar = float(convert_internal_back_to_user(ks_p_out_pa - ks_p_in_pa, "bar"))
    ks_t_c = float(convert_internal_back_to_user(ks_t_in_k, "degC"))
    ks_pwr_kw = float(convert_internal_back_to_user(ks_pwr_w, "kW"))
    ks_flow_m3h = float(convert_internal_back_to_user(ks_flow_m3s, "m3/h"))
    speed_csv = float(convert_internal_back_to_user(speed_rpm, "rpm"))
    dens_csv = float(convert_internal_back_to_user(density_kg_m3, "kg/m3"))
    visc_csv_cp = float(viscosity_pa_s * 1000.0)  # 1 Pa*s = 1000 cP

    # ML predict (force user units)
    ml_flow_pred_m3h = float("nan")
    ml_wear_prob_pct = float("nan")
    flow_err_pct = float("nan")

    if USE_ML_MODEL and predictor is not None:
        cols_for_model = predictor.input_cols if getattr(predictor, "input_cols", None) else INPUT_COLS

        # Exclude derived feature names; Preprocessor computes them
        derived_cols = _derived_cols_from_data_cfg(DATA_CFG)
        base_cols = [c for c in cols_for_model if _internal_name(c) not in derived_cols]

        feature_inputs = build_feature_inputs_for_predictor(
            base_cols,
            ks_p_in_pa=ks_p_in_pa,
            ks_p_out_pa=ks_p_out_pa,
            ks_p_sink_pa=ks_p_sink_pa,
            ks_t_in_k=ks_t_in_k,
            ks_pwr_w=ks_pwr_w,
            speed_rpm=speed_rpm,
            density_kg_m3=density_kg_m3,
            viscosity_pa_s=viscosity_pa_s,
            valve_opening_frac=valve_opening_frac,
            hydraulic_wear_frac=hydraulic_wear_frac,
        )

        if PRINT_ML_DEBUG and hasattr(predictor, "debug_print_model_inputs"):
            predictor.debug_print_model_inputs(feature_inputs)

        pred = predictor.predict(
            feature_inputs=feature_inputs,
            wear_probability_cfg=WEAR_PROB_CFG if WEAR_PROB_CFG.get("enabled", False) else None,
            return_units="user",
        )

        flow_key = _pick_flow_key(pred, TARGET_COLS)
        if flow_key is not None and isinstance(pred.get(flow_key), (int, float)):
            ml_flow_pred_m3h = float(pred[flow_key])

        # Probability key from predictor interface
        for k in ("hydraulic_wear_probability_pct", "degradation_probability_pct"):
            if k in pred and isinstance(pred.get(k), (int, float)):
                ml_wear_prob_pct = float(pred[k])
                break

        if np.isfinite(ml_flow_pred_m3h) and abs(ks_flow_m3h) > 1e-9:
            flow_err_pct = 100.0 * abs(ml_flow_pred_m3h - ks_flow_m3h) / abs(ks_flow_m3h)

    rows.append([
        ks_p_in_bar,
        ks_t_c,
        ks_p_out_bar,
        ks_p_sink_bar,
        ks_dp_bar,
        speed_csv,
        dens_csv,
        visc_csv_cp,
        ks_pwr_kw,
        hydraulic_wear_frac,
        valve_opening_frac,
        ks_flow_m3h,
        ml_flow_pred_m3h,
        ml_wear_prob_pct,
        flow_err_pct,
    ])

# ============================================================
# Save CSV
# ============================================================
header = [
    "ks_p_in__bar",
    "ks_t_in__degC",
    "ks_discharge_p__bar",
    "ks_sink_p__bar",
    "ks_dp__bar",
    "speed__rpm",
    "fluid_density__kg/m3",
    "fluid_viscosity__cP",
    "ks_pump_power__kW",
    "hydraulic_wear__frac",
    "valve_opening__frac",
    "ks_flow__m3_h",
    "ml_flow_pred__m3_h",
    "ml_hydraulic_wear_probability__pct",
    "flow_abs_error__pct",
]

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)

print(f"\n✅ Saved file: {os.path.abspath(CSV_PATH)}")
print("Done.")
