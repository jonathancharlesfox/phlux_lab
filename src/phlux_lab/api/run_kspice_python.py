import os
import sys
import csv
import random
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Tuple

import numpy as np
import yaml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 2=hide INFO, 3=hide INFO+WARNING
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional: disables oneDNN message + op

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
#   - We set sink pressure downstream of the valve (P_SINK)
#   - We set inlet pressure and inlet temperature
#   - Discharge pressure (P_OUT) is NOT set; it is measured
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
TAG_VISCOSITY_PA_S = "PipeFlow1:Viscosity"      # Pa*s  (you said KS provides Pa*s)

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
from phlux_lab.utils.units import convert_internal_back_to_user

# ============================================================
# Training-config-driven feature definitions
# ============================================================
TRAINING_CFG_PATH = Path("configs") / "training_config.yaml"

def load_training_profile() -> Tuple[str, dict, dict, List[str], List[str]]:
    """
    Returns:
      active_profile_name, profile_cfg, model_cfg, input_cols, target_cols

    IMPORTANT:
      - input_cols and target_cols are now defined explicitly in training_config.yaml
        under profiles:<active_profile>:
          input_features: [...]
          targets: [...]
    """
    with open(TRAINING_CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    profiles = cfg.get("profiles", {})
    if not profiles:
        raise KeyError("training_config.yaml must contain a non-empty 'profiles:' block.")

    active = cfg.get("active_profile") or next(iter(profiles.keys()))
    if active not in profiles:
        raise KeyError(f"active_profile='{active}' not found. Available: {list(profiles.keys())}")

    profile_cfg = profiles[active]
    model_cfg = cfg.get("model", {}) or {}

    input_cols = profile_cfg.get("input_features")
    target_cols = profile_cfg.get("targets")

    if not input_cols or not isinstance(input_cols, list):
        raise KeyError(
            f"Profile '{active}' must define 'input_features' as a list of CSV column names "
            f"(including units, e.g. suction_pressure__bar)."
        )
    if not target_cols or not isinstance(target_cols, list):
        raise KeyError(
            f"Profile '{active}' must define 'targets' as a list of CSV column names "
            f"(including units, e.g. q_liquid__m3/h)."
        )

    return active, profile_cfg, model_cfg, input_cols, target_cols


# ============================================================
# ML Predictor (artifact-based)
# ============================================================
USE_ML_MODEL = True
PRINT_ML_DEBUG = False  # set True if you want to print every case inputs

predictor = None
active_profile, profile_cfg, model_cfg, INPUT_COLS, TARGET_COLS = load_training_profile()

DEFAULT_MODEL_PATH = Path("models") / active_profile / f"vfm_{active_profile}.keras"
ML_MODEL_PATH = str(DEFAULT_MODEL_PATH)

if USE_ML_MODEL:
    try:
        from phlux_lab.utils.predictor import VfmPredictor
        predictor = VfmPredictor.from_paths(ML_MODEL_PATH)
        print(f"✅ Loaded ML predictor from: {ML_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Could not load ML predictor: {e}")
        USE_ML_MODEL = False

# ============================================================
# Sampling controls (internal SI for K-Spice boundaries)
# ============================================================
N_CASES = 100
CSV_PATH = "kspice_vs_ml.csv"

P_IN_RANGE_PA = (1.0e5, 3.0e5)
P_SINK_RANGE_PA = (2.0e5, 5.0e5)
T_IN_RANGE_K = (288.15, 298.15)  # 15–25 °C

VALVE_OPENING_RANGE_FRAC = (0.4, 0.7)
HYDRAULIC_WEAR_RANGE = (0.06, 0.06)
DEFAULT_SPEED_RPM = 1800.0

# ============================================================
# Feature mapping helpers
# ============================================================
def _col_unit_suffix(col: str) -> str:
    return col.split("__", 1)[1] if "__" in col else ""


def _internal_name(col: str) -> str:
    return col.split("__", 1)[0] if "__" in col else col


def _derived_output_cols_from_profile(profile_cfg: dict) -> set[str]:
    derived = profile_cfg.get("derived_features", []) or []
    out = set()
    for d in derived:
        name = d.get("name")
        unit = d.get("unit")
        if name and unit:
            out.add(f"{name}__{unit}")
    return out


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
    viscosity_pa_s: float,  # SI from K-Spice (Pa*s)
    valve_opening_frac: float,
    hydraulic_wear_frac: float,
) -> float:
    unit = _col_unit_suffix(col) or ""
    c = col.lower()

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
        # KS gives kg/m3 already; convert only if model wants something else
        return float(convert_internal_back_to_user(density_kg_m3, unit or "kg/m3"))

    if "viscos" in c:
        # IMPORTANT: KS gives Pa*s, but your ML features may be in cP.
        # Do explicit conversion for the ML model when unit suffix requests cP.
        if unit.lower() in {"cp", "cpoise"}:
            return float(viscosity_pa_s * 1000.0)  # 1 Pa*s = 1000 cP
        # otherwise use the shared converter (or return SI)
        return float(convert_internal_back_to_user(viscosity_pa_s, unit or "Pa*s"))

    if "valve" in c and ("opening" in c or "open" in c or "position" in c):
        return float(valve_opening_frac)

    if "wear" in c or "plug" in c or "degrad" in c:
        return float(hydraulic_wear_frac)

    if "pump_power" in c or ("power" in c and "pump" in c):
        return float(convert_internal_back_to_user(ks_pwr_w, unit or "W"))

    if col.startswith("head_proxy"):
        dp_bar = float(convert_internal_back_to_user(ks_p_out_pa - ks_p_in_pa, "bar"))
        dens = float(convert_internal_back_to_user(density_kg_m3, "kg/m3"))
        return dp_bar / max(dens, 1e-12)

    raise RuntimeError(f"Unhandled input feature column: '{col}'")


def build_feature_values(
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
) -> Dict[str, float]:
    fv: Dict[str, float] = {}
    for col in cols:
        fv[_internal_name(col)] = _user_value_for_col(
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
    return fv


def _extract_ml_outputs(pred_dict: Dict[str, float]) -> Tuple[float, float]:
    """
    Returns:
      (ml_flow_pred, ml_deg_prob_pct)

    - Works for single-head: deg_prob_pct = NaN
    - Works for two-head: both returned
    """
    deg_prob_pct = float("nan")
    if "degradation_probability_pct" in pred_dict:
        deg_prob_pct = float(pred_dict["degradation_probability_pct"])

    # Flow: pick the first key that is NOT the degradation score
    flow_keys = [k for k in pred_dict.keys() if k != "degradation_probability_pct"]
    if not flow_keys:
        return float("nan"), deg_prob_pct

    ml_flow_pred = float(pred_dict[flow_keys[0]])
    return ml_flow_pred, deg_prob_pct


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
    if READ_SINK_MEAS:
        ks_p_sink_pa = out[TAG_P_SINK_MEAS]
    else:
        ks_p_sink_pa = float(p_sink)

    speed_rpm = float(DEFAULT_SPEED_RPM)

    # fluid props (SI from K-Spice)
    density_kg_m3 = out[TAG_DENSITY_KG_M3]       # kg/m3
    viscosity_pa_s = out[TAG_VISCOSITY_PA_S]     # Pa*s

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

    # 1 Pa*s = 1000 cP
    visc_csv_cp = float(viscosity_pa_s * 1000.0)

    head_proxy = ks_dp_bar / max(dens_csv, 1e-12)

    # ML predict
    ml_flow_pred = float("nan")
    ml_deg_prob_pct = float("nan")
    flow_err_pct = float("nan")

    if USE_ML_MODEL and predictor is not None:
        cols_for_model = predictor.input_cols if getattr(predictor, "input_cols", None) else INPUT_COLS

        # don't build values for derived features here (Preprocessor will compute them)
        derived_cols = _derived_output_cols_from_profile(profile_cfg)
        base_cols = [c for c in cols_for_model if c not in derived_cols]

        feature_values = build_feature_values(
            base_cols,
            ks_p_in_pa=ks_p_in_pa,
            ks_p_out_pa=ks_p_out_pa,
            ks_p_sink_pa=ks_p_sink_pa,
            ks_t_in_k=ks_t_in_k,
            ks_pwr_w=ks_pwr_w,
            speed_rpm=speed_rpm,
            density_kg_m3=density_kg_m3,
            viscosity_pa_s=viscosity_pa_s,   # Pa*s from KS (conversion happens per feature unit)
            valve_opening_frac=valve_opening_frac,
            hydraulic_wear_frac=hydraulic_wear_frac,
        )

        if PRINT_ML_DEBUG and hasattr(predictor, "debug_print_model_inputs"):
            predictor.debug_print_model_inputs(feature_values)

        pred_dict = predictor.predict(feature_values)
        ml_flow_pred, ml_deg_prob_pct = _extract_ml_outputs(pred_dict)

        if np.isfinite(ml_flow_pred) and abs(ks_flow_m3h) > 1e-9:
            flow_err_pct = 100.0 * abs(ml_flow_pred - ks_flow_m3h) / abs(ks_flow_m3h)

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
        ml_flow_pred,
        ml_deg_prob_pct,
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
    "ml_degradation_probability__pct",
    "flow_abs_error__pct",
]

with open(CSV_PATH, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)

print(f"\n✅ Saved file: {os.path.abspath(CSV_PATH)}")
print("Done.")
