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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ============================================================
# Paths
# ============================================================
LAB_ROOT = Path(__file__).resolve().parents[1]  # .../phlux_lab
TRAINING_CFG_PATH = LAB_ROOT / "configs" / "training_config.yaml"

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
TAG_P_SINK = "Streamboundary2:InputPressure"

TAG_HYDRAULIC_WEAR = "PumpAndMotor1:Plugging"
TAG_VALVE_OPENING = "ControlValve1:LocalControlSignalIn"

# ============================================================
# Measured tags (K-Spice returns internal SI)
# ============================================================
TAG_P_IN_MEAS = "P_In:MeasuredValue"        # Pa
TAG_T_IN_MEAS = "T_In:MeasuredValue"        # K
TAG_P_OUT_MEAS = "P_Out:MeasuredValue"      # Pa
TAG_FLOW_MEAS = "Flow:MeasuredValue"        # m3/s
TAG_PWR_MEAS = "Pump_Power:MeasuredValue"   # W

TAG_P_SINK_MEAS = "Streamboundary2:InputPressure"  # Pa

TAG_DENSITY_KG_M3 = "PipeFlow1:OutletStream.r"     # kg/m3
TAG_VISCOSITY_PA_S = "PipeFlow1:Viscosity"         # Pa*s

READ_SINK_MEAS = True

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

DEFAULT_USER_UNITS: Dict[str, str] = {
    "suction_pressure": "bar",
    "discharge_pressure": "bar",
    "sink_pressure": "bar",
    "delta_pressure": "bar",
    "temperature": "degC",
    "speed": "rpm",
    "fluid_density": "kg/m3",
    "fluid_viscosity": "cP",
    "pump_power": "kW",
    "valve_opening": "frac",
    "q_liquid": "m3/h",
    "hydraulic_wear": "frac",
    "valve_dp": "bar",
    "delta_q_liquid": "m3/h",
    # stacked / synthetic features that may appear in preprocessors:
    "q_liquid_pred": "m3/h",
    "flow_pred": "m3/h",
    "hydraulic_wear_pred": "frac",
    "wear_pred": "frac",
}

# ============================================================
# Training-config reader
# ============================================================
def load_training_config_multi() -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    with open(TRAINING_CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    client_name = str(cfg.get("client_name", "")).strip()
    if not client_name:
        models_root = LAB_ROOT / "models"
        clients = [d.name for d in models_root.iterdir() if d.is_dir()]
        if len(clients) == 1:
            client_name = clients[0]
            print(f"ℹ️ Inferred client_name='{client_name}' from models directory")
        else:
            raise KeyError(
                "training_config.yaml must define 'client_name:' when multiple client model directories exist"
            )

    training_defaults = cfg.get("training_defaults", {}) or {}
    stages_cfg = cfg.get("models", {}) or {}
    if not isinstance(stages_cfg, dict) or not stages_cfg:
        raise KeyError("training_config.yaml must define 'models:' with stage blocks")

    return client_name, training_defaults, stages_cfg

# ============================================================
# Feature mapping helpers
# ============================================================
def _col_unit_suffix(col: str) -> str:
    return col.split("__", 1)[1] if "__" in col else ""

def _internal_name(col: str) -> str:
    return col.split("__", 1)[0] if "__" in col else col

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
) -> float:
    unit = _resolved_unit(col) or ""
    c = _internal_name(col).lower()

    # Stacked features must be provided via extra_values
    if c.endswith("_pred") or c in {"q_liquid_pred", "flow_pred", "hydraulic_wear_pred", "wear_pred"}:
        raise RuntimeError(
            f"Unhandled input feature column: '{col}' (stacked feature). "
            "This must be provided via extra_values when building feature_inputs."
        )

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
        if unit.lower() in {"cp", "cpoise"}:
            return float(viscosity_pa_s * 1000.0)
        return float(convert_internal_back_to_user(viscosity_pa_s, unit or "Pa*s"))

    if "valve" in c and ("opening" in c or "open" in c or "position" in c):
        return float(valve_opening_frac)

    if "pump_power" in c or ("power" in c and "pump" in c):
        return float(convert_internal_back_to_user(ks_pwr_w, unit or "W"))

    if c.startswith("head_proxy"):
        dp_bar = float(convert_internal_back_to_user(ks_p_out_pa - ks_p_in_pa, "bar"))
        dens = float(convert_internal_back_to_user(density_kg_m3, "kg/m3"))
        return dp_bar / max(dens, 1e-12)

    if "valve_dp" in c:
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
    extra_values: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    extras = extra_values or {}

    for col in cols:
        if col in extras:
            out[col] = float(extras[col])
            continue

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
        )
        out[col] = float(val)

    # Allow extra values beyond cols (predictor may compute derived features)
    for k, v in extras.items():
        if k not in out:
            out[k] = float(v)

    return out

def _pick_first_numeric(pred: Dict[str, Any], preferred: List[str]) -> Optional[str]:
    for k in preferred:
        if k in pred and isinstance(pred.get(k), (int, float)):
            return k
    for k, v in pred.items():
        if isinstance(v, (int, float)):
            return k
    return None

# ============================================================
# Load predictors
# ============================================================
USE_ML_MODEL = True
CLIENT_NAME, TRAINING_DEFAULTS, STAGES_CFG = load_training_config_multi()

try:
    from phlux_lab.utils.predictor import VfmPredictor  # type: ignore
except Exception as e:
    print(f"⚠️ Could not import VfmPredictor: {e}")
    USE_ML_MODEL = False
    VfmPredictor = None  # type: ignore

def _stage_paths(client: str, stage: str) -> Tuple[Path, Path]:
    stage_dir = LAB_ROOT / "models" / client / stage
    model_path = stage_dir / f"vfm_{stage}.keras"
    pp_path = stage_dir / f"preprocessor_{stage}.joblib"
    return model_path, pp_path

def _load_stage_predictor(stage: str) -> Optional[Any]:
    if not USE_ML_MODEL or VfmPredictor is None:
        return None
    model_path, pp_path = _stage_paths(CLIENT_NAME, stage)
    if not model_path.exists() or not pp_path.exists():
        print(f"⚠️ Stage '{stage}' artifacts not found:")
        print(f"   - model: {model_path}")
        print(f"   - pp:    {pp_path}")
        return None
    p = VfmPredictor.from_paths(model_path=str(model_path), preprocessor_path=str(pp_path))
    print(f"✅ Loaded '{stage}' predictor:")
    print(f"   - model: {model_path}")
    print(f"   - pp:    {pp_path}")
    return p

predictor_flow = _load_stage_predictor("flow")
predictor_wear = _load_stage_predictor("wear")
predictor_corr = _load_stage_predictor("flow_correction")

def _predict_any(predictor: Any, features: Dict[str, float]) -> Dict[str, float]:
    if predictor is None:
        return {}
    if hasattr(predictor, "predict_one"):
        return predictor.predict_one(features)
    if hasattr(predictor, "predict_batch"):
        out = predictor.predict_batch([features])
        if isinstance(out, list) and out:
            return out[0]
        raise TypeError("predict_batch returned unexpected output")
    raise AttributeError("VfmPredictor has no predict_one or predict_batch method")

# ============================================================
# Sampling controls (internal SI for K-Spice boundaries)
# ============================================================
N_CASES = 100

P_IN_RANGE_PA = (1.0e5, 3.0e5)
P_SINK_RANGE_PA = (2.5e5, 5.0e5)
T_IN_RANGE_K = (288.15, 298.15)  # 15–25 °C
VALVE_OPENING_RANGE_FRAC = (0.25, 0.5)

HYDRAULIC_WEAR_RANGE = (0.001, 0.15)
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

    tl.set_value(APP, TAG_P_IN, float(p_in))
    tl.set_value(APP, TAG_T_IN, float(t_in))
    tl.set_value(APP, TAG_P_SINK, float(p_sink))

    tl.set_value(APP, TAG_HYDRAULIC_WEAR, float(hydraulic_wear_frac))

    if TAG_VALVE_OPENING:
        tl.set_value(APP, TAG_VALVE_OPENING, float(valve_opening_frac))

    tl.run_for(timedelta(seconds=60))

    vals = tl.get_values(APP, ALARM_TAGS)
    out: Dict[str, float] = {}
    for tag, val in zip(ALARM_TAGS, vals):
        if val is None:
            raise RuntimeError(f"K-Spice returned None for tag '{tag}'.")
        out[tag] = float(val)

    ks_p_in_pa = out[TAG_P_IN_MEAS]
    ks_t_in_k = out[TAG_T_IN_MEAS]
    ks_p_out_pa = out[TAG_P_OUT_MEAS]
    ks_flow_m3s = out[TAG_FLOW_MEAS]
    ks_pwr_w = out[TAG_PWR_MEAS]

    ks_p_sink_pa = out[TAG_P_SINK_MEAS] if READ_SINK_MEAS else float(p_sink)
    speed_rpm = float(DEFAULT_SPEED_RPM)

    density_kg_m3 = out[TAG_DENSITY_KG_M3]
    viscosity_pa_s = out[TAG_VISCOSITY_PA_S]

    # User-unit conversions for CSV reporting
    ks_p_in_bar = float(convert_internal_back_to_user(ks_p_in_pa, "bar"))
    ks_p_out_bar = float(convert_internal_back_to_user(ks_p_out_pa, "bar"))
    ks_p_sink_bar = float(convert_internal_back_to_user(ks_p_sink_pa, "bar"))
    ks_dp_bar = float(convert_internal_back_to_user(ks_p_out_pa - ks_p_in_pa, "bar"))
    ks_t_c = float(convert_internal_back_to_user(ks_t_in_k, "degC"))
    ks_pwr_kw = float(convert_internal_back_to_user(ks_pwr_w, "kW"))
    ks_flow_m3h = float(convert_internal_back_to_user(ks_flow_m3s, "m3/h"))
    speed_csv = float(convert_internal_back_to_user(speed_rpm, "rpm"))
    dens_csv = float(convert_internal_back_to_user(density_kg_m3, "kg/m3"))
    visc_csv_cp = float(viscosity_pa_s * 1000.0)

    # ============================================================
    # ML: stage 1 (flow)
    # ============================================================
    ml_flow_pred_m3h = float("nan")
    flow_err_pct = float("nan")

    if predictor_flow is not None:
        cols_for_model = predictor_flow.input_cols if getattr(predictor_flow, "input_cols", None) else []
        if not cols_for_model:
            raise RuntimeError("flow predictor has no input_cols; cannot build inputs.")

        feature_inputs = build_feature_inputs_for_predictor(
            cols_for_model,
            ks_p_in_pa=ks_p_in_pa,
            ks_p_out_pa=ks_p_out_pa,
            ks_p_sink_pa=ks_p_sink_pa,
            ks_t_in_k=ks_t_in_k,
            ks_pwr_w=ks_pwr_w,
            speed_rpm=speed_rpm,
            density_kg_m3=density_kg_m3,
            viscosity_pa_s=viscosity_pa_s,
            valve_opening_frac=valve_opening_frac,
        )

        pred = _predict_any(predictor_flow, feature_inputs)
        flow_key = _pick_first_numeric(pred, preferred=["q_liquid", "flow"])
        if flow_key is not None:
            ml_flow_pred_m3h = float(pred[flow_key])

        if np.isfinite(ml_flow_pred_m3h) and abs(ks_flow_m3h) > 1e-9:
            flow_err_pct = 100.0 * abs(ml_flow_pred_m3h - ks_flow_m3h) / abs(ks_flow_m3h)

    # ============================================================
    # ML: stage 2 (wear)  <-- FIX: pass q_liquid_pred if required
    # ============================================================
    ml_wear_pred_frac = float("nan")
    if predictor_wear is not None:
        cols_for_model = predictor_wear.input_cols if getattr(predictor_wear, "input_cols", None) else []
        if not cols_for_model:
            raise RuntimeError("wear predictor has no input_cols; cannot build inputs.")

        wear_extra: Dict[str, float] = {}
        if np.isfinite(ml_flow_pred_m3h):
            wear_extra["q_liquid_pred"] = float(ml_flow_pred_m3h)
            wear_extra["flow_pred"] = float(ml_flow_pred_m3h)

        feature_inputs = build_feature_inputs_for_predictor(
            cols_for_model,
            ks_p_in_pa=ks_p_in_pa,
            ks_p_out_pa=ks_p_out_pa,
            ks_p_sink_pa=ks_p_sink_pa,
            ks_t_in_k=ks_t_in_k,
            ks_pwr_w=ks_pwr_w,
            speed_rpm=speed_rpm,
            density_kg_m3=density_kg_m3,
            viscosity_pa_s=viscosity_pa_s,
            valve_opening_frac=valve_opening_frac,
            extra_values=wear_extra if wear_extra else None,
        )

        pred = _predict_any(predictor_wear, feature_inputs)
        wear_key = _pick_first_numeric(pred, preferred=["hydraulic_wear", "wear"])
        if wear_key is not None:
            ml_wear_pred_frac = float(pred[wear_key])

    # ============================================================
    # ML: stage 3 (flow_correction)
    # ============================================================
    ml_corr_delta_m3h = float("nan")
    ml_flow_corrected_m3h = float("nan")
    corr_err_pct = float("nan")

    if predictor_corr is not None:
        cols_for_model = predictor_corr.input_cols if getattr(predictor_corr, "input_cols", None) else []
        if not cols_for_model:
            raise RuntimeError("flow_correction predictor has no input_cols; cannot build inputs.")

        extra: Dict[str, float] = {}
        if np.isfinite(ml_flow_pred_m3h):
            extra["q_liquid_pred"] = float(ml_flow_pred_m3h)
            extra["flow_pred"] = float(ml_flow_pred_m3h)
        if np.isfinite(ml_wear_pred_frac):
            extra["hydraulic_wear_pred"] = float(ml_wear_pred_frac)
            extra["wear_pred"] = float(ml_wear_pred_frac)

        feature_inputs = build_feature_inputs_for_predictor(
            cols_for_model,
            ks_p_in_pa=ks_p_in_pa,
            ks_p_out_pa=ks_p_out_pa,
            ks_p_sink_pa=ks_p_sink_pa,
            ks_t_in_k=ks_t_in_k,
            ks_pwr_w=ks_pwr_w,
            speed_rpm=speed_rpm,
            density_kg_m3=density_kg_m3,
            viscosity_pa_s=viscosity_pa_s,
            valve_opening_frac=valve_opening_frac,
            extra_values=extra if extra else None,
        )

        pred = _predict_any(predictor_corr, feature_inputs)

        if "delta_q_liquid" in pred and isinstance(pred.get("delta_q_liquid"), (int, float)):
            ml_corr_delta_m3h = float(pred["delta_q_liquid"])
            if np.isfinite(ml_flow_pred_m3h):
                ml_flow_corrected_m3h = float(ml_flow_pred_m3h + ml_corr_delta_m3h)

        if not np.isfinite(ml_flow_corrected_m3h):
            k = _pick_first_numeric(pred, preferred=["q_liquid", "flow"])
            if k is not None:
                ml_flow_corrected_m3h = float(pred[k])

        if np.isfinite(ml_flow_corrected_m3h) and abs(ks_flow_m3h) > 1e-9:
            corr_err_pct = 100.0 * abs(ml_flow_corrected_m3h - ks_flow_m3h) / abs(ks_flow_m3h)

        if np.isfinite(ml_wear_pred_frac) and abs(hydraulic_wear_frac) > 1e-9:
            wear_err_pct = 100.0 * abs(ml_wear_pred_frac - hydraulic_wear_frac) / abs(hydraulic_wear_frac)
    
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
        valve_opening_frac,
        hydraulic_wear_frac,
        ml_wear_pred_frac,
        wear_err_pct,
        ks_flow_m3h,
        ml_flow_pred_m3h,
        flow_err_pct,
        ml_corr_delta_m3h,
        ml_flow_corrected_m3h,
        corr_err_pct,
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
    "valve_opening__frac",
    "kspice_wear_input__frac", 
    "ml_wear_pred__frac",
    "wear_err_pct",
    "ks_flow__m3_h",
    "ml_flow_pred__m3_h",
    "flow_abs_error__pct",
    "ml_corr_delta__m3_h",
    "ml_flow_corrected__m3_h",
    "corr_abs_error__pct",
]

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)

print(f"\n✅ Saved file: {os.path.abspath(CSV_PATH)}")
print("Done.")
