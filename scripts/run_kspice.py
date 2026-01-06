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
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ============================================================
# Paths
# ============================================================
HERE = Path(__file__).resolve().parent              # .../phlux_lab/scripts
LAB_ROOT = HERE.parent                              # .../phlux_lab
REPO_ROOT = LAB_ROOT.parent                         # .../phlux
TRAINING_CFG_PATH = LAB_ROOT / "configs" / "training_config.yaml"

OUTPUT_DIR = LAB_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = str(OUTPUT_DIR / "kspice_vs_ml.csv")


def _resolve_path(p: str | Path) -> Path:
    pp = Path(str(p)).expanduser()
    if pp.is_absolute():
        return pp
    s = str(p).replace("\\", "/")
    if s.startswith("phlux_lab/") or s.startswith("src/") or s.startswith("logs/"):
        return (REPO_ROOT / s).resolve()
    return (LAB_ROOT / s).resolve()


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
    # stacked / synthetic features that may appear:
    "q_liquid_pred": "m3/h",
    "flow_pred": "m3/h",
    "hydraulic_wear_pred": "frac",
    "wear_pred": "frac",
    "specific_power": "",  # unit depends on your design; leave blank unless you suffix columns explicitly
}

# ============================================================
# Training-config reader (master truth)
# ============================================================
def load_training_config_multi() -> Tuple[Dict[str, Any], str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    with open(TRAINING_CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    project = cfg.get("project", {}) or {}
    client_name = str(cfg.get("client_name") or project.get("client_name") or project.get("client") or "").strip()

    if not client_name:
        # try inferring from model root directory if unambiguous
        base = cfg.get("paths", {}).get("model_root", str(LAB_ROOT / "models"))
        base = str(base).replace("{client}", "").replace("{client_name}", "")
        models_root = _resolve_path(base)
        if models_root.exists() and models_root.is_dir():
            clients = [d.name for d in models_root.iterdir() if d.is_dir()]
            if len(clients) == 1:
                client_name = clients[0]
                print(f"ℹ️ Inferred client_name='{client_name}' from models directory")
            else:
                raise KeyError(
                    "training_config.yaml must define project.client_name (or client_name) when multiple client model directories exist"
                )
        else:
            raise KeyError("training_config.yaml must define project.client_name (or client_name)")

    training_defaults = cfg.get("training_defaults", {}) or {}
    stages_cfg = cfg.get("models", {}) or {}
    if not isinstance(stages_cfg, dict) or not stages_cfg:
        raise KeyError("training_config.yaml must define 'models:' with stage blocks")

    preprocessing = cfg.get("preprocessing", {}) or {}
    return cfg, client_name, training_defaults, stages_cfg, preprocessing


def _canonical_client_dir(cfg: Dict[str, Any], client: str) -> Path:
    raw = str((cfg.get("paths", {}) or {}).get("model_root", str(LAB_ROOT / "models")))
    raw = raw.replace("{client}", client).replace("{client_name}", client)
    base = _resolve_path(raw)
    if base.name.lower() == client.lower():
        return base
    return base / client


def _merge_model_cfg(global_cfg: Dict[str, Any], stage_cfg: Dict[str, Any]) -> Dict[str, Any]:
    defaults = dict(global_cfg.get("training_defaults", {}) or {})
    stage_model = dict(stage_cfg.get("model", {}) or {})

    merged: Dict[str, Any] = {}
    merged.update(defaults)
    merged.update(stage_model)

    if "early_stopping" in merged and "earlystop" not in merged:
        merged["earlystop"] = bool(merged["early_stopping"])
    if "patience" in merged and "earlystop_patience" not in merged:
        merged["earlystop_patience"] = int(merged["patience"])
    if "validation_split" in merged and "val_split" not in merged:
        merged["val_split"] = float(merged["validation_split"])

    rlr = dict(global_cfg.get("reduce_lr_on_plateau", {}) or {})
    if bool(rlr.get("enabled", False)) and "lr_scheduler_cfg" not in merged:
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

    # IMPORTANT:
    # We do NOT build stacked or engineered features here.
    # Those get injected later in this script.
    if c.endswith("_pred") or c in {"q_liquid_pred", "flow_pred", "hydraulic_wear_pred", "wear_pred"}:
        raise RuntimeError(
            f"'{col}' is a stacked feature and must be injected (computed from upstream models)."
        )
    if c == "specific_power":
        raise RuntimeError("'specific_power' is engineered and must be injected/derived after q_liquid_pred exists.")

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

    if "valve_dp" in c:
        dp_pa = ks_p_out_pa - ks_p_sink_pa
        return float(convert_internal_back_to_user(dp_pa, unit or "Pa"))

    raise RuntimeError(f"Unhandled input feature column: '{col}'")


def build_feature_inputs_for_cols(
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
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in cols:
        out[col] = float(
            _user_value_for_col(
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
        )
    return out


# ============================================================
# ML artifact loading (do NOT rely on VFMPredictor for stacking here)
# ============================================================
USE_ML_MODEL = True

MASTER_CFG, CLIENT_NAME, TRAINING_DEFAULTS, STAGES_CFG, PREPROCESSING_CFG = load_training_config_multi()
CLIENT_DIR = _canonical_client_dir(MASTER_CFG, CLIENT_NAME)

try:
    from phlux_lab.utils.preprocessor import Preprocessor  # type: ignore
    from phlux_lab.ml.vfm_model import VFMModel  # type: ignore
except Exception as e:
    print(f"⚠️ Could not import ML stack: {e}")
    USE_ML_MODEL = False

flow_model = wear_model = corr_model = None
flow_pp = wear_pp = corr_pp = None
RAW_INPUT_COLS: List[str] = []


def _stage_artifact_paths(client_dir: Path, stage: str) -> Tuple[Path, Path]:
    stage_cfg = STAGES_CFG.get(stage, {}) or {}
    save_cfg = stage_cfg.get("save", {}) or {}
    model_name = save_cfg.get("model_name") or f"{stage}.keras"
    pp_name = save_cfg.get("preprocessor") or f"preprocessor_{stage}.joblib"
    stage_dir = client_dir / stage
    return stage_dir / model_name, stage_dir / pp_name


def _load_stage(stage: str) -> Tuple[Any, Any]:
    model_path, pp_path = _stage_artifact_paths(CLIENT_DIR, stage)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model for stage '{stage}': {model_path}")
    if not pp_path.exists():
        raise FileNotFoundError(f"Missing preprocessor for stage '{stage}': {pp_path}")
    pp_obj = Preprocessor.load(str(pp_path))
    model_cfg = _merge_model_cfg(MASTER_CFG, STAGES_CFG.get(stage, {}) or {})
    mdl = VFMModel.load(model_path=model_path, preprocessor=pp_obj, model_cfg=model_cfg)
    return mdl, pp_obj


def _predict_in_units(mdl: Any, df: pd.DataFrame) -> np.ndarray:
    if hasattr(mdl, "predict_in_units"):
        out = mdl.predict_in_units(df)
        return np.asarray(out)
    # fallback if needed
    X = mdl.preprocessor.transform_X(df)
    y_scaled = mdl.predict(X)
    return mdl.preprocessor.inverse_transform_y(y_scaled)


def _set_feature_with_possible_suffix(df: pd.DataFrame, feature_base: str, value: float, required_cols: List[str]) -> None:
    """
    If the preprocessor expects 'feature_base__unit', set that exact column.
    Always also set the plain 'feature_base' (harmless, and useful for debugging).
    """
    df[feature_base] = float(value)
    for c in required_cols:
        if _internal_name(c) == feature_base:
            df[c] = float(value)


def _compute_specific_power(pump_power_col: str, q_col: str, df: pd.DataFrame) -> float:
    """
    specific_power = pump_power / q
    Assumes df already has both values in the SAME units the model was trained on.
    """
    p = float(df[pump_power_col].iloc[0])
    q = float(df[q_col].iloc[0])
    if abs(q) < 1e-12:
        return float("nan")
    return p / q


if USE_ML_MODEL:
    try:
        flow_model, flow_pp = _load_stage("flow")
        wear_model, wear_pp = _load_stage("wear")
        corr_model, corr_pp = _load_stage("flow_correction")

        print("✅ Loaded models (direct stage calls in run_kspice):")
        print(f"  - flow: {CLIENT_DIR / 'flow'}")
        print(f"  - wear: {CLIENT_DIR / 'wear'}")
        print(f"  - corr: {CLIENT_DIR / 'flow_correction'}")

        # Build RAW_INPUT_COLS from union of (flow + wear) required inputs
        # EXCLUDING:
        #   - stacked preds (q_liquid_pred, hydraulic_wear_pred)
        #   - engineered features we will compute AFTER flow (specific_power)
        cols_flow = flow_pp.get_input_feature_names()
        cols_wear = wear_pp.get_input_feature_names()

        raw: List[str] = []
        for c in list(cols_flow) + list(cols_wear):
            base = _internal_name(c).lower()
            if base.endswith("_pred") or base in {"q_liquid_pred", "hydraulic_wear_pred"}:
                continue
            if base == "specific_power":
                continue
            raw.append(c)

        RAW_INPUT_COLS = list(dict.fromkeys(raw))

    except Exception as e:
        print(f"⚠️ Disabling ML due to load error: {e}")
        USE_ML_MODEL = False


# ============================================================
# Sampling controls (internal SI for K-Spice boundaries)
# ============================================================
N_CASES = 100

P_IN_RANGE_PA = (1.0e5, 3.0e5)
P_SINK_RANGE_PA = (2.5e5, 5.0e5)
T_IN_RANGE_K = (288.15, 298.15)  # 15–25 °C
VALVE_OPENING_RANGE_FRAC = (0.25, 0.5)

HYDRAULIC_WEAR_RANGE = (0.01, 0.01)
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
    # ML predictions (manual stacked execution)
    # ============================================================
    ml_flow_pred_m3h = float("nan")
    ml_wear_pred_frac = float("nan")
    ml_flow_corrected_m3h = float("nan")
    ml_corr_delta_m3h = float("nan")

    flow_err_pct = float("nan")
    wear_err_pct = float("nan")
    corr_err_pct = float("nan")

    if USE_ML_MODEL and flow_model is not None and wear_model is not None and corr_model is not None:
        if not RAW_INPUT_COLS:
            raise RuntimeError("RAW_INPUT_COLS empty; cannot build inference row.")

        # base raw inputs (no stacked, no engineered)
        feature_inputs = build_feature_inputs_for_cols(
            RAW_INPUT_COLS,
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
        df_base = pd.DataFrame([feature_inputs])

        # -------------------------
        # 1) FLOW
        # -------------------------
        q_flow = _predict_in_units(flow_model, df_base)
        q_flow_scalar = float(q_flow.reshape(-1)[0])
        ml_flow_pred_m3h = q_flow_scalar

        # -------------------------
        # 2) WEAR (inject q_liquid_pred + specific_power)
        # -------------------------
        df_wear = df_base.copy()
        wear_required = wear_pp.get_input_feature_names()

        # inject q_liquid_pred (with or without unit suffix)
        _set_feature_with_possible_suffix(df_wear, "q_liquid_pred", ml_flow_pred_m3h, wear_required)

        # compute specific_power if required by wear
        if any(_internal_name(c) == "specific_power" for c in wear_required):
            # choose pump_power column name that exists in df_wear (with suffix, if present)
            pump_power_candidates = [c for c in df_wear.columns if _internal_name(c) == "pump_power"]
            q_candidates = [c for c in df_wear.columns if _internal_name(c) == "q_liquid_pred"]

            if not pump_power_candidates:
                raise RuntimeError("Cannot compute specific_power: pump_power column not present in df_wear.")
            if not q_candidates:
                raise RuntimeError("Cannot compute specific_power: q_liquid_pred column not present in df_wear.")

            sp = _compute_specific_power(pump_power_candidates[0], q_candidates[0], df_wear)
            _set_feature_with_possible_suffix(df_wear, "specific_power", sp, wear_required)

        wear_out = _predict_in_units(wear_model, df_wear)
        wear_scalar = float(wear_out.reshape(-1)[0])
        ml_wear_pred_frac = wear_scalar

        # -------------------------
        # 3) CORRECTION (inject both preds + specific_power if needed)
        # -------------------------
        df_corr = df_base.copy()
        corr_required = corr_pp.get_input_feature_names()

        _set_feature_with_possible_suffix(df_corr, "q_liquid_pred", ml_flow_pred_m3h, corr_required)
        _set_feature_with_possible_suffix(df_corr, "hydraulic_wear_pred", ml_wear_pred_frac, corr_required)

        if any(_internal_name(c) == "specific_power" for c in corr_required):
            pump_power_candidates = [c for c in df_corr.columns if _internal_name(c) == "pump_power"]
            q_candidates = [c for c in df_corr.columns if _internal_name(c) == "q_liquid_pred"]
            if not pump_power_candidates or not q_candidates:
                raise RuntimeError("Cannot compute specific_power for correction: missing pump_power or q_liquid_pred.")
            sp = _compute_specific_power(pump_power_candidates[0], q_candidates[0], df_corr)
            _set_feature_with_possible_suffix(df_corr, "specific_power", sp, corr_required)

        corr_out = _predict_in_units(corr_model, df_corr)
        corr_scalar = float(corr_out.reshape(-1)[0])
        ml_flow_corrected_m3h = corr_scalar

        if np.isfinite(ml_flow_corrected_m3h) and np.isfinite(ml_flow_pred_m3h):
            ml_corr_delta_m3h = float(ml_flow_corrected_m3h - ml_flow_pred_m3h)

        # Errors vs K-Spice flow
        if np.isfinite(ml_flow_pred_m3h) and abs(ks_flow_m3h) > 1e-9:
            flow_err_pct = 100.0 * abs(ml_flow_pred_m3h - ks_flow_m3h) / abs(ks_flow_m3h)

        if np.isfinite(ml_flow_corrected_m3h) and abs(ks_flow_m3h) > 1e-9:
            corr_err_pct = 100.0 * abs(ml_flow_corrected_m3h - ks_flow_m3h) / abs(ks_flow_m3h)

        if np.isfinite(ml_wear_pred_frac) and abs(hydraulic_wear_frac) > 1e-12:
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
    "virplant_p_in__bar",
    "virplant_t_in__degC",
    "virplant_discharge_p__bar",
    "virplant_sink_p__bar",
    "virplant_dp__bar",
    "speed__rpm",
    "fluid_density__kg/m3",
    "fluid_viscosity__cP",
    "virplant_pump_power__kW",
    "valve_opening__frac",
    "virplant_wear_input__frac",
    "ml_wear_pred__frac",
    "wear_err_pct",
    "virplant_flow__m3_h",
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
