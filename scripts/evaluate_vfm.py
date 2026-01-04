from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from phlux_lab.utils.predictor import VfmPredictor  # type: ignore


# =========================================================
# USER SETTINGS (EDIT THESE)
# =========================================================
LAB_ROOT = Path(__file__).resolve().parents[1]  # phlux_lab
CLIENT = "ClientA"

# Stage artifacts
MODEL_FLOW = LAB_ROOT / "models" / CLIENT / "flow" / "vfm_flow.keras"
PP_FLOW    = LAB_ROOT / "models" / CLIENT / "flow" / "preprocessor_flow.joblib"

MODEL_WEAR = LAB_ROOT / "models" / CLIENT / "wear" / "vfm_wear.keras"
PP_WEAR    = LAB_ROOT / "models" / CLIENT / "wear" / "preprocessor_wear.joblib"

MODEL_CORR = LAB_ROOT / "models" / CLIENT / "flow_correction" / "vfm_flow_correction.keras"
PP_CORR    = LAB_ROOT / "models" / CLIENT / "flow_correction" / "preprocessor_flow_correction.joblib"

# Test CSVs per stage (use the same ones your pipeline uses)
TEST_FLOW = LAB_ROOT / "data" / "synthetic" / "ClientA_flow_clean_test.csv"
TEST_WEAR = LAB_ROOT / "data" / "synthetic" / "ClientA_wear_test.csv"
TEST_CORR = LAB_ROOT / "data" / "synthetic" / "ClientA_paired_test.csv"

# Plots
MAKE_PLOTS = True
PLOT_MAX_TARGETS = 1  # parity+residual plots for first N targets per stage

# If correction stage needs stacked inputs but they aren't in TEST_CORR,
# auto-generate them using the flow/wear predictors.
AUTO_GENERATE_STACKED_FOR_CORR = True

# Common stacked feature names used in your configs
STACK_KEYS_FLOW = ["q_liquid_pred", "flow_pred"]
STACK_KEYS_WEAR = ["hydraulic_wear_pred", "wear_pred"]

# Correction semantics:
# - "absolute": delta_q_liquid is in same units as flow (e.g., m3/h) and should be added
# - "relative": delta_q_liquid is dimensionless and should be applied multiplicatively
#
# RECOMMENDATION:
#   Use "auto" until you're 100% sure your paired dataset and trained model are aligned.
CORR_DELTA_MODE = "auto"  # "auto" | "absolute" | "relative"

# Optional safety clamp for relative corrections (dimensionless). Set None to disable.
REL_CORR_CLIP = 0.5  # e.g., 0.5 means +/-50% max correction applied in derived flow calc

# Optional: gate correction application based on predicted wear (helps when stage-1 is already excellent)
ENABLE_WEAR_GATE = False
WEAR_GATE_THRESHOLD = 0.05  # frac
# =========================================================


def _predict_y(predictor: VfmPredictor, df: pd.DataFrame) -> np.ndarray:
    """
    Vectorized prediction in the *same units as the training targets*.
    (Your current pipeline does NOT track schema units in the predictor,
     so we evaluate in raw CSV units, which match inverse_transform_y().)
    """
    X_scaled = predictor.pp.transform_X(df)
    y_scaled_pred = predictor.model.predict(X_scaled, verbose=0)
    y_pred = predictor.pp.inverse_transform_y(y_scaled_pred).astype("float32")
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    return y_pred


def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    err = (y_pred - y_true).astype("float32")
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err * err, axis=0))
    return mae, rmse


def _pick_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_to_float32(x: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(x, dtype="float32")
    return arr


def _infer_corr_delta_mode(
    delta_pred: np.ndarray,
    df_corr: pd.DataFrame,
    *,
    col_name: str = "delta_q_liquid",
) -> str:
    """
    Heuristic inference:
      - If typical magnitude of delta is < ~1, it's probably "relative".
      - If typical magnitude is >> 1, it's probably "absolute".

    Uses both prediction distribution and CSV true distribution (if available),
    because models can be mis-trained / mis-loaded.
    """
    dp = _safe_to_float32(delta_pred)
    dp = dp[np.isfinite(dp)]
    pred_p95 = float(np.percentile(np.abs(dp), 95)) if dp.size else float("nan")

    true_p95 = float("nan")
    if col_name in df_corr.columns:
        dt = _safe_to_float32(df_corr[col_name].to_numpy())
        dt = dt[np.isfinite(dt)]
        if dt.size:
            true_p95 = float(np.percentile(np.abs(dt), 95))

    # If either strongly indicates "absolute", choose absolute.
    # (Because treating absolute as relative causes catastrophic blow-ups.)
    threshold = 1.5  # >~1.5 is very unlikely for a fraction; likely absolute in flow units
    if (np.isfinite(pred_p95) and pred_p95 > threshold) or (np.isfinite(true_p95) and true_p95 > threshold):
        return "absolute"

    # If both are small, relative is likely
    return "relative"


def _ensure_stacked_inputs_for_corr(
    df_corr: pd.DataFrame,
    corr_pred: VfmPredictor,
    flow_pred: Optional[VfmPredictor],
    wear_pred: Optional[VfmPredictor],
) -> pd.DataFrame:
    """
    Ensure df_corr contains any stacked features required by corr_pred.input_cols
    (e.g., q_liquid_pred, hydraulic_wear_pred). If missing and enabled, generate
    them using flow_pred / wear_pred and add as new columns.
    """
    if not AUTO_GENERATE_STACKED_FOR_CORR:
        return df_corr

    required = list(getattr(corr_pred, "input_cols", []) or [])
    if not required:
        return df_corr

    missing = [c for c in required if c not in df_corr.columns]
    missing_stacked = [c for c in missing if c in (STACK_KEYS_FLOW + STACK_KEYS_WEAR)]
    if not missing_stacked:
        return df_corr

    df = df_corr.copy()

    # Generate flow stacked values
    if any(c in STACK_KEYS_FLOW for c in missing_stacked):
        if flow_pred is None:
            raise RuntimeError(f"Correction requires {missing_stacked} but flow predictor is not available.")
        y_flow = _predict_y(flow_pred, df)
        flow_targets = list(flow_pred.target_cols)
        if not flow_targets:
            raise RuntimeError("Flow predictor has no target_cols; cannot generate q_liquid_pred.")
        key = "q_liquid" if "q_liquid" in flow_targets else flow_targets[0]
        idx = flow_targets.index(key)
        q_pred = y_flow[:, idx].astype("float32")

        if "q_liquid_pred" in required and "q_liquid_pred" not in df.columns:
            df["q_liquid_pred"] = q_pred
        if "flow_pred" in required and "flow_pred" not in df.columns:
            df["flow_pred"] = q_pred

    # Generate wear stacked values
    if any(c in STACK_KEYS_WEAR for c in missing_stacked):
        if wear_pred is None:
            raise RuntimeError(f"Correction requires {missing_stacked} but wear predictor is not available.")
        y_wear = _predict_y(wear_pred, df)
        wear_targets = list(wear_pred.target_cols)
        if not wear_targets:
            raise RuntimeError("Wear predictor has no target_cols; cannot generate hydraulic_wear_pred.")
        key = "hydraulic_wear" if "hydraulic_wear" in wear_targets else wear_targets[0]
        idx = wear_targets.index(key)
        w_pred = y_wear[:, idx].astype("float32")

        if "hydraulic_wear_pred" in required and "hydraulic_wear_pred" not in df.columns:
            df["hydraulic_wear_pred"] = w_pred
        if "wear_pred" in required and "wear_pred" not in df.columns:
            df["wear_pred"] = w_pred

    return df


def _evaluate_stage(stage: str, predictor: VfmPredictor, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate a stage and return metrics + arrays. All in raw CSV units.
    """
    targets = list(predictor.target_cols)
    if not targets:
        raise RuntimeError(f"Stage '{stage}' predictor has empty target_cols.")

    missing_t = [t for t in targets if t not in df.columns]
    if missing_t:
        raise KeyError(f"Stage '{stage}' CSV missing target columns: {missing_t}")

    y_true = df[targets].to_numpy(dtype="float32")
    y_pred = _predict_y(predictor, df)

    mae, rmse = _mae_rmse(y_true, y_pred)
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")

    print(f"\n{'='*70}\nSTAGE: {stage}\n{'='*70}")
    print(f"Samples : {len(df)}")
    print(f"Targets : {targets}")

    for i, t in enumerate(targets):
        print(f"\nTarget: {t}")
        print(f"  MAE  : {mae[i]:.6f}")
        print(f"  RMSE : {rmse[i]:.6f}")
        print(f"  R²   : {float(r2[i]):.6f}")

    if MAKE_PLOTS:
        n_plot = min(PLOT_MAX_TARGETS, len(targets))
        for j in range(n_plot):
            t = targets[j]
            yt = y_true[:, j].flatten()
            yp = y_pred[:, j].flatten()

            plt.figure()
            plt.scatter(yt, yp, alpha=0.7)
            lo = float(min(yt.min(), yp.min()))
            hi = float(max(yt.max(), yp.max()))
            plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)
            plt.xlabel(f"True {t}")
            plt.ylabel(f"Predicted {t}")
            plt.title(f"Parity plot – {stage} – TEST")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            resid = yp - yt
            plt.figure()
            plt.hist(resid, bins=20)
            plt.xlabel(f"Prediction error ({t})")
            plt.ylabel("Count")
            plt.title(f"Residuals – {stage} – TEST")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    return {"targets": targets, "y_true": y_true, "y_pred": y_pred}


def main() -> None:
    print("Starting multi-stage evaluation...")

    for p in [MODEL_FLOW, PP_FLOW, MODEL_WEAR, PP_WEAR, MODEL_CORR, PP_CORR, TEST_FLOW, TEST_WEAR, TEST_CORR]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    flow_pred = VfmPredictor.from_paths(MODEL_FLOW, PP_FLOW)
    wear_pred = VfmPredictor.from_paths(MODEL_WEAR, PP_WEAR)
    corr_pred = VfmPredictor.from_paths(MODEL_CORR, PP_CORR)

    print("\nLoaded predictors:")
    print(f"  flow           : {MODEL_FLOW}")
    print(f"  wear           : {MODEL_WEAR}")
    print(f"  flow_correction: {MODEL_CORR}")

    df_flow = pd.read_csv(TEST_FLOW)
    df_wear = pd.read_csv(TEST_WEAR)
    df_corr = pd.read_csv(TEST_CORR)

    _evaluate_stage("flow", flow_pred, df_flow)
    _evaluate_stage("wear", wear_pred, df_wear)

    df_corr_aug = _ensure_stacked_inputs_for_corr(df_corr, corr_pred, flow_pred, wear_pred)
    corr_out = _evaluate_stage("flow_correction", corr_pred, df_corr_aug)

    # Optional derived metric: corrected flow if correction predicts delta_q_liquid
    corr_targets = corr_out["targets"]
    if "delta_q_liquid" in corr_targets:
        # Use flow model to predict baseline flow on the correction dataset
        y_flow_on_corr = _predict_y(flow_pred, df_corr_aug)
        flow_targets = list(flow_pred.target_cols)
        flow_key = "q_liquid" if "q_liquid" in flow_targets else flow_targets[0]
        q_flow_pred = y_flow_on_corr[:, flow_targets.index(flow_key)].astype("float32")

        # Pull correction prediction
        delta_idx = corr_targets.index("delta_q_liquid")
        delta_pred = corr_out["y_pred"][:, delta_idx].astype("float32")

        # Decide correction semantics
        mode = CORR_DELTA_MODE.lower().strip()
        if mode not in ("auto", "absolute", "relative"):
            raise ValueError(f"Invalid CORR_DELTA_MODE='{CORR_DELTA_MODE}'. Use auto|absolute|relative.")

        if mode == "auto":
            mode = _infer_corr_delta_mode(delta_pred, df_corr_aug, col_name="delta_q_liquid")

        # Optional: wear-based gating of correction
        wear_gate = None
        if ENABLE_WEAR_GATE:
            # Try to use stacked wear prediction if present (generated above), else predict wear again.
            if "hydraulic_wear_pred" in df_corr_aug.columns:
                wear_gate = df_corr_aug["hydraulic_wear_pred"].to_numpy(dtype="float32")
            elif "wear_pred" in df_corr_aug.columns:
                wear_gate = df_corr_aug["wear_pred"].to_numpy(dtype="float32")
            else:
                y_wear_on_corr = _predict_y(wear_pred, df_corr_aug)
                wear_targets = list(wear_pred.target_cols)
                wear_key = "hydraulic_wear" if "hydraulic_wear" in wear_targets else wear_targets[0]
                wear_gate = y_wear_on_corr[:, wear_targets.index(wear_key)].astype("float32")

        # Apply correction
        delta_used = delta_pred.copy()

        if mode == "relative" and REL_CORR_CLIP is not None:
            delta_used = np.clip(delta_used, -float(REL_CORR_CLIP), float(REL_CORR_CLIP)).astype("float32")

        if wear_gate is not None:
            apply = (wear_gate >= float(WEAR_GATE_THRESHOLD)).astype("float32")
        else:
            apply = 1.0

        if mode == "relative":
            q_corrected_pred = q_flow_pred * (1.0 + apply * delta_used)
        else:
            q_corrected_pred = q_flow_pred + apply * delta_used

        # Determine truth for corrected-flow comparison
        true_col = _pick_first_col(df_corr_aug, ["q_liquid", "q_liquid_degraded"])
        if true_col is None and ("q_liquid_clean" in df_corr_aug.columns and "delta_q_liquid" in df_corr_aug.columns):
            q_clean = df_corr_aug["q_liquid_clean"].to_numpy(dtype="float32")
            d_true = df_corr_aug["delta_q_liquid"].to_numpy(dtype="float32")
            if mode == "relative":
                q_true = q_clean * (1.0 + d_true)
                true_name = "q_liquid_clean * (1 + delta_q_liquid)"
            else:
                q_true = q_clean + d_true
                true_name = "q_liquid_clean + delta_q_liquid"
        elif true_col is not None:
            q_true = df_corr_aug[true_col].to_numpy(dtype="float32")
            true_name = true_col
        else:
            q_true = None
            true_name = ""

        # Diagnostics (critical for catching absolute/relative mismatches)
        print(f"\n{'-'*70}")
        print("DERIVED: corrected flow")
        if mode == "relative":
            print("  reconstruction: flow_pred * (1 + delta_q_liquid_pred)")
        else:
            print("  reconstruction: flow_pred + delta_q_liquid_pred")
        if ENABLE_WEAR_GATE:
            print(f"  wear gate: enabled (threshold={WEAR_GATE_THRESHOLD})")
        else:
            print("  wear gate: disabled")
        if mode == "relative" and REL_CORR_CLIP is not None:
            print(f"  rel clip: +/-{REL_CORR_CLIP}")

        # Print ranges to immediately spot mismatches
        dp = delta_pred[np.isfinite(delta_pred)]
        du = delta_used[np.isfinite(delta_used)]
        print(f"  delta_pred p95(abs): {float(np.percentile(np.abs(dp),95)) if dp.size else float('nan'):.6f}")
        print(f"  delta_used p95(abs): {float(np.percentile(np.abs(du),95)) if du.size else float('nan'):.6f}")
        print(f"  q_flow_pred min/max: {float(np.nanmin(q_flow_pred)):.6f} / {float(np.nanmax(q_flow_pred)):.6f}")

        if "delta_q_liquid" in df_corr_aug.columns:
            dt = df_corr_aug["delta_q_liquid"].to_numpy(dtype="float32")
            dt = dt[np.isfinite(dt)]
            print(f"  csv delta_q_liquid p95(abs): {float(np.percentile(np.abs(dt),95)) if dt.size else float('nan'):.6f}")

        if q_true is not None:
            mae = float(np.mean(np.abs(q_corrected_pred - q_true)))
            rmse = float(np.sqrt(np.mean((q_corrected_pred - q_true) ** 2)))
            r2 = float(r2_score(q_true, q_corrected_pred))

            print(f"Compared against: {true_name}")
            print(f"  MAE  : {mae:.6f}")
            print(f"  RMSE : {rmse:.6f}")
            print(f"  R²   : {r2:.6f}")

            if MAKE_PLOTS:
                plt.figure()
                plt.scatter(q_true, q_corrected_pred, alpha=0.7)
                lo = float(min(np.nanmin(q_true), np.nanmin(q_corrected_pred)))
                hi = float(max(np.nanmax(q_true), np.nanmax(q_corrected_pred)))
                plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)
                plt.xlabel(f"True ({true_name})")
                plt.ylabel("Predicted corrected flow")
                plt.title("Parity plot – corrected flow – TEST")
                plt.grid(True)
                plt.tight_layout()
                plt.show()

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
