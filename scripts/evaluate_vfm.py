from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import r2_score

from phlux_lab.utils.predictor import VfmPredictor # type: ignore 
from phlux_lab.utils.units import convert_user_to_internal_SI, convert_internal_back_to_user # type: ignore

# MAE (Mean Absolute Error): Measures typical prediction error in original units.
# RMSE (Root Mean Squared Error): Penalizes larger errors more strongly than MAE.
# R² (Coefficient of Determination): Proportion of variance in the target explained by the model; 1.0 indicates a perfect fit.

# =========================================================
# USER INPUTS (EDIT THESE)
# =========================================================
LAB_ROOT = Path(__file__).resolve().parents[1]  # phlux_lab
MODEL_TO_USE = LAB_ROOT / "models" / "ClientA" / "vfm_ClientA.keras"
PREPROCESSOR_TO_USE = LAB_ROOT / "models" / "ClientA" / "preprocessor.joblib"
TEST_CSV = LAB_ROOT / "data" / "synthetic" / "ClientA_centrifugal_dataset_test.csv"

# Choose: "user" (schema units) or "canonical" (SI)
REPORT_UNITS = "user"   # "user" or "canonical"
# =========================================================

# Canonical SI unit labels for plots (extend as needed)
CANONICAL_UNIT_LABELS = {
    "q_liquid": "m3/s",
    "hydraulic_wear": "frac",
}

def _unit_label(target: str, schema_units_by_col: Dict[str, str], report_units: str) -> str:
    """
    Return the unit string to display on graphs (no 'user/canonical' words).
    """
    report_units = str(report_units).lower().strip()
    if report_units == "user":
        return str(schema_units_by_col.get(target, "") or "")
    if report_units == "canonical":
        return str(CANONICAL_UNIT_LABELS.get(target, ""))
    raise ValueError("REPORT_UNITS must be 'user' or 'canonical'")


def _to_canonical_matrix(df: pd.DataFrame, cols: List[str], units_map: Dict[str, str]) -> np.ndarray:
    """
    Convert df[cols] from schema USER units -> canonical SI (vectorized per column).
    """
    out = np.zeros((len(df), len(cols)), dtype="float32")

    for j, c in enumerate(cols):
        if c not in df.columns:
            raise KeyError(f"Missing required target column in CSV: {c}")

        u = units_map.get(c)
        if not u:
            raise KeyError(
                f"Schema units missing for target '{c}'. "
                "Required to convert y_true to canonical SI."
            )

        col = df[c].to_numpy(dtype="float32")
        out[:, j] = np.vectorize(lambda v: convert_user_to_internal_SI(float(v), u), otypes=[np.float32])(col)

    return out

def _canonical_to_user_matrix(y_canonical: np.ndarray, cols: List[str], units_map: Dict[str, str]) -> np.ndarray:
    """
    Convert canonical SI -> schema USER units (vectorized per column).
    """
    out = np.zeros_like(y_canonical, dtype="float32")

    for j, c in enumerate(cols):
        u = units_map.get(c)
        if not u:
            raise KeyError(
                f"Schema units missing for target '{c}'. "
                "Required to convert predictions back to user units."
            )

        col = y_canonical[:, j].astype("float32")
        out[:, j] = np.vectorize(lambda v: convert_internal_back_to_user(float(v), u), otypes=[np.float32])(col)

    return out

def _mae_rmse_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    err = (y_pred - y_true).astype("float32")
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err * err, axis=0))
    return mae, rmse

def main() -> None:
    report_units = str(REPORT_UNITS).lower().strip()
    if report_units not in {"user", "canonical"}:
        raise ValueError("REPORT_UNITS must be 'user' or 'canonical'")

    print("Starting evaluation...")

    # -----------------------------
    # 1) Sanity checks
    # -----------------------------
    for p, label in [
        (MODEL_TO_USE, "Model"),
        (PREPROCESSOR_TO_USE, "Preprocessor"),
        (TEST_CSV, "Test CSV"),
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{label} not found: {p}")

    # -----------------------------
    # 2) Load predictor (model + artifact)
    # -----------------------------
    predictor = VfmPredictor.from_paths(
        MODEL_TO_USE,
        PREPROCESSOR_TO_USE,
    )

    target_cols = list(predictor.target_cols)  # unitless targets
    schema_units_by_col = getattr(predictor, "schema_units_by_col", {}) or {}

    print(f"Model        : {MODEL_TO_USE}")
    print(f"Preprocessor : {PREPROCESSOR_TO_USE}")
    print(f"Test CSV     : {TEST_CSV}")
    print(f"Targets      : {target_cols}")
    print(f"Report units : {report_units}")

    # -----------------------------
    # 3) Load test data (schema USER units)
    # -----------------------------
    df = pd.read_csv(TEST_CSV)
    print(f"Test samples : {len(df)}")

    missing_targets = [c for c in target_cols if c not in df.columns]
    if missing_targets:
        raise KeyError(
            "CSV is missing required target columns:\n"
            + "\n".join([f"  - {c}" for c in missing_targets])
        )

    # -----------------------------
    # 4) Predict (canonical SI)
    #    X uses preprocessor so derived features are handled consistently.
    # -----------------------------
    print("Running predictions (vectorized)...")
    X_scaled = predictor.pp.transform_X(df)
    y_scaled_pred = predictor.model.predict(X_scaled, verbose=0)
    y_pred_canonical = predictor.pp.inverse_transform_y(y_scaled_pred).astype("float32")
    if y_pred_canonical.ndim == 1:
        y_pred_canonical = y_pred_canonical.reshape(-1, 1)

    # -----------------------------
    # 5) Build y_true (canonical SI)
    # -----------------------------
    y_true_canonical = _to_canonical_matrix(df, target_cols, schema_units_by_col)

    # -----------------------------
    # 6) Convert to requested report units
    # -----------------------------
    if report_units == "canonical":
        y_true = y_true_canonical
        y_pred = y_pred_canonical
    else:
        y_true = _canonical_to_user_matrix(y_true_canonical, target_cols, schema_units_by_col)
        y_pred = _canonical_to_user_matrix(y_pred_canonical, target_cols, schema_units_by_col)

    # -----------------------------
    # 7) Metrics (in selected units)
    # -----------------------------
    mae, rmse = _mae_rmse_per_target(y_true, y_pred)
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")

    print("\n=== Evaluation Metrics (TEST SET) ===")
    for i, t in enumerate(target_cols):
        u = _unit_label(t, schema_units_by_col, report_units)
        tname = f"{t} [{u}]" if u else t
        print(f"\nTarget: {tname}")
        print(f"  MAE  : {mae[i]:.6f}")
        print(f"  RMSE : {rmse[i]:.6f}")
        print(f"  R²   : {float(r2[i]):.6f}")

    # -----------------------------
    # 8) Plots (first target)
    # -----------------------------
    t0 = target_cols[0]
    u0 = _unit_label(t0, schema_units_by_col, report_units)
    axis_label = f"{t0} [{u0}]" if u0 else t0

    y_true_flat = y_true[:, 0].flatten()
    y_pred_flat = y_pred[:, 0].flatten()

    plt.figure()
    plt.scatter(y_true_flat, y_pred_flat, alpha=0.7)

    min_val = float(min(y_true_flat.min(), y_pred_flat.min()))
    max_val = float(max(y_true_flat.max(), y_pred_flat.max()))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=2)

    plt.xlabel(f"True {axis_label}")
    plt.ylabel(f"Predicted {axis_label}")
    plt.title("Parity plot – TEST")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    residuals = y_pred_flat - y_true_flat

    plt.figure()
    plt.hist(residuals, bins=20)
    plt.xlabel(f"Prediction error ({axis_label})")
    plt.ylabel("Count")
    plt.title("Residuals – TEST")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
