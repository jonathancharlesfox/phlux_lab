from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# ROOT + HARD-CODED PATHS
# - This script lives in: F:\phlux\phlux_lab\scripts\
# - Project root is:      F:\phlux
# ============================================================
ROOT = Path(__file__).resolve().parents[2]  # -> F:\phlux

LOW_CSV  = ROOT / "phlux_lab" / "outputs" / "kspice_vs_ml (low wear).csv"
HIGH_CSV = ROOT / "phlux_lab" / "outputs" / "kspice_vs_ml (wear).csv"

OUT_ROOT = ROOT / "phlux_website" / "assets" / "results"


# ============================================================
# YOUR RENAMED COLUMNS (AUTHORITATIVE)
# ============================================================
C = {
    # inputs / context
    "p_in": "virplant_p_in__bar",
    "t_in": "virplant_t_in__degC",
    "p_dis": "virplant_discharge_p__bar",
    "p_sink": "virplant_sink_p__bar",
    "dp": "virplant_dp__bar",
    "speed": "speed__rpm",
    "rho": "fluid_density__kg/m3",
    "mu": "fluid_viscosity__cP",
    "power": "virplant_pump_power__kW",
    "valve": "valve_opening__frac",

    # wear
    "wear_true": "virplant_wear_input__frac",
    "wear_pred": "ml_wear_pred__frac",
    "wear_err_pct": "wear_err_pct",

    # flow (baseline)
    "flow_true": "virplant_flow__m3_h",
    "flow_pred": "ml_flow_pred__m3_h",
    "flow_abs_err_pct": "flow_abs_error__pct",

    # flow correction
    "corr_delta": "ml_corr_delta__m3_h",
    "flow_corr": "ml_flow_corrected__m3_h",
    "corr_abs_err_pct": "corr_abs_error__pct",
}

# Slim export columns order (public-facing)
SLIM_COLS = [
    C["p_in"], C["t_in"], C["p_dis"], C["p_sink"], C["dp"],
    C["speed"], C["rho"], C["mu"], C["power"], C["valve"],
    C["wear_true"], C["wear_pred"], C["wear_err_pct"],
    C["flow_true"], C["flow_pred"], C["flow_abs_err_pct"],
    C["corr_delta"], C["flow_corr"], C["corr_abs_err_pct"],
]


# ============================================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _require_cols(df: pd.DataFrame, cols: list[str], csv_path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing expected columns in {csv_path}:\n"
            f"  {missing}\n\n"
            f"Available columns:\n  {list(df.columns)}"
        )


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # core numerics
    out["flow_true"] = _num(out[C["flow_true"]])
    out["flow_pred"] = _num(out[C["flow_pred"]])
    out["flow_corr"] = _num(out[C["flow_corr"]]) if C["flow_corr"] in out.columns else np.nan

    out["wear_true_frac"] = _num(out[C["wear_true"]]) if C["wear_true"] in out.columns else np.nan
    out["wear_pred_frac"] = _num(out[C["wear_pred"]]) if C["wear_pred"] in out.columns else np.nan

    # baseline errors (use provided if present, else compute)
    if C["flow_abs_err_pct"] in out.columns:
        out["flow_abs_error_pct"] = _num(out[C["flow_abs_err_pct"]])
    else:
        denom = out["flow_true"].abs().replace(0, np.nan)
        out["flow_abs_error_pct"] = ((out["flow_pred"] - out["flow_true"]).abs() / denom) * 100.0

    # corrected errors (use provided if present, else compute)
    if C["corr_abs_err_pct"] in out.columns:
        out["corr_abs_error_pct"] = _num(out[C["corr_abs_err_pct"]])
    else:
        denom = out["flow_true"].abs().replace(0, np.nan)
        out["corr_abs_error_pct"] = ((out["flow_corr"] - out["flow_true"]).abs() / denom) * 100.0

    # residuals
    out["flow_residual"] = out["flow_pred"] - out["flow_true"]
    out["corr_residual"] = out["flow_corr"] - out["flow_true"]

    # wear percent versions (for charts)
    out["wear_true_pct"] = out["wear_true_frac"] * 100.0
    out["wear_pred_pct"] = out["wear_pred_frac"] * 100.0

    if C["wear_err_pct"] in out.columns:
        out["wear_err_pct_calc"] = _num(out[C["wear_err_pct"]])
    else:
        denom = out["wear_true_frac"].abs().replace(0, np.nan)
        out["wear_err_pct_calc"] = ((out["wear_pred_frac"] - out["wear_true_frac"]).abs() / denom) * 100.0

    # bucket for “high-flow performance” callouts
    out["flow_bucket"] = pd.cut(
        out["flow_true"],
        bins=[-np.inf, 10, 30, 60, 90, np.inf],
        labels=["<=10", "10–30", "30–60", "60–90", ">=90"],
    )

    return out


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(yhat)
    y = y[mask]
    yhat = yhat[mask]
    if y.size < 2:
        return float("nan")
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def summarize_errors(err_pct: pd.Series) -> Dict[str, Optional[float]]:
    s = err_pct.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {"mape_pct": None, "p50_pct": None, "p90_pct": None, "p95_pct": None, "p99_pct": None}
    return {
        "mape_pct": round(float(np.mean(s)), 2),
        "p50_pct": round(float(np.percentile(s, 50)), 2),
        "p90_pct": round(float(np.percentile(s, 90)), 2),
        "p95_pct": round(float(np.percentile(s, 95)), 2),
        "p99_pct": round(float(np.percentile(s, 99)), 2),
    }


def bucket_summary(df: pd.DataFrame, err_col: str) -> list[dict]:
    d = df.dropna(subset=["flow_bucket", err_col]).copy()
    if d.empty:
        return []
    rows = []
    grp = d.groupby("flow_bucket")[err_col]
    for bucket, s in grp:
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        rows.append({
            "bucket": str(bucket),
            "count": int(s.size),
            "mean_abs_error_pct": round(float(np.mean(s)), 2),
            "median_abs_error_pct": round(float(np.percentile(s, 50)), 2),
            "p95_abs_error_pct": round(float(np.percentile(s, 95)), 2),
        })
    return rows


def compute_kpis(df: pd.DataFrame, source_csv: Path) -> dict:
    d = df.dropna(subset=["flow_true", "flow_pred"]).copy()

    k = {
        "count": int(len(d)),
        "source_csv": str(source_csv),
        "columns": C,
        "flow_baseline": {
            "r2": None if np.isnan(r2_score(d["flow_true"].to_numpy(), d["flow_pred"].to_numpy()))
                  else round(r2_score(d["flow_true"].to_numpy(), d["flow_pred"].to_numpy()), 4),
            **summarize_errors(d["flow_abs_error_pct"]),
            "by_flow_bucket": bucket_summary(d, "flow_abs_error_pct"),
        },
        "flow_corrected": {
            "r2": None,
            **summarize_errors(d["corr_abs_error_pct"]),
            "by_flow_bucket": bucket_summary(d, "corr_abs_error_pct"),
        },
        "wear": None,
    }

    # corrected r2 only if corrected exists
    if d["flow_corr"].notna().any():
        r2c = r2_score(d["flow_true"].to_numpy(), d["flow_corr"].to_numpy())
        k["flow_corrected"]["r2"] = None if np.isnan(r2c) else round(r2c, 4)

    # wear block only if wear exists
    if d["wear_true_frac"].notna().any() and d["wear_pred_frac"].notna().any():
        wt = d["wear_true_pct"].replace([np.inf, -np.inf], np.nan).dropna()
        wp = d["wear_pred_pct"].replace([np.inf, -np.inf], np.nan).dropna()
        we = d["wear_err_pct_calc"].replace([np.inf, -np.inf], np.nan).dropna()
        k["wear"] = {
            "true_mean_pct": None if wt.empty else round(float(np.mean(wt)), 2),
            "pred_mean_pct": None if wp.empty else round(float(np.mean(wp)), 2),
            **summarize_errors(we),  # wear "error%" distribution summary
        }

    return k


# ============================================================
# PLOTS (PNG) - simple, website-friendly
# ============================================================
def plot_parity(x: pd.Series, y: pd.Series, out: Path, title: str, xlabel: str, ylabel: str) -> None:
    d = pd.DataFrame({"x": _num(x), "y": _num(y)}).dropna()
    if d.empty:
        return
    xv, yv = d["x"].to_numpy(), d["y"].to_numpy()

    plt.figure()
    plt.scatter(xv, yv, s=12)
    mn = float(np.min([xv.min(), yv.min()]))
    mx = float(np.max([xv.max(), yv.max()]))
    plt.plot([mn, mx], [mn, mx])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def plot_hist(s: pd.Series, out: Path, title: str, xlabel: str) -> None:
    v = _num(s).replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return
    plt.figure()
    plt.hist(v, bins=40)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def plot_scatter(x: pd.Series, y: pd.Series, out: Path, title: str, xlabel: str, ylabel: str, hline0: bool = False) -> None:
    d = pd.DataFrame({"x": _num(x), "y": _num(y)}).dropna()
    if d.empty:
        return
    plt.figure()
    plt.scatter(d["x"], d["y"], s=12)
    if hline0:
        plt.axhline(0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


# ============================================================
def export_case(csv_path: Path, label: str) -> None:
    print(f"▶ Processing {label}")

    df_raw = pd.read_csv(csv_path)

    # Require *at minimum* the columns we need for website metrics
    _require_cols(
        df_raw,
        [
            C["flow_true"],
            C["flow_pred"],
            C["flow_abs_err_pct"],
            C["flow_corr"],
            C["corr_abs_err_pct"],
        ],
        csv_path,
    )
    # wear/corr_delta optional (but expected)
    # _require_cols(df_raw, [C["wear_true"], C["wear_pred"], C["wear_err_pct"], C["corr_delta"]], csv_path)

    df = add_derived(df_raw)

    out_dir = OUT_ROOT / label
    ensure_dir(out_dir)

    # KPIs
    kpis = compute_kpis(df, csv_path)
    (out_dir / "kpis.json").write_text(json.dumps(kpis, indent=2))

    # Slim CSV: keep only your known renamed columns if present
    keep = [c for c in SLIM_COLS if c in df_raw.columns]
    df_raw[keep].to_csv(out_dir / "results_slim.csv", index=False)

    # --- Flow plots (baseline vs corrected) ---
    plot_parity(
        df["flow_true"], df["flow_pred"],
        out_dir / "parity_flow_baseline.png",
        f"{label}: Flow parity (baseline)",
        "Flow true (m3/h)",
        "Flow predicted (m3/h)",
    )
    plot_parity(
        df["flow_true"], df["flow_corr"],
        out_dir / "parity_flow_corrected.png",
        f"{label}: Flow parity (corrected)",
        "Flow true (m3/h)",
        "Flow corrected (m3/h)",
    )

    plot_hist(df["flow_abs_error_pct"], out_dir / "error_hist_baseline.png", f"{label}: Abs error % (baseline)", "Abs error (%)")
    plot_hist(df["corr_abs_error_pct"], out_dir / "error_hist_corrected.png", f"{label}: Abs error % (corrected)", "Abs error (%)")

    plot_scatter(
        df["flow_true"], df["flow_residual"],
        out_dir / "residual_vs_true_baseline.png",
        f"{label}: Residual vs true (baseline)",
        "Flow true (m3/h)",
        "Residual (pred - true) (m3/h)",
        hline0=True,
    )
    plot_scatter(
        df["flow_true"], df["corr_residual"],
        out_dir / "residual_vs_true_corrected.png",
        f"{label}: Residual vs true (corrected)",
        "Flow true (m3/h)",
        "Residual (corr - true) (m3/h)",
        hline0=True,
    )

    if C["corr_delta"] in df_raw.columns:
        plot_hist(df_raw[C["corr_delta"]], out_dir / "corr_delta_hist.png", f"{label}: Correction delta distribution", "Delta (m3/h)")

    # --- Wear plots (if present) ---
    if (C["wear_true"] in df_raw.columns) and (C["wear_pred"] in df_raw.columns):
        plot_parity(
            df["wear_true_pct"], df["wear_pred_pct"],
            out_dir / "parity_wear.png",
            f"{label}: Wear parity",
            "Wear input (%)",
            "Wear predicted (%)",
        )
        if C["wear_err_pct"] in df_raw.columns:
            plot_hist(df_raw[C["wear_err_pct"]], out_dir / "wear_err_hist.png", f"{label}: Wear error %", "Wear error (%)")

        plot_scatter(
            df["wear_true_pct"], df["flow_abs_error_pct"],
            out_dir / "wear_vs_flowerr_baseline.png",
            f"{label}: Wear vs flow abs error (baseline)",
            "Wear input (%)",
            "Flow abs error (%)",
        )
        plot_scatter(
            df["wear_true_pct"], df["corr_abs_error_pct"],
            out_dir / "wear_vs_flowerr_corrected.png",
            f"{label}: Wear vs flow abs error (corrected)",
            "Wear input (%)",
            "Flow abs error (%)",
        )

    print(f"✅ Exported → {out_dir}")


def main() -> None:
    ensure_dir(OUT_ROOT)

    # If you changed filenames, update LOW_CSV/HIGH_CSV above.
    export_case(LOW_CSV, "pilot_low_wear")
    export_case(HIGH_CSV, "pilot_wear")


if __name__ == "__main__":
    main()
