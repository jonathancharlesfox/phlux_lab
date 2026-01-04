from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt


# ==========================
# PATHS (EDIT IF NEEDED)
# ==========================
LOW_CSV  = Path(r"F:\phlux\phlux_lab\outputs\kspice_vs_ml (low deg).csv")
HIGH_CSV = Path(r"F:\phlux\phlux_lab\outputs\kspice_vs_ml (high deg).csv")

OUT_ROOT = Path(r"F:\phlux\phlux_website\assets\results")


# ==========================
# COLUMN MAPPING (FROM YOUR HEADER)
# ==========================
COL_TRUTH_FLOW = "ks_flow__m3_h"
COL_PRED_FLOW  = "ml_flow_pred__m3_h"
COL_WEAR_PCT   = "ml_hydraulic_wear_probability__pct"
COL_ERR_PCT    = "flow_abs_error__pct"   # already in your CSV


HOVER_COLS = [
    "ks_p_in__bar",
    "ks_t_in__degC",
    "ks_discharge_p__bar",
    "ks_sink_p__bar",
    "ks_dp__bar",
    "speed__rpm",
    "valve_opening__frac",
    "ks_pump_power__kW",
    "hydraulic_wear__frac",
    "fluid_density__kg/m3",
    "fluid_viscosity__cP",
]


# ==========================
# HELPERS
# ==========================
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["flow_true"] = _num(df[COL_TRUTH_FLOW])
    df["flow_pred"] = _num(df[COL_PRED_FLOW])
    df["wear_pct"]  = _num(df[COL_WEAR_PCT])
    df["err_pct"]   = _num(df[COL_ERR_PCT])

    df["abs_err"] = (df["flow_pred"] - df["flow_true"]).abs()
    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["flow_true", "flow_pred"]).copy()

    mape = float(np.nanmean(d["err_pct"]))
    p95_err = float(np.nanpercentile(d["err_pct"], 95))
    p95_abs = float(np.nanpercentile(d["abs_err"], 95))

    wear_mean = float(np.nanmean(d["wear_pct"]))
    wear_p95 = float(np.nanpercentile(d["wear_pct"], 95))
    wear_max = float(np.nanmax(d["wear_pct"]))
    frac_over5 = float(np.mean(d["wear_pct"] > 5.0)) if d["wear_pct"].notna().any() else float("nan")

    return {
        "count": int(len(d)),
        "mape_pct": round(mape, 2),
        "p95_error_pct": round(p95_err, 2),
        "p95_abs_m3_h": round(p95_abs, 3),
        "wear_mean_pct": round(wear_mean, 2),
        "wear_p95_pct": round(wear_p95, 2),
        "wear_max_pct": round(wear_max, 2),
        "wear_over_5pct_frac": round(frac_over5, 3),
    }


def make_hero_plot(df: pd.DataFrame, out_html: Path, title: str) -> None:
    hover = [c for c in HOVER_COLS if c in df.columns]

    fig = px.scatter(
        df,
        x="flow_true",
        y="flow_pred",
        color="wear_pct",
        hover_data=hover + ["err_pct"],
        labels={
            "flow_true": "Virtual plant flow (m³/h)",
            "flow_pred": "ML predicted flow (m³/h)",
            "wear_pct": "Predicted wear (%)",
            "err_pct": "Abs error (%)",
        },
        title=title,
    )

    lo = float(np.nanmin([df["flow_true"].min(), df["flow_pred"].min()]))
    hi = float(np.nanmax([df["flow_true"].max(), df["flow_pred"].max()]))

    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi)
    fig.update_layout(margin=dict(l=30, r=30, t=60, b=30))
    fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)


def save_hist(df: pd.DataFrame, col: str, out_path: Path, title: str, xlabel: str) -> None:
    plt.figure()
    s = df[col].dropna()
    plt.hist(s, bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()


def export_one(label: str, csv_path: Path) -> None:
    out_dir = OUT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = add_metrics(df)

    make_hero_plot(df, out_dir / "hero.html", f"Flow parity (colored by predicted wear) — {label.replace('_',' ')}")
    save_hist(df, "err_pct", out_dir / "error_hist.svg", "Flow absolute error (%) distribution", "Abs error (%)")
    save_hist(df, "wear_pct", out_dir / "wear_hist.svg", "Predicted wear (%) distribution", "Predicted wear (%)")

    with (out_dir / "kpis.json").open("w", encoding="utf-8") as f:
        json.dump(compute_kpis(df), f, indent=2)


def main() -> None:
    export_one("low_deg", LOW_CSV)
    export_one("high_deg", HIGH_CSV)
    print("Export complete ->", OUT_ROOT)


if __name__ == "__main__":
    main()
