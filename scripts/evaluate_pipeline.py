from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import yaml

from phlux_lab.utils.preprocessor import Preprocessor  # type: ignore
from phlux_lab.ml.vfm_model import VFMModel            # type: ignore


# =============================================================================
# Paths / YAML loading
# =============================================================================

HERE = Path(__file__).resolve().parent
LAB_ROOT = HERE.parent
REPO_ROOT = LAB_ROOT.parent

DEFAULT_CONFIG = LAB_ROOT / "configs" / "training_config.yaml"


def _resolve_path(p: str | Path) -> Path:
    pp = Path(str(p)).expanduser()
    if pp.is_absolute():
        return pp
    s = str(p).replace("\\", "/")
    if s.startswith("phlux_lab/") or s.startswith("src/") or s.startswith("logs/"):
        return (REPO_ROOT / s).resolve()
    return (LAB_ROOT / s).resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# =============================================================================
# Metrics helpers (plots → numbers)
# =============================================================================

def _basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    resid = y_pred - y_true

    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    r2 = float(r2_score(y_true, y_pred))

    bias = float(np.mean(resid))
    p95_abs_error = float(np.percentile(np.abs(resid), 95))

    if np.std(y_true) > 0:
        corr_abs_resid = float(np.corrcoef(np.abs(resid), y_true)[0, 1])
    else:
        corr_abs_resid = 0.0

    slope, intercept = np.polyfit(y_true, y_pred, 1)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
        "p95_abs_error": p95_abs_error,
        "corr_abs_resid_vs_true": corr_abs_resid,
        "calib_slope": float(slope),
        "calib_intercept": float(intercept),
        "calib_slope_abs_err": abs(float(slope) - 1.0),
        "calib_intercept_abs_err": abs(float(intercept)),
    }


def _binned_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int,
    *,
    bin_min: Optional[float],
    bin_max: Optional[float],
    edges: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compute per-bin error stats using bins defined on y_true.

    Binning priority:
      1) If `edges` is provided (list of monotonically increasing bin edges), use it.
      2) Else if both `bin_min` and `bin_max` are provided, use uniform bins with `n_bins`.
      3) Else fall back to uniform bins from min(y_true) to max(y_true) using `n_bins`.

    Returns:
      {
        "edges": [...],  # length n_bins+1
        "bins": [
           {"bin": i, "lo": edge_i, "hi": edge_{i+1}, "count": N, "mae": ..., "bias": ..., "p95_abs_error": ...},
           ...
        ]
      }
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    if edges is not None:
        edges_arr = np.asarray([float(x) for x in edges], dtype=float)
        if edges_arr.ndim != 1 or edges_arr.size < 2:
            raise ValueError("edges must be a 1D list with at least 2 values")
        if not np.all(np.isfinite(edges_arr)):
            raise ValueError("edges contains non-finite values")
        if np.any(np.diff(edges_arr) <= 0):
            raise ValueError("edges must be strictly increasing")
        n_bins_local = int(edges_arr.size - 1)
        edges_arr = edges_arr.astype(float)
    else:
        if n_bins <= 0:
            raise ValueError("n_bins must be > 0")
        n_bins_local = int(n_bins)

        if bin_min is not None and bin_max is not None:
            edges_arr = np.linspace(float(bin_min), float(bin_max), n_bins_local + 1)
        else:
            # fallback: data-driven binning on y_true
            edges_arr = np.linspace(float(np.min(y_true)), float(np.max(y_true)), n_bins_local + 1)

    # Assign each sample to a bin by y_true
    idx = np.digitize(y_true, edges_arr, right=False) - 1
    idx = np.clip(idx, 0, n_bins_local - 1)

    bins: List[Dict[str, Any]] = []
    for i in range(n_bins_local):
        lo = float(edges_arr[i])
        hi = float(edges_arr[i + 1])
        m = idx == i
        if not np.any(m):
            bins.append(
                {
                    "bin": i,
                    "lo": lo,
                    "hi": hi,
                    "count": 0,
                    "mae": None,
                    "bias": None,
                    "p95_abs_error": None,
                }
            )
            continue

        resid = (y_pred[m] - y_true[m]).reshape(-1)
        bins.append(
            {
                "bin": i,
                "lo": lo,
                "hi": hi,
                "count": int(np.sum(m)),
                "mae": float(np.mean(np.abs(resid))),
                "bias": float(np.mean(resid)),
                "p95_abs_error": float(np.percentile(np.abs(resid), 95)),
            }
        )

    return {"edges": edges_arr.tolist(), "bins": bins}


def _focus_aggregates(
    binned: Optional[Dict[str, Any]],
    focus_bins: List[float],
    *,
    treat_as_indices: Optional[bool] = None,
) -> Dict[str, Optional[float]]:
    """Aggregate diagnostics over 'focus' regions.

    Supports two interpretations of focus_bins:
      1) Bin indices (e.g., [3,4,5])  -> select those bins directly
      2) Cut points in target units (e.g., [60,80,120]) -> focus ranges [60,80] and [80,120]

    Heuristic: if treat_as_indices is None, treat_as_indices=True only when all focus values are
    integers within plausible bin-index range.
    """
    # If binning is disabled (binned=None), focus metrics are undefined.
    if binned is None:
        return {
            "abs_bias_focus": None,
            "p95_abs_error_focus": None,
        }

    bins = binned.get("bins", []) or []
    if not bins or not focus_bins:
        return {"abs_bias_focus": None, "p95_abs_error_focus": None}

    n_bins = len(bins)

    # Decide interpretation
    if treat_as_indices is None:
        all_int = all(float(x).is_integer() for x in focus_bins)
        maxv = int(max(focus_bins)) if focus_bins else -1
        treat_as_indices = bool(all_int and 0 <= maxv <= n_bins - 1)

    selected: List[Dict[str, Any]] = []

    if treat_as_indices:
        idx_set = set(int(x) for x in focus_bins)
        for b in bins:
            if int(b.get("bin", -1)) in idx_set and (b.get("count", 0) or 0) > 0:
                selected.append(b)
    else:
        # Interpret as cut points in y-units -> ranges
        cuts = sorted(float(x) for x in focus_bins)
        if len(cuts) == 1:
            # Single cut: focus on values >= cut
            ranges = [(cuts[0], float("inf"))]
        else:
            ranges = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]

        for b in bins:
            if (b.get("count", 0) or 0) <= 0:
                continue
            lo = float(b.get("lo", float("nan")))
            hi = float(b.get("hi", float("nan")))
            if not np.isfinite(lo) or not np.isfinite(hi):
                continue

            # Include bin if it overlaps ANY focus range
            for a, c in ranges:
                if hi <= a:
                    continue
                if lo >= c:
                    continue
                selected.append(b)
                break

    abs_bias: List[float] = []
    p95: List[float] = []
    for b in selected:
        bias = b.get("bias", None)
        p95v = b.get("p95_abs_error", None)
        if bias is None or p95v is None:
            continue
        abs_bias.append(abs(float(bias)))
        p95.append(float(p95v))

    return {
        "abs_bias_focus": float(np.mean(abs_bias)) if abs_bias else None,
        "p95_abs_error_focus": float(np.max(p95)) if p95 else None,
    }


# =============================================================================
# Plotting
# =============================================================================

def _save_parity(out: Path, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    plt.figure()
    plt.scatter(y_true, y_pred, s=8)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()


def _save_residual(out: Path, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    plt.figure()
    plt.scatter(y_true, y_pred - y_true, s=8)
    plt.axhline(0.0)
    plt.xlabel("True")
    plt.ylabel("Residual")
    plt.title(title)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()


# =============================================================================
# Main evaluation
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--stage", type=str, default="all")
    args = parser.parse_args()

    cfg = _load_yaml(_resolve_path(args.config))
    client = cfg["project"]["client_name"]

    # model_root can be either models root or already client-scoped
    model_root = _resolve_path(cfg["paths"]["model_root"])
    client_dir = model_root if model_root.name == client else (model_root / client)

    autolab = cfg.get("autolab", {})
    primary = autolab.get("objective", {}).get("primary", {})
    autolab_focus_bins = primary.get("focus_bins", [])
    autolab_focus_target = primary.get("target")  # e.g. "q_liquid"
    metrics_out: Dict[str, Any] = {"client": client, "stages": {}}

    # -----------------------------------------------------------------
    # Optional GLOBAL evaluation bin edges
    # You currently have these under models.evaluation.bins (NOT a model stage).
    # If present, they act as a fallback when a stage does not define its own
    # evaluation.bins.edges.
    # -----------------------------------------------------------------
    global_eval_bins = (cfg.get("models", {}).get("evaluation", {}) or {}).get("bins", {}) or {}
    global_edges = None
    if bool(global_eval_bins.get("enabled", False)):
        ge = global_eval_bins.get("edges")
        if isinstance(ge, list) and len(ge) >= 2:
            global_edges = ge


    for stage_name, stage_cfg in cfg["models"].items():
        # "evaluation" in the YAML is a configuration-only block, not a model stage.
        if stage_name.lower() in {"evaluation", "evaluator"}:
            continue
        if not isinstance(stage_cfg, dict) or "data" not in stage_cfg:
            # Skip any non-stage entries safely
            continue
        if args.stage != "all" and args.stage != stage_name:
            continue

        model_path = client_dir / stage_name / f"vfm_{stage_name}.keras"
        stage_dir = client_dir / stage_name

        # Preprocessor artifact name can vary across iterations; prefer the canonical name,
        # then fall back to auto-discovery within the stage directory.
        pp_path = stage_dir / f"preprocessor_{stage_name}.joblib"
        if not pp_path.exists():
            # Common alternate names / legacy patterns
            candidates = [
                stage_dir / f"preprocessor_{stage_name}.pkl",
                stage_dir / f"preprocessor_{stage_name}.joblib",
                stage_dir / f"preprocessor_{stage_name}.joblib",
                stage_dir / f"preprocessor_{stage_name}.joblib",
                stage_dir / f"preprocessor_{stage_name}.joblib",
                stage_dir / f"preprocessor_{stage_name}.joblib",
                stage_dir / f"preprocessor_{stage_name}.joblib",
                stage_dir / f"preprocessor_{stage_name}.joblib",
                stage_dir / f"preprocessor_{stage_name}.joblib",
            ]
            # Last resort: any preprocessor*.joblib in the stage directory
            candidates.extend(stage_dir.glob("preprocessor*.joblib"))
            for c in candidates:
                if Path(c).exists():
                    pp_path = Path(c)
                    break

        if not pp_path.exists():
            raise FileNotFoundError(
                f"Preprocessor artifact not found for stage '{stage_name}'. "
                f"Expected '{stage_dir / f'preprocessor_{stage_name}.joblib'}'. "
                f"Available files in '{stage_dir}': {[p.name for p in stage_dir.glob('*')][:40]}"
            )

        model_cfg = stage_cfg.get("model") or stage_cfg.get("training") or {}
        model = VFMModel.load(model_path, Preprocessor.load(pp_path), model_cfg)
        data_cfg = stage_cfg["data"]

        # Stage-specific evaluation bins (preferred)
        eval_bins = (stage_cfg.get("evaluation") or {}).get("bins") or {}

        # NEW: if bins are explicitly disabled for this stage, do NOT fall back
        bins_explicitly_disabled = ("enabled" in eval_bins) and (eval_bins.get("enabled") is False)

        stage_edges = None

        if bins_explicitly_disabled:
            # Marker meaning: "do not compute binned metrics for this stage"
            stage_edges = []
        elif bool(eval_bins.get("enabled", False)):
            edges = eval_bins.get("edges")
            if isinstance(edges, list) and len(edges) >= 2:
                stage_edges = edges

        # Only fall back when bins are NOT explicitly disabled
        if stage_edges is None and (not bins_explicitly_disabled) and global_edges is not None:
            stage_edges = global_edges

        df = pd.read_csv(_resolve_path(data_cfg["test_dataset"]))

        y_pred = model.predict_in_units(df)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, None]

        stage_metrics = {"targets": {}}

        for i, tgt in enumerate(data_cfg["targets"]):
            y_true = df[tgt].to_numpy(dtype=float)

            basic = _basic_metrics(y_true, y_pred[:, i])

            # -----------------------------------------------------------------
            # Pick bin edges PER TARGET
            # Priority:
            #   1) target-level bins (if provided)
            #   2) stage-level bins (unless explicitly disabled for the stage)
            #   3) global bins (unless explicitly disabled for the stage)
            #   4) objective bins (bin_min/bin_max/n_bins)
            #
            # Special marker:
            #   stage_edges == []  => explicitly disable binning for this stage
            # -----------------------------------------------------------------
            edges_for_target = None

            # 1) target-level bins (optional, if present in YAML)
            tp = (data_cfg.get("target_policy") or {}).get(tgt) or {}
            tp_bins = ((tp.get("evaluation") or {}).get("bins") or {})
            if bool(tp_bins.get("enabled", False)):
                e = tp_bins.get("edges")
                if isinstance(e, list) and len(e) >= 2:
                    edges_for_target = e

            # 2) stage-level bins (only if stage binning is not explicitly disabled)
            if edges_for_target is None:
                if stage_edges == []:
                    # explicitly disabled at stage level => do not bin unless target-level bins exist
                    edges_for_target = []
                elif isinstance(stage_edges, list) and len(stage_edges) >= 2:
                    edges_for_target = stage_edges

            # 3) global bins (only if not explicitly disabled and no stage bins)
            if edges_for_target is None:
                if stage_edges == []:
                    edges_for_target = []
                elif isinstance(global_edges, list) and len(global_edges) >= 2:
                    edges_for_target = global_edges

            # -----------------------------------------------------------------
            # Compute binned metrics
            # -----------------------------------------------------------------
            binned = None

            if edges_for_target == []:
                # Explicitly disabled => do not compute binned metrics
                binned = None

            elif isinstance(edges_for_target, list) and len(edges_for_target) >= 2:
                # Explicit edges
                binned = _binned_metrics(
                    y_true,
                    y_pred[:, i],
                    n_bins=len(edges_for_target) - 1,
                    bin_min=None,
                    bin_max=None,
                    edges=edges_for_target,
                )

            else:
                # Fallback to objective ruler
                binned = _binned_metrics(
                    y_true,
                    y_pred[:, i],
                    n_bins=primary.get("n_bins", 12),
                    bin_min=primary.get("bin_min"),
                    bin_max=primary.get("bin_max"),
                    edges=None,
                )

            # -----------------------------------------------------------------
            # Focus bins for *evaluation* (NOT autolab objective)
            # Priority:
            #   1) target-level focus_bins (optional)
            #   2) stage-level evaluation.focus_bins
            #   3) none
            # -----------------------------------------------------------------
            focus_bins_eval: List[float] = []

            # 1) target-level focus (optional, future-proof)
            tp = (data_cfg.get("target_policy") or {}).get(tgt) or {}
            tp_eval = (tp.get("evaluation") or {})
            fb = tp_eval.get("focus_bins")
            if isinstance(fb, list) and len(fb) > 0:
                focus_bins_eval = fb
            else:
                # 2) stage-level focus
                stage_eval = (stage_cfg.get("evaluation") or {})
                fb2 = stage_eval.get("focus_bins")
                if isinstance(fb2, list) and len(fb2) > 0:
                    focus_bins_eval = fb2

            focus = _focus_aggregates(binned, focus_bins_eval)

            stage_metrics["targets"][tgt] = {
                **basic,
                **focus,
                "binned": binned,
            }

            eval_dir = client_dir / stage_name / "eval"
            _save_parity(eval_dir / f"parity_{tgt}.png", y_true, y_pred[:, i], f"{stage_name} parity")
            _save_residual(eval_dir / f"residuals_{tgt}.png", y_true, y_pred[:, i], f"{stage_name} residuals")


        metrics_out["stages"][stage_name] = stage_metrics

    eval_root = client_dir / "eval"
    _ensure_dir(eval_root)
    _json_dump(metrics_out, eval_root / "metrics.json")

    print("✅ Evaluation complete")


if __name__ == "__main__":
    main()
