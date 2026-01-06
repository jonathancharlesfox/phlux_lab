from __future__ import annotations

"""
phlux_lab/scripts/autolab_run.py

Phase 1 "Autolab" loop runner (NO LLM):
  (optional) generate data -> train pipeline -> evaluate -> propose YAML patch -> repeat

Design goals:
- Deterministic and safe: only edits allowed keys, validates ranges lightly.
- Works with your existing scripts via subprocess calls.
- Produces a per-iteration run folder with resolved config + logs + metrics + patches.

Typical usage (Windows / PowerShell):
  python phlux_lab/scripts/autolab_run.py --config phlux_lab/configs/training_config.yaml

Notes:
- This script assumes you already have a training pipeline script (train_pipeline.py)
  that accepts a config path as argv[1].
- For evaluation, it *tries* to call an eval script with the config path as argv[1].
  If your eval script is hardcoded, update it to accept argv[1] (recommended).
"""

import argparse

DEFAULT_CONFIG = "phlux_lab/configs/training_config.yaml"
DEFAULT_TRAIN_SCRIPT = "phlux_lab/scripts/train_pipeline.py"
DEFAULT_EVAL_SCRIPT = "phlux_lab/scripts/evaluate_pipeline.py"
import json
import os
import re
import shutil
import subprocess

# Force UTF-8 for subprocess output decoding on Windows
import os
_UTF8_ENV = os.environ.copy()
_UTF8_ENV['PYTHONIOENCODING'] = 'utf-8'
_UTF8_ENV['PYTHONUTF8'] = '1'

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import yaml


# -----------------------------------------------------------------------------
# Helpers: YAML load/save + deep merge
# -----------------------------------------------------------------------------
def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, default_flow_style=False)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override into base, returning a new dict.
    Scalars/arrays override directly. Dicts merge recursively.
    """
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out



# Backward-compatible alias (older drafts referenced merge_cfg)
def merge_cfg(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a patch dict into a base config dict (deep merge)."""
    return deep_merge(base, patch)


# -----------------------------------------------------------------------------
# Run artifacts
# -----------------------------------------------------------------------------
@dataclass
class StageMetric:
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None


@dataclass
class IterationResult:
    run_id: str
    metrics: Dict[str, Dict[str, StageMetric]]  # stage -> target -> metrics
    raw_stdout: str
    raw_stderr: str


# -----------------------------------------------------------------------------
# Parsing evaluation stdout
# -----------------------------------------------------------------------------
_STAGE_RE = re.compile(r"^===\s*Stage:\s*(?P<stage>\w+)\s*===", re.MULTILINE)
_METRIC_RE = re.compile(
    r"^\s*(?P<target>[\w\.\-]+)\s*:\s*MAE=(?P<mae>[-+eE0-9\.]+)\s*RMSE=(?P<rmse>[-+eE0-9\.]+)\s*R2=(?P<r2>[-+eE0-9\.]+)\s*$",
    re.MULTILINE,
)


def parse_eval_output(stdout: str) -> Dict[str, Dict[str, StageMetric]]:
    """
    Parses output like:
      === Stage: flow ===
      ...
        q_liquid:  MAE=0.93695  RMSE=1.25722  R2=0.997863
    Returns: { "flow": {"q_liquid": StageMetric(...)}, ... }
    """
    metrics: Dict[str, Dict[str, StageMetric]] = {}
    # Split by stage headers
    stage_positions = [(m.start(), m.group("stage")) for m in _STAGE_RE.finditer(stdout)]
    if not stage_positions:
        # Some eval scripts don't print stage headers—try global parse
        for mm in _METRIC_RE.finditer(stdout):
            tgt = mm.group("target")
            metrics.setdefault("unknown", {})[tgt] = StageMetric(
                mae=float(mm.group("mae")),
                rmse=float(mm.group("rmse")),
                r2=float(mm.group("r2")),
            )
        return metrics

    stage_positions.append((len(stdout), "__end__"))
    for i in range(len(stage_positions) - 1):
        start, stage = stage_positions[i]
        end, _ = stage_positions[i + 1]
        chunk = stdout[start:end]
        for mm in _METRIC_RE.finditer(chunk):
            tgt = mm.group("target")
            metrics.setdefault(stage, {})[tgt] = StageMetric(
                mae=float(mm.group("mae")),
                rmse=float(mm.group("rmse")),
                r2=float(mm.group("r2")),
            )
    return metrics


# -----------------------------------------------------------------------------
# Subprocess runner
# -----------------------------------------------------------------------------
def run_cmd(cmd: list[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """Run a command and capture stdout/stderr as UTF-8 text safely (Windows-friendly)."""
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=_UTF8_ENV,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    return p.returncode, (p.stdout or ""), (p.stderr or "")



# -----------------------------------------------------------------------------
# Metrics.json loader + objective scoring
# -----------------------------------------------------------------------------
def _resolve_client_name(cfg: Dict[str, Any]) -> Optional[str]:
    """Resolve client name from config.

    Canonical location is:
      project.client_name

    Backward compatible fallbacks:
      client_name, client, customer, account
    """
    proj = cfg.get("project", {}) or {}
    if isinstance(proj, dict):
        v = proj.get("client_name") or proj.get("client")
        if isinstance(v, str) and v.strip():
            return v.strip()

    for k in ("client_name", "client", "customer", "account"):
        v = cfg.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def _client_models_dir(repo_root: Path, cfg: Dict[str, Any]) -> Optional[Path]:
    """Return the directory that contains stage folders for this client."""
    client = _resolve_client_name(cfg)
    if not client:
        return None
    # Prefer config-driven model_root if present
    paths = cfg.get("paths", {}) or {}
    raw_root = paths.get("model_root")
    if isinstance(raw_root, str) and raw_root.strip():
        model_root = (repo_root / raw_root).resolve() if not Path(raw_root).is_absolute() else Path(raw_root)
        return model_root if model_root.name == client else (model_root / client)
    # Fallback to repo default
    default_root = repo_root / "phlux_lab" / "models"
    return default_root / client
def _load_eval_metrics_json(repo_root: Path, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Load evaluator-produced metrics.json if present.

    The evaluator writes to:
      <model_root>/<client>/eval/metrics.json

    Where model_root comes from:
      paths.model_root (defaults to phlux_lab/models)

    We also keep a couple legacy fallbacks.
    """
    client = _resolve_client_name(cfg)
    if not client:
        return None

    def _resolve_path(p: str | Path) -> Path:
        pp = Path(str(p)).expanduser()
        if pp.is_absolute():
            return pp
        s = str(p).replace("\\", "/")
        if s.startswith("phlux_lab/"):
            return (repo_root / s).resolve()
        return (repo_root / s).resolve()

    paths_cfg = cfg.get("paths", {}) or {}
    raw_root = str(paths_cfg.get("model_root", "phlux_lab/models"))
    raw_root = raw_root.replace("{client}", client).replace("{client_name}", client)
    base = _resolve_path(raw_root)

    # If user already pointed at the client folder, don't append again
    client_dir = base if base.name.lower() == client.lower() else (base / client)

    candidates = [
        client_dir / "eval" / "metrics.json",
        # legacy fallbacks
        repo_root / "phlux_lab" / "models" / client / "eval" / "metrics.json",
        repo_root / "models" / client / "eval" / "metrics.json",
    ]

    for p in candidates:
        if p.exists() and p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None
def _get_stage_target_block(metrics: Dict[str, Any], stage: str, target: str) -> Optional[Dict[str, Any]]:
    """
    Supports two schemas:
      A) evaluator schema:
         metrics["stages"][stage]["targets"][target]
      B) legacy autolab schema:
         metrics["parsed"][stage][target]  (overall metrics only)
    """
    if not isinstance(metrics, dict):
        return None

    stages = metrics.get("stages")
    if isinstance(stages, dict):
        st = stages.get(stage)
        if isinstance(st, dict):
            tgts = st.get("targets")
            if isinstance(tgts, dict):
                tgt_block = tgts.get(target)
                if isinstance(tgt_block, dict):
                    return tgt_block

    parsed = metrics.get("parsed")
    if isinstance(parsed, dict):
        st = parsed.get(stage)
        if isinstance(st, dict):
            tgt_block = st.get(target)
            if isinstance(tgt_block, dict):
                return tgt_block

    return None


def _preferred_stage(cfg: Dict[str, Any]) -> str:
    # Prefer correction if enabled
    if _enabled(cfg, "flow_correction"):
        return "flow_correction"
    return "flow"


def _score_overall(metrics: Dict[str, Any], stage: str, target: str, metric_name: str) -> Optional[float]:
    tgt = _get_stage_target_block(metrics, stage, target)
    if not tgt:
        return None
    overall = tgt.get("overall", tgt)  # evaluator uses "overall"; legacy is flat
    if not isinstance(overall, dict):
        return None
    v = overall.get(metric_name)
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def _score_mae_by_bins(
    metrics: Dict[str, Any],
    stage: str,
    target: str,
    focus_bins: list,
) -> Optional[float]:
    """
    Compute a single scalar MAE across selected value ranges (focus bins).

    focus_bins is interpreted as value cut points, e.g. [60, 80, 120] -> ranges:
      60-80 and 80-120

    Uses the evaluator schema:
      target_block["binned"]["bins"] containing entries with lo/hi/count/mae
    """
    tgt = _get_stage_target_block(metrics, stage, target)
    if not tgt:
        return None

    binned = tgt.get("binned")
    if not isinstance(binned, dict):
        return None

    bins_list = binned.get("bins")
    if not isinstance(bins_list, list) or not bins_list:
        return None

    # Build focus ranges
    cuts = [c for c in focus_bins if isinstance(c, (int, float))]
    cuts = sorted(cuts)
    if len(cuts) < 2:
        # no focus range specified -> use all bins
        ranges = None
    else:
        ranges = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]

    def overlaps(lo: float, hi: float, rlo: float, rhi: float) -> bool:
        return max(lo, rlo) < min(hi, rhi)

    weighted_sum = 0.0
    weight_total = 0.0

    for b in bins_list:
        try:
            lo = float(b.get("lo"))
            hi = float(b.get("hi"))
            mae = b.get("mae", None)
            cnt = b.get("count", 0)
            if mae is None:
                continue
            mae = float(mae)
            cnt = float(cnt) if cnt is not None else 0.0
        except Exception:
            continue

        if cnt <= 0:
            continue

        if ranges is not None:
            ok = any(overlaps(lo, hi, rlo, rhi) for (rlo, rhi) in ranges)
            if not ok:
                continue

        weighted_sum += mae * cnt
        weight_total += cnt

    if weight_total <= 0:
        return None
    return weighted_sum / weight_total


def compute_primary_score(cfg: Dict[str, Any], metrics: Dict[str, Any]) -> Optional[float]:
    """
    Computes the scalar "primary objective" score based on cfg.autolab.objective.primary.

    Supported primary.name:
      - mae_overall
      - rmse_overall
      - r2_overall
      - mae_by_bins  (requires evaluator metrics with binned bins)
    """
    obj = cfg.get("autolab", {}).get("objective", {}).get("primary", {}) or {}
    name = str(obj.get("name", "mae_overall")).strip().lower()
    target = str(obj.get("target", "q_liquid")).strip()
    stage = _preferred_stage(cfg)

    if name in ("mae_overall", "mae"):
        return _score_overall(metrics, stage, target, "mae")
    if name in ("rmse_overall", "rmse"):
        return _score_overall(metrics, stage, target, "rmse")
    if name in ("r2_overall", "r2"):
        return _score_overall(metrics, stage, target, "r2")
    if name == "mae_by_bins":
        focus_bins = obj.get("focus_bins", obj.get("focus_ranges", [])) or []
        return _score_mae_by_bins(metrics, stage, target, focus_bins)

    # fallback to old behavior
    return None

# -----------------------------------------------------------------------------
# Autolab policy (rules-based "tuner")
# -----------------------------------------------------------------------------
ALLOWED_PATCH_TOP_LEVEL = {
    "training_defaults",
    "reduce_lr_on_plateau",
    "preprocessing",
    "targets",
    "models",
}


def _enabled(cfg: Dict[str, Any], stage: str) -> bool:
    return bool(cfg.get("models", {}).get(stage, {}).get("enabled", True))


def _primary_metric_name(cfg: Dict[str, Any]) -> str:
    return cfg.get("autolab", {}).get("stop", {}).get("metric", "mae_overall")


def select_primary_mae(cfg: Dict[str, Any], parsed: Dict[str, Dict[str, StageMetric]]) -> Optional[float]:
    """
    Map a "primary" metric to a concrete number to drive stop criteria.
    Default: MAE of q_liquid from the last enabled stage among [flow_correction, flow].
    """
    # Prefer correction if enabled and present
    if _enabled(cfg, "flow_correction") and "flow_correction" in parsed and "q_liquid" in parsed["flow_correction"]:
        return parsed["flow_correction"]["q_liquid"].mae
    if _enabled(cfg, "flow") and "flow" in parsed and "q_liquid" in parsed["flow"]:
        return parsed["flow"]["q_liquid"].mae
    return None


def propose_patch_rules(cfg: Dict[str, Any], parsed: Dict[str, Dict[str, StageMetric]], metrics_blob: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Phase-1: simple, deterministic rules.
    Produces a YAML patch dict (NOT full config), restricted to allowed keys.
    """
    patch: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Secondary-objective steering (uses plot-derived diagnostics in metrics.json)
    #
    # Primary objective still ranks / stops the run, but ANY secondary entry
    # that includes mode/value is treated as a constraint that can trigger actions.
    # ------------------------------------------------------------------
    sec_cfg = (cfg.get("autolab", {}) or {}).get("objective", {}) or {}
    secondary = list(sec_cfg.get("secondary", []) or [])

    # Resolve which stage/target we care about for flow diagnostics
    primary_cfg = (sec_cfg.get("primary", {}) or {})
    primary_target = str(primary_cfg.get("target", "q_liquid"))

    def _sec_threshold(name: str, target: str) -> Optional[Tuple[str, float]]:
        for it in secondary:
            if not isinstance(it, dict):
                continue
            if str(it.get("name", "")).strip().lower() != name.strip().lower():
                continue
            if str(it.get("target", "")).strip() != target:
                continue
            mode = str(it.get("mode", "")).strip().lower()
            val = it.get("value", None)
            if mode in ("max", "min") and val is not None:
                try:
                    return mode, float(val)
                except Exception:
                    return None
        return None

    def _violates(name: str, value: Optional[float], target: str) -> bool:
        thr = _sec_threshold(name, target)
        if thr is None or value is None:
            return False
        mode, bound = thr
        if mode == "max":
            return float(value) > bound
        if mode == "min":
            return float(value) < bound
        return False

    # Pull plot-derived diagnostics from evaluator metrics.json (if available)
    diag_stage = _preferred_stage(cfg)
    diag_block: Optional[Dict[str, Any]] = None
    if isinstance(metrics_blob, dict):
        diag_block = _get_stage_target_block(metrics_blob, diag_stage, primary_target)

    def _get_diag(key: str) -> Optional[float]:
        if not isinstance(diag_block, dict):
            return None
        v = diag_block.get(key)
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    abs_bias_focus = _get_diag("abs_bias_focus")
    p95_abs_error_focus = _get_diag("p95_abs_error_focus")
    corr_abs_resid_vs_true = _get_diag("corr_abs_resid_vs_true")
    calib_slope_abs_err = _get_diag("calib_slope_abs_err")
    calib_intercept_abs_err = _get_diag("calib_intercept_abs_err")

    # If we have meaningful tail error in the focus range, stabilize training.
    if _violates("p95_abs_error_focus", p95_abs_error_focus, primary_target):
        patch.setdefault("reduce_lr_on_plateau", {})["enabled"] = True
        patch.setdefault("training_defaults", {}).setdefault("gradient_clipping", {})["enabled"] = True

    # If we have a consistent bias in focus bins, give the model a bit more capacity/time.
    if _violates("abs_bias_focus", abs_bias_focus, primary_target):
        cur = cfg.get("models", {}).get("flow", {}).get("training", {}).get("epochs")
        if isinstance(cur, int):
            patch.setdefault("models", {}).setdefault("flow", {}).setdefault("training", {})["epochs"] = min(cur + 25, 400)

    # Heteroscedasticity indicator: error grows with y_true -> consider log1p target.
    if _violates("corr_abs_resid_vs_true", corr_abs_resid_vs_true, primary_target):
        ttype = (
            cfg.get("models", {})
            .get("flow", {})
            .get("data", {})
            .get("target_policy", {})
            .get(primary_target, {})
            .get("transform", {})
            .get("type", "none")
        )
        if ttype == "none":
            patch.setdefault("models", {}).setdefault("flow", {}).setdefault("data", {}).setdefault("target_policy", {})                 .setdefault(primary_target, {}).setdefault("transform", {})["type"] = "log1p"
            patch["models"]["flow"]["data"]["target_policy"][primary_target]["transform"].setdefault("epsilon", 1.0e-6)

    # Poor calibration (slope/intercept) -> increase capacity slightly (nodes_layers).
    if _violates("calib_slope_abs_err", calib_slope_abs_err, primary_target) or _violates("calib_intercept_abs_err", calib_intercept_abs_err, primary_target):
        nl = cfg.get("models", {}).get("flow", {}).get("model", {}).get("nodes_layers")
        if isinstance(nl, list) and all(isinstance(x, int) for x in nl):
            bumped = [min(int(x) + 16, 256) for x in nl]
            patch.setdefault("models", {}).setdefault("flow", {}).setdefault("model", {})["nodes_layers"] = bumped

    # Example rule 1: Wear looks weak (low R2) -> increase epochs or enable sample weighting
    wear_metric = None
    if "wear" in parsed and "hydraulic_wear" in parsed["wear"]:
        wear_metric = parsed["wear"]["hydraulic_wear"]

    if wear_metric and wear_metric.r2 is not None and wear_metric.r2 < 0.6:
        # Prefer bump wear epochs a bit (bounded)
        cur_epochs = cfg.get("models", {}).get("wear", {}).get("training", {}).get("epochs")
        if isinstance(cur_epochs, int) and cur_epochs < 400:
            new_epochs = min(cur_epochs + 50, 400)
            patch.setdefault("models", {}).setdefault("wear", {}).setdefault("training", {})["epochs"] = new_epochs
        else:
            # fallback: enable gradient clipping globally
            patch.setdefault("training_defaults", {}).setdefault("gradient_clipping", {})["enabled"] = True

    # Example rule 2: Correction worse than base flow by a lot -> add more regularization to correction
    flow_mae = None
    corr_mae = None
    if "flow" in parsed and "q_liquid" in parsed["flow"] and parsed["flow"]["q_liquid"].mae is not None:
        flow_mae = parsed["flow"]["q_liquid"].mae
    if "flow_correction" in parsed and "q_liquid" in parsed["flow_correction"] and parsed["flow_correction"]["q_liquid"].mae is not None:
        corr_mae = parsed["flow_correction"]["q_liquid"].mae

    if flow_mae is not None and corr_mae is not None and corr_mae > flow_mae * 1.2:
        cur_do = cfg.get("models", {}).get("flow_correction", {}).get("training", {}).get("regularization", {}).get("dropout", 0.0)
        try:
            cur_do_f = float(cur_do)
        except Exception:
            cur_do_f = 0.0
        new_do = min(cur_do_f + 0.05, 0.6)
        patch.setdefault("models", {}).setdefault("flow_correction", {}).setdefault("training", {}).setdefault("regularization", {})["dropout"] = new_do
        # and enable gradient clipping if not already
        patch.setdefault("training_defaults", {}).setdefault("gradient_clipping", {})["enabled"] = True

    # Example rule 3: If flow MAE is "high-ish", try log1p on q_liquid (only if not already)
    if flow_mae is not None and flow_mae > 1.2:
        ttype = (
            cfg.get("models", {})
            .get("flow", {})
            .get("data", {})
            .get("target_policy", {})
            .get("q_liquid", {})
            .get("transform", {})
            .get("type", "none")
        )
        if ttype == "none":
            patch.setdefault("models", {}).setdefault("flow", {}).setdefault("data", {}).setdefault("target_policy", {}) \
                .setdefault("q_liquid", {}).setdefault("transform", {})["type"] = "log1p"
            patch["models"]["flow"]["data"]["target_policy"]["q_liquid"]["transform"].setdefault("epsilon", 1.0e-6)

    # Restrict patch to allowed top-level keys
    patch = {k: v for k, v in patch.items() if k in ALLOWED_PATCH_TOP_LEVEL}
    return patch

# -----------------------------------------------------------------------------
# Basic validation (lightweight)
# -----------------------------------------------------------------------------
def validate_patch(patch: Dict[str, Any]) -> None:
    bad = [k for k in patch.keys() if k not in ALLOWED_PATCH_TOP_LEVEL]
    if bad:
        raise ValueError(f"Patch contains disallowed top-level keys: {bad}")



# -----------------------------------------------------------------------------
# Run history (summary.csv) + LLM tuner helpers
# -----------------------------------------------------------------------------

def _summary_path(run_root: Path) -> Path:
    return run_root / "summary.csv"


def _append_summary_row(run_root: Path, row: Dict[str, Any]) -> None:
    """Append a single iteration record to run_root/summary.csv."""
    import csv

    path = _summary_path(run_root)
    is_new = not path.exists()
    # Keep a stable, explicit column order
    fieldnames = [
        "timestamp",
        "run_id",
        "iteration",
        "primary_score",
        "primary_name",
        "primary_target",
        "flow_mae_overall",
        "flow_rmse_overall",
        "flow_r2_overall",
        "wear_mae_overall",
        "notes",
    ]
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            w.writeheader()
        out = {k: row.get(k, "") for k in fieldnames}
        w.writerow(out)


def _read_last_summary_rows(run_root: Path, n: int = 5) -> List[Dict[str, str]]:
    """Read the last N rows from summary.csv (if it exists)."""
    import csv
    path = _summary_path(run_root)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-n:]


def _get_stage_target_overall(metrics_blob: Dict[str, Any], stage: str, target: str) -> Dict[str, Any]:
    """
    Safely extract overall metrics for a given stage/target from metrics.json.

    Supports:
      A) evaluator schema: metrics["stages"][stage]["targets"][target]["overall"]
      B) legacy schema:   metrics["parsed"][stage][target]  (flat: mae/rmse/r2)
    """
    try:
        # A) evaluator schema
        overall = (metrics_blob.get("stages", {})
                   .get(stage, {})
                   .get("targets", {})
                   .get(target, {})
                   .get("overall", {}))
        if isinstance(overall, dict) and overall:
            return overall

        # B) legacy schema (flat)
        legacy = (metrics_blob.get("parsed", {})
                  .get(stage, {})
                  .get(target, {}))
        if isinstance(legacy, dict) and legacy:
            return legacy

        return {}
    except Exception:
        return {}


def _format_history_for_prompt(history_rows: List[Dict[str, str]]) -> str:
    if not history_rows:
        return "No prior iterations recorded."
    lines = []
    for r in history_rows:
        lines.append(
            f"- iter {r.get('iteration','?')}: primary={r.get('primary_score','')} (run_id={r.get('run_id','')})"
        )
    return "\n".join(lines)


def _extract_leaf_paths(d: Any, prefix: str = "") -> List[str]:
    """Return dotted paths for leaf values in a nested mapping."""
    paths: List[str] = []
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            paths.extend(_extract_leaf_paths(v, p))
    else:
        paths.append(prefix)
    return paths


def _get_by_path(root: Dict[str, Any], path: str) -> Any:
    cur: Any = root
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _validate_patch_constraints(
    patch: Dict[str, Any],
    *,
    allowed_paths: Optional[List[str]] = None,
    bounds: Optional[Dict[str, List[float]]] = None,
    max_changes: Optional[int] = None,
) -> None:
    """
    Validate a patch beyond top-level keys:
    - enforce allowed dotted paths (leaf-only) if provided
    - enforce numeric bounds if provided
    - enforce max number of leaf changes if provided
    """
    validate_patch(patch)

    # Normalize to leaf paths under the patch root
    leaf_paths = _extract_leaf_paths(patch)

    # Optional max_changes (count leaf assignments)
    if max_changes is not None and max_changes >= 0:
        if len(leaf_paths) > int(max_changes):
            raise ValueError(f"Patch changes too many fields ({len(leaf_paths)} > {max_changes}).")

    # Optional allowed_paths check
    if allowed_paths:
        # allow if a leaf path is exactly allowed OR is under an allowed prefix
        def ok(p: str) -> bool:
            for a in allowed_paths:
                if p == a or p.startswith(a + "."):
                    return True
            return False

        bad = [p for p in leaf_paths if not ok(p)]
        if bad:
            raise ValueError(f"Patch attempts to modify disallowed paths: {bad}")

    # Optional numeric bounds
    if bounds:
        for p in leaf_paths:
            if p not in bounds:
                continue
            v = _get_by_path(patch, p)
            if v is None:
                continue
            lo, hi = bounds[p]
            try:
                fv = float(v)
            except Exception:
                raise ValueError(f"Bounded field '{p}' must be numeric; got {type(v)}")
            if fv < float(lo) or fv > float(hi):
                raise ValueError(f"Field '{p}' out of bounds: {fv} not in [{lo}, {hi}]")



def _call_openai_chat(prompt: str, *, model: str, temperature: float = 0.2, max_tokens: int = 800) -> str:
    """
    Call OpenAI Chat Completions API.
    Requires OPENAI_API_KEY env var set on the machine running autolab.
    No external dependencies required (uses urllib).
    """
    import os
    import json as _json
    import urllib.request

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": int(max_tokens),
        "messages": [
            {"role": "system", "content": "You are a careful ML tuning assistant. Return ONLY valid YAML with no Markdown, no code fences, and no extra text. Output must start with: patch:"},
            {"role": "user", "content": prompt},
        ],
    }
    data = _json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    obj = _json.loads(raw)
    return obj["choices"][0]["message"]["content"]


def _build_llm_prompt(
    *,
    cfg_used: Dict[str, Any],
    metrics_blob: Dict[str, Any],
    primary_name: str,
    primary_target: str,
    focus_bins: Optional[List[float]],
    history_rows: List[Dict[str, str]],
    allowed_paths: List[str],
    bounds: Dict[str, List[float]],
    max_changes: int,
) -> str:
    """Create a compact, auditable prompt for the LLM tuner."""
    import yaml as _yaml

    # Keep the prompt small and decision-focused.
    flow_overall = _get_stage_target_overall(metrics_blob, "flow", primary_target)
    wear_overall = _get_stage_target_overall(metrics_blob, "wear", "hydraulic_wear")

    # Include a short metrics summary plus the objective spec.
    metrics_summary = {
        "primary_objective": {
            "name": primary_name,
            "target": primary_target,
            "focus_bins": focus_bins,
        },
        "overall": {
            "flow": flow_overall,
            "wear": wear_overall,
        },
    }

    # Provide only the relevant config slices to avoid accidental rewrites.
    cfg_slice = {
        "model": cfg_used.get("model"),
        "models": cfg_used.get("models"),
        "preprocessing": cfg_used.get("preprocessing"),
        "targets": cfg_used.get("targets"),
        "training_defaults": cfg_used.get("training_defaults"),
        "reduce_lr_on_plateau": cfg_used.get("reduce_lr_on_plateau"),
        "autolab_objective": (cfg_used.get("autolab", {}) or {}).get("objective"),
    }

    instructions = {
        "output_format": {
            "patch": "Return a YAML object with a top-level key 'patch'. The patch must be a partial config to merge into the existing YAML.",
            "reasoning": "Include a short 'rationale' string explaining why the changes should improve the primary objective.",
        },
        "constraints": {
            "max_changes": max_changes,
            "allowed_paths": allowed_paths,
            "bounds": bounds,
            "dont_touch": [
                "data paths",
                "client_name",
                "stages",
                "output directories",
                "unit system",
            ],
        },
    }

    prompt = f"""We are running an AutoLab loop for a multi-stage regression model.

Goal: improve the PRIMARY objective ONLY:
- {primary_name} for target '{primary_target}' (focus_bins={focus_bins})

Recent iteration history:
{_format_history_for_prompt(history_rows)}

Current iteration metrics summary (JSON-like YAML):
{_yaml.safe_dump(metrics_summary, sort_keys=False)}

Relevant config slice (YAML):
{_yaml.safe_dump(cfg_slice, sort_keys=False)}

Constraints (MUST follow):
{_yaml.safe_dump(instructions, sort_keys=False)}

Task:
1) Propose a small patch (<= {max_changes} leaf changes) that should improve the primary objective.
2) Return ONLY YAML in this exact structure:

patch:
  <top-level keys like preprocessing/model/training_defaults/...>:
    ...
rationale: "<1-3 sentences>"
"""
    return prompt



def _strip_markdown_fences(s: str) -> str:
    """
    Remove common Markdown code fences around YAML blocks, e.g.:

      ```yaml
      patch: ...
      ```

    Returns the inner text if fenced; otherwise returns s unchanged.
    """
    if not isinstance(s, str):
        return ""
    ss = s.strip()
    # Fast path: must start with ```
    if not ss.startswith("```"):
        return ss
    # Remove leading fence line
    lines = ss.splitlines()
    if not lines:
        return ss
    if lines[0].startswith("```"):
        lines = lines[1:]
    # Remove trailing fence line if present
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_llm_output_to_patch(raw_text: str) -> Tuple[Dict[str, Any], str]:
    """Parse the LLM output (YAML) into (patch_dict, rationale)."""
    import yaml as _yaml
    cleaned = _strip_markdown_fences(raw_text)
    obj = _yaml.safe_load(cleaned) or {}
    if not isinstance(obj, dict):
        raise ValueError("LLM output did not parse into a YAML mapping.")
    patch = obj.get("patch") or {}
    if not isinstance(patch, dict):
        raise ValueError("LLM output 'patch' must be a mapping.")
    rationale = obj.get("rationale") or ""
    if not isinstance(rationale, str):
        rationale = str(rationale)
    return patch, rationale


# -----------------------------------------------------------------------------
# Main autolab loop
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="Path to training config YAML")
    ap.add_argument("--train-script", default=DEFAULT_TRAIN_SCRIPT, help="Path to train pipeline script")
    ap.add_argument("--eval-script", default=DEFAULT_EVAL_SCRIPT, help="Path to evaluation script")
    ap.add_argument("--workdir", default=".", help="Working directory (repo root)")
    ap.add_argument("--max-iters", type=int, default=None, help="Override autolab.max_iters")
    ap.add_argument("--no-tune", action="store_true", help="Run exactly one iteration (no patching)")
    args = ap.parse_args()

    repo_root = Path(args.workdir).resolve()
    cfg_path = (repo_root / args.config).resolve()
    train_script = (repo_root / args.train_script).resolve()
    eval_script = (repo_root / args.eval_script).resolve()

    cfg = load_yaml(cfg_path)

    autolab_cfg = cfg.get("autolab", {})
    artifacts_cfg = autolab_cfg.get("artifacts", {})
    SAVE_ARTIFACTS = artifacts_cfg.get("save_run_artifacts", False)

    autolab_cfg = cfg.get("autolab", {}) or {}

    # ------------------------------------------------------------------
    # Experiment layout (cleaner than dumping everything into phlux_lab/runs)
    #
    # Writes to:
    #   phlux_lab/models/<client>/experiments/<exp_id>/
    #     manifest.yaml
    #     summary.csv
    #     iterations/it01/...  (minimal artifacts by default)
    #
    # The legacy autolab.artifacts.run_root is still supported as an override,
    # but the default is now the client model folder for easier navigation.
    # ------------------------------------------------------------------
    client_name = _resolve_client_name(cfg) or "Client"
    models_root = repo_root / "phlux_lab" / "models" / client_name
    models_root.mkdir(parents=True, exist_ok=True)

    artifacts_cfg = (autolab_cfg.get("artifacts", {}) or {})
    legacy_run_root = artifacts_cfg.get("run_root", None)

    tag = str(artifacts_cfg.get("experiment_tag", "") or "").strip()
    ts_exp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"{ts_exp}" + (f"_{tag}" if tag else "")
    exp_root = (repo_root / legacy_run_root).resolve() / exp_id if legacy_run_root else (models_root / "experiments" / exp_id)
    exp_root.mkdir(parents=True, exist_ok=True)

    iterations_root = exp_root / "iterations"
    iterations_root.mkdir(parents=True, exist_ok=True)

    # Write an experiment manifest for traceability
    manifest = {
        "experiment_id": exp_id,
        "created_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "client_name": client_name,
        "base_config": str(cfg_path),
        "train_script": str(train_script),
        "eval_script": str(eval_script),
        "tuner": {
            "enabled": bool(((cfg.get("autolab", {}) or {}).get("tuner", {}) or {}).get("enabled", False)),
            "mode": str(((cfg.get("autolab", {}) or {}).get("tuner", {}) or {}).get("mode", "off")),
        },
    }
    save_yaml(manifest, exp_root / "manifest.yaml")

    max_iters = args.max_iters if args.max_iters is not None else int(autolab_cfg.get("max_iters", 1))
    if args.no_tune:
        max_iters = 1

    stop_cfg = autolab_cfg.get("stop", {}) or {}
    patience = int(stop_cfg.get("patience", 2))
    min_impr = float(stop_cfg.get("min_improvement", 0.0))

    tuner_cfg = (autolab_cfg.get("tuner", {}) or {})
    tuner_enabled = bool(tuner_cfg.get("enabled", False))
    tuner_mode = str(tuner_cfg.get("mode", "off")).lower()  # off | rules | llm
    tuner_history_n = int(tuner_cfg.get("history_n", 5))
    tuner_max_changes = int(tuner_cfg.get("max_changes", 6))
    tuner_allowed_paths = list(tuner_cfg.get("allowed_paths", []) or [])
    tuner_bounds = dict(tuner_cfg.get("bounds", {}) or {})
    # OpenAI settings (only used if tuner_mode == "llm")
    tuner_openai_model = str(tuner_cfg.get("openai_model", "gpt-4.1-mini"))
    tuner_temperature = float(tuner_cfg.get("temperature", 0.2))
    tuner_max_tokens = int(tuner_cfg.get("max_tokens", 800))

    best_val: Optional[float] = None
    no_impr_count = 0

    cur_cfg_path = cfg_path

    print("\n=== AUTOLAB RUN ===")
    print(f"Repo root : {repo_root}")
    print(f"Experiment: {exp_root}")
    print(f"Train     : {train_script}")
    print(f"Eval      : {eval_script}")
    print(f"Iters     : {max_iters}")

    for it in range(1, max_iters + 1):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_id = f"it{it:02d}"
        run_dir = iterations_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Recent history (from previous iterations) for prompt context
        prior_history = _read_last_summary_rows(exp_root, n=tuner_history_n)

        # Load iteration config (no folder yet)
        iter_cfg = load_yaml(cur_cfg_path)

        # 1) TRAIN
        print(f"\n--- Iteration {it}/{max_iters}: TRAIN ({run_id}) ---")
        rc_t, out_t, err_t = run_cmd(
            [sys.executable, str(train_script), str(cur_cfg_path)],
            cwd=repo_root
        )

        # Save artifacts ONLY if enabled or on failure
        if SAVE_ARTIFACTS or rc_t != 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            save_yaml(iter_cfg, run_dir / "config_used.yaml")
            (run_dir / "train_stdout.txt").write_text(out_t or "", encoding="utf-8", errors="replace")
            (run_dir / "train_stderr.txt").write_text(err_t or "", encoding="utf-8", errors="replace")

        if rc_t != 0:
            print("❌ Training failed. See logs:")
            print(f"  {run_dir / 'train_stderr.txt'}")
            sys.exit(rc_t)

        # 2) EVAL
        print(f"\n--- Iteration {it}/{max_iters}: EVAL ({run_id}) ---")
        rc_e, out_e, err_e = run_cmd(
            [sys.executable, str(eval_script), "--config", str(cur_cfg_path)],
            cwd=repo_root
        )

        if SAVE_ARTIFACTS or rc_e != 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "eval_stdout.txt").write_text(out_e or "", encoding="utf-8", errors="replace")
            (run_dir / "eval_stderr.txt").write_text(err_e or "", encoding="utf-8", errors="replace")

        if rc_e != 0:
            print("❌ Evaluation failed. See logs:")
            print(f"  {run_dir / 'eval_stderr.txt'}")
            sys.exit(rc_e)

        # Parse evaluation output
        parsed = parse_eval_output(out_e)

        # ALWAYS persist metrics (this is your optimization signal)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Prefer evaluator-produced metrics.json (contains binned metrics, edges, etc.)
        eval_metrics = _load_eval_metrics_json(repo_root, iter_cfg)
        
        if isinstance(eval_metrics, dict):
            metrics_blob = eval_metrics
            # Ensure run_id/iteration are present for traceability
            metrics_blob.setdefault("run_id", run_id)
            metrics_blob.setdefault("iteration", it)
        else:
            # Fallback: legacy parsed-from-stdout metrics only
            metrics_blob = {
                "run_id": run_id,
                "iteration": it,
                "parsed": {
                    stg: {
                        tgt: {"mae": m.mae, "rmse": m.rmse, "r2": m.r2}
                        for tgt, m in tgts.items()
                    }
                    for stg, tgts in parsed.items()
                },
            }
        # Primary objective score (preferred), with fallback to legacy MAE selection
        primary = compute_primary_score(iter_cfg, metrics_blob)
        if primary is None:
            primary = select_primary_mae(iter_cfg, parsed)

        # Append a compact row to summary.csv (for history + LLM context)
        primary_cfg = (iter_cfg.get("autolab", {}) or {}).get("objective", {}).get("primary", {}) or {}
        primary_name = str(primary_cfg.get("name", "mae_overall"))
        primary_target = str(primary_cfg.get("target", "q_liquid"))
        flow_overall = _get_stage_target_overall(metrics_blob, "flow", primary_target)
        wear_overall = _get_stage_target_overall(metrics_blob, "wear", "hydraulic_wear")

        _append_summary_row(
            exp_root,
            {
                "timestamp": ts,
                "run_id": run_id,
                "iteration": it,
                "primary_score": primary,
                "primary_name": primary_name,
                "primary_target": primary_target,
                "flow_mae_overall": flow_overall.get("mae", ""),
                "flow_rmse_overall": flow_overall.get("rmse", ""),
                "flow_r2_overall": flow_overall.get("r2", ""),
                "wear_mae_overall": wear_overall.get("mae", ""),
                "notes": "",
            },
        )
        
        print(f"Primary objective: {primary}")

        # Persist metrics for this iteration (single source of truth)
        if isinstance(metrics_blob, dict):
            metrics_blob["primary_score"] = primary
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics_blob, indent=2),
            encoding="utf-8"
        )

        # 3) STOP CHECK (minimize MAE)
        if primary is not None:
            if best_val is None or primary < best_val - min_impr:
                best_val = primary
                no_impr_count = 0
            else:
                no_impr_count += 1

            if no_impr_count >= patience:
                print(f"\n✅ Stop: no improvement >= {patience} iterations (best={best_val}).")
                break

        if args.no_tune:
            print("\n✅ Single-iteration run complete (--no-tune).")
            break

        
        # 4) PROPOSE PATCH (rules or LLM)
        patch: Dict[str, Any] = {}
        rationale: str = ""

        if tuner_enabled and tuner_mode == "llm":
            try:
                # Build an auditable prompt
                primary_cfg = (iter_cfg.get("autolab", {}) or {}).get("objective", {}).get("primary", {}) or {}
                primary_name = str(primary_cfg.get("name", "mae_overall"))
                primary_target = str(primary_cfg.get("target", "q_liquid"))
                focus_bins = primary_cfg.get("focus_bins", None)

                prompt = _build_llm_prompt(
                    cfg_used=iter_cfg,
                    metrics_blob=metrics_blob,
                    primary_name=primary_name,
                    primary_target=primary_target,
                    focus_bins=focus_bins,
                    history_rows=prior_history,
                    allowed_paths=tuner_allowed_paths,
                    bounds=tuner_bounds,
                    max_changes=tuner_max_changes,
                )

                if SAVE_ARTIFACTS:
                    (run_dir / "llm_prompt.txt").write_text(prompt, encoding="utf-8")

                raw = _call_openai_chat(
                    prompt,
                    model=tuner_openai_model,
                    temperature=tuner_temperature,
                    max_tokens=tuner_max_tokens,
                )
                if SAVE_ARTIFACTS:
                    (run_dir / "llm_response.txt").write_text(raw, encoding="utf-8", errors="replace")

                patch, rationale = _parse_llm_output_to_patch(raw)

                # Validate LLM patch against constraints
                _validate_patch_constraints(
                    patch,
                    allowed_paths=tuner_allowed_paths or None,
                    bounds=tuner_bounds or None,
                    max_changes=tuner_max_changes,
                )

            except Exception as e:
                # Record the failure and fall back to rules if enabled, otherwise no patch.
                (run_dir / "llm_error.txt").write_text(str(e), encoding="utf-8")
                if str(tuner_cfg.get("fallback", "rules")).lower() == "rules":
                    patch = propose_patch_rules(iter_cfg, parsed, metrics_blob=metrics_blob)
                    rationale = f"LLM failed; used rules fallback. Error: {e}"
                else:
                    patch = {}
                    rationale = f"LLM failed; no patch applied. Error: {e}"

        elif tuner_enabled and tuner_mode == "rules":
            patch = propose_patch_rules(iter_cfg, parsed, metrics_blob=metrics_blob)
            rationale = "Rules-based tuner."

        else:
            # Default behavior: keep legacy rules proposer (safe), but do not run if --no-tune
            patch = propose_patch_rules(iter_cfg, parsed, metrics_blob=metrics_blob)
            rationale = "Legacy rules proposer (tuner disabled)."

        # Persist patch + rationale for audit
        save_yaml({"patch": patch, "rationale": rationale}, run_dir / "tuner_decision.yaml")

        # Save a raw patch.yaml only when artifacts are enabled (keeps iteration folders clean)
        if SAVE_ARTIFACTS:
            save_yaml(patch, run_dir / "patch.yaml")

        if not patch:
            print("\n✅ No patch proposed. Stopping.")
            break

        validate_patch(patch)

        next_cfg = merge_cfg(iter_cfg, patch)
        next_cfg_path = run_dir / "config_next.yaml"
        save_yaml(next_cfg, next_cfg_path)

        if SAVE_ARTIFACTS:
            print(f"\n🧪 Patch saved        : {run_dir / 'patch.yaml'}")
        else:
            print("\n🧪 Patch proposed     : (not written; save_run_artifacts=false)")
        print(f"🧠 Tuner decision     : {run_dir / 'tuner_decision.yaml'}")
        if tuner_enabled and tuner_mode == "llm":
            print(f"📝 LLM prompt/response : {run_dir / 'llm_prompt.txt'} / {run_dir / 'llm_response.txt'}")
        print(f"➡️  Next config saved  : {next_cfg_path}")
        # Move to next iteration config
        cur_cfg_path = next_cfg_path

    print("\n=== AUTOLAB COMPLETE ===")


if __name__ == "__main__":
    main()