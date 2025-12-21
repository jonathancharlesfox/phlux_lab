from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml

from .physics import (
    PumpCurve,
    ValveCvCurve,
    SystemCurve,
    solve_operating_point,
    head_to_delta_p_pa,
)
from phlux_lab.utils.units import convert_user_to_internal_SI, convert_internal_back_to_user


class CentrifugalPumpDatasetGenerator:
    """
    Canonical approach:
      - Internal computations in SI (Pa, K, m3/s, W, ...)
      - CSV column names are unitless (no __unit suffix)
      - Units stored in a schema sidecar YAML (and in synthgen config under `data`)
      - Supports train/test split via config key: test_split (percent)
    """

    @classmethod
    def from_yaml(cls, path: str) -> "CentrifugalPumpDatasetGenerator":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg["_config_dir"] = os.path.dirname(os.path.abspath(path))
        return cls(cfg)

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.rng = np.random.default_rng(int(cfg.get("random_seed", 42)))

        self.op_cfg = cfg.get("operating_ranges", {}) or {}
        self.data_cfg = cfg.get("data", []) or []

        # ---- Required operating ranges (with alias support) ----
        self._get_range("suction_pressure_range", "suction_p_range", "psuc_range")
        self._get_range("temperature_range", "temperature_C_range", "temp_range")
        self._get_range("speed_range", "speed_rpm_range", "rpm_range")
        self._get_range("density_range", "fluid_density_range")
        self._get_range("viscosity_range", "fluid_viscosity_range")
        self._get_range("k_friction_range", "k_range")

        # ---- Pump config (support new nesting OR legacy top-level) ----
        pump_cfg = (self.op_cfg.get("pump") or cfg.get("pump") or {})
        if not pump_cfg:
            raise KeyError(
                "Missing pump block. Expected operating_ranges.pump (preferred) or top-level pump."
            )

        base_speed_rpm = float(pump_cfg["base_speed_rpm"])
        curve_points_user: List[Tuple[float, float]] = pump_cfg["curve_points"]
        curve_points_user = [(float(q), float(h)) for q, h in curve_points_user]
        curve_points_user.sort(key=lambda t: t[0])

        # PumpCurve expects internal flow units (m3/s).
        q_unit = self._get_feature_unit("q_liquid", default="m3/h")
        curve_points_si: List[Tuple[float, float]] = []
        for q_user, head_m in curve_points_user:
            q_m3_s = float(convert_user_to_internal_SI(float(q_user), q_unit))  # -> m3/s
            curve_points_si.append((q_m3_s, float(head_m)))

        self.pump = PumpCurve(base_speed_rpm=base_speed_rpm, curve_points=curve_points_si)

        # ---- Valve config (support new nesting OR legacy top-level) ----
        self.valve_cv_curve: Optional[ValveCvCurve] = None
        valve_cfg = (self.op_cfg.get("valve") or cfg.get("valve") or {})
        self.valve_enabled = bool(valve_cfg.get("enabled", False))

        if self.valve_enabled:
            if "cv_curve_points" not in valve_cfg:
                raise KeyError("Missing valve.cv_curve_points")
            if "valve_opening_range" not in valve_cfg:
                raise KeyError("Missing valve.valve_opening_range")
            if "sink_pressure_range" not in valve_cfg:
                raise KeyError("Missing valve.sink_pressure_range")

            cv_flow_unit = str(valve_cfg.get("cv_flow_unit", "m3/h"))
            cv_dp_unit = str(valve_cfg.get("cv_dp_unit", "bar"))

            cv_points_internal: List[Tuple[float, float]] = []
            for opening_user, cv_user in valve_cfg["cv_curve_points"]:
                opening = float(opening_user)
                cv_internal = self._convert_cv_to_internal(float(cv_user), cv_flow_unit, cv_dp_unit)
                cv_points_internal.append((opening, cv_internal))

            cv_points_internal.sort(key=lambda t: t[0])
            self.valve_cv_curve = ValveCvCurve(cv_curve_points=cv_points_internal)

        eff_model_cfg = self.op_cfg.get("efficiency_model", {}) or {}
        self.base_efficiency = float(eff_model_cfg.get("base_efficiency", 0.75))

    # ============================================================
    # Helpers
    # ============================================================

    def _get_range(self, primary: str, *aliases: str) -> Tuple[float, float]:
        """Read a [lo, hi] range from operating_ranges with alias support."""
        for k in (primary, *aliases):
            if k in self.op_cfg:
                lo, hi = self.op_cfg[k]
                return float(lo), float(hi)
        raise KeyError(f"Missing operating_ranges.{primary} (tried aliases: {aliases})")

    def _get_feature_unit(self, name: str, default: str | None = None) -> str:
        """
        Look up unit for a feature in config.data.
        If not found, return default (or raise).
        """
        for d in self.data_cfg:
            if d.get("name") == name:
                u = d.get("unit")
                if u is None:
                    break
                return str(u)
        if default is not None:
            return default
        raise KeyError(f"Unit for '{name}' not found in config.data")

    def _convert_cv_to_internal(self, cv_user: float, flow_unit: str, dp_unit: str) -> float:
        """
        ValveCvCurve expects Cv_internal with units: (m3/s)/sqrt(Pa)

        If user specifies Cv in (flow_unit)/sqrt(dp_unit), then:
          Cv_internal = Cv_user * (flow_unit->m3/s) / sqrt(dp_unit->Pa)
        """
        flow_scale = float(convert_user_to_internal_SI(1.0, flow_unit))  # m3/s per flow_unit
        dp_scale = float(convert_user_to_internal_SI(1.0, dp_unit))      # Pa per dp_unit
        return float(cv_user) * flow_scale / float(np.sqrt(dp_scale))

    def _train_test_split(
        self, df: pd.DataFrame, test_pct: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe into train/test using a percentage.
        Deterministic based on config random_seed.
        """
        test_pct = float(test_pct)

        if test_pct <= 0.0:
            return df.reset_index(drop=True), df.iloc[0:0].copy()

        if test_pct >= 100.0:
            return df.iloc[0:0].copy(), df.reset_index(drop=True)

        n = len(df)
        if n == 0:
            return df.copy(), df.copy()

        n_test = int(round(n * test_pct / 100.0))
        n_test = max(1, min(n - 1, n_test))  # ensure both splits non-empty

        rng = np.random.default_rng(int(self.cfg.get("random_seed", 42)))
        idx = np.arange(n)
        rng.shuffle(idx)

        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        return df_train, df_test

    # ============================================================
    # Samplers (user ranges -> internal SI)
    # ============================================================

    def _sample_suction_pressure_pa(self) -> float:
        lo, hi = self._get_range("suction_pressure_range", "suction_p_range", "psuc_range")
        unit = self._get_feature_unit("suction_pressure", default="bar")
        v_user = float(self.rng.uniform(lo, hi))
        return float(convert_user_to_internal_SI(v_user, unit))

    def _sample_temperature_K(self) -> float:
        lo, hi = self._get_range("temperature_range", "temperature_C_range", "temp_range")
        unit = self._get_feature_unit("temperature", default="degC")
        v_user = float(self.rng.uniform(lo, hi))
        return float(convert_user_to_internal_SI(v_user, unit))

    def _sample_speed_rpm(self) -> float:
        lo, hi = self._get_range("speed_range", "speed_rpm_range", "rpm_range")
        return float(self.rng.uniform(lo, hi))

    def _sample_density_kg_m3(self) -> float:
        lo, hi = self._get_range("density_range", "fluid_density_range")
        return float(self.rng.uniform(lo, hi))

    def _sample_viscosity_pa_s(self) -> float:
        lo, hi = self._get_range("viscosity_range", "fluid_viscosity_range")
        unit = self._get_feature_unit("fluid_viscosity", default="cP")
        v_user = float(self.rng.uniform(lo, hi))
        return float(convert_user_to_internal_SI(v_user, unit))  # -> Pa*s

    def _sample_k_friction(self) -> float:
        lo, hi = self._get_range("k_friction_range", "k_range")
        return float(self.rng.uniform(lo, hi))

    def _sample_hydraulic_wear_frac(self) -> float:
        wear_cfg = self.op_cfg.get("hydraulic_wear", {}) or {}
        if not wear_cfg.get("enabled", False):
            return 0.0
        lo, hi = wear_cfg.get("range", [0.0, 0.0])
        return float(self.rng.uniform(float(lo), float(hi)))

    def _sample_valve_opening_frac(self) -> float:
        valve_cfg = (self.op_cfg.get("valve") or self.cfg.get("valve") or {})
        lo, hi = valve_cfg["valve_opening_range"]
        return float(self.rng.uniform(float(lo), float(hi)))

    def _sample_sink_pressure_pa(self) -> float:
        valve_cfg = (self.op_cfg.get("valve") or self.cfg.get("valve") or {})
        lo, hi = valve_cfg["sink_pressure_range"]
        unit = self._get_feature_unit("sink_pressure", default="bar")
        v_user = float(self.rng.uniform(lo, hi))
        return float(convert_user_to_internal_SI(v_user, unit))

    # ============================================================
    # Core generation
    # ============================================================

    def generate_sample(self) -> Dict[str, float]:
        suction_p_pa = self._sample_suction_pressure_pa()
        temp_K = self._sample_temperature_K()
        speed_rpm = self._sample_speed_rpm()
        rho = self._sample_density_kg_m3()
        mu_pa_s = self._sample_viscosity_pa_s()
        k_fric = self._sample_k_friction()
        hydraulic_wear = self._sample_hydraulic_wear_frac()

        valve_opening = np.nan
        sink_p_pa = None
        valve_curve = None

        if self.valve_enabled and self.valve_cv_curve is not None:
            valve_opening = self._sample_valve_opening_frac()
            sink_p_pa = self._sample_sink_pressure_pa()
            valve_curve = self.valve_cv_curve

        # Viscosity not used in the hydraulic solve (SystemCurve signature),
        # but we keep it as an ML feature.
        _ = mu_pa_s

        system = SystemCurve(
            h_static_m=0.0,  # static head removed from config
            k_friction=k_fric,
            valve=valve_curve,
            valve_opening=float(valve_opening) if np.isfinite(valve_opening) else 1.0,
            density_kg_m3=rho,
        )

        q_min = self.pump.curve_points[0][0]
        q_max = self.pump.curve_points[-1][0]

        prev_wear = float(getattr(self.pump, "hydraulic_wear_frac", 0.0))
        self.pump.hydraulic_wear_frac = float(hydraulic_wear)

        q_m3_s = solve_operating_point(
            pump=self.pump,
            system=system,
            speed_rpm=speed_rpm,
            q_min_m3_s=q_min,
            q_max_m3_s=q_max,
        )

        head_m = self.pump.head_at(q_m3_s, speed_rpm)
        dp_pump_pa = head_to_delta_p_pa(head_m, rho)
        discharge_p_pa = suction_p_pa + dp_pump_pa

        eff = max(self.base_efficiency, 1e-6)
        hydraulic_power_W = rho * 9.80665 * q_m3_s * head_m
        pump_power_W = hydraulic_power_W / eff

        self.pump.hydraulic_wear_frac = prev_wear

        return {
            "suction_pressure": float(suction_p_pa),
            "discharge_pressure": float(discharge_p_pa),
            "delta_pressure": float(dp_pump_pa),
            "speed": float(speed_rpm),
            "temperature": float(temp_K),
            "fluid_density": float(rho),
            "fluid_viscosity": float(mu_pa_s),
            "q_liquid": float(q_m3_s),
            "hydraulic_wear": float(hydraulic_wear),
            "pump_power": float(pump_power_W),
            "valve_opening": float(valve_opening) if np.isfinite(valve_opening) else float(np.nan),
            "sink_pressure": float(sink_p_pa) if sink_p_pa is not None else float(np.nan),
        }

    def generate_dataset(self, n_samples: int) -> pd.DataFrame:
        rows = [self.generate_sample() for _ in range(int(n_samples))]
        return pd.DataFrame(rows)

    # ============================================================
    # Export
    # ============================================================

    def _build_schema(self) -> Dict[str, Any]:
        cols = []
        for d in self.data_cfg:
            cols.append({"name": d["name"], "user_unit": d["unit"]})

        return {
            "project_name": self.cfg.get("project_name"),
            "client_name": self.cfg.get("client_name"),
            "equipment_type": self.cfg.get("equipment_type"),
            "canonical_internal_system": "SI",
            "columns": cols,
        }

    def export_csv_and_schema(self) -> Tuple[str, str]:
        """
        Writes:
          - train CSV
          - test CSV
          - schema YAML (shared)

        Returns:
          (train_csv_path, test_csv_path)
        """
        out_cfg = self.cfg.get("output", {}) or {}
        out_dir = out_cfg.get("directory", "data/synthetic")
        os.makedirs(out_dir, exist_ok=True)

        # Require a split placeholder for proper naming.
        # If user didn’t include {split}, we’ll auto-upgrade.
        fname_tmpl = str(out_cfg.get("filename_template", "{client_name}_dataset.csv"))
        if "{split}" not in fname_tmpl:
            root, ext = os.path.splitext(fname_tmpl)
            ext = ext or ".csv"
            fname_tmpl = f"{root}_{{split}}{ext}"

        overwrite = bool(out_cfg.get("overwrite", False))

        n_samples = int(self.cfg.get("n_samples", 1000))
        test_pct = float(self.cfg.get("test_split", 0.0))

        df_internal = self.generate_dataset(n_samples)
        df_train_int, df_test_int = self._train_test_split(df_internal, test_pct)

        def _convert_df_to_user_units(df_int: pd.DataFrame) -> pd.DataFrame:
            df_export = df_int.copy()
            for d in self.data_cfg:
                name = d["name"]
                unit = d["unit"]
                if name not in df_export.columns:
                    continue
                df_export[name] = df_export[name].apply(
                    lambda v: convert_internal_back_to_user(float(v), unit) if pd.notna(v) else v
                )
            return df_export

        # --- write train ---
        train_name = fname_tmpl.format(split="train", **self.cfg)
        train_path = os.path.join(out_dir, train_name)
        if os.path.exists(train_path) and not overwrite:
            raise FileExistsError(f"Output exists and overwrite=false: {train_path}")
        _convert_df_to_user_units(df_train_int).to_csv(train_path, index=False)

        # --- write test ---
        test_name = fname_tmpl.format(split="test", **self.cfg)
        test_path = os.path.join(out_dir, test_name)
        if os.path.exists(test_path) and not overwrite:
            raise FileExistsError(f"Output exists and overwrite=false: {test_path}")
        _convert_df_to_user_units(df_test_int).to_csv(test_path, index=False)

        # --- write schema (shared) ---
        schema_name = fname_tmpl.format(split="schema", **self.cfg)
        schema_root, _ = os.path.splitext(schema_name)
        schema_path = os.path.join(out_dir, f"{schema_root}.yaml")
        with open(schema_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._build_schema(), f, sort_keys=False)

        return train_path, test_path
