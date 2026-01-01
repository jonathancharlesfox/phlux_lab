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
    Synthetic dataset generator for a centrifugal pump discharging through a valve to a sink pressure.

    Internal computations are canonical SI:
      - pressures: Pa
      - flow: m3/s
      - temp: K
      - density: kg/m3
      - viscosity: Pa*s
      - head: m
      - power: W

    K-Spice parity (for pump power):
      - Use pump efficiency curve (spline) when provided.
      - Compute shaft power like K-Spice "By efficiency":
            P_shaft = rho_ref * g * Q * H / eta(Q)
        where rho_ref is a configured reference density (typically 1000 kg/m3).
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

        # Required ranges
        self._require_range("suction_pressure_range")
        self._require_range("temperature_range")
        self._require_range("speed_range")
        self._require_range("density_range")
        self._require_range("viscosity_range")
        self._require_range("k_friction_range")

        # -------------------------
        # Pump (head + efficiency)
        # -------------------------
        pump_cfg = (self.op_cfg.get("pump") or cfg.get("pump") or {})
        if not pump_cfg:
            raise KeyError("Missing pump block. Expected operating_ranges.pump or top-level pump.")

        base_speed_rpm = float(pump_cfg["base_speed_rpm"])

        q_unit = self._get_feature_unit("q_liquid", default="m3/h")

        # Head curve points (Q in user units -> SI m3/s)
        head_points_user: List[Tuple[float, float]] = [(float(q), float(h)) for q, h in pump_cfg["curve_points"]]
        head_points_user.sort(key=lambda t: t[0])
        head_points_si: List[Tuple[float, float]] = [
            (float(convert_user_to_internal_SI(q_user, q_unit)), float(head_m))
            for q_user, head_m in head_points_user
        ]

        # Efficiency curve points (optional). Values in percent in YAML.
        eff_points_si: Optional[List[Tuple[float, float]]] = None
        if pump_cfg.get("efficiency_curve_points") is not None:
            eff_points_user: List[Tuple[float, float]] = [
                (float(q), float(eff_pct)) for q, eff_pct in pump_cfg["efficiency_curve_points"]
            ]
            eff_points_user.sort(key=lambda t: t[0])
            eff_points_si = [
                (float(convert_user_to_internal_SI(q_user, q_unit)), float(eff_pct) / 100.0)
                for q_user, eff_pct in eff_points_user
            ]

        # K-Spice reference density for power curve (kg/m3)
        rho_ref = float(pump_cfg.get("power_ref_density_kg_m3", 1000.0))

        self.pump = PumpCurve(
            base_speed_rpm=base_speed_rpm,
            head_points=head_points_si,
            eff_points=eff_points_si,
            power_ref_density_kg_m3=rho_ref,
        )

        # -------------------------
        # Valve
        # -------------------------
        valve_cfg = (self.op_cfg.get("valve") or cfg.get("valve") or {})
        self.valve_enabled = bool(valve_cfg.get("enabled", False))
        self.valve_curve: Optional[ValveCvCurve] = None

        if self.valve_enabled:
            if "cv_curve_points" not in valve_cfg:
                raise KeyError("Missing operating_ranges.valve.cv_curve_points")
            if "valve_opening_range" not in valve_cfg:
                raise KeyError("Missing operating_ranges.valve.valve_opening_range")
            if "sink_pressure_range" not in valve_cfg:
                raise KeyError("Missing operating_ranges.valve.sink_pressure_range")

            cv_flow_unit = str(valve_cfg.get("cv_flow_unit", "m3/h"))
            cv_dp_unit = str(valve_cfg.get("cv_dp_unit", "bar"))

            k_points_internal: List[Tuple[float, float]] = []
            for opening_user, k_user in valve_cfg["cv_curve_points"]:
                opening = float(opening_user)
                k_internal = self._convert_k_to_internal(float(k_user), cv_flow_unit, cv_dp_unit)
                k_points_internal.append((opening, k_internal))
            k_points_internal.sort(key=lambda t: t[0])

            valve_diam_m = float(self._valve_size_internal())
            self.valve_curve = ValveCvCurve(k_curve_points=k_points_internal, nominal_diameter_m=valve_diam_m)

    # ============================================================
    # Config helpers
    # ============================================================

    def _require_range(self, key: str) -> None:
        if key not in self.op_cfg:
            raise KeyError(f"Missing operating_ranges.{key}")

    def _get_feature_unit(self, name: str, default: Optional[str] = None) -> str:
        for d in self.data_cfg:
            if d.get("name") == name:
                u = d.get("unit")
                if u is not None:
                    return str(u)
        if default is None:
            raise KeyError(f"Unit for feature '{name}' not found in config.data")
        return default

    def _convert_k_to_internal(self, k_user: float, flow_unit: str, dp_unit: str) -> float:
        """Convert user K (Q/sqrt(dP)) into internal K [(m3/s)/sqrt(Pa)]."""
        flow_scale = float(convert_user_to_internal_SI(1.0, flow_unit))  # -> m3/s
        dp_scale = float(convert_user_to_internal_SI(1.0, dp_unit))      # -> Pa
        return float(k_user) * flow_scale / float(np.sqrt(dp_scale))

    # ============================================================
    # Samplers (internal SI)
    # ============================================================

    def _sample_uniform_user(self, key: str) -> float:
        lo, hi = self.op_cfg[key]
        return float(self.rng.uniform(float(lo), float(hi)))

    def _sample_suction_pressure_pa(self) -> float:
        unit = self._get_feature_unit("suction_pressure", default="bar")
        return float(convert_user_to_internal_SI(self._sample_uniform_user("suction_pressure_range"), unit))

    def _sample_sink_pressure_pa(self) -> float:
        unit = self._get_feature_unit("sink_pressure", default="bar")
        valve_cfg = (self.op_cfg.get("valve") or self.cfg.get("valve") or {})
        lo, hi = valve_cfg["sink_pressure_range"]
        v_user = float(self.rng.uniform(float(lo), float(hi)))
        return float(convert_user_to_internal_SI(v_user, unit))

    def _sample_temperature_K(self) -> float:
        unit = self._get_feature_unit("temperature", default="degC")
        return float(convert_user_to_internal_SI(self._sample_uniform_user("temperature_range"), unit))

    def _sample_speed_rpm(self) -> float:
        lo, hi = self.op_cfg["speed_range"]
        return float(self.rng.uniform(float(lo), float(hi)))

    def _sample_density_kg_m3(self) -> float:
        lo, hi = self.op_cfg["density_range"]
        return float(self.rng.uniform(float(lo), float(hi)))

    def _sample_viscosity_pa_s(self) -> float:
        unit = self._get_feature_unit("fluid_viscosity", default="cP")
        return float(convert_user_to_internal_SI(self._sample_uniform_user("viscosity_range"), unit))

    def _sample_k_friction(self) -> float:
        lo, hi = self.op_cfg["k_friction_range"]
        return float(self.rng.uniform(float(lo), float(hi)))

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

    def _valve_size_internal(self) -> float:
        """Valve nominal diameter is configured under operating_ranges.pump.valve_size."""
        unit = self._get_feature_unit("valve_size", default="inches")
        valve_cfg = (self.op_cfg.get("pump") or self.cfg.get("pump") or {})
        size_user = float(valve_cfg.get("valve_size", 12.0))
        return float(convert_user_to_internal_SI(size_user, unit))

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

        valve_opening = float("nan")
        sink_p_pa: Optional[float] = None
        valve_curve = None

        if self.valve_enabled and self.valve_curve is not None:
            valve_opening = self._sample_valve_opening_frac()
            sink_p_pa = self._sample_sink_pressure_pa()
            valve_curve = self.valve_curve

        # Static head term to represent sink boundary pressure
        h_static_m = 0.0
        if sink_p_pa is not None:
            h_static_m = float(sink_p_pa - suction_p_pa) / (rho * 9.80665)

        system = SystemCurve(
            h_static_m=h_static_m,
            k_friction=k_fric,
            density_kg_m3=rho,
            viscosity_pa_s=mu_pa_s,
            suction_pressure_pa=suction_p_pa,
            valve=valve_curve,
            valve_opening=float(valve_opening) if np.isfinite(valve_opening) else 1.0,
        )

        q_min = self.pump.head_points[0][0]
        q_max = self.pump.head_points[-1][0]

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

        # Reported discharge pressure uses sampled density
        dp_pump_pa = head_to_delta_p_pa(head_m, rho)
        discharge_p_pa = suction_p_pa + dp_pump_pa

        # K-Spice-like shaft power uses reference density + efficiency curve
        eta = self.pump.efficiency_at(q_m3_s, speed_rpm)
        rho_ref = float(self.pump.power_ref_density_kg_m3)
        q_pos = max(float(q_m3_s), 0.0)
        head_pos = max(float(head_m), 0.0)
        pump_power_W = (rho_ref * 9.80665 * q_pos * head_pos) / max(eta, 1e-6)

        self.pump.hydraulic_wear_frac = prev_wear

        row: Dict[str, float] = {
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
            "valve_opening": float(valve_opening) if np.isfinite(valve_opening) else float("nan"),
            "sink_pressure": float(sink_p_pa) if sink_p_pa is not None else float("nan"),
        }

        return row

    def generate_dataset(self, n_samples: int) -> pd.DataFrame:
        return pd.DataFrame([self.generate_sample() for _ in range(int(n_samples))])

    # ============================================================
    # Export
    # ============================================================

    def _build_schema(self) -> Dict[str, Any]:
        cols = [{"name": d["name"], "user_unit": d.get("unit")} for d in self.data_cfg]
        return {
            "project_name": self.cfg.get("project_name"),
            "client_name": self.cfg.get("client_name"),
            "equipment_type": self.cfg.get("equipment_type"),
            "canonical_internal_system": "SI",
            "columns": cols,
        }

    def export_csv_and_schema(self) -> Tuple[str, str]:
        out_cfg = self.cfg.get("output", {}) or {}
        out_dir = out_cfg.get("directory", "data/synthetic")
        os.makedirs(out_dir, exist_ok=True)

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

        def _convert_df(df_int: pd.DataFrame) -> pd.DataFrame:
            df_export = df_int.copy()
            for d in self.data_cfg:
                name = d["name"]
                unit = d.get("unit")
                if unit is None or name not in df_export.columns:
                    continue
                df_export[name] = df_export[name].apply(
                    lambda v: convert_internal_back_to_user(float(v), unit) if pd.notna(v) else v
                )
            return df_export

        train_name = fname_tmpl.format(split="train", **self.cfg)
        train_path = os.path.join(out_dir, train_name)
        if os.path.exists(train_path) and not overwrite:
            raise FileExistsError(f"Output exists and overwrite=false: {train_path}")
        _convert_df(df_train_int).to_csv(train_path, index=False)

        test_name = fname_tmpl.format(split="test", **self.cfg)
        test_path = os.path.join(out_dir, test_name)
        if os.path.exists(test_path) and not overwrite:
            raise FileExistsError(f"Output exists and overwrite=false: {test_path}")
        _convert_df(df_test_int).to_csv(test_path, index=False)

        schema_name = fname_tmpl.format(split="schema", **self.cfg)
        schema_root, _ = os.path.splitext(schema_name)
        schema_path = os.path.join(out_dir, f"{schema_root}.yaml")
        with open(schema_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._build_schema(), f, sort_keys=False)

        return train_path, test_path

    def _train_test_split(self, df: pd.DataFrame, test_pct: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        test_pct = float(test_pct)
        if test_pct <= 0.0:
            return df.reset_index(drop=True), df.iloc[0:0].copy()
        if test_pct >= 100.0:
            return df.iloc[0:0].copy(), df.reset_index(drop=True)

        n = len(df)
        n_test = int(round(n * test_pct / 100.0))
        n_test = max(1, min(n - 1, n_test))

        rng = np.random.default_rng(int(self.cfg.get("random_seed", 42)))
        idx = np.arange(n)
        rng.shuffle(idx)

        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
