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

    INTERNAL computations are canonical SI:
      - pressures: Pa
      - flow: m3/s
      - temp: K
      - density: kg/m3
      - viscosity: Pa*s
      - head: m
      - power: W
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

        # ------------------------------------------------------------
        # Config structure:
        #   data:
        #     inputs:    [{name, unit, range, (optional) sampling}, ...]
        #     variables: [{name, unit}, ...]  # equation-derived outputs only
        #   equipment_paramterization:
        #     pump: ...
        #     valve: ...
        #   generation: (optional)
        #     datasets: [...]
        # ------------------------------------------------------------
        data = cfg.get("data", {}) or {}
        self.inputs_cfg: List[Dict[str, Any]] = list(data.get("inputs", []) or [])
        self.vars_cfg: List[Dict[str, Any]] = list(data.get("variables", []) or [])

        if not self.inputs_cfg:
            raise KeyError("Missing data.inputs block (list of input definitions).")
        if not self.vars_cfg:
            raise KeyError("Missing data.variables block (list of variable definitions).")

        # Units by feature name (inputs + variables)
        self._units: Dict[str, str] = {}
        for d in self.inputs_cfg + self.vars_cfg:
            n = str(d.get("name", "")).strip()
            if not n:
                continue
            u = d.get("unit")
            if u is not None:
                self._units[n] = str(u)

        # Input definitions by name (for range + sampling)
        self._inputs_by_name: Dict[str, Dict[str, Any]] = {
            str(d["name"]): d for d in self.inputs_cfg if "name" in d
        }

        # Validate required inputs exist
        for req in (
            "suction_pressure",
            "sink_pressure",
            "temperature",
            "speed",
            "fluid_density",
            "fluid_viscosity",
            "k_friction",
            "hydraulic_wear",
            "valve_opening",
        ):
            if req not in self._inputs_by_name:
                raise KeyError(f"Missing required input in data.inputs: '{req}'")

        # -------------------------
        # Equipment parameterization
        # -------------------------
        equip = cfg.get("equipment_paramterization", {}) or {}

        # -------------------------
        # Pump (head + efficiency)
        # -------------------------
        pump_cfg = equip.get("pump", {}) or {}
        if not pump_cfg:
            raise KeyError("Missing equipment_paramterization.pump block.")

        base_speed_rpm = float(pump_cfg["base_speed_rpm"])

        # Units for pump curves (prefer explicit, else fall back)
        q_unit = str(pump_cfg.get("curve_q_unit") or self._get_unit("q_liquid", default="m3/h"))
        head_unit = str(pump_cfg.get("head_unit", "m"))
        eff_unit = str(pump_cfg.get("efficiency_unit", "pct")).lower()

        # Head curve points (Q in user units -> SI m3/s, head -> m)
        head_points_user: List[Tuple[float, float]] = [(float(q), float(h)) for q, h in pump_cfg["curve_points"]]
        head_points_user.sort(key=lambda t: t[0])
        head_points_si: List[Tuple[float, float]] = [
            (float(convert_user_to_internal_SI(q_user, q_unit)), float(convert_user_to_internal_SI(h_user, head_unit)))
            for q_user, h_user in head_points_user
        ]

        # Efficiency curve points (optional)
        eff_points_si: Optional[List[Tuple[float, float]]] = None
        if pump_cfg.get("efficiency_curve_points") is not None:
            eff_points_user: List[Tuple[float, float]] = [
                (float(q), float(eff_val)) for q, eff_val in pump_cfg["efficiency_curve_points"]
            ]
            eff_points_user.sort(key=lambda t: t[0])

            def _eff_to_frac(v: float) -> float:
                if eff_unit in ("pct", "percent", "%"):
                    return float(v) / 100.0
                return float(v)

            eff_points_si = [
                (float(convert_user_to_internal_SI(q_user, q_unit)), _eff_to_frac(eff_val))
                for q_user, eff_val in eff_points_user
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
        valve_cfg = equip.get("valve", {}) or {}
        self.valve_enabled = bool(valve_cfg.get("enabled", False))
        self.valve_curve: Optional[ValveCvCurve] = None

        if self.valve_enabled:
            if "cv_curve_points" not in valve_cfg:
                raise KeyError("Missing equipment_paramterization.valve.cv_curve_points")

            # Valve size and its unit
            valve_size_user = float(valve_cfg.get("valve_size", 12.0))
            valve_size_unit = str(valve_cfg.get("valve_size_unit", "inches"))
            valve_diam_m = float(convert_user_to_internal_SI(valve_size_user, valve_size_unit))

            # Interpret Cv-style points as: Q[cv_flow_unit] = Cv(open)*sqrt(dP[cv_dp_unit])
            cv_flow_unit = valve_cfg.get("cv_flow_unit")
            cv_dp_unit = valve_cfg.get("cv_dp_unit")

            k_points_internal: List[Tuple[float, float]] = []
            for opening_user, cv_user in valve_cfg["cv_curve_points"]:
                opening = float(opening_user)
                cv_val = float(cv_user)

                if cv_flow_unit is not None and cv_dp_unit is not None:
                    k_internal = self._convert_k_to_internal(cv_val, str(cv_flow_unit), str(cv_dp_unit))
                else:
                    # Assume user already provided internal-basis K [(m3/s)/sqrt(Pa)]
                    k_internal = cv_val

                k_points_internal.append((opening, k_internal))

            k_points_internal.sort(key=lambda t: t[0])

            self.valve_curve = ValveCvCurve(
                k_curve_points=k_points_internal,
                nominal_diameter_m=valve_diam_m,
                fp=float(valve_cfg.get("fp", 1.0)),
                xt=float(valve_cfg.get("xt", 0.7)),
                fl=float(valve_cfg.get("fl", 0.9)),
                fd=float(valve_cfg.get("fd", 0.48)),
            )

    # ============================================================
    # Config helpers
    # ============================================================

    def _get_unit(self, name: str, default: Optional[str] = None) -> str:
        u = self._units.get(name)
        if u is not None:
            return str(u)
        if default is None:
            raise KeyError(f"Unit for feature '{name}' not found in config (data.inputs/data.variables).")
        return default

    def _input_def(self, name: str) -> Dict[str, Any]:
        if name not in self._inputs_by_name:
            raise KeyError(f"Input '{name}' not found in data.inputs")
        return self._inputs_by_name[name]

    def _convert_k_to_internal(self, k_user: float, flow_unit: str, dp_unit: str) -> float:
        """Convert user K (Q/sqrt(dP)) into internal K [(m3/s)/sqrt(Pa)]."""
        flow_scale = float(convert_user_to_internal_SI(1.0, flow_unit))  # -> m3/s
        dp_scale = float(convert_user_to_internal_SI(1.0, dp_unit))      # -> Pa
        return float(k_user) * flow_scale / float(np.sqrt(dp_scale))

    # ============================================================
    # Samplers (input ranges are in USER units)
    # ============================================================

    def _sample_from_range_cfg(self, range_cfg: Any, sampling_cfg: Optional[Dict[str, Any]] = None) -> float:
        """
        Sample a value in USER units.

        range_cfg:
          - [min, max] -> uniform by default

        sampling_cfg (optional):
          - {method: uniform|beta|mixture, ...}
        """
        if not (isinstance(range_cfg, (list, tuple)) and len(range_cfg) == 2):
            raise TypeError("Input 'range' must be a [min, max] list.")

        lo, hi = float(range_cfg[0]), float(range_cfg[1])
        sampling = sampling_cfg or {}
        method = str(sampling.get("method", "uniform")).lower()

        if method in ("uniform", "flat"):
            return float(self.rng.uniform(lo, hi))

        if method == "beta":
            a = float(sampling.get("a", 1.0))
            b = float(sampling.get("b", 1.0))
            u = float(self.rng.beta(a, b))
            return float(lo + u * (hi - lo))

        if method == "mixture":
            comps = sampling.get("components") or []
            if not isinstance(comps, list) or len(comps) == 0:
                return float(self.rng.uniform(lo, hi))

            weights = np.array([float(c.get("weight", 1.0)) for c in comps], dtype=float)
            s = float(weights.sum())
            if s <= 0:
                weights = np.ones(len(comps), dtype=float) / float(len(comps))
            else:
                weights = weights / s

            idx = int(self.rng.choice(len(comps), p=weights))
            c = comps[idx] or {}
            c_type = str(c.get("type", "uniform")).lower()

            if c_type in ("uniform", "flat"):
                return float(self.rng.uniform(lo, hi))

            if c_type == "beta":
                a = float(c.get("a", 1.0))
                b = float(c.get("b", 1.0))
                u = float(self.rng.beta(a, b))
                return float(lo + u * (hi - lo))

            return float(self.rng.uniform(lo, hi))

        return float(self.rng.uniform(lo, hi))

    def _sample_input_user(self, name: str) -> float:
        d = self._input_def(name)
        if "range" not in d:
            raise KeyError(f"Input '{name}' missing 'range' field.")
        return float(self._sample_from_range_cfg(d["range"], d.get("sampling")))

    def _sample_input_internal(self, name: str) -> float:
        unit = self._get_unit(name)
        v_user = self._sample_input_user(name)
        return float(convert_user_to_internal_SI(v_user, unit))

    def _sample_operating_point_internal(self) -> Dict[str, float]:
        """
        Sample one operating point (internal SI) INCLUDING hydraulic_wear sample.
        This is the single source of randomness. Downstream datasets will reuse the
        same operating points and the same train/test split indices.
        """
        return {
            "suction_pressure": float(self._sample_input_internal("suction_pressure")),
            "sink_pressure": float(self._sample_input_internal("sink_pressure")),
            "temperature": float(self._sample_input_internal("temperature")),
            "speed": float(self._sample_input_internal("speed")),  # rpm stays rpm
            "fluid_density": float(self._sample_input_internal("fluid_density")),
            "fluid_viscosity": float(self._sample_input_internal("fluid_viscosity")),
            "k_friction": float(self._sample_input_internal("k_friction")),
            "hydraulic_wear": float(self._sample_input_internal("hydraulic_wear")),  # sampled wear
            "valve_opening": float(self._sample_input_internal("valve_opening")),
        }

    # ============================================================
    # Physics solve (single operating point)
    # ============================================================

    def _solve_outputs_internal(self, inp: Dict[str, float], hydraulic_wear_internal: float) -> Dict[str, float]:
        suction_p_pa = float(inp["suction_pressure"])
        sink_p_pa = float(inp["sink_pressure"])
        temp_K = float(inp["temperature"])
        speed_rpm = float(inp["speed"])
        rho = float(inp["fluid_density"])
        mu_pa_s = float(inp["fluid_viscosity"])
        k_fric = float(inp["k_friction"])
        valve_opening = float(inp["valve_opening"])

        # Static head term to represent sink boundary pressure
        h_static_m = float(sink_p_pa - suction_p_pa) / (max(rho, 1e-9) * 9.80665)

        system = SystemCurve(
            h_static_m=h_static_m,
            k_friction=k_fric,
            density_kg_m3=rho,
            viscosity_pa_s=mu_pa_s,
            suction_pressure_pa=suction_p_pa,
            sink_pressure_pa=sink_p_pa,
            valve=self.valve_curve if self.valve_enabled else None,
            valve_opening=valve_opening,
        )

        q_min = self.pump.head_points[0][0]
        q_max = self.pump.head_points[-1][0]

        prev_wear = float(getattr(self.pump, "hydraulic_wear_frac", 0.0))
        self.pump.hydraulic_wear_frac = float(hydraulic_wear_internal)

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

        return {
            "discharge_pressure": float(discharge_p_pa),
            "delta_pressure": float(dp_pump_pa),
            "q_liquid": float(q_m3_s),
            "pump_power": float(pump_power_W),
        }

    # ============================================================
    # Dataset generation (single-pass sampling + shared split)
    # ============================================================

    def _generate_operating_points_internal(self, n_samples: int) -> pd.DataFrame:
        rows = [self._sample_operating_point_internal() for _ in range(int(n_samples))]
        return pd.DataFrame(rows)

    def _compute_clean_and_wear_internal(self, df_inp: pd.DataFrame, reference_wear_internal: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        For each operating point, compute:
          - clean outputs at reference_wear_internal
          - wear outputs at sampled wear (df_inp['hydraulic_wear'])
        Returns (df_clean_vars, df_wear_vars) with variable columns.
        """
        clean_rows: List[Dict[str, float]] = []
        wear_rows: List[Dict[str, float]] = []

        for _, r in df_inp.iterrows():
            inp = r.to_dict()
            wear_sample = float(inp["hydraulic_wear"])

            clean_rows.append(self._solve_outputs_internal(inp, reference_wear_internal))
            wear_rows.append(self._solve_outputs_internal(inp, wear_sample))

        return pd.DataFrame(clean_rows), pd.DataFrame(wear_rows)

    # ============================================================
    # Export / schema
    # ============================================================

    def _columns_definition(self) -> List[Dict[str, Any]]:
        cols: List[Dict[str, Any]] = []
        for d in self.inputs_cfg:
            cols.append({"name": d["name"], "user_unit": d.get("unit"), "role": "input"})
        for d in self.vars_cfg:
            cols.append({"name": d["name"], "user_unit": d.get("unit"), "role": "variable"})
        return cols

    def _build_schema(self) -> Dict[str, Any]:
        return {
            "project_name": self.cfg.get("project_name"),
            "client_name": self.cfg.get("client_name"),
            "equipment_type": self.cfg.get("equipment_type"),
            "canonical_internal_system": "SI",
            "columns": self._columns_definition(),
        }

    def _convert_df_to_user_units(self, df_int: pd.DataFrame) -> pd.DataFrame:
        df_export = df_int.copy()
        col_defs = self._columns_definition()
        for c in col_defs:
            name = c["name"]
            unit = c.get("user_unit")
            if unit is None or name not in df_export.columns:
                continue
            df_export[name] = df_export[name].apply(
                lambda v: convert_internal_back_to_user(float(v), str(unit)) if pd.notna(v) else v
            )
        return df_export

    def export_csv_and_schema(self) -> Tuple[str, str] | List[str]:
        """
        Backwards compatible:
          - If config does NOT contain generation.datasets, export a single train/test pair
            using output.filename_template (existing behavior).
          - If config contains generation.datasets, export multiple train/test pairs
            according to generation.datasets[*].outputs.{train_filename,test_filename}.
        """
        out_cfg = self.cfg.get("output", {}) or {}
        out_dir = str(out_cfg.get("directory", "data/synthetic"))
        os.makedirs(out_dir, exist_ok=True)
        overwrite = bool(out_cfg.get("overwrite", False))

        gen_cfg = self.cfg.get("generation", {}) or {}
        datasets_cfg = gen_cfg.get("datasets")

        n_samples = int(self.cfg.get("n_samples", 1000))
        test_pct = float(self.cfg.get("test_split", 0.0))

        # -------------------------------
        # New multi-dataset mode
        # -------------------------------
        if isinstance(datasets_cfg, list) and len(datasets_cfg) > 0:
            paired_ref_cfg = (gen_cfg.get("paired_reference") or {})
            # Prefer top-level generation.paired_reference.reference_wear if present,
            # otherwise per-dataset wear_reference is used (fallback is 0.0).
            ref_wear_user_default = float(paired_ref_cfg.get("reference_wear", 0.0))
            ref_wear_internal_default = float(convert_user_to_internal_SI(ref_wear_user_default, self._get_unit("hydraulic_wear", default="frac")))

            # Sample operating points ONCE and split ONCE
            df_inp = self._generate_operating_points_internal(n_samples)
            df_train_inp, df_test_inp = self._train_test_split(df_inp, test_pct)

            # For efficiency, compute clean/wear variables once for full df
            df_clean_vars_full, df_wear_vars_full = self._compute_clean_and_wear_internal(df_inp, ref_wear_internal_default)

            # Attach to inputs
            df_clean_full = pd.concat([df_inp.reset_index(drop=True), df_clean_vars_full.reset_index(drop=True)], axis=1)
            df_wear_full = pd.concat([df_inp.reset_index(drop=True), df_wear_vars_full.reset_index(drop=True)], axis=1)

            # Paired dataset columns (derived)
            df_paired_full = df_wear_full.copy()
            # add clean q + delta if present
            df_paired_full["q_liquid_clean"] = df_clean_full["q_liquid"]
            delta_q_abs = df_wear_full["q_liquid"] - df_clean_full["q_liquid"]
            # optional clamp to keep training stable
            delta_q_abs = np.clip(delta_q_abs, -20.0, 20.0)
            df_paired_full["delta_q_liquid"] = delta_q_abs

            # For non-paired datasets, keep paired-only vars as NaN if they exist in schema
            if "q_liquid_clean" in df_clean_full.columns:
                df_clean_full["q_liquid_clean"] = np.nan
            if "delta_q_liquid" in df_clean_full.columns:
                df_clean_full["delta_q_liquid"] = np.nan
            if "q_liquid_clean" in df_wear_full.columns:
                df_wear_full["q_liquid_clean"] = np.nan
            if "delta_q_liquid" in df_wear_full.columns:
                df_wear_full["delta_q_liquid"] = np.nan

            # Build train/test indices once (based on df_inp split)
            # We re-split by regenerating indices from the same RNG to keep consistent selection.
            # The helper returns reindexed frames; we rebuild by using the original RNG split on df_inp index.
            # To preserve exact indices, compute index arrays here:
            train_idx, test_idx = self._train_test_indices(len(df_inp), test_pct)

            outputs_written: List[str] = []

            for dset in datasets_cfg:
                name = str(dset.get("name", "")).strip() or "dataset"
                mode = str(dset.get("mode", "")).strip().lower()

                out_spec = dset.get("outputs", {}) or {}
                train_fn_tmpl = out_spec.get("train_filename")
                test_fn_tmpl = out_spec.get("test_filename")
                if not train_fn_tmpl or not test_fn_tmpl:
                    raise KeyError(f"generation.datasets[{name}] must define outputs.train_filename and outputs.test_filename")

                # Select base frame
                if mode == "clean":
                    df_full = df_clean_full.copy()
                    # override wear to reference (internal)
                    df_full["hydraulic_wear"] = float(ref_wear_internal_default)
                elif mode == "wear":
                    df_full = df_wear_full.copy()
                elif mode == "paired":
                    df_full = df_paired_full.copy()
                else:
                    raise ValueError(f"Unknown dataset mode '{mode}' for dataset '{name}'. Use clean|wear|paired.")

                df_train = df_full.iloc[train_idx].reset_index(drop=True)
                df_test = df_full.iloc[test_idx].reset_index(drop=True)

                # Convert to user units
                df_train_u = self._convert_df_to_user_units(df_train)
                df_test_u = self._convert_df_to_user_units(df_test)

                train_name = str(train_fn_tmpl).format(**self.cfg)
                test_name = str(test_fn_tmpl).format(**self.cfg)

                train_path = os.path.join(out_dir, train_name)
                test_path = os.path.join(out_dir, test_name)

                if os.path.exists(train_path) and not overwrite:
                    raise FileExistsError(f"Output exists and overwrite=false: {train_path}")
                if os.path.exists(test_path) and not overwrite:
                    raise FileExistsError(f"Output exists and overwrite=false: {test_path}")

                df_train_u.to_csv(train_path, index=False)
                df_test_u.to_csv(test_path, index=False)

                outputs_written.extend([train_path, test_path])

            # Write a single schema for the run
            schema_path = os.path.join(out_dir, f"{self.cfg.get('client_name','client')}_schema.yaml")
            with open(schema_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self._build_schema(), f, sort_keys=False)
            outputs_written.append(schema_path)

            return outputs_written

        # -------------------------------
        # Legacy single-dataset mode
        # -------------------------------
        fname_tmpl = str(out_cfg.get("filename_template", "{client_name}_dataset.csv"))
        if "{split}" not in fname_tmpl:
            root, ext = os.path.splitext(fname_tmpl)
            ext = ext or ".csv"
            fname_tmpl = f"{root}_{{split}}{ext}"

        df_internal = self.generate_dataset(n_samples)
        df_train_int, df_test_int = self._train_test_split(df_internal, test_pct)

        train_name = fname_tmpl.format(split="train", **self.cfg)
        train_path = os.path.join(out_dir, train_name)
        if os.path.exists(train_path) and not overwrite:
            raise FileExistsError(f"Output exists and overwrite=false: {train_path}")
        self._convert_df_to_user_units(df_train_int).to_csv(train_path, index=False)

        test_name = fname_tmpl.format(split="test", **self.cfg)
        test_path = os.path.join(out_dir, test_name)
        if os.path.exists(test_path) and not overwrite:
            raise FileExistsError(f"Output exists and overwrite=false: {test_path}")
        self._convert_df_to_user_units(df_test_int).to_csv(test_path, index=False)

        schema_name = fname_tmpl.format(split="schema", **self.cfg)
        schema_root, _ = os.path.splitext(schema_name)
        schema_path = os.path.join(out_dir, f"{schema_root}.yaml")
        with open(schema_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._build_schema(), f, sort_keys=False)

        return train_path, test_path

    def generate_sample(self) -> Dict[str, float]:
        """
        Legacy single-sample generator (wear sampled, single physics pass).
        """
        inp = self._sample_operating_point_internal()
        wear = float(inp["hydraulic_wear"])
        vars_out = self._solve_outputs_internal(inp, wear)

        row: Dict[str, float] = {**inp, **vars_out}
        return row

    def generate_dataset(self, n_samples: int) -> pd.DataFrame:
        return pd.DataFrame([self.generate_sample() for _ in range(int(n_samples))])

    # ============================================================
    # Split helpers
    # ============================================================

    def _train_test_indices(self, n: int, test_pct: float) -> Tuple[np.ndarray, np.ndarray]:
        test_pct = float(test_pct)
        if test_pct <= 0.0:
            return np.arange(n), np.array([], dtype=int)
        if test_pct >= 100.0:
            return np.array([], dtype=int), np.arange(n)

        n_test = int(round(n * test_pct / 100.0))
        n_test = max(1, min(n - 1, n_test))

        rng = np.random.default_rng(int(self.cfg.get("random_seed", 42)))
        idx = np.arange(n)
        rng.shuffle(idx)

        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        return train_idx, test_idx

    def _train_test_split(self, df: pd.DataFrame, test_pct: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_idx, test_idx = self._train_test_indices(len(df), test_pct)
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
