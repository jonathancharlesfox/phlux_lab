# physics.py
# ---------------------------------------------------------
# Physics helpers for pump behavior:
# - PumpCurve (affinity-law scaled pump curves)
# - ValveCvCurve (Cv vs valve opening)
# - SystemCurve (static head + friction + optional valve)
# - solve_operating_point (pump curve ∩ system curve)
# - head_to_delta_p_pa (convert pump head → pressure rise)
#
# INTERNAL SI BASIS FOR THIS MODULE:
#   Flow          -> m3/s
#   Head          -> m
#   Density       -> kg/m3
#   Pressure drop -> Pa
#   Speed         -> rpm
#   Valve Cv      -> (m3/s) / sqrt(Pa)
#   Valve opening -> fraction (0-1)
# ---------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


# =========================================================
# Pump Curve (with affinity laws)
# =========================================================

@dataclass
class PumpCurve:
    """
    Represents a pump curve defined at a base speed.
    The curve is a set of (flow, head) points at base speed.

    INTERNAL UNITS:
        flow (q)   : m3/s
        head (H)   : m
        speed      : rpm

    We use pump affinity laws:
        Q ∝ N
        H ∝ N^2
    """

    base_speed_rpm: float
    curve_points: List[Tuple[float, float]]  # [(q_m3_s, head_m), ...]
    hydraulic_wear_frac: float = 0.0  # fraction head degradation (0-1)


    def _interp_head_base_speed(self, q_m3_s: float) -> float:
        """Interpolates head [m] from the base-speed curve at a given flow [m3/s]."""
        q_vals = [p[0] for p in self.curve_points]
        h_vals = [p[1] for p in self.curve_points]
        return float(np.interp(q_m3_s, q_vals, h_vals))

    def head_at(self, q_m3_s: float, speed_rpm: float) -> float:
        """
        Returns pump head [m] at given flow [m3/s] and speed [rpm].

        Uses affinity law scaling relative to base speed:

            H2 = H1 * (N2/N1)^2
        """
        h_base = self._interp_head_base_speed(q_m3_s)
        ratio = float(speed_rpm) / float(self.base_speed_rpm)
        return (h_base * ratio * ratio) * (1.0 - float(self.hydraulic_wear_frac))


# =========================================================
# Valve Cv curve (optional)
# =========================================================

@dataclass
class ValveCvCurve:
    """
    Represents a control valve Cv curve defined as points of:

        (opening_frac, Cv_internal)

    where:
      - opening_frac is in [0, 1]
      - Cv_internal has units (m3/s) / sqrt(Pa)

    We use the common simplified valve equation form:

        ΔP = (Q / Cv)^2     [Pa]   (with consistent units)

    This ignores choked flow, Reynolds corrections, etc.
    It is sufficient for synthetic data generation.
    """

    cv_curve_points: List[Tuple[float, float]]  # [(opening_frac, Cv_internal), ...]

    def cv_at(self, opening_frac: float) -> float:
        pts = sorted(self.cv_curve_points, key=lambda t: t[0])
        o_vals, cv_vals = zip(*pts)
        o = float(np.clip(opening_frac, 0.0, 1.0))
        cv = float(np.interp(o, o_vals, cv_vals, left=cv_vals[0], right=cv_vals[-1]))
        # avoid divide-by-zero
        return float(max(cv, 1e-12))

    def delta_p_pa(self, q_m3_s: float, opening_frac: float) -> float:
        cv = self.cv_at(opening_frac)
        q = float(q_m3_s)
        return float((q / cv) ** 2)


# =========================================================
# System Curve (static head + k * Q^2 friction term + optional valve)
# =========================================================

@dataclass
class SystemCurve:
    """
    System head curve:

        H(Q) = H_static + k * Q^2 + H_valve(Q, opening)

    where:
        H_static : m
        k        : (m) / (m3/s)^2   (chosen so H is in meters when Q in m3/s)

    If a valve is present, we model:

        ΔP_valve = (Q / Cv(opening))^2      [Pa]
        H_valve  = ΔP_valve / (rho g)      [m]
    """

    h_static_m: float
    k_friction: float
    valve: Optional[ValveCvCurve] = None
    valve_opening: float = 1.0
    density_kg_m3: float = 1000.0

    def head_at(self, q_m3_s: float) -> float:
        q = float(q_m3_s)
        head = self.h_static_m + self.k_friction * q * q

        if self.valve is not None:
            g = 9.80665
            dp_valve = self.valve.delta_p_pa(q, self.valve_opening)
            head += dp_valve / (float(self.density_kg_m3) * g)

        return float(head)


# =========================================================
# Solve operating point (pump curve ∩ system curve)
# =========================================================

def solve_operating_point(
    pump: PumpCurve,
    system: SystemCurve,
    speed_rpm: float,
    q_min_m3_s: float,
    q_max_m3_s: float,
    n_grid: int = 400,
) -> float:
    """
    Computes the operating point flow rate [m3/s] where
    pump head = system head.

    We discretize Q across the pump curve range and find the
    minimum absolute difference between pump head and system head.

    This is robust and fast enough for synthetic data generation.
    """

    q_grid = np.linspace(q_min_m3_s, q_max_m3_s, n_grid)

    pump_heads = np.array([pump.head_at(q, speed_rpm) for q in q_grid])
    sys_heads  = np.array([system.head_at(q) for q in q_grid])

    diff = np.abs(pump_heads - sys_heads)
    idx = int(np.argmin(diff))
    return float(q_grid[idx])


# =========================================================
# Convert pump head [m] → ΔP [Pa]
# =========================================================

def head_to_delta_p_pa(head_m: float, density_kg_m3: float) -> float:
    """
    Convert pump head to pressure rise:

        ΔP = ρ g H
    """
    g = 9.80665  # m/s²
    return float(density_kg_m3 * g * head_m)
