from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

G_STD = 9.80665

# ============================================================
# Hard-coded valve parameters (always applied)
# ============================================================

VALVE_FL = 0.9
VALVE_XT = 0.7
VALVE_FD = 0.48

# Piping geometry factor (Fp) - reduces effective capacity slightly (K-Spice includes piping effects)
VALVE_FP = 0.87

# If you later want cavitation / vapor pressure effects, wire them here.
VALVE_FF = 0.96
VALVE_PV_PA = 0.0


# ============================================================
# 1D Natural Cubic Spline (no external deps)
# ============================================================

class NaturalCubicSpline1D:
    """
    Natural cubic spline interpolation for strictly increasing x.

    This mirrors K-Spice's "Spline" option closely enough for our use:
      - smooth first/second derivatives
      - natural boundary conditions (2nd derivative = 0 at endpoints)

    Note: values are extrapolated linearly outside the data range (clamped slope).
    """

    def __init__(self, x: List[float], y: List[float]) -> None:
        if len(x) != len(y) or len(x) < 2:
            raise ValueError("Spline requires at least 2 points and matching x/y lengths.")
        # Ensure strictly increasing
        for i in range(len(x) - 1):
            if not (x[i + 1] > x[i]):
                raise ValueError("Spline x values must be strictly increasing.")
        self.x = [float(v) for v in x]
        self.y = [float(v) for v in y]
        self._m = self._second_derivatives(self.x, self.y)

    @staticmethod
    def _second_derivatives(x: List[float], y: List[float]) -> List[float]:
        n = len(x)
        h = [x[i + 1] - x[i] for i in range(n - 1)]

        # Tridiagonal system for second derivatives (natural spline)
        a = [0.0] * n
        b = [0.0] * n
        c = [0.0] * n
        d = [0.0] * n

        b[0] = 1.0
        d[0] = 0.0
        b[-1] = 1.0
        d[-1] = 0.0

        for i in range(1, n - 1):
            a[i] = h[i - 1]
            b[i] = 2.0 * (h[i - 1] + h[i])
            c[i] = h[i]
            d[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

        # Thomas algorithm
        for i in range(1, n):
            w = a[i] / b[i - 1] if b[i - 1] != 0 else 0.0
            b[i] = b[i] - w * c[i - 1]
            d[i] = d[i] - w * d[i - 1]

        m = [0.0] * n
        m[-1] = d[-1] / b[-1] if b[-1] != 0 else 0.0
        for i in range(n - 2, -1, -1):
            m[i] = (d[i] - c[i] * m[i + 1]) / b[i] if b[i] != 0 else 0.0
        return m

    def __call__(self, xq: float) -> float:
        xq = float(xq)
        x = self.x
        y = self.y
        m = self._m

        if xq <= x[0]:
            # linear extrapolation using first segment slope
            dx = x[1] - x[0]
            slope = (y[1] - y[0]) / dx if dx != 0 else 0.0
            return y[0] + slope * (xq - x[0])
        if xq >= x[-1]:
            dx = x[-1] - x[-2]
            slope = (y[-1] - y[-2]) / dx if dx != 0 else 0.0
            return y[-1] + slope * (xq - x[-1])

        # find interval
        lo = 0
        hi = len(x) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if x[mid] <= xq:
                lo = mid
            else:
                hi = mid

        i = lo
        h = x[i + 1] - x[i]
        if h == 0:
            return y[i]

        a = (x[i + 1] - xq) / h
        b = (xq - x[i]) / h
        return (
            a * y[i]
            + b * y[i + 1]
            + ((a**3 - a) * m[i] + (b**3 - b) * m[i + 1]) * (h**2) / 6.0
        )


# ============================================================
# Pump curve (Head + optional Efficiency)
# ============================================================

@dataclass
class PumpCurve:
    """
    Pump curve points are (q_m3_s, head_m) at base_speed_rpm.
    Efficiency points are (q_m3_s, eff_frac) at base_speed_rpm.

    K-Spice uses spline interpolation; we replicate that with NaturalCubicSpline1D.
    """
    base_speed_rpm: float
    head_points: List[Tuple[float, float]]
    eff_points: Optional[List[Tuple[float, float]]] = None
    power_ref_density_kg_m3: float = 1000.0
    hydraulic_wear_frac: float = 0.0  # 0..1, reduces head uniformly

    def __post_init__(self) -> None:
        self.head_points = sorted([(float(q), float(h)) for q, h in self.head_points], key=lambda t: t[0])
        qh = [q for q, _ in self.head_points]
        hh = [h for _, h in self.head_points]
        self._head_spline = NaturalCubicSpline1D(qh, hh)

        self._eff_spline: Optional[NaturalCubicSpline1D] = None
        if self.eff_points:
            self.eff_points = sorted([(float(q), float(e)) for q, e in self.eff_points], key=lambda t: t[0])
            qe = [q for q, _ in self.eff_points]
            ee = [e for _, e in self.eff_points]
            self._eff_spline = NaturalCubicSpline1D(qe, ee)

    def head_at(self, q_m3_s: float, speed_rpm: float) -> float:
        # Affinity laws: Q ~ N, H ~ N^2
        if speed_rpm <= 0:
            return 0.0
        n_ratio = float(speed_rpm) / float(self.base_speed_rpm)
        q_base = float(q_m3_s) / max(n_ratio, 1e-12)

        head_base = float(self._head_spline(q_base))
        head = head_base * (n_ratio ** 2)

        wear = max(0.0, min(1.0, float(self.hydraulic_wear_frac)))
        return head * (1.0 - wear)

    def efficiency_at(self, q_m3_s: float, speed_rpm: float) -> float:
        """
        Returns efficiency as a fraction (0..1). If no efficiency curve exists, returns 1.0.
        We use q scaling (Q ~ N) to map to base curve.
        """
        if self._eff_spline is None or speed_rpm <= 0:
            return 1.0
        n_ratio = float(speed_rpm) / float(self.base_speed_rpm)
        q_base = float(q_m3_s) / max(n_ratio, 1e-12)
        eta = float(self._eff_spline(q_base))
        # Clamp to sane values
        return max(1e-6, min(1.0, eta))


def head_to_delta_p_pa(head_m: float, rho_kg_m3: float) -> float:
    return float(rho_kg_m3) * G_STD * float(head_m)


# ============================================================
# Valve model (capacity curve + Reynolds correction + FL cap)
# ============================================================

@dataclass
class ValveCvCurve:
    """
    Capacity curve:
        Q = K(opening) * sqrt(dP)
    where:
        Q [m3/s]
        dP [Pa]
        K [ (m3/s)/sqrt(Pa) ]
    """
    k_curve_points: List[Tuple[float, float]]
    nominal_diameter_m: float

    def __post_init__(self) -> None:
        self.k_curve_points = sorted([(float(o), float(k)) for o, k in self.k_curve_points], key=lambda t: t[0])
        xo = [o for o, _ in self.k_curve_points]
        yk = [k for _, k in self.k_curve_points]
        self._k_spline = NaturalCubicSpline1D(xo, yk)

    def k_at_opening(self, opening_frac: float) -> float:
        op = max(0.0, min(1.0, float(opening_frac)))
        k = float(self._k_spline(op))
        return max(k, 0.0)

    def _reynolds_number_liquid(self, q_m3_s: float, rho: float, mu_pa_s: float) -> float:
        D = max(float(self.nominal_diameter_m), 1e-6)
        A = math.pi * (D ** 2) / 4.0
        if A <= 0 or mu_pa_s <= 0:
            return 1e12
        v = abs(float(q_m3_s)) / A
        return float(rho) * v * D / float(mu_pa_s)

    @staticmethod
    def _fr_factor_liquid(re: float) -> float:
        re = max(float(re), 1.0)
        if re >= 10000.0:
            return 1.0

        fr_trans = 1.0 + 0.33 * math.sqrt(max(VALVE_FL, 1e-6)) * math.log10(re / 10000.0)
        fr_trans = max(min(fr_trans, 1.0), 0.05)

        fr_lam = 0.026 * math.sqrt(re) / math.sqrt(max(VALVE_FL, 1e-6))
        fr_lam = max(min(fr_lam, 1.0), 0.02)

        return min(fr_trans, fr_lam)

    def delta_p_pa(
        self,
        q_m3_s: float,
        opening_frac: float,
        rho_kg_m3: float,
        mu_pa_s: float,
        p_upstream_pa: Optional[float] = None,
    ) -> float:
        K = self.k_at_opening(opening_frac)
        if K <= 0:
            return 1e30

        re = self._reynolds_number_liquid(q_m3_s=q_m3_s, rho=rho_kg_m3, mu_pa_s=mu_pa_s)
        FR = self._fr_factor_liquid(re)

        K_eff = max(K * FR * VALVE_FP, 1e-18)
        dp = (float(q_m3_s) / K_eff) ** 2

        if p_upstream_pa is not None:
            dp_max = (VALVE_FL ** 2) * max(float(p_upstream_pa) - (VALVE_FF * VALVE_PV_PA), 0.0)
            dp = min(dp, dp_max)

        return float(dp)


# ============================================================
# System curve / operating point
# ============================================================

@dataclass
class SystemCurve:
    h_static_m: float
    k_friction: float
    density_kg_m3: float
    viscosity_pa_s: float
    suction_pressure_pa: float
    valve: Optional[ValveCvCurve] = None
    valve_opening: float = 1.0

    def required_head_m(self, q_m3_s: float, head_pump_m: float) -> float:
        rho = float(self.density_kg_m3)
        mu = float(self.viscosity_pa_s)
        q = float(q_m3_s)

        h = float(self.h_static_m) + float(self.k_friction) * (q ** 2)

        if self.valve is not None:
            p1 = float(self.suction_pressure_pa) + rho * G_STD * float(head_pump_m)
            dp_v = self.valve.delta_p_pa(
                q_m3_s=q,
                opening_frac=float(self.valve_opening),
                rho_kg_m3=rho,
                mu_pa_s=mu,
                p_upstream_pa=p1,
            )
            h += dp_v / (rho * G_STD)

        return h


def solve_operating_point(
    pump: PumpCurve,
    system: SystemCurve,
    speed_rpm: float,
    q_min_m3_s: float,
    q_max_m3_s: float,
    max_iter: int = 80,
) -> float:
    """
    Solve H_pump(Q) = H_required(Q) using bisection.
    """
    q_lo = float(q_min_m3_s)
    q_hi = float(q_max_m3_s)

    def f(q: float) -> float:
        Hp = pump.head_at(q, speed_rpm)
        Hr = system.required_head_m(q, Hp)
        return Hp - Hr

    f_lo = f(q_lo)
    f_hi = f(q_hi)

    if f_lo == 0.0:
        return q_lo
    if f_hi == 0.0:
        return q_hi
    if f_lo * f_hi > 0.0:
        return q_lo if abs(f_lo) < abs(f_hi) else q_hi

    for _ in range(max_iter):
        q_mid = 0.5 * (q_lo + q_hi)
        f_mid = f(q_mid)
        if abs(f_mid) < 1e-9:
            return q_mid
        if f_lo * f_mid <= 0.0:
            q_hi, f_hi = q_mid, f_mid
        else:
            q_lo, f_lo = q_mid, f_mid

    return 0.5 * (q_lo + q_hi)
