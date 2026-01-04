from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

G_STD = 9.80665


# ============================================================
# Utility: Natural cubic spline (1D)
# ============================================================

class NaturalCubicSpline1D:
    """
    Natural cubic spline interpolator with endpoint second-derivative = 0.

    We clamp outside the x-range to endpoints, which matches typical pump-curve usage.
    """
    def __init__(self, x: List[float], y: List[float]) -> None:
        if len(x) != len(y):
            raise ValueError("x and y must be same length")
        if len(x) < 2:
            raise ValueError("Need at least 2 points for spline")

        # x must be strictly increasing
        for i in range(1, len(x)):
            if x[i] <= x[i - 1]:
                raise ValueError("x must be strictly increasing")

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

        # Natural spline boundary conditions
        b[0] = 1.0
        d[0] = 0.0
        b[n - 1] = 1.0
        d[n - 1] = 0.0

        for i in range(1, n - 1):
            a[i] = h[i - 1]
            b[i] = 2.0 * (h[i - 1] + h[i])
            c[i] = h[i]
            d[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

        # Thomas algorithm
        cp = [0.0] * n
        dp = [0.0] * n

        cp[0] = c[0] / b[0] if b[0] != 0 else 0.0
        dp[0] = d[0] / b[0] if b[0] != 0 else 0.0

        for i in range(1, n):
            denom = b[i] - a[i] * cp[i - 1]
            if abs(denom) < 1e-18:
                denom = 1e-18
            cp[i] = c[i] / denom if i < n - 1 else 0.0
            dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

        m = [0.0] * n
        m[n - 1] = dp[n - 1]
        for i in range(n - 2, -1, -1):
            m[i] = dp[i] - cp[i] * m[i + 1]

        return m

    def __call__(self, xq: float) -> float:
        # Clamp
        if xq <= self.x[0]:
            return float(self.y[0])
        if xq >= self.x[-1]:
            return float(self.y[-1])

        # Binary search interval
        lo = 0
        hi = len(self.x) - 1
        while hi - lo > 1:
            mid = (hi + lo) // 2
            if self.x[mid] <= xq:
                lo = mid
            else:
                hi = mid

        i = lo
        h = self.x[i + 1] - self.x[i]
        if h <= 0:
            return float(self.y[i])

        A = (self.x[i + 1] - xq) / h
        B = (xq - self.x[i]) / h

        yq = (
            A * self.y[i]
            + B * self.y[i + 1]
            + ((A**3 - A) * self._m[i] + (B**3 - B) * self._m[i + 1]) * (h**2) / 6.0
        )
        return float(yq)


# ============================================================
# Pump curve model (head/eff vs Q with affinity laws)
# ============================================================

@dataclass
class PumpCurve:
    """
    Pump curve points are (q_m3_s, head_m) at base_speed_rpm.
    Efficiency points are (q_m3_s, eff_frac) at base_speed_rpm.

    Wear model:
      - head reduced by (1 - wear)
      - efficiency reduced by (1 - eff_sens * wear)
    """
    base_speed_rpm: float
    head_points: List[Tuple[float, float]]
    eff_points: Optional[List[Tuple[float, float]]] = None

    power_ref_density_kg_m3: float = 1000.0

    hydraulic_wear_frac: float = 0.0  # 0..1
    hydraulic_wear_eff_sensitivity: float = 0.7  # 0..1
    hydraulic_wear_head_alpha: float = 1.0

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
        if speed_rpm <= 0:
            return 0.0

        # Affinity laws: Q ~ N, H ~ N^2
        n_ratio = float(speed_rpm) / float(self.base_speed_rpm)
        q_base = float(q_m3_s) / max(n_ratio, 1e-12)

        head_base = float(self._head_spline(q_base))
        head = head_base * (n_ratio ** 2)

        wear = max(0.0, min(1.0, float(self.hydraulic_wear_frac)))

        # fudge knob: alpha > 1 makes wear hit harder (K-Spice-like)
        alpha = float(getattr(self, "hydraulic_wear_head_alpha", 1.0))  # default 1.0
        return head * ((1.0 - wear) ** alpha)

    def efficiency_at(self, q_m3_s: float, speed_rpm: float) -> float:
        """
        Returns hydraulic pump efficiency (0..1).
        Vendor pump efficiency curves are hydraulic efficiency (hydraulic / shaft),
        not motor electrical efficiency.
        """
        if self._eff_spline is None or speed_rpm <= 0:
            return 1.0

        n_ratio = float(speed_rpm) / float(self.base_speed_rpm)
        q_base = float(q_m3_s) / max(n_ratio, 1e-12)

        eta_healthy = float(self._eff_spline(q_base))

        wear = max(0.0, min(1.0, float(self.hydraulic_wear_frac)))
        sens = max(0.0, min(1.0, float(self.hydraulic_wear_eff_sensitivity)))
        eta = eta_healthy * (1.0 - sens * wear)

        return max(1e-6, min(1.0, eta))


def head_to_delta_p_pa(head_m: float, rho_kg_m3: float) -> float:
    return float(rho_kg_m3) * G_STD * float(head_m)


# ============================================================
# Valve model (capacity curve + Reynolds correction)
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

    Important: K(opening) here is a *base* capacity. Piping geometry factor Fp
    (typically <= 1) is applied as a multiplier to the capacity:
        K_eff = Fp * K
    which increases dP for the same Q if Fp < 1. This aligns with common ISA/IEC
    sizing where Fp reduces effective Cv due to fittings/reducers.
    """
    k_curve_points: List[Tuple[float, float]]
    nominal_diameter_m: float

    # Optional sizing modifiers (defaults are conservative)
    fp: float = 1.0  # piping geometry factor (<=1 reduces capacity)
    xt: float = 0.7  # pressure recovery factor (used only for optional cap)
    fl: float = 0.9  # liquid pressure recovery (not used directly yet)
    fd: float = 0.48 # valve style modifier (not used directly yet)

    def __post_init__(self) -> None:
        self.k_curve_points = sorted([(float(o), float(k)) for o, k in self.k_curve_points], key=lambda t: t[0])
        xo = [o for o, _ in self.k_curve_points]
        yk = [k for _, k in self.k_curve_points]
        self._k_spline = NaturalCubicSpline1D(xo, yk)

        # Clamp fp to sane range
        self.fp = float(self.fp)
        if not (0.1 <= self.fp <= 2.0):
            self.fp = 1.0

        self.xt = max(0.0, min(1.0, float(self.xt)))

    def k_at_opening(self, opening_frac: float) -> float:
        op = max(0.0, min(1.0, float(opening_frac)))
        k = float(self._k_spline(op))
        k = max(k, 0.0)

        # Apply piping factor to capacity
        return k * float(self.fp)

    def _reynolds_number_liquid(self, q_m3_s: float, rho: float, mu_pa_s: float) -> float:
        D = max(float(self.nominal_diameter_m), 1e-6)
        A = math.pi * (D ** 2) / 4.0
        if A <= 0 or mu_pa_s <= 0:
            return 1e12
        v = abs(float(q_m3_s)) / A
        return float(rho) * v * D / float(mu_pa_s)

    @staticmethod
    def _fr_factor_liquid(re: float) -> float:
        # Mild Reynolds correction (you can replace with a stricter ISA form later)
        re = max(float(re), 1.0)
        if re >= 10000.0:
            return 1.0
        fr_trans = 1.0 + (10_000.0 - re) / 10_000.0
        fr_lam = 1.0 + (10_000.0 - re) / 2_000.0
        return max(1.0, min(fr_lam, fr_trans))

    def valve_dp_pa(
        self,
        q_m3_s: float,
        opening_frac: float,
        rho_kg_m3: float,
        mu_pa_s: float,
        p_up_pa: float,
        p_dn_pa: float,
    ) -> float:
        q = float(q_m3_s)
        op = max(0.0, min(1.0, float(opening_frac)))
        rho = max(float(rho_kg_m3), 1e-9)
        mu = max(float(mu_pa_s), 1e-12)

        k_eff = self.k_at_opening(op)
        if k_eff <= 1e-18:
            return 0.0

        # Base dp from capacity curve
        dp = (q / k_eff) ** 2  # Pa

        # Reynolds correction
        re = self._reynolds_number_liquid(q, rho, mu)
        fr = self._fr_factor_liquid(re)
        dp *= fr

        # Optional "cap" for extreme recovery effects (usually irrelevant for liquid, moderate dP)
        # Keep conservative and based on upstream absolute pressure.
        p1 = max(float(p_up_pa), 0.0)
        dp_cap = max(0.0, self.xt * p1)
        if dp_cap > 0.0:
            dp = min(dp, dp_cap)

        return max(0.0, float(dp))


# ============================================================
# System curve (what generator expects)
# ============================================================

@dataclass
class SystemCurve:
    """
    System-side requirements seen by the pump.

    head_required = h_static + k_friction*q^2 + head_valve
    """
    h_static_m: float
    k_friction: float
    density_kg_m3: float
    viscosity_pa_s: float
    suction_pressure_pa: float
    sink_pressure_pa: float
    valve: Optional[ValveCvCurve]
    valve_opening: float = 1.0

    def head_required_m(self, q_m3_s: float, dp_pump_pa: float) -> float:
        rho = max(float(self.density_kg_m3), 1e-9)
        mu = max(float(self.viscosity_pa_s), 1e-12)

        head_fric = float(self.k_friction) * (float(q_m3_s) ** 2)

        p_up = float(self.suction_pressure_pa) + float(dp_pump_pa)
        p_dn = float(self.sink_pressure_pa)

        head_valve = 0.0
        if self.valve is not None:
            dp_valve = self.valve.valve_dp_pa(
                q_m3_s=float(q_m3_s),
                opening_frac=float(self.valve_opening),
                rho_kg_m3=rho,
                mu_pa_s=mu,
                p_up_pa=p_up,
                p_dn_pa=p_dn,
            )
            head_valve = dp_valve / (rho * G_STD)

        return float(self.h_static_m) + head_fric + head_valve


# ============================================================
# Operating point solver (what generator expects)
# ============================================================

def solve_operating_point(
    pump: PumpCurve,
    system: SystemCurve,
    speed_rpm: float,
    q_min_m3_s: float,
    q_max_m3_s: float,
    n_steps: int = 2000,
) -> float:
    """
    Brute-force search for q where pump head matches system required head.
    Returns q [m3/s] only (generator expects q only).
    """
    best_q = float(q_min_m3_s)
    best_err = float("inf")

    rho = max(float(system.density_kg_m3), 1e-9)

    for i in range(n_steps + 1):
        frac = i / max(n_steps, 1)
        q = float(q_min_m3_s) + (float(q_max_m3_s) - float(q_min_m3_s)) * frac

        head_pump = pump.head_at(q, speed_rpm)
        dp_pump = head_to_delta_p_pa(head_pump, rho)

        head_req = system.head_required_m(q_m3_s=q, dp_pump_pa=dp_pump)

        err = abs(head_pump - head_req)
        if err < best_err:
            best_err = err
            best_q = q

    return float(best_q)
