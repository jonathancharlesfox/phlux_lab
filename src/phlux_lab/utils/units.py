
# =========================================================
# Script explaination at bottom of the py file
# =========================================================

from __future__ import annotations

# =========================================================
# Pressure Conversions (internal: Pa)
# =========================================================

def _kpa_to_pa(kpa: float) -> float:
    return kpa * 1_000.0

def _bar_to_pa(bar: float) -> float:
    return bar * 100_000.0

def _psi_to_pa(psi: float) -> float:
    return psi * 6_894.75729

def _pa_to_kpa(pa: float) -> float:
    return pa / 1_000.0

def _pa_to_bar(pa: float) -> float:
    return pa / 100_000.0

def _pa_to_psi(pa: float) -> float:
    return pa / 6_894.75729

# =========================================================
# Flow Conversions (internal: m3/s)
# =========================================================

def _m3h_to_m3s(m3h: float) -> float:
    # 1 h = 3_600 s
    return m3h / 3_600.0

def _m3s_to_m3h(m3s: float) -> float:
    return m3s * 3_600.0

def _bbl_day_to_m3s(bbl_d: float) -> float:
    # 1 m3 = 6.28981 bbl
    # 1 day = 86_400 s
    return (bbl_d / 6.28981) / 86_400.0

def _m3s_to_bbl_day(m3s: float) -> float:
    return (m3s * 6.28981) * 86_400.0

def _Lps_to_m3s(Lps: float) -> float:
    # 1 L = 1e-3 m3
    return Lps * 1e-3

def _m3s_to_Lps(m3s: float) -> float:
    return m3s / 1e-3

# =========================================================
# Temperature Conversions (internal: K)
# =========================================================

def _C_to_K(c: float) -> float:
    return c + 273.15

def _K_to_C(K: float) -> float:
    return K - 273.15

def _F_to_K(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0 + 273.15

def _K_to_F(K: float) -> float:
    return (K - 273.15) * 9.0 / 5.0 + 32.0

# =========================================================
# Density Conversions (internal: kg/m3)
# =========================================================

def _gcm3_to_kgm3(gcm3: float) -> float:
    # 1 g/cm3 = 1000 kg/m3
    return gcm3 * 1_000.0

def _kgm3_to_gcm3(kgm3: float) -> float:
    return kgm3 / 1_000.0

def _lbft3_to_kgm3(lbft3: float) -> float:
    # 1 kg/m3 = 0.062428 lb/ft3
    return lbft3 / 0.062428

def _kgm3_to_lbft3(kgm3: float) -> float:
    return kgm3 * 0.062428

# =========================================================
# Viscosity Conversions (internal: Pa*s)
# =========================================================

def _cp_to_pas(cp: float) -> float:
    # 1 cP = 0.001 Pa*s
    return cp * 1e-3

def _pas_to_cp(pas: float) -> float:
    # 1 Pa*s = 1000 cP
    return pas * 1_000.0

# =========================================================
# Power Conversions (internal: W)
# =========================================================

def _kW_to_W(kw: float) -> float:
    return kw * 1_000.0

def _W_to_kW(W: float) -> float:
    return W / 1_000.0

def _hp_to_W(hp: float) -> float:
    # Mechanical horsepower
    return hp * 745.699872

def _W_to_hp(W: float) -> float:
    return W / 745.699872

# =========================================================
# Speed Conversions (internal: rpm)
# =========================================================

def _rps_to_rpm(rps: float) -> float:
    return rps * 60.0

def _rpm_to_rps(rpm: float) -> float:
    return rpm / 60.0


# =========================================================
# Length / Size Conversions (internal: m)
# =========================================================

def _mm_to_m(mm: float) -> float:
    return mm / 1_000.0

def _cm_to_m(cm: float) -> float:
    return cm / 100.0

def _in_to_m(inches: float) -> float:
    return inches * 0.0254

def _ft_to_m(ft: float) -> float:
    return ft * 0.3048

def _m_to_mm(m: float) -> float:
    return m * 1_000.0

def _m_to_cm(m: float) -> float:
    return m * 100.0

def _m_to_in(m: float) -> float:
    return m / 0.0254

def _m_to_ft(m: float) -> float:
    return m / 0.3048



# =========================================================
# Canonical internal units (the unit system used inside Phlux)
# =========================================================

CANONICAL_INTERNAL_UNITS: dict[str, str] = {
    "dimensionless": "1",
    "pressure": "Pa",
    "flow": "m3/s",
    "temperature": "K",
    "density": "kg/m3",
    "viscosity": "Pa*s",
    "speed": "rpm",
    "power": "W",
    "length": "m",
}

def unit_dimension(unit: str) -> str:
    """
    Map a unit string to a high-level physical dimension.

    This is used for schema/export metadata and runtime validation.
    """
    u = unit.strip()

    if u in ("frac", "fraction", "-", "1", "%", "pct", "percent"):
        return "dimensionless"

    if u in ("Pa", "kPa", "bar", "bara", "barg", "psi", "psia", "psig"):
        return "pressure"

    if u in ("m3/s", "m^3/s", "m3/h", "m^3/h", "m3/hr", "m^3/hr", "bbl/d", "stb/d", "STB/d", "bpd", "L/s", "l/s"):
        return "flow"

    if u in ("K", "degC", "C", "°C", "degF", "F", "°F"):
        return "temperature"

    if u in ("kg/m3", "kg/m^3", "g/cm3", "g/cc", "lb/ft3", "lb/ft^3"):
        return "density"

    if u in ("Pa*s", "Pa.s", "cP"):
        return "viscosity"

    if u in ("rpm", "rps"):
        return "speed"

    if u in ("W", "kW", "hp", "HP"):
        return "power"

    if u in ("m", "meter", "meters", "mm", "cm", "in", "inch", "inches", "ft", "feet"):
        return "length"

    raise ValueError(f"Unknown/unsupported unit for dimension inference: '{unit}'")

# =========================================================
# Public API
# =========================================================

def convert_user_to_internal_SI(value: float, unit: str) -> float:
    """
    Convert a user-facing value with the given unit string
    into the INTERNAL SI-style basis used by the physics equations:
     
        Pressure:   Pa
        Flow:       m3/s
        Temp:       K
        Density:    kg/m3
        Viscosity:  Pa*s
        Speed:      rpm
        Power:      W

    """
    u = unit.strip()


    # ---------------- Dimensionless ----------------
    if u in ("frac", "fraction", "-", "1"):
        return float(value)
    if u in ("%", "pct", "percent"):
        return float(value) / 100.0

    # ---------------- Pressure -> Pa ----------------
    if u in ("Pa",):
        return value
    if u in ("kPa",):
        return _kpa_to_pa(value)
    if u in ("bar", "bara", "barg"):  # treat all as same here
        return _bar_to_pa(value)
    if u in ("psi", "psia", "psig"):
        return _psi_to_pa(value)

    # ---------------- Flow -> m3/s ------------------
    if u in ("m3/s", "m^3/s"):
        return value
    if u in ("m3/h", "m^3/h", "m3/hr", "m^3/hr"):
        return _m3h_to_m3s(value)
    if u in ("bbl/d", "stb/d", "STB/d", "bpd"):
        return _bbl_day_to_m3s(value)
    if u in ("L/s", "l/s"):
        return _Lps_to_m3s(value)

    # -------------- Temperature -> K ----------------
    if u in ("K",):
        return value
    if u in ("degC", "C", "°C"):
        return _C_to_K(value)
    if u in ("degF", "F", "°F"):
        return _F_to_K(value)

    # -------------- Density -> kg/m3 ----------------
    if u in ("kg/m3", "kg/m^3"):
        return value
    if u in ("g/cm3", "g/cc"):
        return _gcm3_to_kgm3(value)
    if u in ("lb/ft3", "lb/ft^3"):
        return _lbft3_to_kgm3(value)

    # -------------- Viscosity -> Pa*s ---------------
    if u in ("Pa*s", "Pa.s"):
        return value
    if u in ("cP",):
        return _cp_to_pas(value)

    # -------------- Speed -> rpm --------------------
    if u in ("rpm",):
        return value
    if u in ("rps",):
        return _rps_to_rpm(value)

    # -------------- Power -> W ----------------------
    if u in ("W",):
        return value
    if u in ("kW",):
        return _kW_to_W(value)
    if u in ("hp", "HP"):
        return _hp_to_W(value)

    # -------------- Length -> m ---------------------
    if u in ("m", "meter", "meters"):
        return float(value)
    if u in ("mm",):
        return _mm_to_m(float(value))
    if u in ("cm",):
        return _cm_to_m(float(value))
    if u in ("in", "inch", "inches"):
        return _in_to_m(float(value))
    if u in ("ft", "feet"):
        return _ft_to_m(float(value))

    raise ValueError(f"Unsupported unit in convert_user_to_internal_SI(): '{unit}'")

def convert_internal_back_to_user(value: float, unit: str) -> float:
    """
    Convert a value FROM the INTERNAL SI-style basis:

        Pressure:   Pa
        Flow:       m3/s
        Temp:       K
        Density:    kg/m3
        Viscosity:  Pa*s
        Speed:      rpm
        Power:      W

    """
    u = unit.strip()


    # ---------------- Dimensionless ----------------
    if u in ("frac", "fraction", "-", "1"):
        return float(value)
    if u in ("%", "pct", "percent"):
        return float(value) * 100.0

    # ---------------- Pa -> Pressure ----------------
    if u in ("Pa",):
        return value
    if u in ("kPa",):
        return _pa_to_kpa(value)
    if u in ("bar", "bara", "barg"):
        return _pa_to_bar(value)
    if u in ("psi", "psia", "psig"):
        return _pa_to_psi(value)

    # ---------------- m3/s -> Flow ------------------
    if u in ("m3/s", "m^3/s"):
        return value
    if u in ("m3/h", "m^3/h", "m3/hr", "m^3/hr"):
        return _m3s_to_m3h(value)
    if u in ("bbl/d", "stb/d", "STB/d", "bpd"):
        return _m3s_to_bbl_day(value)
    if u in ("L/s", "l/s"):
        return _m3s_to_Lps(value)

    # ---------------- K -> Temperature -------------
    if u in ("K",):
        return value
    if u in ("degC", "C", "°C"):
        return _K_to_C(value)
    if u in ("degF", "F", "°F"):
        return _K_to_F(value)

    # -------------- kg/m3 -> Density ----------------
    if u in ("kg/m3", "kg/m^3"):
        return value
    if u in ("g/cm3", "g/cc"):
        return _kgm3_to_gcm3(value)
    if u in ("lb/ft3", "lb/ft^3"):
        return _kgm3_to_lbft3(value)

    # -------------- Pa*s -> Viscosity --------------
    if u in ("Pa*s", "Pa.s"):
        return value
    if u in ("cP",):
        return _pas_to_cp(value)

    # -------------- rpm -> Speed --------------------
    if u in ("rpm",):
        return value
    if u in ("rps",):
        return _rpm_to_rps(value)

    # -------------- W -> Power ----------------------
    if u in ("W",):
        return value
    if u in ("kW",):
        return _W_to_kW(value)
    if u in ("hp", "HP"):
        return _W_to_hp(value)

    # -------------- m -> Length ---------------------
    if u in ("m", "meter", "meters"):
        return float(value)
    if u in ("mm",):
        return _m_to_mm(float(value))
    if u in ("cm",):
        return _m_to_cm(float(value))
    if u in ("in", "inch", "inches"):
        return _m_to_in(float(value))
    if u in ("ft", "feet"):
        return _m_to_ft(float(value))

    raise ValueError(f"Unsupported unit in convert_internal_back_to_user(): '{unit}'")

"""
units.py
---------------------------------------------------------
Central unit conversion utilities for Phlux.

INTERNAL / PHYSICS BASIS (used by generator + physics):

    Pressure      -> Pa
    Flow rate     -> m3/s
    Temperature   -> K
    Density       -> kg/m3
    Viscosity     -> Pa*s
    Speed         -> rpm
    Power         -> W

The idea:

- User specifies ranges + units in synthgen_config.yaml
- generator.py uses:
    convert_user_to_internal_SI(value, unit)
  to map user units -> internal SI basis for all physics equations.

- When exporting raw data (for human inspection), you can map
  back from internal SI to user-specified units with:
    convert_internal_back_to_user(value, unit)

Supported (common) units:

    Pressure:
        Pa, kPa, bar, bara, psi, psia

    Flow:
        m3/s, m3/h, bbl/d, stb/d, STB/d, L/s

    Temperature:
        K, degC, C, degF, F

    Density:
        kg/m3, g/cm3, lb/ft3

    Viscosity:
        Pa*s, Pa.s, cP

    Speed:
        rpm, rps

    Power:
        W, kW, hp

    Length / size:
        m, mm, cm, in, ft
---------------------------------------------------------
"""