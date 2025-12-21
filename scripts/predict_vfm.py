from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from phlux_lab.utils.predictor import VfmPredictor

# ============================================================
# USER SETTINGS (easy to edit)
# ============================================================

MODEL_TO_USE = r"models\ClientA\vfm_ClientA.keras"
PREPROCESSOR_TO_USE = r"models\ClientA\preprocessor.joblib"
RETURN_UNITS = "user"   # options: "user", "canonical"


# YAML-like structure: you explicitly declare (value, unit) per input.
# These units can be DIFFERENT than the dataset schema units.
FEATURE_INPUTS = {
    "suction_pressure":   {"value": 2.547,   "unit": "bar"},
    "discharge_pressure": {"value": 3.644,   "unit": "bar"},
    "delta_pressure":     {"value": 1.096,   "unit": "bar"},
    "speed":              {"value": 1800.0,  "unit": "rpm"},
    "temperature":        {"value": 19.388,  "unit": "degC"},
    "fluid_density":      {"value": 1005.0,  "unit": "kg/m3"},
    "fluid_viscosity":    {"value": 0.91,    "unit": "cP"},
    "pump_power":         {"value": 6.11,    "unit": "kW"},
    "valve_opening":      {"value": 0.78155, "unit": "frac"},
    "sink_pressure":      {"value": 1.512,   "unit": "bar"},

    # Optional: you can provide derived features too (if your model expects them),
    # but you usually DON'T need to because Preprocessor.transform_X(df) can compute them.
    # "valve_dp": {"value": 2.1, "unit": "bar"},
}

# Optional: if you want to post-process a probability from hydraulic_wear
# (only used if your model predicts hydraulic_wear).
WEAR_PROBABILITY = {
    "enabled": True,
    "method": "linear",   # "linear" only here
    "wear_lo": 0.04,
    "wear_hi": 0.08,
}


def main() -> None:
    predictor = VfmPredictor.from_paths(
        model_path=MODEL_TO_USE,
        preprocessor_path=PREPROCESSOR_TO_USE,
    )

    predictor.debug_print_model_inputs(FEATURE_INPUTS)

    pred = predictor.predict(
        feature_inputs=FEATURE_INPUTS,
        wear_probability_cfg=WEAR_PROBABILITY if WEAR_PROBABILITY.get("enabled", False) else None,
        return_units=RETURN_UNITS,
    )

    print("\nPredicted outputs:")
    for name, value in pred.items():
        # Keep formatting flexible (some outputs might be strings)
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.6f}")
        else:
            print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
