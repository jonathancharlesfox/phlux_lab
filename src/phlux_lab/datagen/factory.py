"""
Factory for creating synthetic data generators based on the YAML config.

Expected config structure:

pump:
  family: "centrifugal"     # or "positivedisplacement", "compressor"
  id: "P1"
  curves:
    - id: "P1"
      base_speed_rpm: 1800
      curve_points: [...]
      power_curve_points: [...]

This factory returns the correct generator class so the user can call:

gen = create_generator(cfg)
df  = gen.generate_dataset()
"""

from __future__ import annotations

from typing import Any, Dict

# Import available generator classes
from .centrifugal.generator import CentrifugalPumpDatasetGenerator
# If you add these modules later, uncomment:
# from .positivedisplacement.generator import PositiveDisplacementPumpDatasetGenerator
# from .compressor.generator import CompressorDatasetGenerator


def create_generator(cfg: Dict[str, Any]):
    """
    Create the correct synthetic data generator instance based on the config.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration dictionary.

    Returns
    -------
    generator : object
        An instance of the appropriate dataset generator.

    Raises
    ------
    ValueError if pump.family is missing or unknown.
    """

    # Ensure pump.family exists
    if "pump" not in cfg or "family" not in cfg["pump"]:
        raise ValueError(
            "Configuration missing 'pump.family'. Example:\n"
            "pump:\n"
            "  family: centrifugal"
        )

    family = cfg["pump"]["family"].lower().strip()

    # --- CENTRIFUGAL PUMP ---
    if family == "centrifugal":
        return CentrifugalPumpDatasetGenerator(cfg)

    # --- POSITIVE DISPLACEMENT PUMP ---
    elif family in ("positivedisplacement", "pd", "positive_displacement"):
        raise NotImplementedError(
            "Positive displacement generator not implemented yet.\n"
            "Add module: phlux/datagen/positivedisplacement/generator.py"
        )
        # return PositiveDisplacementPumpDatasetGenerator(cfg)

    # --- COMPRESSOR ---
    elif family == "compressor":
        raise NotImplementedError(
            "Compressor generator not implemented yet.\n"
            "Add module: phlux/datagen/compressor/generator.py"
        )
        # return CompressorDatasetGenerator(cfg)

    # --- UNKNOWN ---
    else:
        raise ValueError(
            f"Unknown pump.family '{family}'. Supported families:\n"
            "- centrifugal\n"
            "- positivedisplacement (pd)\n"
            "- compressor\n"
        )
