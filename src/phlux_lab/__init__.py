"""
phlux package: synthetic data generation + ML-based virtual flow metering
"""

# Expose ML public API
from .ml.vfm_model import VFMModel
from .utils.preprocessor import Preprocessor

# Expose datagen factory if desired
from .datagen.factory import create_generator


__all__ = [
    "VFMModel",
    "Preprocessor",
    "create_generator",
]
