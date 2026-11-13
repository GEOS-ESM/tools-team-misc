"""
Observation Data Readers
"""

from .registry import OBS_TYPES

# Import modules so they self-register (import the modules, not functions from them)
from . import aerdb_l2_viirs_snpp, aqua_myd04_l2

__all__ = ["OBS_TYPES"]
