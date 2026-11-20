"""
Observation Data Readers
"""

from .registry import OBS_TYPES

# Import modules so they self-register (import the modules, not functions from them)
from . import aerdb_l2_viirs_snpp, aqua_myd04_l2, terra_myd04_l2, omi_aura_l2_omaero

__all__ = ["OBS_TYPES"]
