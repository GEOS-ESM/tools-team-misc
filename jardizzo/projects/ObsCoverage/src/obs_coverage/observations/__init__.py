"""
Observation Data Readers
"""

from .registry import OBS_READERS

# Import modules so they self-register (import the modules, not functions from them)
from . import aerdb_l2_viirs_snpp

__all__ = ["OBS_READERS"]
