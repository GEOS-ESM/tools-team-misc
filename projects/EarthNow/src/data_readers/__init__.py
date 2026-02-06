"""
Data reader package for WxMaps
"""

from .registry import DATA_READERS

# Import modules so they self-register (import the modules, not functions from them)
from . import geos_cycled_replays
from . import geos_forward_processing
from . import gencast_geos_fp

__all__ = ["DATA_READERS"]

