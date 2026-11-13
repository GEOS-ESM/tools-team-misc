"""
Observation Data Writers
"""

from .registry import OBS_WRITERS

# Import modules so they self-register (import the modules, not functions from them)
from . import obs_writers

__all__ = ["OBS_WRITERS"]
