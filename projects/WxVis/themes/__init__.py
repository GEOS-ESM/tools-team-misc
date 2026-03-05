"""
Themes package for WxVis
"""

from .registry import THEMES

# import modules so they self-register
from .theme import theme
from .wxmapsclassicpub import wxmapsclassicpub

__all__ = ["THEMES"]

