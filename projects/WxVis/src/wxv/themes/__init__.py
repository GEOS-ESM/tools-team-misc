"""
Themes package for WxVis
"""

from wxv.themes.registry import THEMES

# import modules so they self-register
from wxv.themes.theme import theme
from wxv.themes.wxmapsclassicpub import wxmapsclassicpub

__all__ = ["THEMES"]

