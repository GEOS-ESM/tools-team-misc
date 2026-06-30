"""
Themes package for WxVis
"""

from wxvis.themes.registry import THEMES

# import modules so they self-register
from wxvis.themes.theme import theme
from wxvis.themes.wxmapsclassicpub import wxmapsclassicpub

__all__ = ["THEMES"]
