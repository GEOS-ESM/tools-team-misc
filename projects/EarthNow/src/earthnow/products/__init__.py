"""
Product package for EarthNow/WxMaps
"""

# from earthnow.products.registry import PRODUCTS
from .registry import PRODUCTS

# import modules so they self-register
from . import GEOS_WxMaps
from . import EarthNow

__all__ = ["PRODUCTS"]
