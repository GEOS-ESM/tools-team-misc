"""
Product package for EarthNow/WxMaps
"""

# from earthnow.products.registry import PRODUCTS
from .registry import PRODUCTS

# import modules so they self-register
# COMMENTED OUT BELOW - We don't need to load all the the GEOS_WxMaps functions because we don't use them and they slow down run times
# from . import GEOS_WxMaps
from . import EarthNow

__all__ = ["PRODUCTS"]
