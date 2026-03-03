"""
Product package for GOES WxMaps
"""

from .registry import PRODUCTS

# import modules so they self-register
from .basemap import plot_basemap
from .wxmap import plot_wxmap
from .max_reflectivity import plot_max_reflectivity
from .longwave_window_ir import plot_longwave_window_ir
from .co2_longwave_ir import plot_co2_longwave_ir
from .ozone_longwave_ir import plot_ozone_longwave_ir
from .low_level_water_vapor import plot_low_level_water_vapor
from .mid_level_water_vapor import plot_mid_level_water_vapor
from .upr_level_water_vapor import plot_upr_level_water_vapor
from .shortwave_window_ir import plot_shortwave_window_ir
from .geocolor_rgb import plot_geocolor_rgb
from .dust_rgb import plot_dust_rgb
from .temperature_2m import plot_temperature_2m
from .snow_accumulation_total import plot_snow_accumulation_total
from .precip_accumulation_total import plot_precip_accumulation_total
from .vorticity_heights_500mb import plot_vorticity_heights_500mb
from .vorticity_heights_700mb import plot_vorticity_heights_700mb
from .vorticity_heights_850mb import plot_vorticity_heights_850mb
from .slp_winds_10m import plot_slp_winds_10m
from .sea_level_pressure import plot_sea_level_pressure
from .winds_heights_250mb import plot_winds_heights_250mb
from .winds_heights_500mb import plot_winds_heights_500mb
from .winds_heights_700mb import plot_winds_heights_700mb
from .winds_heights_850mb import plot_winds_heights_850mb

__all__ = ["PRODUCTS"]

