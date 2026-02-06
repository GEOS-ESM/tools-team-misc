#!/usr/bin/env python3
"""
Generate colorbar PNGs for all WxMaps products
"""

import sys
sys.path.insert(0, '/discover/nobackup/projects/gmao/g6dev/pub/WxMaps')

print("Generating WxMaps colorbars...")
print("=" * 60)

# wxmap (2x2 grid)
from products.wxmap import generate_colorbar as gen_wxmap
print("\n1. wxmap (2x2 grid)")
gen_wxmap()

# Single colorbars
from products.max_reflectivity import generate_colorbar as gen_refl
print("\n2. max_reflectivity")
gen_refl()

from products.longwave_window_ir import generate_colorbar as gen_lw
print("\n3. longwave_window_ir")
gen_lw()

from products.co2_longwave_ir import generate_colorbar as gen_co2
print("\n4. co2_longwave_ir")
gen_co2()

from products.ozone_longwave_ir import generate_colorbar as gen_ozone
print("\n5. ozone_longwave_ir")
gen_ozone()

from products.low_level_water_vapor import generate_colorbar as gen_low_wv
print("\n6. low_level_water_vapor")
gen_low_wv()

from products.mid_level_water_vapor import generate_colorbar as gen_mid_wv
print("\n7. mid_level_water_vapor")
gen_mid_wv()

from products.upr_level_water_vapor import generate_colorbar as gen_upr_wv
print("\n8. upr_level_water_vapor")
gen_upr_wv()

from products.shortwave_window_ir import generate_colorbar as gen_sw
print("\n9. shortwave_window_ir")
gen_sw()

from products.temperature_2m import generate_colorbar as gen_t2m
print("\n10. temperature_2m")
gen_t2m()

from products.snow_accumulation_total import generate_colorbar as gen_snow
print("\n11. snow_accumulation_total")
gen_snow()

from products.precip_accumulation_total import generate_colorbar as gen_precip
print("\n12. precip_accumulation_total")
gen_precip()

from products.slp_winds_10m import generate_colorbar as gen_slp
print("\n13. slp_winds_10m")
gen_slp()

from products.vorticity_heights_500mb import generate_colorbar as gen_v500
print("\n14. vorticity_heights_500mb")
gen_v500()

from products.vorticity_heights_700mb import generate_colorbar as gen_v700
print("\n15. vorticity_heights_700mb")
gen_v700()

from products.vorticity_heights_850mb import generate_colorbar as gen_v850
print("\n16. vorticity_heights_850mb")
gen_v850()

from products.vorticity_slp_850mb import generate_colorbar as gen_vslp850
print("\n17. vorticity_slp_850mb")
gen_vslp850()

print("\n" + "=" * 60)
print("All colorbars generated successfully!")
print("Location: /discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/")
