"""
Based on
/home/wputman/IDL_BASE/ploteic_sandwich.pro
NOTE: This product, like radar reflectivity, appears to read data
directly from cube-sphere files and interpolate them in situ
to a lat/lon grid. 
Uses IDL method called `read_and_interpolate_cube2` which has
not yet been translated to Python.
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from earthnow.products.registry import register
from earthnow.wxmaps_utils import load_color_table
from earthnow import paths

# ------------------------------------------------------------------
# Reflectivity colormap + levels (wxmaps-style)
# ------------------------------------------------------------------

COLORS = load_color_table(paths.colortable("NESDIS_IR_10p3micron.txt"))

clevs = [-110.0, -59, -20, 6, 31, 57]  # Celcius
LEVELS = np.interp(5 * np.arange(256) / 255.0, np.arange(len(clevs)), clevs)

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------


@register("blended_LWIR_veg_EarthNow")
def plot_blended_LWIR_veg(fig, ax, plotter, reader, args):
    """
    Plot longwave window IR and Veggie VIS
    """
    # Read from reader (reader decides the collection)
    lwir, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["TBRB06RG"]
    )
    lwir = data.astype(np.float32) - 273.15  # Celcius
    # lccdataG in IDL

    swtdn, lats, lons, meta = reader.read_varaible(
        args.fdate,
        args.pdate,
        variables=["SWTDN"],
    )
    # lccdata_a in IDL

    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(LEVELS, ncolors=cmap.N, clip=True)

    # ------------------------------------------------------------
    # Plot field
    # ------------------------------------------------------------
    ax.pcolormesh(
        lons,
        lats,
        data,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        zorder=4,
    )


def generate_colorbar():
    """Generate colorbar for longwave window IR"""
    from earthnow.wxmaps_utils import save_colorbar_single

    # Use representative tick levels instead of all 256
    tick_levels = np.array([-110, -80, -50, -20, 0, 20, 40, 57])

    output = paths.colorbar_output("longwave_window_ir.png")
    save_colorbar_single(
        COLORS,
        LEVELS,
        output,
        label="11.2 μm Longwave Window Brightness Temperature (°C)",
        extend="both",
    )
