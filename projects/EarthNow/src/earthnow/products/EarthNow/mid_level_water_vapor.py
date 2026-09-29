"""
Mid-Level Water Vapor Product
6.9 micron - Mid-Level Water Vapor - GOES Band 09
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from earthnow.products.registry import register
from earthnow.wxmaps_utils import load_color_table
from earthnow import paths

variable = "mid_level_water_vapor_EarthNow"
create_colorbar = True

# ------------------------------------------------------------------
# Colormap + levels
# ------------------------------------------------------------------

COLORS = load_color_table(paths.colortable("NESDIS_WV_6p9micron.txt"))

clevs = [-93.0, -54, -30, -18, -5, 7]  # Celsius
LEVELS = np.interp(5 * np.arange(256) / 255.0, np.arange(len(clevs)), clevs)

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------


@register("mid_level_water_vapor_EarthNow")
def plot_mid_level_water_vapor(fig, ax, plotter, reader, args):
    """
    Plot mid-level water vapor brightness temperature (6.9 micron)
    GOES Band 09 → TBRB10RG
    """
    # Read from reader
    data, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["TBRB10RG"]
    )
    data = data.astype(np.float32) - 273.15  # Celsius

    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(LEVELS, ncolors=cmap.N, clip=True)

    # ------------------------------------------------------------
    # Plot field
    # ------------------------------------------------------------
    plot = ax.pcolormesh(
        lons,
        lats,
        data,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        zorder=4,
    )

    if create_colorbar == True:
        """Generate colorbar for mid-level water vapor"""
        from earthnow.wxmaps_utils import save_colorbar_single

        colorbar_output = (
            f"/discover/nobackup/eibell/EarthNow/plots/{variable}_colorbar.png"
        )

        save_colorbar_single(
            plot,
            colorbar_output,
            label="6.9 micron - Mid-Level Water Vapor - IR",
            ticks=clevs,
        )
