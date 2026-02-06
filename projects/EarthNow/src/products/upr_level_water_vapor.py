"""
Upper-Level Water Vapor Product
6.2 micron - Upper-Level Water Vapor - GOES Band 08
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from .registry import register
from wxmaps_utils import load_color_table

# ------------------------------------------------------------------
# Colormap + levels
# ------------------------------------------------------------------

COLORS = load_color_table(
    "/discover/nobackup/projects/gmao/g6dev/pub/ColorTables/NESDIS_WV_6p2micron.txt"
)

clevs = [-93., -54, -30, -18, -5, 7]  # Celsius
LEVELS = np.interp(5 * np.arange(256) / 255.0, np.arange(len(clevs)), clevs)

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------

@register("upr_level_water_vapor")
def plot_upr_level_water_vapor(fig, ax, plotter, reader, args):
    """
    Plot upper-level water vapor brightness temperature (6.2 micron)
    GOES Band 08 → TBRB11RG
    """
    # Read from reader
    data, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["TBRB11RG"]
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
    """Generate colorbar for upper-level water vapor"""
    from wxmaps_utils import save_colorbar_single
    
    output = "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/upr_level_water_vapor.png"
    save_colorbar_single(
        COLORS, 
        LEVELS, 
        output, 
        label="6.2 μm Upper-Level Water Vapor Brightness Temperature (°C)",
        extend='both'
    )
