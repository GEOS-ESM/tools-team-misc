"""
Ozone Longwave IR Product
9.6 micron - Ozone Band - GOES Band 12
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
    "/discover/nobackup/projects/gmao/g6dev/pub/ColorTables/NESDIS_IR_9p6micron.txt"
)

clevs = [-110., -59, -20, 6, 31, 57]  # Celsius
LEVELS = np.interp(5 * np.arange(256) / 255.0, np.arange(len(clevs)), clevs)

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------

@register("ozone_longwave_ir")
def plot_ozone_longwave_ir(fig, ax, plotter, reader, args):
    """
    Plot ozone longwave IR brightness temperature (9.6 micron)
    GOES Band 12 → TBRB07RG
    """
    # Read from reader
    data, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["TBRB07RG"]
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
    """Generate colorbar for ozone longwave IR"""
    from wxmaps_utils import save_colorbar_single
    
    output = "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/ozone_longwave_ir.png"
    save_colorbar_single(
        COLORS, 
        LEVELS, 
        output, 
        label="9.6 μm Ozone Band Brightness Temperature (°C)",
        extend='both'
    )
