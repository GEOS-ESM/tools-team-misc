"""
Shortwave Window IR Product
3.9 micron - Shortwave Window - GOES Band 07
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
    "/discover/nobackup/projects/gmao/g6dev/pub/ColorTables/NESDIS_IR_3p9micron.txt"
)

clevs = [-110., -59, -20, 6, 31, 57]  # Celsius
LEVELS = np.interp(5 * np.arange(256) / 255.0, np.arange(len(clevs)), clevs)

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------

@register("shortwave_window_ir")
def plot_shortwave_window_ir(fig, ax, plotter, reader, args):
    """
    Plot shortwave window IR brightness temperature (3.9 micron)
    GOES Band 07 → TBRB15RG
    """
    # Read from reader
    data, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["TBRB15RG"]
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
    """Generate colorbar for shortwave window IR"""
    from wxmaps_utils import save_colorbar_single
    
    output = "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/shortwave_window_ir.png"
    save_colorbar_single(
        COLORS, 
        LEVELS, 
        output, 
        label="3.9 μm Shortwave Window Brightness Temperature (°C)",
        extend='both'
    )
