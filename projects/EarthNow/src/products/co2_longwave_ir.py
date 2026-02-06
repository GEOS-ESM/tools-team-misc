"""
CO2 Longwave IR Product
13.3 micron - CO2 Longwave Band - GOES Band 16
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
    "/discover/nobackup/projects/gmao/g6dev/pub/ColorTables/ColorBar450Band16_horz.txt"
)

clevs = [-87.,-60,-30,-15,0,4,12,24]  # Celsius
LEVELS = np.interp(7 * np.arange(256) / 255.0, np.arange(len(clevs)), clevs)

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------

@register("co2_longwave_ir")
def plot_co2_longwave_ir(fig, ax, plotter, reader, args):
    """
    Plot CO2 longwave IR brightness temperature (13.3 micron)
    GOES Band 16 → TBRB05RG
    """
    # Read from reader
    data, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["TBRB05RG"]
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
    """Generate colorbar for CO2 longwave IR"""
    from wxmaps_utils import save_colorbar_single
    
    output = "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/co2_longwave_ir.png"
    save_colorbar_single(
        COLORS, 
        LEVELS, 
        output, 
        label="13.3 μm CO2 Longwave Brightness Temperature (°C)",
        extend='both'
    )
