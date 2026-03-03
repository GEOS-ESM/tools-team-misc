"""
Radar Reflectivity Product
Maximum Composite Reflectivity (DBZ_MAX / REFC)
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from .registry import register
from wxmaps_utils import load_color_table

# ------------------------------------------------------------------
# Reflectivity colormap + levels (wxmaps-style)
# ------------------------------------------------------------------

COLORS = load_color_table(
        "/discover/nobackup/projects/gmao/g6dev/pub/ColorTables/NESDIS_IR_10p3micron.txt"
    )

clevs = [-110., -59, -20, 6, 31, 57] # Celcius
LEVELS = np.interp(5 * np.arange(256) / 255.0, np.arange(len(clevs)), clevs)

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------

@register("longwave_window_ir")
def plot_longwave_window_ir(fig, ax, plotter, reader, args):
    """
    Plot longwave window IR (C)
    """
    # Read from reader (reader decides the collection)
    data, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["TBRB06RG"]
    )
    data = data.astype(np.float32) - 273.15 # Celcius


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
    from wxmaps_utils import save_colorbar_single
    
    # Use representative tick levels instead of all 256
    tick_levels = np.array([-110, -80, -50, -20, 0, 20, 40, 57])
    
    output = "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/longwave_window_ir.png"
    save_colorbar_single(
        COLORS, 
        LEVELS, 
        output, 
        label="11.2 μm Longwave Window Brightness Temperature (°C)",
        extend='both'
    )
