"""
Temperature at 2-meters Product
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from .registry import register

# ------------------------------------------------------------------
# Reflectivity colormap + levels (wxmaps-style)
# ------------------------------------------------------------------

LEVELS = [ -90, -80 , -70 , -60 , -50 , -40 , -30 , -20 , -10 , 0 , 10 , 15 , 20 ,25 , 30 , 35 , 40 , 45 , 50 , 55 , 60 , 65 , 70 , 75 , 80 , 85 , 90 , 95 , 100 , 105 , 110 , 115]

COLORS = np.array([
    [ 30,  30,  75],
    [ 60,  60, 125],
    [105, 120, 190],
    [125, 150, 206],
    [153, 190, 230],
    [200, 225, 240],
    [235, 226, 235],
    [203, 139, 190],
    [180,  56, 152],
    [137,   0, 123],
    [ 90,   0, 129],
    [ 60,   0, 131],
    [ 21,   0, 135],
    [ 14,   0, 216],
    [ 26,  51, 241],
    [ 43, 129, 241],
    [ 68, 213, 240],
    [ 90, 240, 200],
    [127, 239, 136],
    [186, 239,  54],
    [235, 235,   0],
    [233, 175,   0],
    [232, 123,   0],
    [231,  46,   0],
    [218,   0,   0],
    [160,   0,   0],
    [110,   0,   0],
    [ 80,   0,   0],
    [ 65,   0,   0],
    [ 50,   0,   0],
    [ 75,  65,  65],
    [110, 110, 110],
    [180, 180, 180],
]) / 255.0

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------

@register("temperature_2m")
def plot_temperature_2m(fig, ax, plotter, reader, args):
    """
    Plot Temperature at 2-meters (F)
    """
    # Read from reader (reader decides the collection)
    data, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["TMP_2M", "T2M"]
    )
    data = (data.astype(np.float32) - 273.15)* 1.8000 + 32.0

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

    # Add city temperature labels
    plotter.add_city_temperatures(
        data, 
        lons, 
        lats, 
        temperature_unit='F'
    )

def generate_colorbar():
    """Generate colorbar for 2m temperature"""
    from wxmaps_utils import save_colorbar_single
    
    output = "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/temperature_2m.png"
    save_colorbar_single(
        COLORS, 
        LEVELS, 
        output, 
        label="2-Meter Temperature (°F)",
        extend='both'
    )
