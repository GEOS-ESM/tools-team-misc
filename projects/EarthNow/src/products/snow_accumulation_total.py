"""
Snow accumulation total for forecast Product
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from .registry import register

# ------------------------------------------------------------------
# Reflectivity colormap + levels (wxmaps-style)
# ------------------------------------------------------------------

LEVELS = [0.1,0.25,0.5,0.75,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,7,8,9,10,11,12,14,16,18,20,22,24,26,28,30,32,34,36,40,44,48,52,56,60]

COLORS = np.array([
    [221, 221, 221],
    [193, 193, 193],
    [165, 165, 165],
    [139, 139, 139],
    [160, 227, 239],
    [104, 183, 207],
    [ 53, 141, 176],
    [ 11, 101, 147],
    [  0,  61, 162],
    [ 35,  89, 175],
    [ 68, 116, 189],
    [105, 149, 202],
    [144, 182, 216],
    [185, 215, 231],
    [193, 164, 212],
    [184, 138, 201],
    [176, 115, 191],
    [166,  92, 181],
    [157,  70, 171],
    [149,  49, 162],
    [123,  16,  62],
    [139,  28,  79],
    [156,  41,  97],
    [172,  56, 116],
    [190,  72, 136],
    [207,  88, 156],
    [231, 157, 174],
    [227, 145, 151],
    [225, 132, 128],
    [222, 119, 107],
    [220, 107,  86],
    [216,  96,  67],
    [214, 118,  76],
    [219, 137,  96],
    [226, 159, 119],
    [231, 182, 145],
    [238, 203, 168],
    [243, 226, 194],
    [250, 248, 219],
]) / 255.0

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------

@register("snow_accumulation_total")
def plot_snow_accumulation_total(fig, ax, plotter, reader, args):
    """
    Plot total snow accumulation (inches)
    """
    # Read from reader (reader decides the collection)
    data, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["SNOWACCUM"],
        var_type='accum'
    )
    if data is None:
        return  False # Signal to skip this plot

    data = data.astype(np.float32)/25.4

    # Mask low values
    data = np.ma.masked_where(data < 0.1, data)

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

    # Add city data labels
    plotter.add_city_data_values(
        data,
        lons,
        lats,
    )

def generate_colorbar():
    """Generate colorbar for snow accumulation"""
    from wxmaps_utils import save_colorbar_single
    
    output = "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/snow_accumulation_total.png"
    save_colorbar_single(
        COLORS, 
        LEVELS, 
        output, 
        label="Total Snow Accumulation (inches)",
        extend='max'
    )

