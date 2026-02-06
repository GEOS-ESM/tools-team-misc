"""
precip accumulation total for forecast Product
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from .registry import register

# ------------------------------------------------------------------
# Reflectivity colormap + levels (wxmaps-style)
# ------------------------------------------------------------------

LEVELS = [0.01,0.1,0.25,0.5,1,1.5,2,3,4,5,6,8,10,12,14,16,18,20,22,24,30]

COLORS = np.array([
    [255, 255, 255],
    [175, 210, 235],
    [130, 160, 230],
    [ 90, 105, 220],
    [ 65, 175,  45],
    [ 95, 215,  75],
    [140, 240, 130],
    [175, 255, 165],
    [250, 215,  30],
    [245, 150,  30],
    [230,  30,  30],
    [200,   0,  25],
    [140,   0,  10],
    [205, 130, 130],
    [230, 185, 185],
    [245, 225, 225],
    [215, 200, 230],
    [185, 165, 210],
    [155, 125, 185],
    [140, 100, 170],
    [120,  70, 160],
]) / 255.0

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------

@register("precip_accumulation_total")
def plot_precip_accumulation_total(fig, ax, plotter, reader, args):
    """
    Plot total precip accumulation (inches)
    """
    # Read from reader (reader decides the collection)
    data, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["APCP","PRECACCUM"],
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
    """Generate colorbar for precipitation accumulation"""
    from wxmaps_utils import save_colorbar_single
    
    output = "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/precip_accumulation_total.png"
    save_colorbar_single(
        COLORS, 
        LEVELS, 
        output, 
        label="Total Precipitation Accumulation (inches)",
        extend='max'
    )

