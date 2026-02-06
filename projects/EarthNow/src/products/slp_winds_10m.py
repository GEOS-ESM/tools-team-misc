"""
SLP at Winds at 10m Product
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
from .registry import register
from wxmaps_utils import load_color_table
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

# ------------------------------------------------------------------
# Reflectivity colormap + levels (wxmaps-style)
# ------------------------------------------------------------------

COLORS = np.array([
    [240, 240, 240],
    [210, 210, 210],
    [187, 187, 187],
    [150, 150, 150],
    [ 20, 100, 210],
    [ 40, 130, 240],
    [ 82, 165, 243],
    [ 53, 213,  51],
    [ 80, 242,  79],
    [147, 246, 137],
    [203, 253, 194],
    [255, 246, 169],
    [255, 233, 124],
    [253, 193,  63],
    [255, 161,   0],
    [255,  95,   4],
    [255,  50,   0],
    [225,  20,   0],
    [197,   0,   0],
    [163,   0,   0],
    [118,  82,  70],
    [138, 102,  90],
    [177, 143, 133],
    [225, 191, 182],
    [238, 220, 210],
    [255, 200, 200],
    [245, 160, 160],
    [225, 136, 130],
    [232, 108, 100],
    [229,  96,  87],
], dtype=float) / 255.0

LEVELS = [ 1 , 3 , 6 , 9 , 12 , 15 , 18, 20 , 22 , 24 , 27 , 30 , 35 , 40 , 45 , 50 , 55 , 60 , 65 , 70 , 75 , 80 , 85 , 90 , 100 , 105 , 110 , 120 , 135 ]

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------

@register("slp_winds_10m")
def plot_slp_winds_10m(fig, ax, plotter, reader, args):
    """
    Plot SLP and winds at 10m
    """
    # Read from reader (reader decides the collection)
    u10m, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["UGRD_10M","U10M"]
    )
    u10m = u10m.astype(np.float32)*2.23694
    v10m, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["VGRD_10M","V10M"]
    )
    v10m = v10m.astype(np.float32)*2.23694
    wspd = np.sqrt(u10m**2 + v10m**2)
    # Mask invalid wspd
    wspd = np.ma.masked_where(wspd < 1, wspd)

    #phis, lats, lons, meta = reader.read_variable(
    #    args.fdate,
    #    args.pdate,
    #    variables=["HGT_SFC","PHIS"]
    #)
    #phis = phis.astype(np.float32)/9.81

    slp, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["PRMSL","SLP"]
    )
    slp = slp.astype(np.float32)/100.0
    # mask and smooth
    slp = gaussian_filter(slp, sigma=4.0)
    #slp = np.ma.masked_where(phis > 1000, slp)

    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(LEVELS, ncolors=cmap.N, clip=True)

    # ------------------------------------------------------------
    # Plot vorticity field
    # ------------------------------------------------------------
    ax.pcolormesh(
        lons,
        lats,
        wspd,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        zorder=4,
    )

    # ------------------------------------------------------------
    # Plot SLP contours
    # ------------------------------------------------------------
    clevs = np.concatenate([
        np.arange(860, 980, 12),   # Low pressure: 900-960 mb (every 20mb)
        np.arange(980, 1080, 4)    # Normal range: 980-1080 mb (every 4mb)
    ])
    colors = plt.cm.RdBu(np.linspace(0, 1, len(clevs)))
    cmap = ListedColormap(colors)
    contour_colors = [cmap(i / (len(clevs) - 1)) for i in range(len(clevs))]

    cs = ax.contour(
        lons,
        lats,
        slp,
        levels=clevs,
        colors="darkslategrey",
        linewidths=1.25,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )
    # Create labels
    clabels = ax.clabel(
        cs,
        fmt="%d",
        fontsize=9,
        inline=True,
        inline_spacing=5,
    )
    # Make labels bold/thicker
    for label in clabels:
        label.set_fontweight('bold')

def generate_colorbar():
    """Generate colorbar for SLP/winds at 10m"""
    from wxmaps_utils import save_colorbar_single
    
    output = "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/slp_winds_10m.png"
    save_colorbar_single(
        COLORS, 
        LEVELS, 
        output, 
        label="10-Meter Wind Speed (mph) with Sea Level Pressure Contours (mb)",
        extend='max'
    )
