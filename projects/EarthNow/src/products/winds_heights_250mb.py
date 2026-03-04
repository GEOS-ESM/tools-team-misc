"""
Winds and Heights at 250mb Product
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from .registry import register
from wxmaps_utils import load_color_table

# ------------------------------------------------------------------
# Reflectivity colormap + levels (wxmaps-style)
# ------------------------------------------------------------------

# Color table (30 colors, normalized)
vCOLORS = np.array([
    [240, 240, 240],  # Light gray
    [210, 210, 210],  # Gray
    [187, 187, 187],  # Gray
    [150, 150, 150],  # Dark gray
    [ 20, 100, 210],  # Blue
    [ 40, 130, 240],  # Light blue
    [ 82, 165, 243],  # Sky blue
    [ 53, 213,  51],  # Green
    [ 80, 242,  79],  # Light green
    [147, 246, 137],  # Light green
    [203, 253, 194],  # Pale green
    [255, 246, 169],  # Pale yellow
    [255, 233, 124],  # Yellow
    [253, 193,  63],  # Yellow-orange
    [255, 161,   0],  # Orange
    [255,  95,   4],  # Dark orange
    [255,  50,   0],  # Red-orange
    [225,  20,   0],  # Red
    [197,   0,   0],  # Dark red
    [163,   0,   0],  # Darker red
    [118,  82,  70],  # Brown
    [138, 102,  90],  # Light brown
    [177, 143, 133],  # Tan
    [225, 191, 182],  # Light tan
    [238, 220, 210],  # Pale tan
    [255, 200, 200],  # Light pink
    [245, 160, 160],  # Pink
    [225, 136, 130],  # Rose
    [232, 108, 100],  # Salmon
    [229,  96,  87]   # Dark salmon
]) / 255.0

vLEVELS = [ 0 , 10, 20 , 30 , 40 , 50 , 60 , 70 , 80 , 90 , 100, 110, 120, 130, 140, 150, 160, 170 , 180 , 190 , 200 , 210 , 220 , 230, 240 ]

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------

@register("winds_heights_250mb")
def plot_winds_heights_250mb(fig, ax, plotter, reader, args):
    """
    Plot Winds (knots) and Heights (m) at 250mb
    """
    # Read from reader (reader decides the collection)
    uwnd, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["U250"]
    )
    uwnd = uwnd.astype(np.float32)*1.94384

    vwnd, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["V250"]
    )
    vwnd = vwnd.astype(np.float32)*1.94384

    wspd = np.sqrt(uwnd**2 + vwnd**2)
    print(f"wspd: min={np.nanmin(wspd):.6f}, max={np.nanmax(wspd):.6f}, mean={np.nanmean(wspd):.6f}")

    hgts, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["H250"]
    )
    hgts = hgts.astype(np.float32)/10.0

    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    cmap = ListedColormap(vCOLORS)
    norm = BoundaryNorm(vLEVELS, ncolors=cmap.N, clip=True)

    # ------------------------------------------------------------
    # Plot wind field
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
    # Plot height contours
    # ------------------------------------------------------------
    hlevs = np.arange(960, 1146, 6)  # 9600m to 11400m every 60m
    cs = ax.contour(
        lons,
        lats,
        hgts,
        levels=hlevs,
        colors="darkslategrey",
        linewidths=2.0,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )
    # Create labels
    clabels = ax.clabel(
        cs,
        fmt="%d",
        fontsize=12,
        inline=True,
        inline_spacing=5,
    )
    # Make labels bold/thicker
    for label in clabels:
        label.set_fontweight('bold')

def generate_colorbar():
    """Generate colorbar for 250mb winds/heights"""
    from wxmaps_utils import save_colorbar_single
    
    output = "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/winds_heights_250mb.png"
    save_colorbar_single(
        vCOLORS, 
        vLEVELS, 
        output, 
        label="250mb Wind Speed (knots) with Geopotential Height Contours (dam)",
        extend='max'
    )

