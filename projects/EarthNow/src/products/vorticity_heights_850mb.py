"""
Vorticity and Heights at 850mb Product
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from .registry import register
from wxmaps_utils import load_color_table

# ------------------------------------------------------------------
# Reflectivity colormap + levels (wxmaps-style)
# ------------------------------------------------------------------

vCOLORS = load_color_table(
        "/discover/nobackup/projects/gmao/g6dev/pub/ColorTables/idl_colortable_5_reversed.txt"
    )

vLEVELS= 60.0 * np.arange(256) / 255.0 # seconds^-1

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------

@register("vorticity_heights_850mb")
def plot_vorticity_heights_850mb(fig, ax, plotter, reader, args):
    """
    Plot Vorticity (s-1) and Heights (m) at 850mb
    """
    # Read from reader (reader decides the collection)
    vort, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["VORT850"]
    )
    vort = vort.astype(np.float32)*1.e5

    hgts, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["H850"]
    )
    hgts = hgts.astype(np.float32)

    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    cmap = ListedColormap(vCOLORS)
    norm = BoundaryNorm(vLEVELS, ncolors=cmap.N, clip=True)

    # ------------------------------------------------------------
    # Plot vorticity field
    # ------------------------------------------------------------
    ax.pcolormesh(
        lons,
        lats,
        vort,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        zorder=4,
    )


    # ------------------------------------------------------------
    # Plot height contours
    # ------------------------------------------------------------
    hlevs = np.arange(900, 2100, 30)  # 900m to 2070m every 30m
    cs = ax.contour(
        lons,
        lats,
        hgts,
        levels=hlevs,
        colors="black",
        linewidths=1.0,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )
    # labels
    ax.clabel(
         cs,
        fmt="%d",
        fontsize=9,
        inline=True,
        inline_spacing=5,
    )

def generate_colorbar():
    """Generate colorbar for 850mb vorticity/heights"""
    from wxmaps_utils import save_colorbar_single
    
    output = "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/vorticity_heights_850mb.png"
    save_colorbar_single(
        vCOLORS, 
        vLEVELS, 
        output, 
        label="850mb Relative Vorticity (×10⁻⁵ s⁻¹) with Geopotential Height Contours (m)",
        extend='max'
    )
