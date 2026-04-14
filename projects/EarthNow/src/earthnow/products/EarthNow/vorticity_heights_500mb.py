"""
Vorticity and Heights at 500mb Product
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from earthnow.products.registry import register
from earthnow.wxmaps_utils import load_color_table

# ------------------------------------------------------------------
# Reflectivity colormap + levels (wxmaps-style)
# ------------------------------------------------------------------

vCOLORS = load_color_table(
    "/discover/nobackup/projects/gmao/g6dev/pub/ColorTables/idl_colortable_5_reversed.txt"
)

vLEVELS = 60.0 * np.arange(256) / 255.0  # seconds^-1

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------


@register("vorticity_heights_500mb_EarthNow")
def plot_vorticity_heights_500mb(fig, ax, plotter, reader, args):
    """
    Plot Vorticity (s-1) and Heights (m) at 500mb
    """
    # Read from reader (reader decides the collection)
    vort, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["VORT500"]
    )
    vort = vort.astype(np.float32) * 1.0e5

    hgts, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["H500"]
    )
    hgts = hgts.astype(np.float32)

    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    cmap = ListedColormap(vCOLORS)
    cmap_colors = cmap(np.arange(cmap.N))
    cmap_colors[:2, -1] = 0  # Set the first 2 colors transparent
    cmap = ListedColormap(cmap_colors)

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
    hlevs = np.arange(4800, 6300, 60)  # 4800m to 6240m every 60m
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
    """Generate colorbar for 500mb vorticity/heights"""
    from earthnow.wxmaps_utils import save_colorbar_single

    output = "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/vorticity_heights_500mb.png"
    save_colorbar_single(
        vCOLORS,
        vLEVELS,
        output,
        label="500mb Relative Vorticity (×10⁻⁵ s⁻¹) with Geopotential Height Contours (m)",
        extend="max",
    )
