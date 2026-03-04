"""
Vorticity and SLP at 850mb Product
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from .registry import register
from wxmaps_utils import load_color_table
from scipy.ndimage import gaussian_filter

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

@register("vorticity_slp_850mb")
def plot_vorticity_slp_850mb(fig, ax, plotter, reader, args):
    """
    Plot Vorticity (s-1) and SLP (mb) at 850mb
    """
    # Read from reader (reader decides the collection)
    vort, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["VORT850"]
    )
    vort = vort.astype(np.float32)*1.e5

    phis, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["PHIS"]
    )
    phis = phis.astype(np.float32)/9.81

    slp, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["SLP"]
    )
    slp = slp.astype(np.float32)/100.0
    # mask and smooth
    slp_masked = np.ma.masked_where(phis > 1500, slp)
    lat_weight = np.cos(np.deg2rad(lats))[:, None]
    slp_filled = slp_masked.filled(np.nan)
    slp_weighted = slp_filled * lat_weight
    slp = gaussian_filter(slp_weighted, sigma=1.0)
    slp = slp / lat_weight
    slp = np.ma.masked_invalid(slp)

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
    # Plot SLP contours
    # ------------------------------------------------------------
    levs0 =  884 + np.arange(12) * 8
    levs1 =  980 + np.arange(5)  * 4
    levs2 = 1000 + np.arange(31) * 2
    clevs = np.concatenate([levs0, levs1, levs2])
    cs = ax.contour(
        lons,
        lats,
        slp,
        levels=clevs,
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
    """Generate colorbar for 850mb vorticity/SLP"""
    from wxmaps_utils import save_colorbar_single
    
    output = "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/vorticity_slp_850mb.png"
    save_colorbar_single(
        vCOLORS, 
        vLEVELS, 
        output, 
        label="850mb Relative Vorticity (×10⁻⁵ s⁻¹) with Sea Level Pressure Contours (mb)",
        extend='max'
    )
