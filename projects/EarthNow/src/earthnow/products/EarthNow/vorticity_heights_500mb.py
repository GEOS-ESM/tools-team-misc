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

    # Bills alphas for vort: alevs= [0,2.5], Match this
    cmap_colors[:3, -1] = 0
    cmap = ListedColormap(cmap_colors)

    norm = BoundaryNorm(vLEVELS, ncolors=cmap.N, clip=True)

    # ------------------------------------------------------------
    # Plot vorticity field (cyclonic)
    # ------------------------------------------------------------
    hemispheres = ["NH","SH"]
    for hemis in hemispheres: # plot cyclonic in each hemisphere
        mask=lats>0 
        if hemis=="NH":
            lats_mask = lats[mask]
            vort_mask = vort[mask]
        else:
            lats_mask = lats[~mask]
            vort_mask = vort[~mask]*-1
        ax.pcolormesh(
            lons,
            lats_mask,
            vort_mask,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
            shading="nearest",
            zorder=4,
        )

    # ------------------------------------------------------------
    # Plot height contours
    # ------------------------------------------------------------

    # Smooth heights
    import scipy.ndimage as ndimage

    pngImgIdim = 3840
    pngImgJdim = 2160

    # Gaussian Filter: This doesn't look the exact same
    # sigma = 20.0
    # hgts_smoothed = ndimage.gaussian_filter(hgts, sigma=sigma, mode="wrap")

    # Boxcar filter as in IDL SMOOTH (IDL: idata = SMOOTH(h500, pngImgIdim*0.025, /NAN, /EDGE_TRUNCATE))
    window_size = int(pngImgIdim * 0.025)
    # hgts_smoothed = ndimage.generic_filter(hgts, np.nanmean, size=window_size, mode="constant", cval=np.nan) # This is SO SLOW (not optimized in C)
    # have to mask out NANs so they are incorporated into the smoothing
    mask = ~np.isnan(hgts)
    hgts_valid = np.where(mask, hgts, 0)
    sum_data = ndimage.uniform_filter(
        hgts_valid, size=window_size, mode="constant", cval=0.0
    )
    sum_weights = ndimage.uniform_filter(
        mask.astype(float), size=window_size, mode="constant", cval=0.0
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        hgts_smoothed = sum_data / sum_weights

    hgts_smoothed[sum_weights == 0] = np.nan
    hgts_smoothed[~mask] = np.nan
    hlevs = np.arange(4500, 6300, 30)  # 4800m to 6240m every 30m

    cs = ax.contour(
        lons,
        lats,
        hgts_smoothed,
        levels=hlevs,
        colors="black",
        linewidths=0.5,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )
    # labels
    ax.clabel(
        cs,
        cs.levels[::2],
        fmt="%d",
        fontsize=5,
        inline=True,
        inline_spacing=5,
    )

    # Generate a new colorbar tests
    # ticks = np.arange(0,65,5)
    # generate_colorbar(
    #     cmap_colors, # If we apply transparency to a colormap, we want it to be reflected in the colobar we create... so perhaps we need to regenerate this more often/inside the function?
    #     vLEVELS,
    #     ticks,
    # )

def generate_colorbar(colors, levels, ticks):
    """Generate colorbar for 500mb vorticity/heights"""
    from earthnow.wxmaps_utils import save_colorbar_single

    #NOTE: Temp output location until we determine a central location
    output = "/discover/nobackup/hzafar/EarthNow/plots/vorticity_heights_500mb.png"
    # output = "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/vorticity_heights_500mb.png"
    save_colorbar_single(
        colors,
        levels,
        output,
        label="500mb Cyclonic Relative Vorticity (×10⁻⁵ s⁻¹) and Height (m)",
        extend="neither",
        ticks=ticks,
    )
