"""
Winds and Heights at 250mb Product
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.pyplot as plt
from earthnow.products.registry import register
from earthnow.wxmaps_utils import load_color_table
import sys
from earthnow.products.EarthNow.vorticity_heights_500mb import boxcar_smooth_2D

# ------------------------------------------------------------------
# Windspeed colormap + levels (wxmaps-style)
# ------------------------------------------------------------------

# Color table (30 colors, normalized)
# This is jet. ew.
wCOLORS = (
    np.array(
        [
            # IDL EarthNow color table
            [0, 0, 0],  # Black
            [0, 51, 102],  # Dark Navy Blue
            [0, 102, 204],  # Medium Blue
            [51, 153, 255],  # Sky Blue
            [153, 204, 255],  # Light Blue
            [0, 102, 0],  # Dark Green
            [0, 153, 0],  # Medium Green
            [26, 190, 26],  # Green
            [102, 225, 102],  # Light Green
            [204, 255, 204],  # Pale Green
            [0, 204, 204],  # Teal/Cyan
            [255, 255, 0],  # Yellow
            [255, 153, 51],  # Orange
            [204, 0, 0],  # Dark Red
            [127, 0, 255],  # Purple/Violet
            [255, 51, 255],  # Pink/Magenta
            [255, 0, 255],  # Magenta
            [255, 255, 255],  # White
        ]
    )
    / 255.0
)

wLEVELS = np.linspace(0, 100, 10)

# Wind speed alpha values
aLEVELS = [0, 12.5]  # opacity kicks in at 12.5 m/s

# ------------------------------------------------------------
# Colormap + normalization
# ------------------------------------------------------------
# cmap = ListedColormap(wCOLORS)
# norm = BoundaryNorm(wLEVELS, ncolors=cmap.N, clip=True)
# cmap_base = LinearSegmentedColormap.from_list("custom_wind", wCOLORS, N=256)
cmap_base = plt.get_cmap("turbo")
norm = Normalize(vmin=wLEVELS.min(), vmax=wLEVELS.max())

# Divide colorbar into 256 colors
# Here are those data values
clevs = np.linspace(wLEVELS.min(), wLEVELS.max(), 256)
# Here are those colors
rgba_table = cmap_base(np.linspace(0, 1, 256))  # shape (256,4)

# Calculate alpha values and clip to min = 0, max = 1
alphas = np.clip((clevs - aLEVELS[0]) / (aLEVELS[1] - aLEVELS[0]), 0.0, 1.0)
print("alphas: ", alphas)
rgba_table[:, 3] = alphas  # overwrite alpha channel
cmap = ListedColormap(rgba_table, name="custom_wind")

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------


@register("winds_heights_250mb_EarthNow")
def plot_winds_heights_250mb(fig, ax, plotter, reader, args):
    """
    Plot Winds (knots) and Heights (m) at 250mb
    """
    # ------------------------------------------------------------
    # Read wind fields
    # ------------------------------------------------------------

    # Read from reader (reader decides the collection)
    uwnd, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["U250"]
    )
    print("dims after uwnd read ===")
    print("uwnd: ", uwnd.size)
    print("lats: ", lats.size)
    print("lons: ", lons.size)
    print("meta: ", meta)
    # uwnd = uwnd.astype(np.float32) * 1.94384 # convert m/s to knots

    vwnd, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["V250"]
    )
    print("dims after vwnd read ===")
    print("vwnd: ", vwnd.size)
    print("lats: ", lats.size)
    print("lons: ", lons.size)
    print("meta: ", meta)
    # vwnd = vwnd.astype(np.float32) * 1.94384 # convert m/s to knots

    wspd = np.sqrt(uwnd**2 + vwnd**2)
    print(
        f"wspd: min={np.nanmin(wspd):.6f}, max={np.nanmax(wspd):.6f}, mean={np.nanmean(wspd):.6f}"
    )
    print("dims after wspd calc ===")
    print("wspd: ", wspd.size)
    print("uwnd: ", lats.size)
    print("vwnd: ", lons.size)

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
    # Read SLP
    # ------------------------------------------------------------
    slp, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["SLP"]
    )
    slp = slp.astype(np.float32) / 100.0
    print("dims after slp read ===")
    print("slp: ", slp.size)
    print("lats: ", lats.size)
    print("lons: ", lons.size)
    print("meta: ", meta)

    pngImgIdim = 3840
    pngImgJdim = 2160
    window_size = int(pngImgIdim * 0.025)  # This is the window size bill's IDL uses

    slp_smoothed = boxcar_smooth_2D(slp, window_size=window_size)

    # ------------------------------------------------------------
    # Plot SLP contours
    # ------------------------------------------------------------
    hlevs = np.arange(958, 1138, 2)  # 960 mb to 1140 mb every 4 mb
    cs = ax.contour(
        lons,
        lats,
        slp_smoothed,
        levels=hlevs,
        colors="white",
        linewidths=0.5,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )
    # Create labels
    clabels = ax.clabel(
        cs,
        fmt="%d",
        fontsize=6,
        inline=True,
        inline_spacing=5,
    )
    # Make labels bold/thicker
    # for label in clabels:
    #    label.set_fontweight("bold")


def generate_colorbar():
    """Generate colorbar for 250mb winds/heights"""
    from earthnow.wxmaps_utils import save_colorbar_single

    output = (
        "/discover/nobackup/eibell/EarthNow/colorbars/winds_heights_250mb_EarthNow.png"
    )
    save_colorbar_single(
        wCOLORS,
        wLEVELS,
        output,
        label="250mb Wind Speed (m/s) with Sea Level Pressure (mb)",
        extend="max",
    )
