"""
Winds at 10m Product
Adapted from /home/wputmna/IDL_BASE/ploteic_winds.pro
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.pyplot as plt
from earthnow.products.registry import register
from earthnow.wxmaps_utils import load_color_table
from earthnow import paths
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

# ------------------------------------------------------------------
# Reflectivity colormap + levels (wxmaps-style)
# ------------------------------------------------------------------

COLORS = (
    np.array(
        [
            # Dark Blue -> Green -> YOR -> Purple -> Pink -> White
            [1, 1, 1],
            [0, 51, 102],
            [0, 102, 204],
            [51, 153, 255],
            [153, 204, 255],
            [0, 102, 0],
            [0, 153, 0],
            [26, 190, 26],
            [102, 225, 102],
            [204, 255, 204],
            [0, 204, 204],
            [255, 255, 0],
            [255, 153, 51],
            [204, 0, 0],
            [127, 0, 255],
            [255, 51, 255],
            [255, 0, 255],
            [255, 255, 255],
        ],
        dtype=float,
    )
    / 255.0
)

LEVELS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 33, 43, 50, 58, 70, 80]

# Interpolate specified colors and levels to 256 values
## Approach 1: stable + BoundaryNorm
n_in = len(COLORS)
n_out = 256

x_in = np.linspace(0, n_out - 1, n_in)
x_out = np.arange(n_out)

r256 = np.interp(x_out, x_in, COLORS[:, 0])
g256 = np.interp(x_out, x_in, COLORS[:, 1])
b256 = np.interp(x_out, x_in, COLORS[:, 2])

colors_256 = np.stack([r256, g256, b256], axis=1)  # Return to shape (256,3)
levels_256 = np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(LEVELS)), LEVELS)

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------


@register("winds_10m_EarthNow")
def plot_winds_10m(fig, ax, plotter, reader, args):
    """
    Plot winds at 10m
    """
    # Read from reader (reader decides the collection)
    ## Read Winds
    u10m, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["UGRD_10M", "U10M"]
    )
    # u10m = u10m.astype(np.float32) * 2.23694    # Converts from m/s to MPH
    v10m, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["VGRD_10M", "V10M"]
    )
    # v10m = v10m.astype(np.float32) * 2.23694   # Converts from m/s to MPH
    wspd = np.sqrt(u10m**2 + v10m**2)
    # Mask invalid wspd
    # wspd = np.ma.masked_where(wspd < 1, wspd)

    # phis, lats, lons, meta = reader.read_variable(
    #    args.fdate,
    #    args.pdate,
    #    variables=["HGT_SFC","PHIS"]
    # )
    # phis = phis.astype(np.float32)/9.81

    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    cmap = ListedColormap(colors_256)
    norm = BoundaryNorm(levels_256, ncolors=cmap.N, clip=True)

    # ------------------------------------------------------------
    # Plot field
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


def generate_colorbar():
    """Generate colorbar for SLP/winds at 10m"""
    from earthnow.wxmaps_utils import save_colorbar_single

    output = paths.colorbar_output("winds_10m_EarthNow.png")

    save_colorbar_single(
        colors_256,
        levels_256,
        output,
        label="Wind Speed at 10-meters (m/s)",
        # extend="max",
        ticks=LEVELS,
    )
