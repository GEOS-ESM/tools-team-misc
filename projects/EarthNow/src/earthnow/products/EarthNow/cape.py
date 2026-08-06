"""
Surface Based Convective Available Potential Energy (CAPE) (J/kg)
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from earthnow.products.registry import register

import logging

logger = logging.getLogger(__name__)
from wxvis import colors
from matplotlib import colormaps

cmap = colormaps["EN-cape"]

variable = "cape"

# ------------------------------------------------------------------
# CAPE colormap + levels
# ------------------------------------------------------------------
# NOTE: Alpha values from IDL: alevs= [99,100] - I don't really know what this means I just clipped below 100?


CAPE_LEVELS = np.array(
    [
        100,
        325,
        550,
        775,
        1000,
        1250,
        1500,
        1750,
        2000,
        2250,
        2500,
        2750,
        3000,
        3500,
        4000,
        4500,
        5000,
        6000,
        7000,
        8000,
        9000,
        10000,
        11000,
    ]
)


# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------
@register("cape_EarthNow")
def plot_cape(fig, ax, plotter, reader, args):
    """
    Plot CAPE (J/kg)
    """

    # Read from reader (reader decides the collection)
    data, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["CAPE"]
    )

    data = data.astype(np.float32)

    # Mask values below 100
    data = np.ma.masked_where(data < CAPE_LEVELS[0], data)
    # data = np.ma.masked_where(data > CAPE_LEVELS[-1], data) # Don't mask out above values, will default to top color

    logger.info(f"CAPE min: {data.min()}")
    logger.info(f"CAPE max: {data.max()}")

    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    norm = BoundaryNorm(CAPE_LEVELS, ncolors=cmap.N, clip=False)

    # ------------------------------------------------------------
    # Plot field
    # ------------------------------------------------------------
    plot = ax.contourf(
        lons,
        lats,
        data,
        cmap=cmap,
        norm=norm,
        levels=CAPE_LEVELS,
        transform=ccrs.PlateCarree(),
    )

    # generate_colorbar(plot) # Uncomment to generate colorbar w every frame


def generate_colorbar(plot):
    from earthnow.wxmaps_utils import save_colorbar_single

    colorbar_output = (
        f"/discover/nobackup/hzafar/EarthNow/plots/{variable}_colorbar.png"
    )
    save_colorbar_single(
        plot,
        colorbar_output,
        label="Surface-Based CAPE [J/kg]",
        ticks=CAPE_LEVELS[::4],
    )
