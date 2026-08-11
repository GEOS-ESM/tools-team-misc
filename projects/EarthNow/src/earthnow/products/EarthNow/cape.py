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

variable = "cape"
create_colorbar = False

# ------------------------------------------------------------------
# CAPE colormap + levels
# ------------------------------------------------------------------


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
    # data = np.ma.masked_where(
    #     data > CAPE_LEVELS[-1], data
    # )  # Don't mask out above values, will default to top color

    logger.info(f"CAPE min: {data.min()}")
    logger.info(f"CAPE max: {data.max()}")

    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    cmap = colormaps["EN-cape"]
    norm = BoundaryNorm(CAPE_LEVELS, ncolors=cmap.N, clip=False)

    # NOTE: Alpha values from IDL: alevs= [99,100] but this doesn't match his cbar, the 100s would have no color then, decide what we want to preserve, colorshading or colormap
    # alpha_fade_pct = 99 / 10000
    # cmap = colorbar_alpha_fade(cmap, alpha_fade_pct)

    # ------------------------------------------------------------
    # Plot field
    # ------------------------------------------------------------
    plot = ax.pcolormesh(
        lons,
        lats,
        data,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
    )

    if create_colorbar == True:
        generate_colorbar(plot)


def generate_colorbar(plot):
    from earthnow.wxmaps_utils import build_and_save_colorbars

    colorbar_output = (
        f"/discover/nobackup/hzafar/EarthNow/plots/{variable}_colorbar.png"
    )
    build_and_save_colorbars(
        plot,
        CAPE_LEVELS[::4],
        colorbar_output,
        "Surface-Based CAPE [J/kg]",
    )
