"""
Surface Based Convective Available Potential Energy (CAPE) (J/kg)
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from earthnow.products.registry import register

import logging

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# CAPE colormap + levels
# ------------------------------------------------------------------
CAPE_RGB = (
    np.array(
        [
            [200, 200, 200],
            [160, 160, 160],
            [125, 125, 125],
            [91, 91, 91],
            [116, 170, 255],
            [83, 125, 226],
            [52, 84, 197],
            [23, 43, 158],
            [117, 255, 117],
            [79, 197, 79],
            [45, 142, 45],
            [16, 91, 16],
            [255, 255, 91],
            [221, 170, 60],
            [188, 91, 31],
            [156, 24, 0],
            [255, 142, 255],
            [212, 106, 212],
            [170, 70, 170],
            [129, 39, 129],
            [90, 10, 90],
            [50, 10, 60],
        ]
    )
    / 255.0
)
CAPE_ALPHA = np.ones(CAPE_RGB.shape[0])
# NOTE: Alpha values from IDL: alevs= [99,100] - I don't really know what this means I just clipped below 100?

CAPE_COLORS = np.column_stack((CAPE_RGB, CAPE_ALPHA))

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
    # logger.info(f"colors: {len(CAPE_COLORS)}")
    # logger.info(f"levels: {len(CAPE_LEVELS)}")

    # Read from reader (reader decides the collection)
    data, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["CAPE"]
    )

    data = data.astype(np.float32)
    logger.info(f"CAPE min: {data.min()}")
    # breakpoint()

    # Mask values below 100
    data = np.ma.masked_where(data < CAPE_LEVELS[0], data)
    # data = np.ma.masked_where(data > CAPE_LEVELS[-1], data) # Don't mask out above values, will default to top color

    logger.info(f"CAPE min: {data.min()}")
    logger.info(f"CAPE max: {data.max()}")

    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    cmap = ListedColormap(CAPE_COLORS)
    norm = BoundaryNorm(CAPE_LEVELS, ncolors=cmap.N, clip=False)

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
        shading="nearest",
        zorder=4,
        rasterized=True,
    )
    # fig.colorbar(plot)  # Confirm plot colorbar matches

    generate_colorbar()


def generate_colorbar():
    """Generate colorbar"""
    from earthnow.wxmaps_utils import save_colorbar_single

    # Reflectivity colorbar
    refl_output = "/discover/nobackup/hzafar/EarthNow/plots/cape_colorbar.png"

    import numpy as np

    save_colorbar_single(
        CAPE_COLORS,
        CAPE_LEVELS,
        refl_output,
        label="Surface-Based CAPE [J/kg]",
        extend="max",
        ticks=CAPE_LEVELS[::4],
    )
