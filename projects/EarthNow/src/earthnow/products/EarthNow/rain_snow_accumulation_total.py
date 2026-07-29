"""
Accumulation total of both rain & snow
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
from earthnow.products.registry import register
from earthnow import paths

# ------------------------------------------------------------------
# Reflectivity colormap + levels (wxmaps-style)
# ------------------------------------------------------------------

snowLEVELS = [
    0.1,
    0.25,
    0.5,
    0.75,
    1,
    1.5,
    2,
    2.5,
    3,
    3.5,
    4,
    4.5,
    5,
    5.5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    16,
    18,
    20,
    22,
    24,
    26,
    28,
    30,
    32,
    34,
    36,
    40,
]
#    44,
#    48,
#    52,
#    56,
#    60,
# ]

rainLEVELS = [
    0.01,
    0.1,
    0.25,
    0.5,
    1,
    1.5,
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
]
#    22,
#    24,
#    30,
# ]

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------


@register("rain_snow_accumulation_total_EarthNow")
def plot_rain_snow_accumulation_total(fig, ax, plotter, reader, args):
    """
    Plot total rain & snow accumulation (inches)
    """
    # ============
    # SNOW FIRST
    # ============
    # Read from reader (reader decides the collection)
    data, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["SNOWACCUM"]  # , var_type="accum"
    )

    # NOTE: old data readers required var_type keyword, new ones do not.

    if data is None:

        return False  # Signal to skip this plot

    data = data.astype(np.float32) / 25.4

    # Mask low values
    data = np.ma.masked_where(data < 0.1, data)

    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    cmap = plt.get_cmap("PuBu")
    norm = BoundaryNorm(snowLEVELS, ncolors=cmap.N, clip=True)

    # ------------------------------------------------------------
    # Plot field
    # ------------------------------------------------------------
    ax.pcolormesh(
        lons,
        lats,
        data,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        zorder=4,
    )

    # ============
    # RAIN NEXT
    # ============
    # Read from reader (reader decides the collection)
    data, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["APCP", "PRECACCUM"]  # , var_type="accum"
    )

    # NOTE: old data readers required var_type keyword, new ones do not.

    if data is None:
        return False  # Signal to skip this plot
    data = data.astype(np.float32) / 25.4

    # Mask low values
    data = np.ma.masked_where(data < 0.1, data)

    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    cmap = plt.get_cmap("YlGn")
    norm = BoundaryNorm(rainLEVELS, ncolors=cmap.N, clip=True)
    # ------------------------------------------------------------
    # Plot field
    # ------------------------------------------------------------
    ax.pcolormesh(
        lons,
        lats,
        data,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        zorder=4,
    )


def generate_colorbar():
    """Generate colorbar for snow accumulation"""
    from earthnow.wxmaps_utils import save_colorbar_single

    output = paths.colorbar_output("snow_accumulation_total.png")
    save_colorbar_single(
        COLORS, LEVELS, output, label="Total Snow Accumulation (mm)", extend="max"
    )
