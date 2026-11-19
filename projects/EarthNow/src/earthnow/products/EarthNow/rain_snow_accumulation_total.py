"""
Accumulation total of both rain & snow
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
from earthnow.products.registry import register
from earthnow import paths
import logging
import sys

variable = "precip"
create_colorbar = False

# Create mappable dict for multi-variable colorbar
mappables = []
levels = []
labels = []
ticks = []
# ------------------------------------------------------------------
# Reflectivity colormap + levels (wxmaps-style)
# ------------------------------------------------------------------

snowLEVELS = [
    0,
    50,
    60,
    70,
    80,
    90,
    100,
    120,
    140,
    150,
    160,
    180,
    200,
    250,
    300,
    350,
    400,
    500,
    600,
    700,
    800,
    1000,
    1200,
    1500,
]

rainLEVELS = [
    0,
    50,
    60,
    70,
    80,
    90,
    100,
    120,
    140,
    150,
    160,
    180,
    200,
    250,
    300,
    350,
    400,
]

MIN_ACCUM_MM = 5.0  # matches IDL alpha ramp: values <= 5 mm are transparent

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------


@register("rain_snow_accumulation_total_EarthNow")
def plot_rain_snow_accumulation_total(fig, ax, plotter, reader, args):
    """
    Plot total rain & snow accumulation (mm)
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

    data = data.astype(np.float32)

    # Mask low values
    data = np.ma.masked_where(data <= MIN_ACCUM_MM, data)

    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    cmap = plt.get_cmap("PuBu")
    norm = BoundaryNorm(snowLEVELS, ncolors=cmap.N, clip=True)

    # ------------------------------------------------------------
    # Plot field
    # ------------------------------------------------------------
    lats, lons, data = plotter.crop_data_to_view(lats, lons, data)
    snowplot = ax.pcolormesh(
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
    data = data.astype(np.float32)

    # Mask low values
    data = np.ma.masked_where(data <= MIN_ACCUM_MM, data)

    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    cmap = plt.get_cmap("YlGn")
    norm = BoundaryNorm(rainLEVELS, ncolors=cmap.N, clip=True)
    # ------------------------------------------------------------
    # Plot field
    # ------------------------------------------------------------
    lats, lons, data = plotter.crop_data_to_view(lats, lons, data)
    rainplot = ax.pcolormesh(
        lons,
        lats,
        data,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        zorder=4,
    )

    if args.colorbar:
        # Custom ticks to match Bill's
        snow_ticks = [0, 60, 80, 100, 140, 160, 200, 300, 400, 600, 800, 1200]
        mappables.append(snowplot)
        levels.append(snow_ticks)
        labels.append("Accumulated Snow [mm]")

        mappables.append(rainplot)
        levels.append(rainLEVELS)
        labels.append("Accumulated Rain [mm]")

        from earthnow.wxmaps_utils import build_and_save_colorbars

        build_and_save_colorbars(
            mappables,
            levels,
            variable,
            labels,
        )
