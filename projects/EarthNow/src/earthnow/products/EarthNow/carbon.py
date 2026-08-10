"""
Carbon Aerosol Optical Thickness
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
from earthnow.products.registry import register

import logging

from earthnow.wxmaps_utils import colorbar_alpha_fade

logger = logging.getLogger(__name__)

from wxvis import colors
from matplotlib import colormaps

variable = "carbon"

create_colorbar = True


# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------
@register("carbon_EarthNow")
def plot_aerosols(fig, ax, plotter, reader, args):
    """
    Plot Carbon AOT
    """
    # Read from reader
    oc_data, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["OCEXTTAU"],
    )

    br_data, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["BREXTTAU"],
    )

    bc_data, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["BCEXTTAU"],
    )

    # Sum all carbons
    data = oc_data + br_data + bc_data

    # Data has an extra 1d dimension
    data = data.squeeze()
    data = data.astype(np.float32)

    levels = np.linspace(0, 1, 11)

    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    cmap = colormaps["EN-carbon"]
    cmap = colorbar_alpha_fade(cmap, 0.125)

    norm = Normalize(levels[0], levels[-1])

    # ------------------------------------------------------------
    # Plot fields
    # ------------------------------------------------------------
    plot = ax.pcolormesh(
        lons,
        lats,
        data,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        antialiased=True,
    )

    # plt.colorbar(
    #     plot, orientation="horizontal", shrink=0.2, aspect=15, pad=0.01
    # )  # Testing the colorbar matches

    if create_colorbar == True:
        from earthnow.wxmaps_utils import save_colorbar_single

        colorbar_output = (
            f"/discover/nobackup/hzafar/EarthNow/plots/{variable}_colorbar.png"
        )
        save_colorbar_single(
            plot,
            colorbar_output,
            label="Carbon Aerosol Optical Thickness",
            ticks=levels,
        )
