"""
Aerosol Optical Thickness (Sea Salt, Dust Sulfates, Nitrates)
"""

import numpy as np
from matplotlib.colors import BoundaryNorm
from earthnow.products.registry import register

import logging

logger = logging.getLogger(__name__)

from wxvis import colors
from matplotlib import colormaps

cmap = colormaps["AOT-SEASALT"]
variable = "aerosols"


SS_levels = np.linspace(0, 0.333, 11)


# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------
@register("aerosol_EarthNow")
def plot_aerosols(fig, ax, plotter, reader, args):
    """
    Plot AOT (SS, DU, SU, NI)
    """
    # Read from reader
    data_SS, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["SSEXTTAU"]
    )

    # data_DU, lats, lons, meta = reader.read_variable(
    #     args.fdate, args.pdate, variables=["DUEXTTAU"]
    # )
    #
    # data_SU, lats, lons, meta = reader.read_variable(
    #     args.fdate, args.pdate, variables=["SUEXTTAU"]
    # )
    #
    # data_NI, lats, lons, meta = reader.read_variable(
    #     args.fdate, args.pdate, variables=["NIEXTTAU"]
    # )

    data = data_SS.squeeze()
    data = data.astype(np.float32)
    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    norm = BoundaryNorm(SS_levels, ncolors=cmap.N, clip=False)

    # ------------------------------------------------------------
    # Plot fields
    # ------------------------------------------------------------
    plot = ax.contourf(
        lons,
        lats,
        data,
        cmap=cmap,
        norm=norm,
        levels=SS_levels,
    )

    from earthnow.wxmaps_utils import save_colorbar_single

    colorbar_output = (
        f"/discover/nobackup/hzafar/EarthNow/plots/{variable}_colorbar.png"
    )
    save_colorbar_single(
        plot,
        colorbar_output,
        label="Sea Salt Aerosol Optical Thickness",
        format="%.2f",
        ticks=SS_levels,
    )

