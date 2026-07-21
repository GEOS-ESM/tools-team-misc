"""
Aerosol Optical Thickness (Sea Salt, Dust, Sulfates, Nitrates)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from earthnow.products.registry import register

import logging

logger = logging.getLogger(__name__)

from wxvis import colors
from matplotlib import colormaps

variable = "aerosols"
aerosols = {
    "Nitrate": {
        "varname": "NIEXTTAU",
        "cmap": "AOT-NITRATE",
        "max_val": 1.0,
    },
    "Sulfate": {
        "varname": "SUEXTTAU",
        "cmap": "AOT-SULFATE",
        "max_val": 0.5,
    },
    "Dust": {
        "varname": "DUEXTTAU",
        "cmap": "AOT-DUST",
        "max_val": 0.5,
    },
    "Sea Salt": {
        "varname": "SSEXTTAU",
        "cmap": "AOT-SEASALT",
        "max_val": 0.33,
    },
}

n_vars = len(aerosols.keys())


# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------
@register("aerosol_EarthNow")
def plot_aerosols(fig, ax, plotter, reader, args):
    """
    Plot AOT (NI, SU, DU, SS)
    """
    # Loop over aerosols:
    for name, values in aerosols.items():
        # Read from reader
        data, lats, lons, meta = reader.read_variable(
            args.fdate,
            args.pdate,
            variables=[values["varname"]],
        )

        # Data has an extra 1d dimension
        data = data.squeeze()
        # print(data.shape)
        data = data.astype(np.float32)

        levels = np.linspace(0, float(values["max_val"]), 11)

        # ------------------------------------------------------------
        # Colormap + normalization
        # ------------------------------------------------------------
        cmap = colormaps[values["cmap"]]

        # Assign alpha transparency
        num_colors = 256
        color_matrix = cmap(np.linspace(0, 1, num_colors))
        alphas = np.ones(num_colors)
        n_fade = int(256 * 0.2)
        alphas[:n_fade] = np.linspace(0, 1, n_fade)
        color_matrix[:, -1] = alphas

        cmap = ListedColormap(color_matrix)

        # ------------------------------------------------------------
        # Plot fields
        # ------------------------------------------------------------
        plot = ax.pcolormesh(  # I think to get the aerosol effect we want we need to use pcolormesh, or a bunch of contours, but not sure that is worth it over just pcolormesh?
            lons,
            lats,
            data,
            cmap=cmap,
            # norm=norm,
            vmin=levels[0],
            vmax=levels[-1],
            # vmin, vmax maps colormap to min/max levels, values below are assigned first cmap color, values above are assigned last cmap color
        )

        # plt.colorbar(
        #     plot, orientation="horizontal", shrink=0.2, aspect=15, pad=0.01
        # )  # Testing the colorbar matches

        create_colorbar = True
        if create_colorbar == True:
            # Store vars for generation of cbars out of loop
            import matplotlib.cm as cm

            aerosols[name]["ticks"] = levels
            aerosols[name]["cm"] = cm.ScalarMappable(norm=plot.norm, cmap=plot.cmap)

def generate_colorbar(variable, plot, label, ticks):
    """Generate colorbar for each var"""
    from earthnow.wxmaps_utils import save_colorbar_single

    colorbar_output = (
        f"/discover/nobackup/hzafar/EarthNow/plots/{variable}_colorbar.png"
    )

    save_colorbar_single(
        plot,
        colorbar_output,
        label=label,
        format="%.2f",
        ticks=ticks,
    )
