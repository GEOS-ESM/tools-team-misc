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
create_colorbar = True


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
        ## For testing, subsample
        # stride = 6
        # lons = lons[::stride]
        # lats = lats[::stride]
        # data = data[::stride, ::stride]

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

        if create_colorbar == True:
            # Store vars for generation of cbars out of loop
            import matplotlib.cm as cm

            aerosols[name]["ticks"] = levels
            aerosols[name]["cm"] = cm.ScalarMappable(norm=plot.norm, cmap=plot.cmap)

    if create_colorbar == True:
        generate_colorbar(aerosols)


def generate_colorbar(
    mappables_dict: dict,
    dpi=100,
    width=6600,
    hspace=1,
):
    """
    Generate colorbar grid for multi-variable plot
    Parameters
    ----------
    mappables_dict:
    label : str
        Label for the colorbar
    width, height : int
        Image dimensions in pixels
    hspace :
    """
    from earthnow.wxmaps_utils import build_colorbar

    nvars = len(mappables_dict.keys())
    height = 600 * int(nvars)
    figsize = (width / dpi, height / dpi)
    fig_cbar, axes_cbar = plt.subplots(
        nvars,
        ncols=1,
        figsize=figsize,
    )

    fig_cbar.subplots_adjust(hspace=hspace)

    for i, (name, values) in enumerate(mappables_dict.items()):
        build_colorbar(
            fig=fig_cbar,
            ax=axes_cbar[i],
            mappable=values["cm"],
            ticks=values["ticks"],
            label=f"{name} Aerosol Optical Thickness",
            # format="%.2f",
        )

    colorbar_output = (
        f"/discover/nobackup/hzafar/EarthNow/plots/{variable}_colorbar.png"
    )
    fig_cbar.savefig(colorbar_output)
    print(f"Saved cbar to {colorbar_output}")
