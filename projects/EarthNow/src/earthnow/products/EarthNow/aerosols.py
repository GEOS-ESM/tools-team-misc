"""
Aerosol Optical Thickness (Sea Salt, Dust, Sulfates, Nitrates)
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

variable = "aerosols"
aerosols = {
    "Sea Salt": {
        "varname": "SSEXTTAU",
        "cmap": "EN-seasalt",
        "max_val": 0.33,
    },
    "Dust": {
        "varname": "DUEXTTAU",
        "cmap": "EN-dust",
        "max_val": 0.5,
    },
    "Sulfate": {
        "varname": "SUEXTTAU",
        "cmap": "EN-sulfate",
        "max_val": 0.5,
    },
    "Nitrate": {
        "varname": "NIEXTTAU",
        "cmap": "EN-nitrate",
        "max_val": 1.0,
    },
}

nvars = len(aerosols.keys())
create_colorbar = False


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

        # ------------------------------------------------------------
        # Colormap + normalization
        # ------------------------------------------------------------
        # Alpha varying linearly 0 to 1 from 0 to 0.125
        alpha_fade_pct = 0.125 / float(values["max_val"])

        cmap = colormaps[values["cmap"]]
        cmap = colorbar_alpha_fade(cmap, alpha_fade_pct)

        norm = Normalize(0, float(values["max_val"]))

        # Experiments with contourf:
        # levels = np.linspace(0, float(values["max_val"]), 24)
        # Pass levels=levels into contourf

        # ------------------------------------------------------------
        # Plot fields
        # ------------------------------------------------------------
        ## For testing, subsample
        # stride = 6
        # lons = lons[::stride]
        # lats = lats[::stride]
        # data = data[::stride]

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
        #     plot,
        #     orientation="horizontal",
        #     aspect=40,
        #     pad=0.05,
        #     ticks=(np.linspace(0, float(values["max_val"]), 11)),
        #     fraction=0.05,
        # )  # Testing the colorbar matches

        if create_colorbar == True:
            # Store vars for generation of cbars out of loop
            import matplotlib.cm as cm

            aerosols[name]["cm"] = cm.ScalarMappable(norm=plot.norm, cmap=plot.cmap)

    if create_colorbar == True:
        generate_colorbar(aerosols)


def generate_colorbar(
    mappables_dict: dict,
    dpi=100,
    width=6600,
    hspace: float = 1,
):
    """
    Generate colorbar grid for multi-variable plot (via dictionary of mapples)
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
        nrows=nvars,
        ncols=1,
        figsize=figsize,
        dpi=dpi,
    )

    fig_cbar.patch.set_facecolor("none")

    for i, (name, values) in enumerate(mappables_dict.items()):

        axes_cbar[i].set_position([0.15, 0.2 * (i + 1), 0.70, 0.25 / nvars])

        ticks = np.linspace(0, float(values["max_val"]), 11)
        build_colorbar(
            fig=fig_cbar,
            ax=axes_cbar[i],
            mappable=values["cm"],
            ticks=ticks,
            label=f"{name} Aerosol Optical Thickness",
            format="%.2f",
        )

    colorbar_output = (
        f"/discover/nobackup/hzafar/EarthNow/plots/{variable}_colorbar.png"
    )
    fig_cbar.savefig(
        colorbar_output,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.2,
        transparent=True,
    )
    print(f"Saved cbar to {colorbar_output}")
