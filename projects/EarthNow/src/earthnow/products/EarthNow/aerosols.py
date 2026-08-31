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
mappables = []
levels = []
labels = []


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

            mappables.append(cm.ScalarMappable(norm=plot.norm, cmap=plot.cmap))
            levels.append(np.linspace(0, float(values["max_val"]), 11))
            labels.append(f"{name} Aerosol Optical Thickness")

    if create_colorbar == True:
        generate_colorbars(mappables[::-1], levels[::-1], labels[::-1])


def generate_colorbars(mappables: list, levels: list, labels: list):
    """Generate colorbar for all aerosols plotted"""

    from earthnow.wxmaps_utils import build_and_save_colorbars

    colorbar_output = (
        f"/discover/nobackup/hzafar/EarthNow/plots/{variable}_colorbar.png"
    )
    build_and_save_colorbars(
        mappables,
        levels,
        colorbar_output,
        labels,
    )
