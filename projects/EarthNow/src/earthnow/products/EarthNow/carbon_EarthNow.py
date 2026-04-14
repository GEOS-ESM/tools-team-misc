"""
Carbon Aerosol Optical Thickness Product
"""

from __future__ import annotations

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import LinearSegmentedColormap, Normalize

from earthnow.products.registry import register

# ------------------------------------------------------------------
# Variable definitions
# ------------------------------------------------------------------

CARBON_VARIABLES = ["OCEXTTAU", "BREXTTAU", "BCEXTTAU"]

# ------------------------------------------------------------------
# Colorbar configuration
# ------------------------------------------------------------------
# IDL source:
# cTable(0,*) = CONGRID([255,185,206,254,252,106, 23,102,  0], 256, ...)
# cTable(1,*) = CONGRID([255,185,206,254, 16,  7,  1,  0,  0], 256, ...)
# cTable(2,*) = CONGRID([255,185,113,113, 22,  9,  2,102,204], 256, ...)
#
# These are anchor RGB values that IDL interpolates to 256 colors.

COLOR_ANCHORS = (
    np.array(
        [
            [255, 255, 255],
            [185, 185, 185],
            [206, 206, 113],
            [254, 254, 113],
            [252, 16, 22],
            [106, 7, 9],
            [23, 1, 2],
            [102, 0, 102],
            [0, 0, 204],
        ],
        dtype=np.float32,
    )
    / 255.0
)

# IDL winds up using a continuous-looking scale for the plotted image.
# These are the main plot/value bounds from the script.
VMIN = 0.0
VMAX = 1.0

# Alpha scaling bounds in IDL:
# alevs = [0, 0.125]
# So optical depth below ~0 is transparent, and opacity ramps up quickly.
ALPHA_MIN = 0.0
ALPHA_MAX = 0.125


def build_carbon_colormap(name: str = "carbon_aot") -> LinearSegmentedColormap:
    """Build the continuous carbon aerosol colormap from IDL anchor colors."""
    return LinearSegmentedColormap.from_list(name, COLOR_ANCHORS, N=256)


def build_alpha(
    data: np.ndarray, vmin: float = ALPHA_MIN, vmax: float = ALPHA_MAX
) -> np.ndarray:
    """
    Mimic the IDL alpha behavior:
    aImageTV = 255 * (image_bytscl(data, MIN=MIN(alevs), MAX=MAX(alevs)) / 255.0)

    Returned alpha is 0..1 for matplotlib.
    """
    scaled = (data - vmin) / (vmax - vmin)
    return np.clip(scaled, 0.0, 1.0)


# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------


@register("carbon_aot")
def plot_carbon_aot(fig, ax, plotter, reader, args):
    """
    Plot carbon aerosol optical thickness from:
      OCEXTTAU + BREXTTAU + BCEXTTAU

    This is a first-pass Python equivalent of the IDL ploteic_carbon
    product logic.
    """
    # ------------------------------------------------------------
    # Read component fields
    # ------------------------------------------------------------
    fields, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=CARBON_VARIABLES,
    )

    # Depending on your reader, `fields` may come back as:
    #   - dict[str, np.ndarray]
    #   - tuple/list of arrays
    # The dict pattern is the cleanest, so handle both.
    if isinstance(fields, dict):
        oc = np.asarray(fields["OCEXTTAU"], dtype=np.float32)
        br = np.asarray(fields["BREXTTAU"], dtype=np.float32)
        bc = np.asarray(fields["BCEXTTAU"], dtype=np.float32)
    else:
        oc, br, bc = [np.asarray(arr, dtype=np.float32) for arr in fields]

    carbon = oc + br + bc

    # # Mask obviously bad values if needed.
    # carbon = np.where(np.isfinite(carbon), carbon, np.nan)

    # ------------------------------------------------------------
    # Optional sea ice overlay
    # ------------------------------------------------------------
    # The IDL version builds a sea-ice image and overlays it above the
    # background map. In Python, I’d strongly suggest pushing that into a
    # shared helper on `plotter`, rather than embedding lots of logic here.
    #
    # Example future shape:
    #   plotter.add_seaice_overlay(ax, args.fdate, args.region)
    #
    # For now this is just a placeholder hook:
    if hasattr(plotter, "add_seaice_overlay"):
        plotter.add_seaice_overlay(ax, args.fdate, args.pdate)

    # ------------------------------------------------------------
    # Optional background map / land image
    # ------------------------------------------------------------
    # IDL uses a Natural Earth background with NOICE. That also feels like
    # shared infrastructure, not product-specific plotting logic.
    #
    # Example future shape:
    #   plotter.add_natural_earth_background(ax, style="dark")
    #
    if hasattr(plotter, "add_background"):
        plotter.add_background(ax, style="dark")

    # ------------------------------------------------------------
    # Plot carbon AOT field
    # ------------------------------------------------------------
    cmap = build_carbon_colormap()
    norm = Normalize(vmin=VMIN, vmax=VMAX, clip=True)
    alpha = build_alpha(carbon)

    ax.pcolormesh(
        lons,
        lats,
        carbon,
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        zorder=4,
    )

    # ------------------------------------------------------------
    # Optional coastlines / borders
    # ------------------------------------------------------------
    # Your shared plotter may already handle this globally.
    if hasattr(plotter, "add_boundaries"):
        plotter.add_boundaries(ax)


def generate_colorbar():
    """Generate colorbar for carbon aerosol optical thickness."""
    from earthnow.wxmaps_utils import save_colorbar_single

    cmap = build_carbon_colormap()
    colors = cmap(np.linspace(0, 1, 256))[:, :3]

    output = "path"
    save_colorbar_single(
        colors,
        [VMIN, VMAX],
        output,
        label="Carbon Aerosol Optical Thickness",
        extend="max",
    )
