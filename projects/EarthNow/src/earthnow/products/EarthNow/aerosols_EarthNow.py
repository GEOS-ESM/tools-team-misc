"""
Temperature at 2-meters Product
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from earthnow.products.registry import register


# ------------------------------------------------------------------
# Variable definitions
# ------------------------------------------------------------------

CARBON_VARIABLES = ["OCEXTTAU", "BREXTTAU", "BCEXTTAU"]

# ------------------------------------------------------------------
# Colorbar configuration
# ------------------------------------------------------------------

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



# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------


@register("temperature_2m_EarthNow")
def plot_temperature_2m(fig, ax, plotter, reader, args):
    """
    Plot Temperature at 2-meters (F)
    """
    # Read from reader (reader decides the collection)

    from earthnow.wxmaps_utils import normalize

    data, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["TBRB05RG"]
    )
    bt05 = data.astype(np.float32) - 273.15  # Celcius

    data, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["TBRB06RG"]
    )
    bt06 = data.astype(np.float32) - 273.15  # Celcius

    data, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["TBRB07RG"]
        
    data, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["TMP_2M", "T2M"]
    )
    data = (data.astype(np.float32) - 273.15) * 1.8000 + 32.0

    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    vmin = min(LEVELS)
    vmax = max(LEVELS)
    # Create a function to normalize over specified range
    norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = LinearSegmentedColormap.from_list(
        "EarthNow_temperature_2m",
        list(zip(norm(LEVELS), COLORS[:-1])),  # see note below
        N=256,
    )

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
    if args.station_values:
        # Add city temperature labels
        plotter.add_city_temperatures(data, lons, lats, temperature_unit="F")


def generate_colorbar():
    """Generate colorbar for 2m temperature"""
    from earthnow.wxmaps_utils import save_colorbar_single

    output = (
        "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/temperature_2m.png"
    )
    save_colorbar_single(
        COLORS, LEVELS, output, label="2-Meter Temperature (°F)", extend="both"
    )

