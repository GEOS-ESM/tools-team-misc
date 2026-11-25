"""
2-Meter Temperature Anomaly Product

Generates temperature anomaly plots (°C deviation from climatology) with lapse rate
correction for elevation differences between model and reference topography.

Based on IDL script: ploteic_tanom2.pro
Conversion: ploteic_tanom2.pro (instantaneous, 3-day climatology smoothing)

See docs/T2M_ANOMALY.md for full documentation.
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import LinearSegmentedColormap, Normalize

from earthnow.products.registry import register
from earthnow.wxmaps_climatology_utils import (
    read_climatology_3day_smooth,
    read_merra2_reference_topography,
    calculate_temperature_anomaly,
    cap_extremes,
)

variable = "t2m_anom"

# ------------------------------------------------------------------
# Temperature Anomaly Colormap + Levels (°C)
# ------------------------------------------------------------------
# -21°C to +21°C anomaly range with 256 smooth color interpolation

ANOMALY_LEVELS = [
    -21,
    -20,
    -19,
    -18,
    -17,
    -16,
    -15,
    -14,
    -13,
    -12,
    -11,
    -10,
    -9,
    -8,
    -7,
    -6,
    -5,
    -4,
    -3,
    -2,
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
]

# Color palette: cold (blue) -> normal (white) -> warm (red)
ANOMALY_COLORS = (
    np.array(
        [
            [30, 30, 75],  # -21°C: dark blue
            [30, 60, 110],
            [40, 90, 150],
            [60, 120, 180],
            [90, 150, 210],
            [130, 180, 230],
            [170, 210, 245],
            [200, 230, 250],
            [220, 240, 255],
            [235, 245, 255],
            [245, 250, 255],
            [255, 255, 255],  # Normal (0°C): white
            [255, 255, 255],  # Normal (0°C): white
            [255, 250, 245],
            [255, 245, 230],
            [255, 230, 200],
            [255, 210, 160],
            [255, 180, 100],
            [255, 150, 50],
            [255, 120, 0],
            [245, 100, 0],
            [230, 80, 0],  # +21°C: red
            [215, 60, 0],
        ]
    )
    / 255.0
)


# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------


@register("temperature_2m_anomaly_EarthNow")
def plot_temperature_2m_anomaly(fig, ax, plotter, reader, args):
    """
    Plot 2-meter Temperature Anomaly (°C) with lapse rate correction.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    plotter : wxvis.Plotter
        Plotter instance with map configuration
    reader : DataReader
        Data reader instance
    args : argparse.Namespace
        Command-line arguments with fdate, pdate, colorbar, etc.

    Notes
    -----
    Anomaly calculation includes:
    - 3-day running mean climatology (previous, current, next day)
    - Lapse rate correction for elevation differences (6°C/km)
    - Capping of extreme values to prevent color artifacts

    Data sources:
    - T2M: From model forecast data (GEOS)
    - Climatology: 1991-2020 baseline (3-day smoothing)
    - Topography: Static model elevation grid + MERRA-2 reference
    """

    # Read T2M data from model
    data, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["TMP_2M", "T2M"]
    )

    if data is None:
        raise ValueError("Unable to read T2M data")

    # Get grid dimensions
    nlats, nlons = data.shape

    # Read climatology with 3-day smoothing
    climatology = read_climatology_3day_smooth(args.pdate, nlons, nlats)

    # Read model topography (PHIS) from the same data file
    model_elevation, _, _, _ = reader.read_variable(
        args.fdate, args.pdate, variables=["PHIS"]
    )
    
    if model_elevation is None:
        raise ValueError("Unable to read model topography (PHIS)")
    
    # Convert geopotential to height in meters
    model_elevation = model_elevation / 9.80665

    # Read MERRA-2 reference topography (constant baseline)
    reference_elevation = read_merra2_reference_topography(nlons, nlats)

    # Calculate anomaly
    anomaly = calculate_temperature_anomaly(
        data, climatology, model_elevation, reference_elevation
    )

    # Cap extremes to prevent color artifacts
    vmin, vmax = min(ANOMALY_LEVELS), max(ANOMALY_LEVELS)
    anomaly = cap_extremes(anomaly, vmin, vmax)

    # Crop data to view
    lats, lons, anomaly = plotter.crop_data_to_view(lats, lons, anomaly)

    # ------------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------------
    norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = LinearSegmentedColormap.from_list(
        "temperature_2m_anomaly",
        list(zip(norm(ANOMALY_LEVELS), ANOMALY_COLORS)),
        N=256,
    )

    # ------------------------------------------------------------------
    # Plot field
    # ------------------------------------------------------------------
    plot = ax.pcolormesh(
        lons,
        lats,
        anomaly,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        zorder=4,
    )

    # Generate colorbar if requested
    if args.colorbar:
        from earthnow.wxmaps_utils import build_and_save_colorbars

        build_and_save_colorbars(
            plot,
            ANOMALY_LEVELS[1:-1],
            variable,
            "2-Meter Temperature Anomaly (°C)",
            tick_labelsize=42,
        )

    return plot
