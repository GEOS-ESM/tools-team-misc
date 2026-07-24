"""
Radar Reflectivity Product
Maximum Composite Reflectivity (DBZ_MAX / REFC)
and Maximum Updraft Helicity
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from earthnow.products.registry import register
from wxvis import colors
from matplotlib import colormaps

import logging

logger = logging.getLogger(__name__)

cmap = colormaps["DBZ_MAX"]
uphl_cmap = colormaps["UPHL"]


# Level generation for Updraft Helicity
UPHL_LEVELS = np.arange(50, 800, 50)  # 50–750 m²/s²

REFL_LEVELS = np.arange(5.0, 80.0, 5.0)  # 5–75 dBZ


# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------


@register("max_reflectivity_EarthNow")
def plot_max_reflectivity(fig, ax, plotter, reader, args):
    """
    Plot maximum composite radar reflectivity (dBZ) and updraft helicity
    """
    # Read from reader (reader decides the collection)
    data, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["REFC", "DBZ_MAX"]
    )

    # Read updraft helicity
    uphl_data, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["UH25", "UPHL_2-5KM"]
    )

    data = data.astype(np.float32)
    uphl_data = uphl_data.astype(np.float32)

    # Mask invalid reflectivity
    data = np.ma.masked_where(data < REFL_LEVELS[0], data)
    data = np.ma.masked_where(data > REFL_LEVELS[-1], data)

    # logger.info(
    #     f"Before masking: UPHL min: {uphl_data.min()}, UPHL max: {uphl_data.max()}"
    # )

    # Mask invalid updraft helicity
    uphl_data = np.ma.masked_where(uphl_data < UPHL_LEVELS[0], uphl_data)
    uphl_data = np.ma.masked_where(uphl_data > UPHL_LEVELS[-1], uphl_data)

    # logger.info(
    #     f"After masking: UPHL min: {uphl_data.min()}, UPHL max: {uphl_data.max()}"
    # )

    # ------------------------------------------------------------
    # Report data resolution
    # ------------------------------------------------------------
    data_shape = data.shape
    print(
        f"Reflectivity data resolution: {data_shape[0]} x {data_shape[1]} (height x width)"
    )
    print(f"  Reflectivity total data points: {data_shape[0] * data_shape[1]:,}")

    # Calculate approximate grid spacing
    if lons.ndim == 1 and lats.ndim == 1:
        lon_spacing = (lons.max() - lons.min()) / (len(lons) - 1)
        lat_spacing = (lats.max() - lats.min()) / (len(lats) - 1)
        print(f"  Grid spacing: {lon_spacing:.4f}° lon x {lat_spacing:.4f}° lat")
    elif lons.ndim == 2 and lats.ndim == 2:
        lon_spacing = np.median(np.diff(lons[0, :]))
        lat_spacing = np.median(np.diff(lats[:, 0]))
        print(
            f"  Grid spacing (median): {lon_spacing:.4f}° lon x {lat_spacing:.4f}° lat"
        )

    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    norm = BoundaryNorm(REFL_LEVELS, ncolors=cmap.N, clip=True)

    uphl_norm = BoundaryNorm(UPHL_LEVELS, ncolors=uphl_cmap.N, clip=True)
    # logger.info((uphl_cmap.N))
    # logger.info(UPHL_LEVELS)

    # ------------------------------------------------------------
    # Plot fields
    # ------------------------------------------------------------
    # Plot Radar Reflectivity
    radar_plot = ax.pcolormesh(
        lons,
        lats,
        data,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        zorder=4,
        rasterized=True,
    )
    # fig.colorbar(radar_plot) # Confirm plot colorbar matches

    # Updraft Helicity
    uphl_plot = ax.contourf(
        lons,
        lats,
        uphl_data,
        cmap=uphl_cmap,
        norm=uphl_norm,
        extend="neither",
        levels=UPHL_LEVELS,  # you MUST pass this with contourf so it doesn't auto-calculate levels
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    logger.info(f"vmin: {UPHL_LEVELS.min()} vmax: {UPHL_LEVELS.max()}")

    # ------------------------------------------------------------
    # Report image resolution
    # ------------------------------------------------------------
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_inches, height_inches = bbox.width, bbox.height
    dpi = fig.dpi

    img_width_px = int(width_inches * dpi)
    img_height_px = int(height_inches * dpi)

    print(f"Image resolution: {img_height_px} x {img_width_px} pixels (height x width)")
    print(f"  Figure size: {width_inches:.2f} x {height_inches:.2f} inches")
    print(f"  DPI: {dpi}")
    print(f"  Total image pixels: {img_height_px * img_width_px:,}")
    print(
        f"  Pixel ratio (image/data): {(img_height_px * img_width_px) / (data_shape[0] * data_shape[1]):.2f}x"
    )

    # generate_colorbar("dbz_max", radar_plot, "Simulated Radar (dBZ)", REFL_LEVELS)
    # generate_colorbar(
    #     "uphl", uphl_plot, "Maximum Updraft Helicity (m2 s-2)", UPHL_LEVELS[1::2]
    # )


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
        ticks=ticks,
    )
