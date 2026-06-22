"""
Radar Reflectivity Product
Maximum Composite Reflectivity (DBZ_MAX / REFC)
and Maximum Updraft Helicity
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from earthnow.products.registry import register

import logging

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Updraft Helicity colormap + levels
# ------------------------------------------------------------------
# IDL values for colormapping
r = [128, 200, 54, 123, 59, 153]
g = [128, 231, 224, 62, 54, 0]
b = [128, 255, 224, 210, 135, 153]

UPHL_colors_list = np.column_stack((r, g, b)) / 255

UPHL_COLORS = np.array(
    [
        (128 / 255, 128 / 255, 128 / 255),  # Dark grey
        (200 / 255, 231 / 255, 255 / 255),  # Light Blue
        (54 / 255, 224 / 255, 224 / 255),  # Bright Cyan
        (123 / 255, 62 / 255, 210 / 255),  # Vibrant Purple
        (59 / 255, 54 / 255, 135 / 255),  #  Dark Indigo
        (153 / 255, 0 / 255, 153 / 255),  # Deep Magenta
    ]
)

# Level generation for Updraft Helicity
UPHL_LEVELS = np.arange(50, 800, 50)  # 50–750 m²/s²

# ------------------------------------------------------------------
# Reflectivity colormap + levels (wxmaps-style)
# ------------------------------------------------------------------

REFL_COLORS = (
    np.array(
        [
            [108, 237, 239],
            [50, 129, 246],
            [0, 33, 245],
            [117, 250, 76],
            [86, 187, 55],
            [55, 125, 34],
            [255, 253, 84],
            [246, 192, 66],
            [239, 134, 51],
            [234, 57, 36],
            [175, 35, 24],
            [117, 20, 12],
            [230, 61, 244],
            [134, 106, 198],
        ]
    )
    / 255.0
)

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
    cmap = ListedColormap(REFL_COLORS)
    norm = BoundaryNorm(REFL_LEVELS, ncolors=cmap.N, clip=True)

    uphl_cmap = LinearSegmentedColormap.from_list(
        "custom_smooth", UPHL_colors_list, N=len(UPHL_LEVELS)
    )
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
    # fig.colorbar(uphl_plot)  # Confirm plot colorbar matches

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig_cbar, ax_cbar = plt.subplots(figsize=(10, 2))

    sm = cm.ScalarMappable(cmap=uphl_cmap, norm=uphl_norm)
    plt.colorbar(
        sm,
        cax=ax_cbar,
        orientation="horizontal",
    )
    uphl_output = (
        "/discover/nobackup/hzafar/EarthNow/plots/updraft_helicity_colorbar.png"
    )
    fig_cbar.savefig(uphl_output)

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

    generate_colorbar()


# We should just get rid of this, I want to look into replacing this with Joe's config next
def generate_colorbar():
    """Generate colorbar for max reflectivity"""
    from earthnow.wxmaps_utils import save_colorbar_single

    # Reflectivity colorbar
    refl_output = (
        "/discover/nobackup/hzafar/EarthNow/plots/max_reflectivity_colorbar.png"
    )
    save_colorbar_single(
        REFL_COLORS,
        REFL_LEVELS,
        refl_output,
        label="Composite Reflectivity (dBZ)",
        extend="neither",
        ticks=REFL_LEVELS,
    )
