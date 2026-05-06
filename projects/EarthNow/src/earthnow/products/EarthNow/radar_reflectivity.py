"""
Radar Reflectivity Product
Maximum Composite Reflectivity (DBZ_MAX / REFC)
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from earthnow.products.registry import register
import logging
import sys

# ------------------------------------------------------------------
# Reflectivity colormap + levels (wxmaps-style)
# ------------------------------------------------------------------

REFL_COLORS = (
    np.array(
        [
            [28, 254, 54],
            [22, 197, 42],
            [85, 175, 60],
            [73, 145, 60],
            [40, 115, 50],
            [16, 90, 30],
            [255, 255, 59],
            [255, 197, 48],
            [254, 92, 30],
            [254, 16, 22],
            [196, 47, 20],
            [141, 9, 12],
            [253, 19, 249],
            [141, 46, 192],
            [240, 200, 255],
            [250, 225, 255],
        ]
    )
    / 255.0
)

REFL_LEVELS = np.arange(5.0, 85.0, 5.0)  # 5–80 dBZ

SNOW_COLORS = (
    np.array(
        [[235, 255, 255], [54, 204, 214], [54, 125, 215], [123, 62, 210], [59, 54, 135]]
    )
    / 255.0
)

MIX_COLORS = (
    np.array(
        [[255, 235, 235], [255, 51, 153], [255, 0, 127], [204, 0, 102], [153, 0, 76]]
    )
    / 255.0
)
# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------


@register("radar_reflectivity_EarthNow")
def plot_radar_reflectivity(fig, ax, plotter, reader, args):
    """
    Plot radar reflectivity (dBZ)
    """
    # Initialize logger
    logger = logging.getLogger(__name__)

    # ========
    # Reflectivity
    # ========
    # Read from reader (reader decides the collection)
    data, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["REFC", "HGT_SFC", "SNOW", "RAIN", "ICE", "TMP_2M"],
    )
    logger.debug("variables in data: ", [print(v) for v in data.variables])
    logger.debug("meta :", meta)

    data = data.astype(np.float32)

    # Mask invalid reflectivity
    data = np.ma.masked_where(data < 0.0, data)
    data = np.ma.masked_where(data > 80.0, data)
    logger.debug("refl min: ", data.min())
    logger.debug("refl max: ", data.max())
    logger.debug("refl mean: ", data.mean())
    # sys.exit()

    # ------------------------------------------------------------
    # Report data resolution
    # ------------------------------------------------------------
    data_shape = data.shape
    #    print(f"Data resolution: {data_shape[0]} x {data_shape[1]} (height x width)")
    #    print(f"  Total data points: {data_shape[0] * data_shape[1]:,}")
    #
    #    # Calculate approximate grid spacing
    #    if lons.ndim == 1 and lats.ndim == 1:
    #        lon_spacing = (lons.max() - lons.min()) / (len(lons) - 1)
    #        lat_spacing = (lats.max() - lats.min()) / (len(lats) - 1)
    #        print(f"  Grid spacing: {lon_spacing:.4f}° lon x {lat_spacing:.4f}° lat")
    #    elif lons.ndim == 2 and lats.ndim == 2:
    #        lon_spacing = np.median(np.diff(lons[0, :]))
    #        lat_spacing = np.median(np.diff(lats[:, 0]))
    #        print(
    #            f"  Grid spacing (median): {lon_spacing:.4f}° lon x {lat_spacing:.4f}° lat"
    #        )

    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    cmap = ListedColormap(REFL_COLORS)
    norm = BoundaryNorm(REFL_LEVELS, ncolors=cmap.N, clip=True)

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
        #        rasterized=True,
    )

    # ------------------------------------------------------------
    # Report image resolution
    # ------------------------------------------------------------
    #    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #    width_inches, height_inches = bbox.width, bbox.height
    #    dpi = fig.dpi
    #
    #    img_width_px = int(width_inches * dpi)
    #    img_height_px = int(height_inches * dpi)
    #
    #    print(f"Image resolution: {img_height_px} x {img_width_px} pixels (height x width)")
    #    print(f"  Figure size: {width_inches:.2f} x {height_inches:.2f} inches")
    #    print(f"  DPI: {dpi}")
    #    print(f"  Total image pixels: {img_height_px * img_width_px:,}")
    #    print(
    #        f"  Pixel ratio (image/data): {(img_height_px * img_width_px) / (data_shape[0] * data_shape[1]):.2f}x"
    #    )

    # ========
    # Snow
    # ========
    # Read from reader (reader decides the collection)
    data, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["SNOW"]
    )


def generate_colorbar():
    """Generate colorbar for max reflectivity"""
    from earthnow.wxmaps_utils import save_colorbar_single

    output = "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/max_reflectivity.png"
    save_colorbar_single(
        REFL_COLORS,
        REFL_LEVELS,
        output,
        label="Composite Reflectivity (dBZ)",
        extend="max",
    )
