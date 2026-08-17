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
    # Read all variables
    # ========
    # Read from reader (reader decides the collection)
    refl, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["REFC"],
    )
    phis, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["HGT_SFC"],
    )
    phis = phis / 9.81
    snow, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["SNOW"],
    )
    rain, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["RAIN"],
    )
    ice, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["ICE"],
    )
    t2m, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["TMP_2M"],
    )

    refl = refl.astype(np.float32)

    # Mask invalid reflectivity
    refl = np.ma.masked_where(refl < 0.0, refl)
    refl = np.ma.masked_where(refl > 80.0, refl)
    logger.debug("refl min: ", refl.min())
    logger.debug("refl max: ", refl.max())
    logger.debug("refl mean: ", refl.mean())
    # sys.exit()

    # Only keep nonzero snow values where there's snow
    # and t2m is below freezing
    # Convert snow kg m-2 s-1 to inches hr-1
    condition = (snow * 3600.0 / 25.4 > 0) & (t2m <= 276.483)
    snow[~condition] = 0.0

    # Only keep nonzero ice values where there's ice
    # and t2m is below freezing
    condition = (ice * 3600.0 / 25.4 > 0) & (twm <= 276.483)
    ice[~condition] = 0.0

    # Elevation correction
    # Use geopotential height to determine precip type
    # -------------------------------------------------
    i1000 = 0
    i925 = 1
    i850 = 2
    i700 = 3
    i500 = 4

    hgt, lats, lons, meat = reader.read_variable(
        args.fdate, args.pdate, variables=["HGT"]
    )

    # Replace missing heights (belowground) with surface pressure
    phis_3d = phis[:, :, np.newaxis]  # dims X,Y,1
    hgt = np.where(hgt == 1.0e15, phis, hgt)
    # syntax here: where hgt is missing (= 1e15), insert phis value.
    # where hgt is valid, keep it

    # 500 - 925 mb thickness
    thck = np.squeeze(hgt[:, :, i500] - hgt[:, :, i925])
    elevFactor = (phis - 305.0) / 915.0
    # apply 0 and 1 caps to elevFactor
    elevFactor[elevFactor < 0] = 0.0
    elevFactor[elevFactor > 1] = 1.0

    elevFactor = elevFactor * 50.0
    thLow = 5425.0 - 625 + elevFactor  # cold edge
    thHigh = 5475.0 - 625 + elevFactor  # warm edge
    # Explanation:
    # between these edges you get sleet, freezing rain, or rain/snow mix.
    # The 625 adjustment accounts for 1000-925 mb thickness:
    # 5425 and 5475 are typicaly thresholds for 1000-500 mb thickness
    # but we're using 925-500 so need to tweak.
    # Add the elevation factor of up to 50 m depending on elevation
    # (warmer/lower elevations snow melts further from the ground)

    # Precip in this chunk of atmosphere is defined here as icefall (sleet)
    ifind = (
        (thck > thLow)
        & (thck < thHigh)
        & ((snow * 3600.0 / 25.4 > 1.0e-6) | (ice * 3600.0 / 25.4 > 1.0e-6))
    )
    # recall this *3600./25.4 converts to inches per hour
    # so that line is checking for EITHER nonzero snowfall OR nonzero icefall

    ice[ifind] = refl[ifind]  # fill reflectivity for ifind
    snow[ifind] = 0.0
    rain[ifind] = 0.0
    ice[~ifind] = 0.0

    # Snow does not fall below thHigh
    snow[thck >= thHigh] = 0.0
    # Rain reflectivity
    rain[rain > 0.0] = refl[rain > 0.0]
    # Snow reflectivity
    snow[snow > 0.0] = ref[snow > 0.0]
    # Freezing rain reflectivity
    frzr = rain
    frzr[t2m > 273.15] = 0.0

    # ------------------------------------------------------------
    # Report data resolution
    # ------------------------------------------------------------
    refl_shape = refl.shape
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
