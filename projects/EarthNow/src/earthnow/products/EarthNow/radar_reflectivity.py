"""
Radar Reflectivity Product
Maximum Composite Reflectivity (DBZ_MAX / REFC)
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap
from earthnow.products.registry import register
import logging

variable = "radar_reflectivity_EarthNow"
colorbar_only = False

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

ICE_COLORS = (
    np.array(
        [
            [252, 236, 254],
            [214, 205, 215],
            [203, 138, 210],
            [185, 72, 194],
            [125, 59, 131],
        ]
    )
    / 255.0
)

MIX_COLORS = (
    np.array(
        [[255, 235, 235], [255, 51, 153], [255, 0, 127], [204, 0, 102], [153, 0, 76]]
    )
    / 255.0
)

if colorbar_only:
    import matplotlib.cm as cm

    mappables = []
    ticks = []
    labels = []

# Generate colormaps
# Rain reflectivity
rain_cmap = ListedColormap(REFL_COLORS)
rain_norm = BoundaryNorm(REFL_LEVELS, ncolors=rain_cmap.N, clip=True)
if colorbar_only:
    mappables.append(cm.ScalarMappable(norm=rain_norm, cmap=rain_cmap))
    ticks.append(
        [
            (REFL_LEVELS.min(), "Light"),
            (np.median(REFL_LEVELS), "Moderate"),
            (REFL_LEVELS.max(), "Heavy"),
        ]
    )
    labels.append("Rain")

# Snow Reflectivity
snow_cmap = LinearSegmentedColormap.from_list(
    "snow_cmap", SNOW_COLORS, N=len(REFL_LEVELS) - 1
)
snow_norm = BoundaryNorm(REFL_LEVELS, ncolors=snow_cmap.N, clip=True)
if colorbar_only:
    mappables.append(cm.ScalarMappable(norm=snow_norm, cmap=snow_cmap))
    ticks.append(
        [
            (REFL_LEVELS.min(), "Light"),
            (np.median(REFL_LEVELS), "Moderate"),
            (REFL_LEVELS.max(), "Heavy"),
        ]
    )
    labels.append("Snow")

# Ice reflectivity
ice_cmap = LinearSegmentedColormap.from_list(
    "ice_cmap", ICE_COLORS, N=len(REFL_LEVELS) - 1
)
ice_norm = BoundaryNorm(REFL_LEVELS, ncolors=ice_cmap.N, clip=True)
if colorbar_only:
    mappables.append(cm.ScalarMappable(norm=ice_norm, cmap=ice_cmap))
    ticks.append(
        [
            (REFL_LEVELS.min(), "Light"),
            (np.median(REFL_LEVELS), "Moderate"),
            (REFL_LEVELS.max(), "Heavy"),
        ]
    )
    labels.append("Ice")

# Frzr reflectivity
frzr_cmap = LinearSegmentedColormap.from_list(
    "frzr_cmap", MIX_COLORS, N=len(REFL_LEVELS) - 1
)
frzr_norm = BoundaryNorm(REFL_LEVELS, ncolors=frzr_cmap.N, clip=True)
if colorbar_only:
    mappables.append(cm.ScalarMappable(norm=frzr_norm, cmap=frzr_cmap))
    ticks.append(
        [
            (REFL_LEVELS.min(), "Light"),
            (np.median(REFL_LEVELS), "Moderate"),
            (REFL_LEVELS.max(), "Heavy"),
        ]
    )
    labels.append("Mix")

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

    if colorbar_only:
        colorbar_creation()
    else:
        noise_floor = 1e-7

        # ========
        # Read all variables
        # ========
        # Read from reader (reader decides the collection)
        refl, lats, lons, meta = reader.read_variable(
            args.fdate,
            args.pdate,
            variables=["DBZ_MAX"],
        )
        refl = refl.astype(np.float32)

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
            variables=["T2M"],
        )

        # Fill values -> NaN
        fillValue = 1e15
        refl = np.where(refl == fillValue, np.nan, refl)
        rain = np.where(rain == fillValue, np.nan, rain)
        snow = np.where(snow == fillValue, np.nan, snow)
        ice = np.where(ice == fillValue, np.nan, ice)
        t2m = np.where(t2m == fillValue, np.nan, t2m)

        # Only keep nonzero snow values where there's snow
        # and t2m is below freezing
        # Convert snow kg m-2 s-1 to inches hr-1
        condition = ((snow * 3600.0 / 25.4) > 0) & (t2m <= 276.483)
        snow[~condition] = 0.0

        # Only keep nonzero ice values where there's ice
        # and t2m is below freezing
        condition = (ice * 3600.0 / 25.4 > 0) & (t2m <= 276.483)
        ice[~condition] = 0.0

        # Elevation correction
        # Use geopotential height to determine precip type
        # -------------------------------------------------
        i1000 = 0
        i925 = 1
        i850 = 2
        i700 = 3
        i500 = 4

        # Calculate/read in layer heights
        # -------------------
        if args.map_type == "conus":
            phis, lats, lons, meta = reader.read_variable(
                args.fdate,
                args.pdate,
                variables=["HGT_SFC"],
            )
            phis = phis / 9.81
            hgt, lats, lons, meta = reader.read_variable(
                args.fdate, args.pdate, variables=["HGT"]
            )
            # Replace missing heights (belowground) with surface pressure
            phis_3d = phis[np.newaxis, :, :]  # dims X,Y,1
            hgt = np.where(hgt == fillValue, phis_3d, hgt)
            # Syntax here: where hgt is missing (= 1e15), insert phis value.
            # Where hgt is valid, keep it.
            # This works bc of "broadcasting" - numpy automatically
            # copies phis_3d to match every Z level of the full 3D hgt array.
            # And can compare it to every Z level
            # for replacing where hgt = 1e15.

            # 500 - 925 mb thickness
            thck = np.squeeze(hgt[i500, :, :] - hgt[i925, :, :])

            elevFactor = (phis - 305.0) / 915.0
            # apply 0 and 1 caps to elevFactor
            elevFactor[elevFactor < 0] = 0.0
            elevFactor[elevFactor > 1] = 1.0
            # print("phis.shape: ",phis.shape)
            # print("elevFactor.shape: ",elevFactor.shape)
            # print("thck.shape: ", thck.shape)
            # print("hgt.shape: ",hgt.shape)

            elevFactor = elevFactor * 50.0
            thLow = 5425.0 - 625 + elevFactor  # warm edge
            thHigh = 5475.0 - 625 + elevFactor  # cold edge
            # Explanation:
            # between these edges you get sleet, freezing rain, or rain/snow mix.
            # The 625 adjustment accounts for 1000-925 mb thickness:
            # 5425 and 5475 are typicaly thresholds for 1000-500 mb thickness
            # but we're using 925-500 so need to tweak.
            # Add the elevation factor of up to 50 m depending on elevation
            # (warmer/lower elevations snow melts further from the ground)

            # print("thLow.shape: ", thLow.shape)
            # print("thHigh.shape: ", thHigh.shape)
            # sys.exit()

        elif args.map_type == "global":
            h1000, lats, lons, meta = reader.read_variable(
                args.fdate, args.pdate, variables=["H1000"]
            )
            h500, lats, lons, meta = reader.read_variable(
                args.fdate, args.pdate, variables=["H500"]
            )
            # h1000 = np.where(np.isnan(h1000) | (h1000 == 1e15), 0.0, h1000)
            # h500 = np.where(np.isnan(h500) | (h500 == 1e15), 0.0, h500)
            thck = h500 - h1000
            thLow = 5425.0
            thHigh = 5475.0

        # -- end if/else block

        # Precip manipulation
        # -------------------
        precip_floor = 1e-6
        # Precip in this chunk of atmosphere is defined here as icefall
        ifind = (
            (thck > thLow)
            & (thck < thHigh)
            & (
                (snow * 3600.0 / 25.4 > precip_floor)
                | (ice * 3600.0 / 25.4 > precip_floor)
            )
        )
        # recall this *3600./25.4 converts to inches per hour
        # so that line is checking for EITHER nonzero snowfall OR nonzero icefall

        ice[ifind] = refl[ifind]  # fill reflectivity for ifind
        snow[ifind] = 0.0
        rain[ifind] = 0.0
        ice[~ifind] = 0.0

        # Snow does not fall below thHigh
        snow[thck >= thHigh] = 0.0
        ## Rain does not fall above thLow
        # rain[thck <= thLow] = 0.0

        # Fill in reflectivity
        # --------------------
        # Rain reflectivity
        rain = np.where((rain * 3600.0 / 25.4) > noise_floor, refl, 0.0)
        # Snow reflectivity
        snow = np.where((snow * 3600.0 / 25.4) > noise_floor, refl, 0.0)

        # Freezing rain reflectivity
        frzr = np.copy(rain)
        # frzr is zero if it's too warm OR if temp data missing
        frzr = np.where((t2m > 273.15) | np.isnan(t2m), 0.0, frzr)

        # Mask invalid reflectivity
        # Note: This was not explicit in the IDL code,
        # but IDL's plotting automatically masks out values
        # beyond the colorbar limits.
        # pcolormesh does not, so we do that here.
        refl_min = min(REFL_LEVELS)
        refl_max = max(REFL_LEVELS)
        rain = np.ma.masked_where(rain < 10.0, rain)
        snow = np.ma.masked_where(snow < 10.0, snow)
        frzr = np.ma.masked_where(frzr < 10.0, frzr)
        ice = np.ma.masked_where(ice < 10.0, ice)

        logger.debug("refl min: %s", refl.min())
        logger.debug("refl max: %s", refl.max())
        logger.debug("refl mean: %s", refl.mean())

        # Plot rain reflectivity
        # ------------------------------------------------------------
        rain_plot = ax.pcolormesh(
            lons,
            lats,
            rain,
            cmap=rain_cmap,
            norm=rain_norm,
            transform=ccrs.PlateCarree(),
            shading="nearest",
            zorder=4,
        )

        mix_plot = ax.pcolormesh(
            lons,
            lats,
            frzr,
            cmap=frzr_cmap,
            norm=frzr_norm,
            transform=ccrs.PlateCarree(),
            shading="nearest",
            zorder=5,
        )

        ice_plot = ax.pcolormesh(
            lons,
            lats,
            ice,
            cmap=ice_cmap,
            norm=ice_norm,
            transform=ccrs.PlateCarree(),
            shading="nearest",
            zorder=6,
        )

        snow_plot = ax.pcolormesh(
            lons,
            lats,
            snow,
            cmap=snow_cmap,
            norm=snow_norm,
            transform=ccrs.PlateCarree(),
            shading="nearest",
            zorder=7,
        )


def colorbar_creation():
    from earthnow.wxmaps_utils import build_and_save_colorbars

    colorbar_output = (
        f"/discover/nobackup/eibell/EarthNow/plots/{variable}_colorbar.png"
    )

    build_and_save_colorbars(
        mappables,
        ticks,
        colorbar_output,
        labels,
        subtitles=True,
        title="Radar Reflectivity [dBZ]",
        share_xaxis=True,
    )
