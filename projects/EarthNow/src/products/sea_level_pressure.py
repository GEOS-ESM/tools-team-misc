"""
SLP Product
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
from .registry import register
from wxmaps_utils import load_color_table
from scipy.ndimage import minimum_filter, gaussian_filter 

def find_local_minima(slp, lats, lons,
                      threshold=1012.0,
                      filter_size=15):
    """
    Find local minima in SLP field.

    Returns list of (lat, lon, value)
    """
    # Apply minimum filter
    local_min = slp == minimum_filter(slp, size=filter_size)

    # Apply threshold
    candidates = np.where(local_min & (slp <= threshold))

    minima = []
    for i, j in zip(*candidates):
        minima.append((lats[i], lons[j], slp[i, j]))

    return minima

# ------------------------------------------------------------------
# colormap + levels (wxmaps-style)
# ------------------------------------------------------------------

sLevs = [
    880, 890, 900, 910, 920, 924, 928, 932,
    936, 940, 944, 948, 952, 956, 960, 964, 968, 972, 976, 980,
    984, 988, 992, 996, 1000, 1004, 1008, 1012, 1016, 1020, 1024, 1028
]

rgb = np.array([
    [255,255,255],
    [ 50, 16,100],
    [ 38,  8,133],
    [ 78,  5,151],
    [119, 13,153],
    [137, 21,147],
    [156, 32,137],
    [172, 44,125],
    [188, 57,113],
    [202, 70,103],
    [215, 85, 92],
    [226,100, 84],
    [237,116, 75],
    [245,134, 68],
    [251,153, 62],
    [253,175, 59],
    [251,198, 58],
    [245,222, 62],
    [255,231, 93],
    [255,252,148],
    [247,245,140],
    [232,240,130],
    [219,230,115],
    [145,202, 90],
    [ 73,180, 71],
    [ 73,163,129],
    [ 72,146,184],
    [ 98,165,215],
    [133,196,234],
    [173,224,248],
    [214,239,252],
    [255,255,255],
]) / 255.0

cmap = ListedColormap(rgb)
norm = BoundaryNorm(sLevs, cmap.N)

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------

@register("sea_level_pressure")
def plot_sea_level_pressure(fig, ax, plotter, reader, args):
    """
    Plot SLP
    """
    # Read from reader (reader decides the collection)

    slp, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["PRMSL","SLP"]
    )
    slp = slp.astype(np.float32)/100.0
    if "cycled" in getattr(reader, "name", "").lower():
        slp = gaussian_filter(slp, sigma=4.0)

    # ------------------------------------------------------------
    # Plot SLP contours
    # ------------------------------------------------------------
    cf = ax.contourf(
        lons,
        lats,
        slp,
        levels=sLevs,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        zorder=3,                 # lower than contour lines
        extend="both",
    )
    cs = ax.contour(
        lons,
        lats,
        slp,
        levels=sLevs,
        colors="darkslategrey",
        linewidths=1.25,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )
    # Create labels
    clabels = ax.clabel(
        cs,
        fmt="%d",
        fontsize=9,
        inline=True,
        inline_spacing=5,
    )
    # Make labels bold/thicker
    for label in clabels:
        label.set_fontweight('bold')

    # ------------------------------------------------------------
    # Plot ensemble member SLP minima (GenCast only)
    # ------------------------------------------------------------

    if getattr(reader, "name", "").lower().startswith("gencast"):
        for m in range(32):
            try:
                slp_m, _, _, _ = reader.read_variable(
                    args.fdate,
                    args.pdate,
                    variables=["PRMSL", "SLP"],
                    ensemble_member=m,
                )

                slp_m = slp_m.astype(np.float32) / 100.0

                minima = find_local_minima(
                    slp_m,
                    lats,
                    lons,
                    threshold=1012.0,
                    filter_size=15,
                )

                # -------------------------
                # Diagnostics
                # -------------------------
                slp_min = np.nanmin(slp_m)
                slp_max = np.nanmax(slp_m)

                if minima:
                    strongest = min(minima, key=lambda x: x[2])
                    print(
                        f"{reader.name} Member {m:02d}: "
                        f"SLP range [{slp_min:.1f}, {slp_max:.1f}] hPa | "
                        f"minima={len(minima):2d} | "
                        f"strongest={strongest[2]:.1f} hPa "
                        f"@ ({strongest[0]:.1f}N, {strongest[1]:.1f}E)"
                    )
                else:
                    print(
                        f"{reader.name} Member {m:02d}: "
                        f"SLP range [{slp_min:.1f}, {slp_max:.1f}] hPa | "
                        f"minima=0"
                    )

                for lat, lon, val in minima:
                    ax.plot(
                        lon,
                        lat,
                        marker="x",
                        markersize=6,
                        markeredgewidth=1.2,
                        color="red",
                        transform=ccrs.PlateCarree(),
                        zorder=6,
                        alpha=0.6,
                    )

            except Exception as e:
                print(f"{reader.name} Member {m:02d}: FAILED ({e})")

def generate_colorbar():
    """Generate colorbar for SLP"""
    from wxmaps_utils import save_colorbar_single
    
    output = "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/sea_level_pressure.png"
    save_colorbar_single(
        COLORS, 
        LEVELS, 
        output, 
        label="10-Meter Wind Speed (mph) with Sea Level Pressure Contours (mb)",
        extend='max'
    )
