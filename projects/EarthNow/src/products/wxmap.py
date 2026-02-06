"""
Radar Reflectivity Product
Maximum Composite Reflectivity (DBZ_MAX / REFC)
"""

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from .registry import register

# ------------------------------------------------------------------
# Reflectivity colormap + levels (wxmaps-style)
# ------------------------------------------------------------------

import numpy as np

# Rain color table (15 colors, normalized)
rain_colors = np.array([
    [119,254, 55],
    [ 75,225, 42], 
    [ 28,195, 54],
    [ 22,165, 42],
    [ 18,135, 32],
    [ 14,110, 27],
    [255, 255,   0],  # Bright yellow
    [255, 220,   0],  # Yellow-orange
    [255, 185,   0],  # Orange-yellow
    [255, 150,   0],  # Orange
    [255, 100,   0],  # Red-orange
    [255,  50,   0],  # Orange-red
    [225,  20,  20],  # Red
    [175,   0,   0],  # Dark red
    [150,   0,   0]   # Darker red
]) / 255.0

# Snow color table (15 colors, normalized)
snow_colors = np.array([
    [144, 226, 255],  # Light blue (starting color)
    [119, 210, 244],  # Light blue
    [ 94, 194, 234],  # Blue
    [ 74, 179, 224],  # Blue
    [ 54, 165, 215],  # Medium blue
    [ 54, 145, 215],  # Medium blue
    [ 60, 125, 215],  # Blue-purple
    [ 71, 109, 213],  # Blue-purple
    [ 82,  93, 212],  # Blue-purple
    [ 94,  81, 211],  # Purple
    [106,  74, 211],  # Purple
    [123,  62, 210],  # Purple
    [115,  60, 188],  # Dark purple
    [107,  58, 167],  # Dark purple
    [ 99,  56, 146]   # Darker purple
]) / 255.0

# mix color table (15 colors, normalized)
mix_colors = np.array([
    [255, 204, 229],  # Light pink (starting color)
    [251, 190, 221],  # Light pink
    [248, 176, 213],  # Pink
    [244, 161, 205],  # Pink
    [240, 147, 197],  # Pink
    [237, 133, 189],  # Medium pink
    [233, 118, 181],  # Medium pink
    [229, 104, 173],  # Pink-purple
    [226,  90, 165],  # Pink-purple
    [222,  76, 157],  # Purple-pink
    [218,  61, 149],  # Purple-pink
    [215,  47, 141],  # Purple
    [211,  33, 133],  # Purple
    [207,  18, 125],  # Magenta
    [204,   0, 102]   # Dark magenta (ending color)
]) / 255.0

# Freezing rain color table (15 colors, normalized)
frzr_colors = np.array([
    [255, 235, 235],  # Very light pink/white
    [255, 191, 214],  # Light pink
    [255, 143, 194],  # Light pink
    [255,  97, 173],  # Pink
    [255,  51, 153],  # Pink
    [255,  34, 146],  # Hot pink
    [255,  17, 140],  # Hot pink
    [255,   8, 133],  # Magenta-pink
    [255,   0, 127],  # Magenta-pink
    [242,   0, 120],  # Red-magenta
    [229,   0, 114],  # Red-magenta
    [216,   0, 108],  # Red
    [204,   0, 102],  # Red
    [191,   0,  95],  # Dark red
    [178,   0,  89]   # Dark red
]) / 255.0
FZRA_COLORS = [
    "#FFF3E0",  # 5–10  very light amber
    "#FFE0B2",  # 10–15
    "#FFCC80",  # 15–20
    "#FFB74D",  # 20–25
    "#FFA726",  # 25–30
    "#FF9800",  # 30–35
    "#FB8C00",  # 35–40
    "#F57C00",  # 40–45
    "#EF6C00",  # 45–50
    "#E65100",  # 50–55
    "#D84315",  # 55–60
    "#BF360C",  # 60–65
    "#A84300",  # 65–70
    "#8D2E00",  # 70–75
    "#5D1F00",  # >75 severe icing
]

REFL_LEVELS = np.arange(5.0, 80.0, 5.0)  # 5–75 dBZ

# ------------------------------------------------------------
# Colormap + normalization
# ------------------------------------------------------------
r_cmap = ListedColormap(rain_colors)
r_norm = BoundaryNorm(REFL_LEVELS, ncolors=r_cmap.N, clip=True)

f_cmap = ListedColormap(FZRA_COLORS)
f_norm = BoundaryNorm(REFL_LEVELS, ncolors=f_cmap.N, clip=True)
    
i_cmap = ListedColormap(mix_colors)
i_norm = BoundaryNorm(REFL_LEVELS, ncolors=i_cmap.N, clip=True)
    
s_cmap = ListedColormap(snow_colors)
s_norm = BoundaryNorm(REFL_LEVELS, ncolors=s_cmap.N, clip=True)

# Test contour levels for topography (in meters)
levels = [0, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
p_cmap = plt.cm.terrain
p_norm = BoundaryNorm(levels, p_cmap.N, extend='max')

h1000_levels = [0, 10, 20, 50, 100, 150, 200, 250, 300, 350]
h1000_cmap = plt.cm.viridis
h1000_norm = BoundaryNorm(h1000_levels, h1000_cmap.N, extend='max')

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------

@register("wxmap")
def plot_wxmap(fig, ax, plotter, reader, args):
    """
    Plot GEOS WxMap Product
    """
    # Read from reader (reader decides the collection)

    eps = 1.0e-20

    phis, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["PHIS", "HGT_SFC"]
    )
    phis = phis.astype(np.float32)/9.81

    t2m, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["T2M", "TMP_2M"]
    )
    t2m = t2m.astype(np.float32)
    
    h500, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["H500"]
    )
    h500 = h500.astype(np.float32)
    
    h1000, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["H1000"]
    )
    h1000 = h1000.astype(np.float32)

    # 500-1000 thickness
    thck =  h500-h1000

    dbz, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["DBZ_MAX", "REFC"]
    )
    dbz = dbz.astype(np.float32)
    # Mask dBZ below 5.0
    dbz = np.ma.masked_where(dbz < 5.0, dbz)

    rain, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["RAIN"]
    )
    rain = rain.astype(np.float32)*3600.0/25.4

    mix, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["ICE"]
    )
    mix = mix.astype(np.float32)*3600.0/25.4

    snow, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["SNOW"]
    )
    snow = snow.astype(np.float32)*3600.0/25.4

    # determine Freezing Rain
    # First, compute freezing rain amount
    frzr = np.where(t2m < 273.15, rain, 0.0)

    print(f"  T2M  range: [{t2m.min():.2e}, {t2m.max():.2e}]")
    print(f"  THCK range: [{thck.min():.2e}, {thck.max():.2e}]")
    print(f"  Rain range: [{rain.min():.2e}, {rain.max():.2e}]")
    print(f"  Snow range: [{snow.min():.2e}, {snow.max():.2e}]") 
    print(f"  Mix  range: [{mix.min():.2e}, {mix.max():.2e}]") 
    print(f"  Frzr range: [{frzr.min():.2e}, {frzr.max():.2e}]") 

    # ------------------------------------------------------------------
    # Zero snow where T2M > 276.483 K
    # ------------------------------------------------------------------
    mask = (snow > 0.0) & (t2m <= 276.483)
    snow[~mask] = 0.0

    # ------------------------------------------------------------------
    # Zero mix where T2M > 276.483 K
    # ------------------------------------------------------------------
    mask = (mix > 0.0) & (t2m <= 276.483)
    mix[~mask] = 0.0

    # ------------------------------------------------------------------
    # Elevation factor
    # ------------------------------------------------------------------
    elevFactor = (phis - 305.0) / 915.0
    # Clamp to [0, 1]
    elevFactor = np.clip(elevFactor, 0.0, 1.0)
    # Scale
    elevFactor = elevFactor * 50.0

    # ------------------------------------------------------------------
    # Thickness thresholds
    # ------------------------------------------------------------------
    thLow  = 5425.0 + elevFactor
    thHigh = 5475.0 + elevFactor

    # ------------------------------------------------------------------
    # Mixed-phase region
    # (Does not support all snow at low elevation — preserved behavior)
    # ------------------------------------------------------------------
    mask = (
        (thck > thLow) &
        (thck < thHigh) &
        ((snow > eps) | (mix > eps))
    )
    mix[mask]  = snow[mask]
    snow[mask] = 0.0
    # Zero mix outside mixed-phase region
    mix[~mask] = 0.0

    # ------------------------------------------------------------------
    # Above upper thickness threshold → no snow
    # ------------------------------------------------------------------
    mask = thck >= thHigh
    snow[mask] = 0.0

    # ------------------------------------------------------------------
    # Intensity-based phase prioritization
    # Ties within 1.e-6 go to mix
    # ------------------------------------------------------------------
    frzr_dom = (
        (frzr > mix + eps) &
        (frzr > snow + eps)
    )
    snow_dom = (
        (snow > frzr + eps) &
        (snow > mix + eps)
    )
    # Mix wins ties or near-ties
    mix_dom = (
        ~(frzr_dom | snow_dom) &
        (mix > eps)
    )
    frzr[~frzr_dom] = 0.0
    snow[~snow_dom] = 0.0
    mix[~mix_dom]   = 0.0


    # Mask below 0.1 mm/hour
    rain_dbz = np.ma.masked_where(rain < eps, dbz)
    frzr_dbz = np.ma.masked_where(frzr < eps, dbz)
    snow_dbz = np.ma.masked_where(snow < eps, dbz)
    mix_dbz = np.ma.masked_where(mix < eps, dbz)

    # ------------------------------------------------------------
    # Plot rain
    # ------------------------------------------------------------
    ax.pcolormesh(
        lons,
        lats,
        rain_dbz,
        cmap=r_cmap,
        norm=r_norm,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )

    # ------------------------------------------------------------
    # Plot frzr
    # ------------------------------------------------------------
    ax.pcolormesh(
        lons,
        lats,
        frzr_dbz,
        cmap=f_cmap,
        norm=f_norm,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )

    # ------------------------------------------------------------
    # Plot snow
    # ------------------------------------------------------------
    ax.pcolormesh(
        lons,
        lats,
        snow_dbz,
        cmap=s_cmap,
        norm=s_norm,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )

    # ------------------------------------------------------------
    # Plot mix
    # ------------------------------------------------------------
    ax.pcolormesh(
        lons,
        lats,
        mix_dbz,
        cmap=i_cmap,
        norm=i_norm,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )


    # TEST: Plot phis to check values
    #ax.pcolormesh(
    #    lons,
    #    lats,
    #    phis,
    #    cmap=p_cmap,
    #    norm=p_norm,
    #    transform=ccrs.PlateCarree(),
    #    zorder=4,
    #)

    # TEST: Plot H1000 to check values
    #ax.pcolormesh(
    #    lons,
    #    lats,
    #    h1000,
    #    cmap=h1000_cmap,
    #    norm=h1000_norm,
    #    transform=ccrs.PlateCarree(),
    #    zorder=4,
    #)
    ## Add colorbar for H1000
    #cbar = plt.colorbar(ax.collections[-1], ax=ax, orientation='horizontal',
    #                    pad=0.05, shrink=0.6)
    #cbar.set_label('H1000 (m)', fontsize=10)

    # ------------------------------------------------------------
    # Plot 500-1000mb thickness
    # ------------------------------------------------------------
    # Cold side (≤5400m) - blue dashed
    #contours_cold = ax.contour(lons, lats, thck,
    #           levels=np.arange(4800, 5440, 40),
    #           colors='blue', linewidths=1., linestyles='dashed',
    #           transform=ccrs.PlateCarree(), zorder=8)
    #ax.clabel(contours_cold, inline=True, fontsize=10, fmt='%d')

    # Warm side (>5400m) - red dashed  
    #contours_warm = ax.contour(lons, lats, thck,
    #           levels=np.arange(5440, 6000, 40),
    #           colors='#bd0202', linewidths=1., linestyles='dashed',
    #           transform=ccrs.PlateCarree(), zorder=8)
    #ax.clabel(contours_warm, inline=True, fontsize=10, fmt='%d')

    # Critical thickness line (5400m) - bold purple
    #contour_crit = ax.contour(lons, lats, thck,
    #           levels=[5400],
    #           colors='purple', linewidths=1.,
    #           transform=ccrs.PlateCarree(), zorder=9)
    #ax.clabel(contour_crit, inline=True, fontsize=10, fmt='%d')

    # T2M test contours
    #contours_t2m = ax.contour(lons, lats, t2m-273.15,
    #           levels=np.arange(-100, 5, 5),
    #           colors='purple', linewidths=1.,
    #           transform=ccrs.PlateCarree(), zorder=8)
    #ax.clabel(contours_t2m, inline=True, fontsize=10, fmt='%d')


def generate_colorbar():
    """Generate 2x2 colorbar grid for wxmap product"""
    from wxmaps_utils import save_colorbar_grid
    
    specs = [
        {
            'colors': rain_colors,
            'levels': REFL_LEVELS,
            'label': 'Rain (dBZ)',
            'extend': 'max'
        },
        {
            'colors': np.array([tuple(int(c.lstrip('#')[i:i+2], 16)/255.0 for i in (0, 2, 4)) 
                               for c in FZRA_COLORS]),  # Convert hex to RGB
            'levels': REFL_LEVELS,
            'label': 'Freezing Rain (dBZ)',
            'extend': 'max'
        },
        {
            'colors': snow_colors,
            'levels': REFL_LEVELS,
            'label': 'Snow (dBZ)',
            'extend': 'max'
        },
        {
            'colors': mix_colors,
            'levels': REFL_LEVELS,
            'label': 'Ice Pellets/Mix (dBZ)',
            'extend': 'max'
        }
    ]
    
    title = "Precipitation Type by Reflectivity\n(1000-500mb Thickness: Blue dashed <=5400m, Red dashed >5400m)"
    
    output = "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/wxmap.png"
    save_colorbar_grid(specs, output, title=title, grid_shape=(2, 2))
