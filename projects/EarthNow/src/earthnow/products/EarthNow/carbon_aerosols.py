"""
carbon aerosol Product
"""

from typing import Sequence, Optional
import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import LinearSegmentedColormap, Normalize
from earthnow.products.registry import register

fieldname = 'carbon_EarthNow'
# ------------------------------------------------------------------
# Variable definitions
# ------------------------------------------------------------------

CARBON_VARIABLES = ["OCEXTTAU", "BREXTTAU", "BCEXTTAU"]

CB_PATH = "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars"

# ------------------------------------------------------------------
# Reflectivity colormap + levels (wxmaps-style)
# ------------------------------------------------------------------

LEVELS = [
    0.0,
    1.0,
]

COLORS = (
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

ALPHA_MIN = 0.0
ALPHA_MAX = 0.125

def build_continuous_colormap(
    name: str = fieldname, 
    colors: Optional[Sequence]=None, 
    levels: Optional[Sequence]=None
    ) -> LinearSegmentedColormap:
    """Build the continuous colormap from anchor colors."""
    colors = colors or COLORS
    levels = levels or LEVELS
    vmin = min(levels)
    vmax = max(levels)

    positions = (np.array(levels) - vmin) / (vmax - vmin)    
    return LinearSegmentedColormap.from_list(
        name, 
        list(zip(positions, colors[:-1])), 
        N=256
    )


def build_alpha(data: np.ndarray, vmin: float = ALPHA_MIN, vmax: float = ALPHA_MAX) -> np.ndarray:
    """
    Mimic the IDL alpha behavior:
    aImageTV = 255 * (image_bytscl(data, MIN=MIN(alevs), MAX=MAX(alevs)) / 255.0)

    Returned alpha is 0..1 for matplotlib.
    """
    scaled = (data - vmin) / (vmax - vmin)
    return np.clip(scaled, 0.0, 1.0)

def print_shape(name,var):
    print(f'Shape of {name}:')
    try:
        print(np.shape(var))
    except Exception:
        print('Nope, not going to print')

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------

def ensure_2d(arr: np.ndarray, name: str, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(arr, dtype=dtype)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim != 2:
        raise ValueError(f"{name} expected 2D or (1, y, x), got shape {arr.shape}")
    return arr
    
@register(fieldname)
def plot_carbon(fig, ax, plotter, reader, args):
    """
    Plot Carbon Aerosols 
    """
    fields = {}
    for field in CARBON_VARIABLES:
        f, lats, lons, meta = reader.read_variable(
            args.fdate,
            args.pdate,
            variables=[field],
        )
        print_shape(field, f)
        print(meta)
        fields[field] = f

    # Depending on your reader, `fields` may come back as:
    #   - dict[str, np.ndarray]
    #   - tuple/list of arrays
    # The dict pattern is the cleanest, so handle both.
    if isinstance(fields, dict):
        # oc = np.asarray(fields["OCEXTTAU"], dtype=np.float32)
        # br = np.asarray(fields["BREXTTAU"], dtype=np.float32)
        # bc = np.asarray(fields["BCEXTTAU"], dtype=np.float32)
        oc = ensure_2d(fields["OCEXTTAU"], "OCEXTTAU", dtype=np.float32)
        br = ensure_2d(fields["BREXTTAU"], "BREXTTAU", dtype=np.float32)
        bc = ensure_2d(fields["BCEXTTAU"], "BCEXTTAU", dtype=np.float32)
    # else:
    #     oc, br, bc = [np.asarray(arr, dtype=np.float32) for arr in fields]

    carbon = oc + br + bc

    # Mask obviously bad values if needed.
    # carbon = np.where(np.isfinite(carbon), carbon, np.nan)


    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    vmin = min(LEVELS)
    vmax = max(LEVELS)

    cmap = build_continuous_colormap()
    norm = Normalize(vmin=vmin, vmax=vmax)
    alpha = build_alpha(carbon)
    print_shape('lons', lons)
    print_shape('lats', lats)
    print_shape('data', carbon)
    print_shape('cmap', cmap)
    print_shape('norm', norm)
    # ------------------------------------------------------------
    # Plot field
    # ------------------------------------------------------------
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


def generate_colorbar(path: Optional[str]=None):
    """Generate colorbar for carbon aerosol optical thickness."""
    from earthnow.wxmaps_utils import save_colorbar_single
    path = path or CB_PATH
    cmap = build_continuous_colormap()
    colors = cmap(np.linspace(0, 1, 256))[:, :3]

    output = (
        f"{path}/{fieldname}.png"
    )
    save_colorbar_single(
        colors,
        [0.0, 1.0],
        output,
        label="Carbon Aerosol Optical Thickness",
        extend="max",
    )
# def generate_colorbar():
#     """Generate colorbar for 2m temperature"""
#     from earthnow.wxmaps_utils import save_colorbar_single

#     output = (
#         "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/temperature_2m.png"
#     )
#     save_colorbar_single(
#         COLORS, LEVELS, output, label="2-Meter Temperature (°F)", extend="both"
#     )
