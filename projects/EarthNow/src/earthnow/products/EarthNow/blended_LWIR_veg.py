"""
Sandwich RGB: longwave window IR (band 14) blended over a vegetation field.
Translation of ploteic_sandwich_eblocal.pro (CONUS branch)

Plots on the native 2-D lons/lats grid, like longwave_window_ir.py.

NOTE: The IDL script reads OSRB08RG (RRTMG) or OSRB09RG (RRTMGP). This
translation always uses OSRB09RG for now.
"""

import datetime
import numpy as np
import cartopy.crs as ccrs
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, Normalize
from earthnow.products.registry import register
from earthnow import paths

# ------------------------------------------------------------------
# Constants (from ploteic_sandwich_eblocal.pro)
# ------------------------------------------------------------------

LWIR_MIN = -93.0  # Celsius, coldest level in the IDL clevs
LWIR_MAX = -15.0  # Celsius, warmest level in the IDL clevs
LWIR_OPACITY = 0.66  # IDL: 0.66*255 alpha for the LWIR layer
ALPHA_LIMIT = 600.0  # IDL: normalize(SWTDN, 0, 600)
FILL_VALUE = 1e14  # IDL undef = 1e15; anything above this is missing

# ------------------------------------------------------------------
# Colormap + levels (turbo replaces IDL colortable 33)
# ------------------------------------------------------------------

COLORS = colormaps["turbo_r"](np.linspace(0, 1, 256))[:, :3]
LEVELS = np.linspace(LWIR_MIN, LWIR_MAX, 256)  # Celsius, one per color index


# ------------------------------------------------------------------
# Helpers (same logic as geocolor_rgb.py)
# ------------------------------------------------------------------


def solar_distance_au(pdate):
    """
    Compute Earth–Sun distance in AU for the valid time.
    LAZY IMPORT astropy here - only when actually called
    """
    from astropy.time import Time
    from astropy.coordinates import get_sun
    import astropy.units as u

    if isinstance(pdate, datetime.datetime):
        t = Time(pdate)
    elif isinstance(pdate, np.datetime64):
        t = Time(pdate.astype("datetime64[s]").astype(datetime.datetime))
    elif isinstance(pdate, str):
        p = pdate.strip()
        if p.lower().endswith("z"):
            p = p[:-1]
        for fmt in ("%Y%m%d_%H%M", "%Y%m%d%H", "%Y-%m-%d %H:%M:%S"):
            try:
                t = Time.strptime(p, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Unrecognized pdate format: {pdate}")
    else:
        raise TypeError(f"Unsupported pdate type: {type(pdate)}")

    return get_sun(t).distance.to(u.au).value


def read_field(reader, args, name):
    data, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=[name]
    )
    data = np.ma.filled(np.ma.asarray(data, dtype=np.float32), np.nan)
    data = np.where(np.abs(data) >= FILL_VALUE, np.nan, data)
    return data, lats, lons


# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------


@register("blended_LWIR_veg_EarthNow")
def plot_blended_LWIR_veg(fig, ax, plotter, reader, args):
    """
    Plot longwave window IR blended over vegetation field
    """
    # ------------------------------------------------------------
    # Longwave IR (band 14) -> Celsius (lccdataG in IDL)
    # ------------------------------------------------------------
    tb, lats, lons = read_field(reader, args, "TBRB06RG")
    lwir_c = tb - 273.15

    # ------------------------------------------------------------
    # Solar alpha (lccdata_a in IDL): normalize(SWTDN, 0, 600)
    # ------------------------------------------------------------
    isr, _, _ = read_field(reader, args, "SWTDN")
    isr = np.nan_to_num(isr, nan=0.0)
    alpha = np.clip(isr / ALPHA_LIMIT, 0.0, 1.0)

    # ------------------------------------------------------------
    # Vegetation field (lccdata_g in IDL)
    # factor = pi / (ISR * d^2); osr NaN -> 0; night (ISR = 0) -> 0
    # ------------------------------------------------------------
    osr, _, _ = read_field(reader, args, "OSRB09RG")
    osr = np.nan_to_num(osr, nan=0.0)
    sun_dist = solar_distance_au(args.pdate)
    with np.errstate(divide="ignore", invalid="ignore"):
        veg = np.where(isr > 0, np.pi * osr / (isr * sun_dist**2), 0.0)
    veg = np.nan_to_num(veg, nan=0.0, posinf=0.0, neginf=0.0)
    veg = veg * alpha

    # IDL: 255*SQRT(bytscl(x, 0, 1)/255) -> sqrt of the clipped field on 0-1
    veg_gray = np.sqrt(np.clip(veg, 0.0, 1.0))
    veg_alpha = alpha

    # ------------------------------------------------------------
    # LWIR colormap (IDL CONUS contour branch)
    # Levels span -93..-15 C; values outside that range are not drawn
    # ------------------------------------------------------------
    valid = np.isfinite(lwir_c) & (lwir_c >= LWIR_MIN) & (lwir_c <= LWIR_MAX)
    frac = (lwir_c - LWIR_MIN) / (LWIR_MAX - LWIR_MIN)
    idx = np.clip(np.floor(np.nan_to_num(frac) * 256.0), 0, 255).astype(np.int64)
    lwir_rgb = COLORS[idx]
    lwir_alpha = np.where(valid, LWIR_OPACITY, 0.0).astype(np.float32)

    # ------------------------------------------------------------
    # Blend: vegetation bottom, LWIR on top (IDL Blend_Func=[3,4])
    # ------------------------------------------------------------
    veg_rgb = np.repeat(veg_gray[..., None], 3, axis=-1)
    out_alpha = lwir_alpha + veg_alpha * (1.0 - lwir_alpha)
    out_rgb = (
        lwir_rgb * lwir_alpha[..., None]
        + veg_rgb * (veg_alpha * (1.0 - lwir_alpha))[..., None]
    )
    out_rgb = out_rgb / np.maximum(out_alpha, 1e-6)[..., None]
    rgba = np.dstack([out_rgb, out_alpha]).astype(np.float32)

    # ------------------------------------------------------------
    # Plot on native grid: one color per cell, carrying the blended RGBA
    # ------------------------------------------------------------
    lats_c, lons_c, rgba_c = plotter.crop_data_to_view(lats, lons, rgba)
    ny, nx = rgba_c.shape[:2]
    n_colors = ny * nx
    cmap = ListedColormap(rgba_c.reshape(-1, 4))
    norm = Normalize(vmin=-0.5, vmax=n_colors - 0.5)
    cell_index = np.arange(n_colors, dtype=np.float64).reshape(ny, nx)

    ax.pcolormesh(
        lons_c,
        lats_c,
        cell_index,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        zorder=4,
    )


def generate_colorbar():
    """Generate colorbar for the blended LWIR / vegetation product"""
    from earthnow.wxmaps_utils import save_colorbar_single

    output = paths.colorbar_output("blended_LWIR_veg.png")
    save_colorbar_single(
        COLORS,
        LEVELS,
        output,
        label="11.2 μm Longwave Window Brightness Temperature (°C)",
        extend="both",
    )
