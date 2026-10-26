"""
Sandwich product: Longwave window IR (band 14) blended with a
vegetation field (green channel) derived from OSRB09RG.
Based on ploteic_sandwich_eblocal.pro
"""

import datetime
import time
import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from earthnow.products.registry import register
from earthnow.wxmaps_utils import load_color_table
from earthnow import paths

# ------------------------------------------------------------------
# Longwave IR colormap + levels (wxmaps-style)
# ------------------------------------------------------------------

COLORS = load_color_table(paths.colortable("NESDIS_IR_10p3micron.txt"))

clevs = [-110.0, -59, -20, 6, 31, 57]  # Celcius
LEVELS = np.interp(5 * np.arange(256) / 255.0, np.arange(len(clevs)), clevs)

IR_ALPHA = 0.66
ALPHA_LIMIT = 600.0  # SWTDN (W m-2) upper bound used by IDL normalize


# ------------------------------------------------------------------
# Solar geometry + vegetation helpers (from geocolor_rgb.py)
# ------------------------------------------------------------------


def solar_distance_au(pdate):
    """
    Compute Earth–Sun distance in AU for the valid time.
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


def vegetation_field(osr, isr, sun_distance_au, min_solar_threshold=0.001):
    """
    Vegetation (green-channel) reflectance from NIR OSR, scaled by the
    incoming solar flux. Matches the IDL factor * osr * data_a construction.
    Returns values clipped to [0, 1] and the solar alpha layer.
    """
    isr_masked = np.where(isr < min_solar_threshold, np.nan, isr)
    factor = np.pi / (isr_masked * sun_distance_au**2)

    osr = np.where(np.isfinite(osr), osr, 0.0)
    alpha = np.clip(isr / ALPHA_LIMIT, 0.0, 1.0)

    veg = np.nan_to_num(factor * osr * alpha, nan=0.0, posinf=0.0, neginf=0.0)
    veg = np.clip(veg, 0.0, 1.0)

    # IDL sqrt enhancement
    veg = np.sqrt(veg)
    return veg, alpha


# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------


@register("blended_LWIR_veg_EarthNow")
def plot_blended_LWIR_veg(fig, ax, plotter, reader, args):
    """
    Plot longwave window IR blended with vegetation field
    """
    # ------------------------------------------------------------
    # Longwave IR (band 14)
    # ------------------------------------------------------------
    data, lats, lons, meta = reader.read_variable(
        args.fdate, args.pdate, variables=["TBRB06RG"]
    )
    data = data.astype(np.float32) - 273.15  # Celcius

    # ------------------------------------------------------------
    # Vegetation field (IDL data_g / lccdata_g)
    # ------------------------------------------------------------
    isr, _, _, _ = reader.read_variable(args.fdate, args.pdate, variables=["SWTDN"])
    osr_nir, _, _, _ = reader.read_variable(
        args.fdate, args.pdate, variables=["OSRB09RG"]
    )
    sun_dist = solar_distance_au(args.pdate)
    veg, veg_alpha = vegetation_field(
        osr_nir.astype(np.float32), isr.astype(np.float32), sun_dist
    )

    # ------------------------------------------------------------
    # IR RGBA (underlay)
    # ------------------------------------------------------------
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(LEVELS, ncolors=cmap.N, clip=True)
    ir_rgba = cmap(norm(data))
    ir_rgba[..., 3] = np.where(np.isfinite(data), IR_ALPHA, 0.0)

    # ------------------------------------------------------------
    # Vegetation RGBA (underlay, grayscale green level)
    # ------------------------------------------------------------
    veg_rgb = np.dstack([veg, veg, veg])
    va = veg_alpha[..., None]
    ia = ir_rgba[..., 3:4]

    # ------------------------------------------------------------
    # Composite IR over vegetation (IDL draws m1ImageObj after m0ImageObj)
    # ------------------------------------------------------------
    out_a = ia + va * (1.0 - ia)
    out_rgb = (ia * ir_rgba[..., :3] + va * (1.0 - ia) * veg_rgb) / np.maximum(
        out_a, 1e-6
    )
    rgba = np.dstack([out_rgb, out_a])

    # ------------------------------------------------------------
    # Plot image on map
    # ------------------------------------------------------------
    start = time.time()
    from earthnow.wxmaps_transform_cache import get_transform_cache

    transform_cache = get_transform_cache()
    target_extent = ax.get_extent()
    target_shape = (2160, 4320)  # height, width for target
    transform_data, cache_key = transform_cache.get_or_compute_transform(
        source_proj=ccrs.PlateCarree(),
        target_proj=ax.projection,
        source_extent=[-180, 180, -90, 90],
        target_extent=target_extent,
        target_shape=target_shape,
    )
    warped_rgba, warped_extent = transform_cache.apply_transform(
        source_data=rgba,
        transform_data=transform_data,
        fill_value=0.0,
        flip_source_y=True,
    )
    ax.imshow(
        warped_rgba,
        origin="upper",
        extent=warped_extent,
        transform=ax.projection,
        interpolation="antialiased",
        resample=False,
        zorder=4,
        rasterized=True,
    )
    print(f"Blended LWIR/veg composite finished: {time.time() - start:.2f}s")


def generate_colorbar():
    """Generate colorbar for longwave window IR"""
    from earthnow.wxmaps_utils import save_colorbar_single

    # Use representative tick levels instead of all 256
    tick_levels = np.array([-110, -80, -50, -20, 0, 20, 40, 57])

    output = paths.colorbar_output("longwave_window_ir.png")
    save_colorbar_single(
        COLORS,
        LEVELS,
        output,
        label="11.2 μm Longwave Window Brightness Temperature (°C)",
        extend="both",
    )
