"""
Dust RGB Product
"""

import time
import numpy as np
import cartopy.crs as ccrs
from .registry import register
from wxmaps_utils import load_color_table

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------

@register("dust_rgb")
def plot_dust_rgb(fig, ax, plotter, reader, args):
    """
    Plot Dust RGB Image
    """
    from wxmaps_utils import normalize

    data, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["TBRB05RG"]
    )
    bt05 = data.astype(np.float32) - 273.15 # Celcius

    data, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["TBRB06RG"]
    )
    bt06 = data.astype(np.float32) - 273.15 # Celcius

    data, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["TBRB07RG"]
    )
    bt07 = data.astype(np.float32) - 273.15 # Celcius

    data, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["TBRB08RG"]
    )
    bt08 = data.astype(np.float32) - 273.15 # Celcius

    data, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["TBRB09RG"]
    )
    bt09 = data.astype(np.float32) - 273.15 # Celcius

    rmin = -6.7
    rmax = 2.6
    rgam = 1.0

    gmin = -0.5
    gmax = 20.0
    ggam = 1.5

    bmin = -11.95
    bmax = 15.5
    bgam = 1.0


    # Dust RGB differences
    red_raw = np.clip(0.5*(bt05+bt06) - 0.5*(bt06+bt07), rmin, rmax)
    grn_raw = np.clip(bt06 - 0.5*(bt08+bt09), gmin, gmax)
    blu_raw = np.clip(0.5*(bt06+bt07), bmin, bmax)

    print(f"R: min={np.nanmin(red_raw):.6f}, max={np.nanmax(red_raw):.6f}, mean={np.nanmean(red_raw):.6f}")
    print(f"G: min={np.nanmin(grn_raw):.6f}, max={np.nanmax(grn_raw):.6f}, mean={np.nanmean(grn_raw):.6f}")
    print(f"B: min={np.nanmin(blu_raw):.6f}, max={np.nanmax(blu_raw):.6f}, mean={np.nanmean(blu_raw):.6f}")

    # normalization
    R = normalize(red_raw, rmin, rmax, rgam)
    G = normalize(grn_raw, gmin, gmax, ggam)
    B = normalize(blu_raw, bmin, bmax, bgam)
    rgb = np.dstack([R, G, B]).astype(np.float32)

    print(f"R: min={np.nanmin(R):.6f}, max={np.nanmax(R):.6f}, mean={np.nanmean(R):.6f}")
    print(f"G: min={np.nanmin(G):.6f}, max={np.nanmax(G):.6f}, mean={np.nanmean(G):.6f}")
    print(f"B: min={np.nanmin(B):.6f}, max={np.nanmax(B):.6f}, mean={np.nanmean(B):.6f}")

    # Limb Darkening
    #rgb = plotter.apply_limb_darkening(
    #    rgb,
    #    lats,
    #    lons
    #)
    # The RGB array is shape (height, width, 3) with values in range [0, 1]
    # Plot image on map - let Cartopy handle the reprojection
    start = time.time()
    from wxmaps_transform_cache import get_transform_cache
    transform_cache = get_transform_cache()
    # Get or load cached transform
    target_extent = ax.get_extent()
    target_shape = (2160, 4320)  # height, width for target
    transform_data, cache_key = transform_cache.get_or_compute_transform(
        source_proj=ccrs.PlateCarree(),
        target_proj=ax.projection,
        source_extent=[-180, 180, -90, 90],  # Adjust to your data extent
        target_extent=target_extent,
        target_shape=target_shape
    )
    # Apply the cached transform to this worker's data
    warped_rgb, warped_extent = transform_cache.apply_transform(
        source_data=rgb,
        transform_data=transform_data,
        fill_value=0.0,
        flip_source_y=True  # Don't flip for NetCDF data
    )
    # Plot the transformed image (FAST!)
    im = ax.imshow(
        warped_rgb,
        origin='upper',
        extent=warped_extent,
        transform=ax.projection,
        interpolation='antialiased',
        resample=False,
        zorder=4,
        rasterized=True
    )
    print(f"Dust RGB finished: {time.time() - start:.2f}s")

def generate_colorbar():
    """Generate colorbar"""
    from wxmaps_utils import save_colorbar_single
    
    # Use representative tick levels instead of all 256
    tick_levels = np.array([-110, -80, -50, -20, 0, 20, 40, 57])
    
    output = "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/dust_rgb.png"
    save_colorbar_single(
        COLORS, 
        LEVELS, 
        output, 
        label="Dust RGB Brightness Temperature (°C)",
        extend='both'
    )


