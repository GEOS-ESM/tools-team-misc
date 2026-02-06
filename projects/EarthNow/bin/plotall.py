#!/usr/bin/env python3
"""
plotall.py

Generic plotting driver for WxMaps.
Basemap is implemented as the first product.

"""

import argparse
import sys
import cartopy.crs as ccrs  # ADD THIS LINE
import numpy as np

from wxmaps_config import WxMapsConfig, StyleConfig
from wxmaps_plotting import WxMapPlotter
from wxmaps_utils import (
    get_output_filepath,
    parse_date_string,
)

from data_readers import DATA_READERS
from products import PRODUCTS

# -----------------------------------------------------------------------------
# ARGPARSE
# -----------------------------------------------------------------------------

def parse_color_arg(color_str: str) -> str:
    return color_str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic WxMaps plotting driver"
    )

    # -------------------------------------------------------------------------
    # Core required args
    # -------------------------------------------------------------------------
    parser.add_argument("--data-reader", 
                        default="geos_cycled_replays",
                        choices=DATA_READERS.keys(),
                        help="Data reader type")

    parser.add_argument("--product", required=True,
                        choices=PRODUCTS.keys(),
                        help="Plot product type")

    parser.add_argument("--map-type", required=True,
                        help="Map domain (e.g. conus, europe)")

    parser.add_argument("--fdate", required=True)

    # -------------------------------------------------------------------------
    # Worker args (parallel over pdates)
    # -------------------------------------------------------------------------
    parser.add_argument("--pdate", default=None)
    parser.add_argument('--nproc', type=int, default=1,
                       help='Number of parallel workers')
    parser.add_argument('--compress-level', type=int, default=6,
                       help='Compression during plotting: 0=none, 1=minimal (default: 0)')
    parser.add_argument('--optimize-after', action='store_true',
                       help='Re-compress PNGs after plotting for smaller files')
    parser.add_argument('--optimize-workers', type=int, default=1,
                       help='Workers for optimization phase (default: 1)')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='Directory for caching transforms (default: $TMPDIR/wxmaps_transform_cache)')

    # -------------------------------------------------------------------------
    # Experiment / IO
    # -------------------------------------------------------------------------
    parser.add_argument("--exp-path",
                        default="/discover/nobackup/projects/gmao/osse2/HWT",
                        help="Experiment path (for cycled replays)")
    parser.add_argument("--exp-res", default="CONUS02KM",
                        help="Experiment resolution (for cycled replays)")
    parser.add_argument("--exp-id", default="Feature-c2160_L137",
                        help="Experiment ID")
    parser.add_argument("--collection", default="inst1_2d_asm_Nx",
                        help="Collection name (if overriding defaults)")

    # Forward Processing specific
    parser.add_argument("--fp-base-path",
                        default="/discover/nobackup/projects/gmao/gmao_ops/pub",
                        help="Base path for Forward Processing data")

    parser.add_argument("--base-path",
                        default="/discover/nobackup/projects/gmao/g6dev/pub/WxMaps")

    parser.add_argument("--resolution",
                        choices=["hd", "fhd", "2k", "4k", "8k"],
                        default="4k")

    parser.add_argument("--timezone", default="US/Eastern")

    # -------------------------------------------------------------------------
    # Map features
    # -------------------------------------------------------------------------
    parser.add_argument("--boundaries", nargs="+",
                        choices=["coastlines", "countries", "states", "counties",
                                 "rivers"])
    parser.add_argument("--boundaries_color", type=parse_color_arg)

    parser.add_argument("--feature_resolution",
                        choices=["10m", "50m", "110m"],
                        default="50m")

    parser.add_argument("--cities", action="store_true")
    parser.add_argument("--roads", action="store_true")
    parser.add_argument("--major-roads-only", action="store_true")

    # -------------------------------------------------------------------------
    # Style
    # -------------------------------------------------------------------------
    parser.add_argument("--style",
                        choices=["wxmaps", "light", "dark", "nightlights", "satellite", "print"],
                        default="wxmaps")

    parser.add_argument("--ocean_color", type=parse_color_arg)
    parser.add_argument("--land_color", type=parse_color_arg)
    parser.add_argument("--road_color", type=parse_color_arg)

    # -------------------------------------------------------------------------
    # NWS warnings
    # -------------------------------------------------------------------------
    parser.add_argument("--show-nws-warnings", action="store_true")
    parser.add_argument("--nws-shapefile-base",
                        default="/discover/nobackup/projects/gmao/osse2/TSE_staging/SHAPE_FILES/ALL")

    # -------------------------------------------------------------------------
    return parser.parse_args()


# -----------------------------------------------------------------------------
# READER FACTORY
# -----------------------------------------------------------------------------

def create_data_reader(args):
    """
    Create appropriate data reader based on data-reader type.
    
    Handles different initialization signatures for different readers.
    """
    ReaderClass = DATA_READERS[args.data_reader]
    
    if args.data_reader == "geos_forward_processing":
        # Forward Processing reader uses different parameters
        reader = ReaderClass(
            base_path=args.fp_base_path,
            exp_id=args.exp_id,
            collection=args.collection if args.collection != "inst1_2d_asm_Nx" else None
        )
    elif args.data_reader == "geos_cycled_replays":
        # Cycled Replays reader
        reader = ReaderClass(
            exp_path=args.exp_path,
            exp_res=args.exp_res,
            exp_id=args.exp_id,
            collection=args.collection,
            map_type=args.map_type
        )
    elif args.data_reader == "gencast_geos_fp":
        # GenCast reader
        reader = ReaderClass(
            exp_path=args.exp_path,
            exp_res=args.exp_res,
            exp_id=args.exp_id
        )
    else:
        # Generic fallback - try all parameters and let reader handle it
        try:
            reader = ReaderClass(
                exp_path=args.exp_path,
                exp_res=args.exp_res,
                exp_id=args.exp_id,
                collection=args.collection,
                map_type=args.map_type
            )
        except TypeError:
            # If that fails, try minimal parameters
            reader = ReaderClass(
                exp_path=args.exp_path,
                exp_id=args.exp_id
            )
    
    return reader


# -----------------------------------------------------------------------------
# STYLE BUILDER
# -----------------------------------------------------------------------------

def build_style(args):
    if args.style == "wxmaps":
        style = StyleConfig.wxmaps()
    elif args.style == "light":
        style = StyleConfig.light()
    elif args.style == "dark":
        style = StyleConfig.dark()
    elif args.style == "nightlights":
        style = StyleConfig.nightlights()
    elif args.style == "satellite":
        style = StyleConfig.satellite()
    elif args.style == "print":
        style = StyleConfig.print_quality()
    else:
        raise ValueError(f"Unknown style: {args.style}")

    if args.ocean_color:
        style.ocean_color = args.ocean_color
    if args.land_color:
        style.land_color = args.land_color

    if args.boundaries_color:
        style.coastline_color = args.boundaries_color
        style.country_color = args.boundaries_color
        style.state_color = args.boundaries_color

    if args.show_nws_warnings:
        style.show_nws_warnings = True
        style.nws_shapefile_base = args.nws_shapefile_base

    return style


# -----------------------------------------------------------------------------
# WORKER FUNCTION
# -----------------------------------------------------------------------------

def plot_single_pdate(pdate, args, style, map_config):
    import copy

    local_args = copy.copy(args)
    local_args.pdate = pdate

    print(f"Working on {local_args.product}: {local_args.pdate}")

    # Plotter (fresh per pdate)
    plotter = WxMapPlotter(
        map_config,
        resolution=local_args.resolution,
        style=style
    )

    fig, ax = plotter.create_basemap(
        boundaries=local_args.boundaries or [],
        feature_resolution=local_args.feature_resolution
    )

    # Create reader for this worker
    reader = create_data_reader(local_args)

    # Call product function
    PRODUCTS[local_args.product](fig, ax, plotter, reader, local_args)

    # Add optional features
    if local_args.roads:
        plotter.add_roads(major_only=local_args.major_roads_only)

    if local_args.cities:
        plotter.add_cities()
    
    # Add NWS warnings if requested (BEFORE timestamp so warnings are below text)
    if style.show_nws_warnings:
        from wxmaps_utils import parse_date_string
        pdate_dt = parse_date_string(local_args.pdate)
        plotter.add_nws_warnings(pdate_dt)

    if style.show_timestamp:
        plotter.add_forecast_timestamp(
            local_args.fdate,
            local_args.pdate,
            timezone=local_args.timezone
        )

    output = get_output_filepath(
        local_args.base_path,
        local_args.exp_res,
        local_args.exp_id,
        local_args.product,
        local_args.map_type,
        local_args.fdate,
        local_args.pdate
    )

    plotter.save(output, optimize=True)
    plotter.close()

    print(f"✓ Created {local_args.product}: {output}")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    # -------------------------------------------------------------------------
    # Map config
    # -------------------------------------------------------------------------
    all_maps = WxMapsConfig.get_all_standard_maps()
    if args.map_type not in all_maps:
        raise ValueError(f"Unknown map-type: {args.map_type}")

    map_config = all_maps[args.map_type]
    style = build_style(args)

    # =========================================================================
    # PRE-COMPUTE COORDINATE TRANSFORMS
    # =========================================================================
    if isinstance(map_config.projection, ccrs.Geostationary):
        from wxmaps_transform_cache import get_transform_cache
        import matplotlib.pyplot as plt
    
        print("\n" + "="*70)
        print("Pre-computing coordinate transformations...")
        print("="*70)
    
        # Get cache with custom directory if specified
        transform_cache = get_transform_cache(cache_dir=getattr(args, 'cache_dir', None))
        transform_cache.clear_cache()   # clear once here only
    
        # Create a minimal figure to get proper extent
        # Use actual output size to ensure extent matches what workers will have
        resolution_config = WxMapsConfig.RESOLUTIONS.get(args.resolution)
        fig_temp = plt.figure(figsize=resolution_config.figsize, dpi=resolution_config.dpi)
        ax_temp = fig_temp.add_subplot(1, 1, 1, projection=map_config.projection)

        if map_config.extent is None:
            ax_temp.set_global()
        else:
            ax_temp.set_extent(map_config.extent, crs=ccrs.PlateCarree())

        xmin, xmax = ax_temp.get_xlim()
        ymin, ymax = ax_temp.get_ylim()

        target_extent = (xmin, xmax, ymin, ymax)
        print(target_extent)

        ax_temp.set_xlim(xmin, xmax)
        ax_temp.set_ylim(ymin, ymax)

        plt.close(fig_temp)
    
        # Pre-compute the transform
        target_shape = (resolution_config.height, resolution_config.width)  # height, width
    
        transform_data, cache_key = transform_cache.get_or_compute_transform(
            source_proj=ccrs.PlateCarree(),
            target_proj=map_config.projection,
            source_extent=[-180, 180, -90, 90],
            target_extent=target_extent,
            target_shape=target_shape
        )
    
        # Store in style config so workers can access it
        style.cached_target_extent = target_extent
        style.cached_target_shape = target_shape
        style.cached_transform_key = cache_key
    
        print(f"✓ Transform cached with key: {cache_key[:16]}")
        print(f"  Target extent: {target_extent}")
        print(f"  Target shape: {target_shape}")
        print("  All workers will use this cached transformation.\n")
        print("="*70 + "\n")
    # =========================================================================

    # =========================================================================
    # PRELOAD BASE IMAGES (after transform is computed)
    # =========================================================================
    from wxmaps_base_images import preload_base_images_from_style
    
    # Determine target resolution based on output resolution
    resolution_map = {
        'hd': 4000,
        'fhd': 4000,
        '2k': 6000,
        '4k': 4000,
        '8k': 8000
    }
    target_resolution = resolution_map.get(args.resolution, 4000)
    
    # Store in style so workers can use the same value
    style.base_image_target_resolution = target_resolution
    
    # Preload base images (will use the same transform we just computed)
    preload_base_images_from_style(style, target_resolution=target_resolution)
    # =========================================================================

    # -------------------------------------------------------------------------
    # Data reader
    # -------------------------------------------------------------------------
    reader = create_data_reader(args)

    # -------------------------------------------------------------------------
    # Determine pdates
    # -------------------------------------------------------------------------
    if args.pdate is None:
        # If pdate not provided, plot ALL available times
        pdates = reader.find_available_times(args.fdate)
        pdates = [dt.strftime("%Y%m%d_%H%Mz") for dt in pdates]
    else:
        pdates = [args.pdate]

    print(f"Processing {len(pdates)} plot times on {args.nproc} CPUs")

    # -------------------------------------------------------------------------
    # PRODUCT DISPATCH (parallel)
    # -------------------------------------------------------------------------
    from functools import partial
    from concurrent.futures import ProcessPoolExecutor

    worker = partial(
        plot_single_pdate,
        args=args,
        style=style,
        map_config=map_config,
    )

    with ProcessPoolExecutor(max_workers=args.nproc) as exe:
        list(exe.map(worker, pdates))


if __name__ == "__main__":
    main()

