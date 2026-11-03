PRODUCT SETTINGS
================
See `tools-team-misc/projects/EarthNow/cycl/config.yml`

BASIC EARTHNOW INFORMATION (EXISTING IDL WORKFLOW)
==========================

## Files directory
**IDL source code**: `/home/wputman/IDL_BASE/`

**Images**: `/discover/nobackup/projects/gmao/g6dev/pub/OpsCONUS02KM_R`

**Movies**: `/discover/nobackup/projects/gmao/g6dev/pub/MOVIES_EIC_NOW`

**ColorBars**: `/discover/nobackup/projects/gmao/g6dev/pub/MOVIES_EIC_NOW/ColorBars`

**Timestamps**: `/discover/nobackup/projects/gmao/g6dev/pub/WebGL/geos_earth/include/geos/`

**Basemap Imagery**: `/discover/nobackup/projects/gmao/g6dev/pub/BMNG`

**Web Scripts**: `/discover/nobackup/projects/gmao/g6dev/pub/WebGL`

## Caching Processes
### CONUS map geometry
The shared `conus` output view matches IDL `setup_region.pro` region 104
(`eicusa_mapset`): oblique cylindrical equidistant centered at 30 N, 97 W,
implemented with Cartopy `RotatedPole(pole_longitude=83, pole_latitude=60)`.
The geographic input limits remain west/east/south/north
`[-125.75, -68.25, 23, 50]`. The plotter uses explicit native rotated-degree
limits `[-26.33985162225, 26.33985162225, -7, 22.43526791581]` to reproduce
IDL's projected viewport; applying geographic `set_extent` alone is not exact.
The map-frame longitude envelope is approximately 134.91380 W to 59.08620 W.
Equal aspect fits this frame inside the 16:9 canvas with small vertical margins.

This output geometry is shared by LCC and regular latitude/longitude products.
Their source grids and plotting transforms are unchanged. Curved edges
of an LCC field mark its limited data coverage, not a separate output projection.
Other regional presets retain their existing projections. Projection and extent
are part of the warped-basemap cache key, so old Lambert entries are not reused.

The shared forecast/replay LCC reader normalizes geographic longitudes to
`[-180, 180)` when needed. Latitude values, coordinate masks, grid ordering,
and data ordering are preserved. These remain geographic coordinates for
`transform=ccrs.PlateCarree()`, not native projected LCC x/y coordinates.
Regular latitude/longitude readers retain their original longitude convention.

=======
### CONUS forecast-field cropping
EarthNow product renderers subset one-dimensional latitude/longitude grids
before drawing meshes, contours, and images for the shared `conus` view.
The crop covers the inverse-projected viewport, including its curved geographic
edges, with a one-degree safety margin and two neighboring source cells.
Both -180..180 and 0..360 longitude conventions are supported.
Native LCC/two-dimensional coordinate grids and other map views are unchanged.

Cropping happens after product calculations and smoothing, preserving their
full-grid values, masks, and normalization. Image extents are adjusted to the
actual cropped pixel boundaries to preserve registration. All related layers
are cropped together. This reduces data sent to Cartopy/Matplotlib; it does not
reduce file reads or the preceding product calculations. Actual runtime gains
depend on the product. INFO logging reports
the original and cropped rendering shapes. This is separate from basemap
source-image caching and does not change it.

#### Cropping benchmark
A sequential, single-threaded benchmark used the actual `SSEXTTAU` field from
the 2026-11-01 00Z replay on its 2881 x 5760 global grid. Three runs per case
in fresh processes, with alternating cropped/uncropped order, gave these medians:

| Output preset | Full-grid render | Cropped render | Speedup | Full/cropped peak RSS |
|---|---:|---:|---:|---:|
| HD | 55.91 s | 4.50 s | 12.4x | 3277 / 326 MiB |
| 8K | 68.22 s | 15.49 s | 4.4x | 3277 / 1037 MiB |

The rendering input shrank to 557 x 1249 cells, a 95.8% reduction; cropping
itself took about 0.025 s. Decoded cropped and uncropped PNG pixels were
identical in the first pair at each resolution.

These are isolated raster-rendering measurements, not end-to-end product or
scheduler timings. Each case includes cropping, mesh creation, an explicit
canvas draw, and PNG encoding to memory (including the save-time redraw).
It uses one opaque viridis-colored aerosol layer without basemap, boundaries,
or labels; the workflow normally uses multiple styled layers. File reads,
imports, and axes setup are excluded from the render times. No files were
written to production output locations. Shared-node load and PNG content can
affect timings.

With the existing tight-save behavior, actual PNG sizes were 1920 x 1072 and
7680 x 4291, not the nominal preset heights; this benchmark does not change
that separate output-sizing behavior.

### Processed base-image cache
EarthNow persists a warped RGBA basemap and its target extent, as one `.npz` file per source image and map geometry. Downsampled source images and coordinate transforms are not cached for this path, as they were originally set up to in the source code.
Source-image downsampling preserves the input aspect ratio; output canvas dimensions belong to the map transform. A regional crop before resizing could retain more CONUS detail, but is not currently applied. A safe crop must cover the inverse-projected viewport, include an interpolation margin, and use its actual pixel-aligned geographic extent. Cropping pixels while retaining the global source extent would misregister the map. A warped-cache hit skips mapping computation and pixel interpolation.

Before submitting parallel frame jobs, render one date for each unique source image, map projection/extent, and output resolution combination. This generates the final warped RGBA and extent. A global render does not warm the CONUS cache; monthly/custom source images and other regions/resolutions have their own entries. Warm-up is optional: missing entries are generated on demand, and concurrent writers publish complete files atomically. Cache keys include the source fingerprint, projections, extents, output shape, and cache format version. Old entries can be deleted manually when jobs are idle.

The cache directory is set by `enow_plot.py --cache-dir`, which currently defaults to the hard-coded `/discover/nobackup/$USER/EarthNow/cache`.  A cache directory is required for basemap styles. Future integration of this argument into `cylc/config.yml` should be explored. Cache files are written directly into it, and the directory is created only when the first warped image is saved.

Reuse across jobs/nodes requires writable shared storage. Missing or invalid entries are rebuilt from the source image. Cache filenames include the source file's path, size, and modification time, so replacing a source image produces a new entry. Entries are derived, can be deleted when jobs are idle, and are not automatically evicted.

### Transform cache (disabled)
Transform geometry is derived from each map's axes/projection and requested output resolution. Basemap rendering checks its warped-image cache first; only a cache miss computes the mapping and warps source pixels. The basemap path does not persist a separate transform.

Persistent transform caching has been disabled for now. Geocolor, which  called `get_transform_cache()` no longer load a cached transform for the rgba plotting, transformation is done on the fly. In the future, Geocolor could benefit from a persisted transform because its geometry is constant across frames (its warped output is not, so the warped-image cache does not apply), but benchmark the gain first.

## Resolutions: 8k, 4k, HD
        "hd": ResolutionConfig(
            name="HD (1080p)", width=1920, height=1080, dpi=100, figsize=(19.2, 10.8)
        ),
        "4k": ResolutionConfig(
            name="4K (2160p)", width=3840, height=2160, dpi=200, figsize=(19.2, 10.8)
        ),
        "8k": ResolutionConfig(
            name="8K (4320p)", width=7680, height=4320, dpi=400, figsize=(19.2, 10.8)
        ),
        "fhd": ResolutionConfig(
            name="Full HD", width=1920, height=1080, dpi=120, figsize=(16.0, 9.0)
        ),
        "2k": ResolutionConfig(
            name="2K", width=2560, height=1440, dpi=150, figsize=(17.067, 9.6)
        )
        
## Movie files
`ploteic_aerosols-globe_8k.mp4`
`ploteic_geocolor-eicusa_mapset_8k.mp4`
`ploteic_precip-eicusa_mapset_8k.mp4`
`ploteic_vort500-globe_8k.mp4`
`ploteic_radar-globe_8k.mp4`
`ploteic_aerosols-eicusa_mapset_8k.mp4`
`ploteic_band16-globe_8k.mp4`
`ploteic_carbon-globe_8k.mp4`
`ploteic_cape-globe_8k.mp4`
`ploteic_winds-globe_8k.mp4`
`ploteic_cape-eicusa_mapset_8k.mp4`
`ploteic_winds-eicusa_mapset_8k.mp4`
`ploteic_sandwich-globe_8k.mp4`
`ploteic_wind250-eicusa_mapset_8k.mp4`
`ploteic_geocolor-globe_8k.mp4`
`ploteic_band16-eicusa_mapset_8k.mp4`
`ploteic_band14-globe_8k.mp4`
`ploteic_t2m-eicusa_mapset_8k.mp4`
`ploteic_precip-globe_8k.mp4`
`ploteic_radar-eicusa_mapset_8k.mp4`
`ploteic_wind250-globe_8k.mp4`
`ploteic_t2m-globe_8k.mp4`
`ploteic_band14-eicusa_mapset_8k.mp4`
`ploteic_carbon-eicusa_mapset_8k.mp4`
`ploteic_band09-globe_8k.mp4`
`ploteic_sandwich-eicusa_mapset_8k.mp4`
`ploteic_band09-eicusa_mapset_8k.mp4`
`ploteic_vort500-eicusa_mapset_8k.mp4`

Shared Directory for Tools Team
===============================

/discover/swdev/gmao-tools
