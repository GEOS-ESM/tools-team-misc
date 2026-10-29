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
