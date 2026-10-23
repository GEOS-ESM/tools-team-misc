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

## Processed base-image cache

EarthNow caches decoded and downsampled base imagery separately from the source basemap files and published plots. On Discover, the default is `/discover/nobackup/$USER/EarthNow/cache/base_images`; elsewhere it defaults to `~/.cache/earthnow/base_images`. Set `EARTHNOW_BASE_IMAGE_CACHE_DIR` to override the cache root. For reuse across jobs or nodes, configure a writable filesystem visible to all those jobs. If the cache cannot be read or written, plotting falls back to loading the source image. Cache entries are derived and can be deleted when not in use; they are recreated on demand and are not automatically evicted.

## Persistent transform cache and Slurm warm-up

Transform geometry is derived from each map's axes/projection and requested output resolution. Rendering looks up or creates the persistent transform using that geometry; no separate driver precompute step is needed.

Coordinate transforms are stored in `/discover/nobackup/$USER/EarthNow/cache/transforms` on Discover and `~/.cache/earthnow/transforms` elsewhere. Override with `EARTHNOW_TRANSFORM_CACHE_DIR` or the plotting CLI's `--cache-dir`. The image and transform overrides are independent. Use a private, trusted, writable cache directory visible on every participating node; transform entries use Python pickle and must not be supplied by untrusted users.

Before submitting parallel frame jobs, render one date for each region/resolution/style combination in the workflow. A completed render creates both the source-image cache and its map transform. A single global render does not warm the CONUS transform, and monthly/custom source assets require their own image entries. Warm-up is an optimization, not a requirement: missing entries are generated on demand and concurrent writers publish complete files atomically. No cache is cleared on normal process initialization. Transform keys include source/target projections, extents, output shape, and a format version; older keys are ignored and can be cleaned up manually when jobs are idle.

Source-image downsampling preserves the input aspect ratio; output canvas dimensions belong to the later map transform. A regional crop before resizing could retain substantially more CONUS detail, but is not currently applied. A safe crop must cover the inverse-projected viewport (not just the nominal lat/lon bounds), include an interpolation margin, and pass the actual pixel-aligned crop extent into the transform and cache key. Cropping pixels while retaining the global source extent would misregister the map. Transform caching avoids recomputing coordinate mappings, but interpolation of image pixels still runs for each frame.

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
