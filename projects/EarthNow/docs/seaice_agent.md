# Sea Ice Agent Notes

## Scope

This document consolidates:
- Prior sea-ice analysis
- Follow-up investigation of IDL land/ocean masking behavior
- Python implementation updates in:
  - `src/earthnow/wxmaps_config.py`
  - `src/earthnow/wxmaps_plotting.py`
- Current status, mismatches vs IDL, and next actions

---

## 1) Prior Findings (from `seaice_summary.md`)

### Two sea-ice plotting paths in repo

1. **Observational sea ice (OSTIA/Reynolds)**: Used for some `ploteic_*` scripts
  - `get_seaice_map.pro` reads Fortran-unformatted binary sea-ice files (multiple daily records) and uses file headers to size arrays.
  - Performs a simple linear time interpolation between the previous day and the current day:
     - `sice = sice*(hour/24.0) + sice0*(1.0 - hour/24.0)`
 - It does not return explicit lat/lon arrays for sea ice.
 - If cubed-sphere shape is detected, calls `map_cubed_sphere`. Cubed-sphere mapper behavior (`map_cubed_sphere.pro`):
     - Reads cube grid geometry from tile NetCDF files (`x`, `y` vertices).
     - Bins data into color levels (`dlevs`), fills polygons.
     - Rasterizes back to 2D field via `TVRD`.
     - No intrinsic land masking logic in this routine.
 - Else regrids to target image dimensions.
 - *Example: `ploteic_carbon.pro` obtains a sea-ice field (sice) via get_seaice_map and composes an image object (ImageTV + alpha) that is combined with a map background and aerosol layers, then writes PNG outputs.*

 - Files that call `get_seaice_map`:
     -  `ploteic_carbon.pro`, `ploteic_aerosols.pro`, `ploteic_aer.pro`, `ploteic_arlindo.pro`, `ploteic_helicity.pro`, `ploteic_radar.pro`, `ploteic_cape.pro`, `ploteic_winds.pro`

2. **Model sea ice data (`FRSEAICE`)**: Used for some `plot_*` scripts
  - Many `plot_*` scripts (e.g., `plot_t850.pro`, `plot_tpw.pro`, `plot_area.pro`, `plot_dyn.pro`, etc) use `read_and_interpolate_cube[2](..., 'FRSEAICE', ...)`.
  - Data source: model/reanalysis files that contain variable `FRSEAICE`.
  - Many of these scripts then apply a nonlinear transform (empirical visualization scaling):
    - `sice = 50.0 * sice^3.0`
  - This is a plotting transform, not data-reader behavior.

  - Some files that read `FRSEAICE` via `read_and_interpolate_cube2`:
    - `plot_tpw.pro`, `plot_t850.pro`, `plot_area.pro`, `plot_dyn.pro`, `plotall_aeros.pro`, `plotall_aero.pro`, `plot_so4volc.pro`, `plot_so2.pro`, `plot_t2m.pro`, `plot_winds10m.pro`, `plot_winds250.pro`, `weather_plots.pro`, `plot_wx.pro`, `plot_heatchill.pro`, `plot_cldvis.pro`, `plot_cloudpath.pro`, `plot_watervapor.pro`, `plot_so2volc.pro`, `plot_slp.pro`, `plot_so4.pro`, `plotall_smoke.pro`, `plot_winds850.pro`, `read_and_interpolate_merra.pro`, `read_and_interpolate_merra2.pro`, `read_and_interpolate_merra3.pro`,

---

## 2) IDL Land Suppression: How Sea Ice Is Masked Over Land

### Where `mapImageObjL` comes from
In `ploteic_*` scripts (e.g., `ploteic_carbon.pro`, `ploteic_aerosols.pro`):
- `mapImageObjL = get_map_image_new(..., OCEAN_ALPHA=[0], /FILL_CONTINENTS, LAND_ALPHA=[255], ...)`

This creates a mask-like map layer:
- Ocean fully transparent
- Land opaque

### Suppression sequence in IDL
1. Build sea-ice image object (`siceImageObj`) with color+alpha.
2. Composite:
   - `snapshot = get_snapshot([siceImageObj, mapImageObjL], ...)`
3. Convert snapshot back into a sea-ice raster:
   - `sice = REFORM(snapshot(0,*,*))/255.0`

Effect: land pixels get overwritten by opaque land layer before final plotting, so sea ice is suppressed over land.

### Is this done for non-sea-ice variables?
- This exact precomposite "land suppression pass" is primarily used for sea ice.
- Other fields generally rely on:
  - field-specific alpha/thresholds,
  - NaN masking,
  - layer ordering with map/ocean background objects (`mapImageObj`, `mapImageObjO`),
  rather than this same explicit sea-ice pre-mask pattern.

---

## 3) Land/Ocean Delineation Sources in IDL

### Base map vectors
- `get_map_image_new.pro` uses IDL map routines (`MAP_SET`, `MAP_CONTINENTS`) and fill settings (`E_CONTINENTS={FILL:1,...}`).

### Optional external datasets
- GSHHS (when requested):
  - `/discover/nobackup/projects/gmao/osse2/GSHHG/v2.3.7/gshhs_f.b`
- Lakes and other extras via shapefiles in external paths.

So: base land/ocean comes from IDL map system, with optional external overrides/supplements.

---

## 4) Python Work in `wxmaps_*` and Requested Direction

### User request
- Keep implementation in `wxmaps_plotting.py` (not config).
- Use 50m land shapefile.

### Implemented updates in `wxmaps_plotting.py`
- Sea ice plotting switched from RGBA `imshow` flow to georeferenced plotting with `pcolormesh`.
- Applied mask before plotting:
  - sea ice over land -> masked

### Why this is better than previous Python state
- Uses sea-ice lat/lon grid coordinates directly.
- Avoids projection-stretch artifacts from image-space placement.
- Supports variable alpha by concentration.

---

## 5) Current Visual Mismatch vs IDL

User reported: sea-ice colors do not match IDL.
Reason:
- IDL sea-ice rendering is not constant white.
- In IDL sea-ice path:
  - color index from `image_bytscl(..., /LOG)` (log-scaled intensity),
  - alpha from linear `BYTSCL(...)`,
  - `ctable=0` (IDL built-in grayscale table).

Current Python:
- white color + alpha ramp (concentration-based),
- therefore brighter/flatter look than IDL grayscale-log style.

---

## 6) Recommended Next Step for IDL-Parity

To better match IDL appearance:
2. Replace pure-white RGB with grayscale ramp (IDL-like `loadct,0` behavior).
3. Apply log-like intensity transform for color channel (IDL `/LOG` analog).
4. Keep alpha linear with concentration (`0` transparent).
5. Consider threshold:
   - strict IDL parity usually implies little/no hard low-end cut (set threshold near 0).

---

## 7) Session Outcome Summary

- Confirmed IDL masking behavior and data flow.
- Traced land/ocean source datasets and map object generation.
- Remaining items:
   - Tune grayscale/log color mapping to reproduce IDL visual style more closely
   - Cache globe land mask

---

## 8) Key Files Referenced

- `docs/2026-09-01-seaice_summary.md`
- `links/IDL_BASE/get_seaice_map.pro`
- `links/IDL_BASE/map_cubed_sphere.pro`
- `links/IDL_BASE/ploteic_carbon.pro`
- `links/IDL_BASE/ploteic_aerosols.pro`
- `links/IDL_BASE/get_map_image_new.pro`
- `links/IDL_BASE/get_snapshot.pro`
- `links/IDL_BASE/plot_gshhs_coastlines.pro`
- `src/earthnow/wxmaps_config.py`
- `src/earthnow/wxmaps_plotting.py`
- `src/earthnow/paths.py`
