Date: 2026-09-01
Title: seaice_summary.md

Conversation copy and summary

This file contains the user/assistant conversation and findings about how sea-ice (sice) is plotted in the repository, the two different patterns used, and recommended Python verification steps.

---

User: Test that my API key works right now

User: Look in the current directory at the file ploteic_carbon.pro. Provide an explanation of how the IDL code and others in this folder plot seaice (sice)

Assistant: (actions)
- Listed IDL (.pro) files in the repository and searched for occurrences of "sice".
- Read the file ploteic_carbon.pro and supporting functions get_seaice_map.pro, image_bytscl.pro, map_cubed_sphere.pro, get_mapped_data.pro.

Findings (key points)

- ploteic_carbon.pro obtains a sea-ice field (sice) via get_seaice_map and composes an image object (ImageTV + alpha) that is combined with a map background and aerosol layers, then writes PNG outputs.

- get_seaice_map.pro:
  - Reads OSTIA/Reynolds binary sea-ice files (multiple daily records) and uses file headers to size arrays.
  - Performs a simple linear time interpolation between the previous day and the current day using sice = sice*(hour/24.0) + sice0*(1.0-hour/24.0).
  - Detects cubed-sphere data shapes and either calls map_cubed_sphere to project cubed-sphere into the PNG grid or, for regular grids, calls regrid to resample into pngImgIdim × pngImgJdim.

- map_cubed_sphere.pro:
  - Handles cubed-sphere grids by reading tile grid files, building cell polygons, assigning colors based on data-bin mapping (dlevs), and drawing using polyfill.
  - Returns a 2D float array mapped to the target image dimensions.

- get_mapped_data.pro and regrid:
  - get_mapped_data wraps map_image and places a smaller mapped sub-image into a full pngImgIdim × pngImgJdim NaN-initialized array.
  - regrid is used to resample lat–lon data to the desired image resolution.

- image_bytscl.pro and create_image_object.pro are used to map floating values to byte images and create image objects with alpha channels suitable for get_snapshot and WRITE_PNG.

Repository pattern discovery

- There are two common patterns for plotting sea ice:
  1) get_seaice_map(...) -> used by ploteic_* scripts (ploteic_carbon.pro, ploteic_aerosols.pro, ploteic_* family).
     - Source: OSTIA/Reynolds observed sea-ice binary files.
     - Handles daily file layout, hourly interpolation, and supports cubed-sphere projection.

  2) read_and_interpolate_cube[2](..., 'FRSEAICE', ...) -> used by plot_* scripts (plot_t850.pro, plot_tpw.pro, plot_area.pro, plot_dyn.pro, and many others).
     - Source: model/reanalysis files that contain variable FRSEAICE.
     - Many of these scripts then apply a nonlinear transform: sice = 50.0 * sice^3.0, an empirical visualization scaling.

Why two patterns exist (concise reasons)

- Different data sources and formats: OSTIA/Reynolds is a separate binary product; model fields are inside forecast/reanalysis NetCDF/HDF files.
- Different projection and grid needs: cubed-sphere model outputs require map_cubed_sphere for accurate projection; lat–lon model grids can use regrid/map_image.
- Temporal handling: get_seaice_map does day→hour interpolation. Model fields may already be hourly and don't need this interpolation.
- Visual scaling: the cubic scaling is a display choice in the plotting scripts to emphasize high fractional ice coverage.

Files that call get_seaice_map (observational path)
- get_seaice_map.pro
- ploteic_carbon.pro
- ploteic_aerosols.pro
- ploteic_aer.pro
- ploteic_arlindo.pro
- ploteic_helicity.pro
- ploteic_radar.pro
- ploteic_cape.pro
- ploteic_winds.pro

Files that read FRSEAICE via read_and_interpolate_cube[2] (model/reanalysis path)
(partial list from repository grep)
- plot_tpw.pro
- plot_t850.pro
- plot_area.pro
- plot_dyn.pro
- plotall_aeros.pro
- plotall_aero.pro
- plot_so4volc.pro
- plot_so2.pro
- plot_t2m.pro
- plot_winds10m.pro
- plot_winds250.pro
- weather_plots.pro
- plot_wx.pro
- plot_heatchill.pro
- plot_cldvis.pro
- plot_cloudpath.pro
- plot_watervapor.pro
- plot_so2volc.pro
- plot_slp.pro
- plot_so4.pro
- plotall_smoke.pro
- plot_winds850.pro
- read_and_interpolate_merra.pro
- read_and_interpolate_merra2.pro
- read_and_interpolate_merra3.pro

Why the cubic transform (50*s^3)

- It's applied in plotting code (not the cube reader) and appears in many plot_* scripts.
- It likely transforms fractional concentration (0..1) into a visual-scale range that suppresses low fractions and emphasizes near-complete coverage for overlays. Example mapping:
  - s=1.0 -> 50
  - s=0.5 -> 6.25
  - s=0.2 -> 0.4

Python verification steps (provided to the user)

The assistant provided Python snippets to:
1) Inspect FRSEAICE in a sample NetCDF using xarray to confirm shape/units/range.
2) Compare raw FRSEAICE vs transformed (50*s^3) side-by-side with matplotlib to see visual effect.
3) Regrid to png target dims via xESMF if a model lat/lon dataset is available.
4) Advice for reading OSTIA/Reynolds Fortran-unformatted binary: either use IDL to export a NetCDF/npy file or attempt direct Python reading with caution about record markers.

Representative Python snippets

Inspect variable:
```python
import xarray as xr
import numpy as np

path = "/path/to/your/model_file.nc4"  # replace with real path
ds = xr.open_dataset(path)
varname = "FRSEAICE"
if varname in ds:
    v = ds[varname]
    print("shape:", v.shape)
    print("dims:", v.dims)
    print("attrs:", v.attrs)
    print("min/max:", float(v.min().values), float(v.max().values))
else:
    print(varname, "not found. Variables:", list(ds.data_vars))
```

Compare raw vs transformed:
```python
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

ds = xr.open_dataset(path)
s = ds['FRSEAICE'].isel(time=0)
svals = s.values.astype(float)
svals[np.isnan(svals)] = 0.0
trans = 50.0 * (svals ** 3)
fig, axs = plt.subplots(1,2,figsize=(12,6))
axs[0].imshow(svals, origin='lower', vmin=0, vmax=1, cmap='Blues')
axs[0].set_title('raw FRSEAICE (0-1)')
im = axs[1].imshow(trans, origin='lower', cmap='inferno')
axs[1].set_title('50*s^3 transform')
fig.colorbar(im, ax=axs[1])
plt.show()
```

OSTIA/Reynolds Fortran-unformatted reader sketch (use cautiously):
```python
import numpy as np

def read_fortran_recorded_f32(f):
    nbytes = np.fromfile(f, dtype=np.int32, count=1)
    if nbytes.size == 0:
        return None
    n = int(nbytes[0])
    data = np.fromfile(f, dtype=np.float32, count=n//4)
    np.fromfile(f, dtype=np.int32, count=1)
    return data

with open('/path/to/dataoceanfile.data', 'rb') as f:
    hdr = read_fortran_recorded_f32(f)
    nx, ny = int(hdr[12]), int(hdr[13])
    arr = read_fortran_recorded_f32(f)
    arr = arr.reshape((nx, ny))
```

Notes & recommendations

- If you'd like a canonical Python wrapper that returns a standardized sice array given a source flag ('model'|'ostia'), the assistant can draft one.
- For OSTIA, the assistant recommended using a small IDL export (get_seaice_map -> write NetCDF) as the most reliable route to get a Python-readable file.

User requested actions following this analysis
- The user asked: "Yes do 1 and 2. I am working in Python now, so all future suggestions should be Python-based"
  - The assistant produced the lists and inspected read_and_interpolate_cube and -cube2 and explained where the cubic transform is applied (visualization by plotting scripts).

User final instruction in the session
- "Save a copy of this conversation in /discover/nobackup/hzafar/tools-team-misc/projects/EarthNow/ with today's date and seaice_summary.md title"

This file was created to satisfy that request. If you want the exact raw transcript including internal tool outputs, or prefer a different filename format (for example: seaice_summary-2026-09-01.md), tell me and I'll update or create the alternate copy.

---

End of file.
