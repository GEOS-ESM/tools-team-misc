# cf_map

Generates publication-quality datagram plots (PNG) and timeseries file exports (XLSX/CSV/JSON) from [GEOS-CF](https://gmao.gsfc.nasa.gov/weather_prediction/GEOS-CF/) model data. Built on top of [`cfapi`](../cfapi/) for data retrieval.

---

## Installation

```bash
cd cf_map
uv sync
```

`cfapi` is declared as a path dependency and installed automatically in editable mode from `../cfapi`.

### Dependencies

- `cfapi` (local, editable)
- `matplotlib` — figure rendering
- `numpy`, `pandas` — array and tabular operations
- `pyyaml`, `requests` — configuration and HTTP fallback
- Python >= 3.11

---

## CLI usage

The `cf-map` entry point supports two modes: **plot** and **file export**.

### Plot mode

Generates a multi-panel datagram PNG image.

```
cf-map --version {v1,v2} --collection COLLECTION --product PRODUCT
       --lat LAT --lon LON --plot_type {pres,sfc,replay}
       [--output_dir DIR]
```

| Plot type | Description |
|-----------|-------------|
| `pres` | Forecast datagram with pressure-level contours |
| `sfc` | Forecast surface datagram (no vertical levels) |
| `replay` | Assimilation (replay) datagram spanning a date range |

**Examples:**

```bash
# Forecast pressure-level ozone datagram
cf-map --version v2 --collection fcast --product O3 \
       --lat 38.9 --lon -77.0 --plot_type pres

# Surface PM2.5 forecast datagram
cf-map --version v2 --collection fcast --product PM25 \
       --lat 38.9 --lon -77.0 --plot_type sfc

# Assimilation replay datagram
cf-map --version v2 --collection assim --product NO2 \
       --lat 38.9 --lon -77.0 --plot_type replay
```

### File export mode

Exports timeseries data to XLSX, CSV, or JSON.

```
cf-map --version {v1,v2} --collection COLLECTION --dataset DATASET
       --product PRODUCT [PRODUCT ...] --lat LAT --lon LON
       --format {xlsx,csv,json}
```

**Examples:**

```bash
# Export surface NO2 and PM2.5 to Excel
cf-map --version v2 --collection fcast --dataset chm \
       --product NO2 PM25 --lat 38.9 --lon -77.0 --format xlsx

# Export as JSON
cf-map --version v2 --collection fcast --dataset chm \
       --product O3 --lat 38.9 --lon -77.0 --format json
```

Run `cf-map --help` for the full argument reference and accepted product aliases.

---

## Plot types

### Pressure datagram (`pres`)

A 4-panel figure combining:
- Cloud-top temperature timeseries
- Pressure-level contour plot (e.g. ozone profile)
- Surface product timeseries
- Meteorology panel (wind speed, direction, precipitation)

Queries: `met_slv`, `chm_slv`, `chm_p23`

### Surface datagram (`sfc`)

Same layout as `pres` but omits the pressure-level contour panel. Includes surface PM2.5 handling.

Queries: `met_slv`, `chm_slv`

### Replay datagram (`replay`)

Assimilation variant that spans a full date range rather than a single forecast initialization. Uses the AQ replay dataset.

Queries: `met_slv`, `aqc_slv`

---

## Outputs

| Type | Default location |
|------|-----------------|
| Plot images (PNG) | `$CACHE/plots/{timestamp}/cf{version}_{type}_{product}_{lat}_{lon}.png` |
| File exports | `$CACHE/file_downloads/cf{version}_{type}_{product}_{lat}_{lon}.{ext}` |

`$CACHE` resolves to a platform-specific project directory at runtime.

---

## Library usage

```python
from cf_map.cf_plots import make_plot
from cf_map.file_maker import CFFile

# Generate a datagram image
make_plot(
    version='v2',
    collection='fcast',
    product='O3',
    lat=38.9,
    lon=-77.0,
    plot_type='pres',
    output_dir='/tmp/plots',
)

# Export data to file
from cfapi.core import build_response_cfg
from cfapi.constants import CliConfig

cfg = CliConfig(...)
response = build_response_cfg(cfg)
cf_file = CFFile(response, products=['O3'], fmt='xlsx')
cf_file.write('/tmp/output.xlsx')
```

---

## Package structure

```
cf_map/
├── cf_map/
│   ├── cli_interface.py    # CLI parsers and entry points
│   ├── cf_plots.py         # Datagram classes (CFDatagram, CFSurfacePlot, CFReplayPlot)
│   ├── datagrams.py        # Base plotting logic, panel layout, axis formatting
│   ├── data_prepper.py     # CFDatagramPrep: orchestrates cfapi queries
│   ├── file_maker.py       # CFFile: XLSX/CSV/JSON export
│   ├── cfmap_config.py     # CLI validators, output templates, aliases
│   └── config/
│       ├── plot_config.yml     # Per-product colorbar, limits, contour settings
│       ├── datagrams.mplstyle  # Matplotlib stylesheet
│       ├── GMAO-logo.png
│       └── nasa-logo.png
├── pyproject.toml
└── uv.lock
```

---

## Data flow

```
CLI args
   │
   ▼
cfmap_config (validate & normalize)
   │
   ▼
CFDatagramPrep
   ├── get_plot_vars()         → determine which datasets to query
   └── cfapi.get_data()        → retrieve netCDF timeseries
          │
          ▼
   prep_met / prep_sfc / prep_contour_data
          │
          ▼
   CFDatagram / CFSurfacePlot / CFReplayPlot
          │
          ▼
   PNG output  ──or──  CFFile → XLSX/CSV/JSON
```
