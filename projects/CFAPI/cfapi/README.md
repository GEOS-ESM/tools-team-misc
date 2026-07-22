# cfapi

Command-line and programmatic interface for extracting point timeseries data from [GEOS-CF](https://gmao.gsfc.nasa.gov/weather_prediction/GEOS-CF/) netCDF output files.

Given a latitude/longitude, date range, and product list, `cfapi` locates the relevant netCDF files on disk, interpolates to the requested point using multiprocessing, and returns a structured JSON response. Results are cached locally to avoid redundant I/O.

---

## Installation

### From the repo (editable)

```bash
cd cfapi
uv sync
```

This installs the package into `.venv/` and registers the `cfapi` CLI entry point.

### Dependencies

- `numpy`, `pandas`, `xarray` — array and tabular data operations
- `netCDF4` — HDF5/netCDF file I/O
- `pyyaml` — configuration loading
- Python >= 3.11

---

## CLI usage

```
cfapi [--version {v1,v2}] --collection COLLECTION --dataset DATASET
      --level LEVEL --product PRODUCT [PRODUCT ...]
      --lat LAT --lon LON [--start START] [--end END]
```

### Key arguments

| Argument | Description |
|----------|-------------|
| `--version` | Model version: `v1` (legacy) or `v2` (current) |
| `--collection` | Data collection: `fcast` (forecast) or `assim` (assimilation) |
| `--dataset` | Dataset: `chm` (chemistry), `met` (meteorology), `aqc` (AQ replay) |
| `--level` | Vertical level: `slv` (surface), `p23` (pressure levels), `v72` (model levels) |
| `--product` | One or more variable names (e.g. `NO2`, `O3`, `PM25`) |
| `--lat` / `--lon` | Target point coordinates |
| `--start` / `--end` | Date range (defaults to latest available) |

### Examples

```bash
# Latest surface ozone forecast at a single point
cfapi --version v2 --collection fcast --dataset chm --level slv \
      --product O3 --lat 38.9 --lon -77.0

# Multi-product query with a date range
cfapi --version v2 --collection fcast --dataset chm --level slv \
      --product NO2 PM25 --lat 38.9 --lon -77.0 \
      --start 2024-01-01 --end 2024-01-07

# Pressure-level ozone profile
cfapi --version v2 --collection fcast --dataset chm --level p23 \
      --product O3 --lat 38.9 --lon -77.0
```

Run `cfapi --help` for the full list of options and accepted aliases.

---

## Library usage

`cfapi` can also be used as a Python library:

```python
from cfapi.core import build_response_cfg
from cfapi.constants import CliConfig

cfg = CliConfig(
    version='v2',
    collection='fcast',
    dataset='chm',
    level='slv',
    product='NO2',
    products=['NO2'],
    lat=38.9,
    lon=-77.0,
    start=None,
    end=None,
    datakey='chm_slv',
)

response = build_response_cfg(cfg, use_cache=True)
# response = {
#   'schema': {'product': ..., 'units': ..., 'level': ...},
#   'values': {'NO2': [...]},
#   'time': ['2024-01-01T00:00:00', ...],
#   'response_time': 1.23,
# }
```

For querying multiple product groups in parallel:

```python
from cfapi.core import get_multifile_vars

responses = get_multifile_vars(cfg_list)
```

---

## Response format

Responses are Python dicts (also written as JSON to the cache) with the following structure:

```json
{
  "schema": {
    "product": "NO2",
    "units": "ppbv",
    "level": "slv",
    "lat": 38.9,
    "lon": -77.0,
    "nearest_lat": 38.875,
    "nearest_lon": -77.0
  },
  "values": {
    "NO2": [0.012, 0.015, ...]
  },
  "time": ["2024-01-01T00:00:00", "2024-01-01T01:00:00", ...],
  "response_time": 1.45
}
```

For pressure-level datasets (`p23`, `v72`), `values` is nested by level:

```json
"values": {
  "O3": {
    "1000": [0.03, ...],
    "925":  [0.04, ...],
    ...
  }
}
```

---

## Package structure

```
cfapi/
├── cfapi/
│   ├── cfapi_interface.py  # CLI argument parsing; entry point
│   ├── core.py             # Data extraction, interpolation, caching
│   ├── data_config.py      # File discovery (fileHelper, cacheHelper)
│   ├── api_config.py       # Product and version configuration
│   ├── validator.py        # Input validation
│   ├── aliases.py          # Alias → canonical name lookups
│   ├── constants.py        # Shared dataclasses, error types, paths
│   └── config/
│       ├── config_data.yml   # File path templates per version/dataset
│       ├── config_params.yml # Variable metadata (units, scaling factors)
│       ├── aliases.yml       # Product name aliases
│       └── options.yml       # Valid collections, datasets, levels
├── pyproject.toml
└── uv.lock
```

---

## Caching

Responses are cached as JSON files under a platform-specific directory (resolved at runtime). The cache is keyed by query parameters. To bypass the cache, pass `use_cache=False` to `build_response_cfg()` or use the `--no-cache` CLI flag.
