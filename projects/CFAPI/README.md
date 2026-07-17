# CF_tools

Tools for extracting and visualizing [GEOS-CF](https://gmao.gsfc.nasa.gov/weather_prediction/GEOS-CF/) (GEOS Composition Forecast) model output. The repository contains two complementary Python packages:

| Package | Purpose |
|---------|---------|
| [`cfapi`](cfapi/) | Query and extract GEOS-CF timeseries data from netCDF files |
| [`cf_map`](cf_map/) | Generate datagram plots (PNG) and data exports (XLSX/CSV/JSON) |

`cfapi` is the data layer; `cf_map` builds on top of it to produce visualizations and file downloads.

---

## Repository structure

```
CF_tools/
├── cfapi/          # Data extraction package
├── cf_map/         # Visualization and export package
└── fluid-1.2.yml   # Conda environment specification
```

Each package is a standalone Python project managed with [uv](https://docs.astral.sh/uv/) and has its own `pyproject.toml`, virtual environment, and lock file.

---

## Supported model versions

Both packages support two GEOS-CF model versions:

- **v2** — current production model
- **v1** — legacy model (backward-compatible CLI)

---

## Environment setup

### Conda (recommended for GMAO systems)

**On Dataportal:**
```bash
conda activate /explore/dataportal/nobackup01/projects/GMAO/development
```

**On Discover:**
```bash
conda activate /discover/nobackup/sjrober4/miniforge3/envs/fluid-1.2
```

**From the spec file (to create a new environment):**
```bash
conda env create -f fluid-1.2.yml
```

### uv (for development)

Install each package in editable mode within its own virtual environment:

```bash
# Install cfapi
cd cfapi && uv sync

# Install cf_map (automatically links to the local cfapi)
cd ../cf_map && uv sync
```

---

## Quick start

```bash
# Extract a timeseries for surface NO2 at a lat/lon point
cfapi --version v2 --collection fcast --dataset chm --level slv \
      --product NO2 --lat 38.9 --lon -77.0

# Generate a forecast datagram plot for ozone
cf-map --version v2 --collection fcast --product O3 \
       --lat 38.9 --lon -77.0 --plot_type pres

# Export a timeseries to Excel
cf-map --version v2 --collection fcast --product NO2 PM25 \
       --lat 38.9 --lon -77.0 --format xlsx
```

See the individual package READMEs for full CLI references and library usage:

- [cfapi README](cfapi/README.md)
- [cf_map README](cf_map/README.md)
