# T2M Anomaly Product

**Product ID**: `temperature_2m_anomaly_EarthNow`

## Implementation Status

**Current Phase**: Topography Data Integration

| Component | Status | Notes |
|-----------|--------|-------|
| Climatology (1991-2020) | ✅ Complete | 3-day smoothing, hourly interpolation |
| MERRA-2 Reference Topography | ✅ Complete | Reads on-the-fly from MERRA-2 (fixed 2015-01-01) |
| Model Topography | 🔄 In Progress | PHIS registered in readers; awaiting data sourcing |
| Core Product | ✅ Complete | Framework integration, registration, testing |
| Date Parsing | ✅ Complete | Automatic string-to-datetime conversion |

**Blocking**: Model topography file creation awaits PHIS data sourcing by user
### Usage

```bash
uv run python bin/plotall.py --product temperature_2m_anomaly_EarthNow \
    --fdate 20260115_00z --pdate 20260115_12z --region conus
```

### Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/earthnow/products/EarthNow/temperature_2m_anomaly.py` | 171 | Product entry point |
| `src/earthnow/wxmaps_climatology_utils.py` | 260 | Climatology/topography utilities |

### Algorithm

```
anomaly = T2M - (climatology - 6.0°C/km * (model_elev - ref_elev) / 1000)
```

With 3-day climatology smoothing and hourly interpolation within the day.

### Two-Part Topography System

The product uses two independent topography datasets for the lapse rate correction:

**1. Model Topography (Primary)**
- Source: PHIS from model's `const_2d_asm_Nx` collection
- Purpose: Actual elevation in the model grid
- Reading: Static NetCDF file (read once at startup)
- Variable: PHIS (surface geopotential → converted to meters)
- Role: `model_elev` in lapse correction formula

**2. MERRA-2 Reference Topography (Baseline)**
- Source: MERRA-2 reanalysis
- Date: Fixed 2015-01-01 (constant across all runs)
- Purpose: Climatological reference elevation
- Reading: Dynamic via MERRA-2 reader (auto-regrid)
- Variable: PHIS (surface geopotential → converted to meters)
- Role: `ref_elev` in lapse correction formula

**Why two?**
- **Consistency**: MERRA-2 reference never changes, ensuring identical anomalies for identical T2M
- **Accuracy**: Model topo may differ from standard reference, lapse correction captures difference
- **Robustness**: Handles topo variations between model updates/versions

### Data

- **Climatology**: 1991-2020 baseline, 365-day climatology with 3-day smoothing
- **Model topography**: Read from model PHIS field (surface geopotential)
- **Reference topography**: MERRA-2 static (fixed 2015-01-01 baseline)
- **Output range**: -21°C to +21°C (256-color gradient)

## Implementation

**Files**:
- Product: `src/earthnow/products/EarthNow/temperature_2m_anomaly.py` (171 lines)
- Utilities: `src/earthnow/wxmaps_climatology_utils.py` (260 lines)

**Algorithm**:
1. Read T2M from model
2. Read climatology (1991-2020 baseline, 3-day smoothing with hourly interpolation)
3. Read model topography (PHIS from model data, converted to meters)
4. Read MERRA-2 reference topography (fixed 2015-01-01 baseline)
5. Apply lapse rate correction: `6.0°C/km * (model_elev - merra2_ref_elev) / 1000`
6. Calculate anomaly: `T2M - (climatology - lapse_correction)`
7. Clip to [-21, +21]°C range
8. Render with 256-color gradient (Blue→White→Red)

## Data Requirements

Data paths are defined in `src/earthnow/paths.py` and can be overridden via environment variables:

**Climatology** (1991-2020 baseline):
```
EARTHNOW_CLIMATOLOGY_FILE=/discover/nobackup/npthomas/MERRA2/dailyavgs/daily_climatology/
                          temperature_climatology_19912020.nc
```

**Topography** (Two-part system):
1. **Model topography (static file)**:
   - Source: PHIS from `const_2d_asm_Nx` collection
   - Now registered in: conus2kmrp, conus2kmfc data readers
   - **Status**: Awaiting PHIS data to create static file
   - See "Setup: Creating Model Topography File" section below

2. **Reference topography (MERRA-2)**:
   - Source: MERRA-2 reanalysis (fixed 2015-01-01)
   - Automatically read on-the-fly via registered data reader
   - Provides consistent climatological baseline

## Functions in `wxmaps_climatology_utils.py`

- `read_climatology_3day_smooth(date, nlons, nlats)` - Read climatology with 3-day smoothing
- `read_topography(nlons, nlats)` - Read static topography
- `calculate_temperature_anomaly(t2m_K, clim_C, model_elev, ref_elev)` - Compute anomaly
- `cap_extremes(data, vmin, vmax)` - Clip to colorbar range
- `regrid_data(data, target_lons, target_lats)` - Regrid using scipy.ndimage.zoom

## Customization

**Lapse rate**: Edit `LAPSE_RATE` in `wxmaps_climatology_utils.py` (default: 6.0°C/km)

**Colors/levels**: Edit `ANOMALY_LEVELS` and `ANOMALY_COLORS` in `temperature_2m_anomaly.py`

**Data files**: Override via environment variable in `paths.py`:
- `EARTHNOW_CLIMATOLOGY_FILE` - Path to climatology NetCDF
- `EARTHNOW_MODEL_TOPOGRAPHY_FILE` - Path to static topography file (see setup section)

**Reference date**: To use a different MERRA-2 reference date, modify the fixed date in `read_merra2_reference_topography()` function (currently 2015-01-01)

## Setup: Creating Model Topography File

**Current Status**: Model topography is sourced from PHIS (surface geopotential) in the `const_2d_asm_Nx` collection.

**Prerequisites**:
- PHIS data available in `const_2d_asm_Nx` collection
- Data reader registered (conus2kmrp or conus2kmfc)

**Steps** (to be implemented once PHIS data is sourced):

1. **Identify PHIS source**:
   - Locate const_2d_asm_Nx files in your model data
   - Verify PHIS variable is accessible

2. **Create static topography file**:
   ```bash
   uv run python -c "
   from earthnow.wxmaps_climatology_utils import create_model_topography_file
   from earthnow.data_readers.conus2kmrp import conus2kmrp
   
   reader = conus2kmrp()
   # Read PHIS from representative date and write static file
   create_model_topography_file(
       fdate='20260101',
       pdate='20260101_00z',
       reader=reader,
       output_path='/path/to/model_topo_static.nc'
   )
   ```

3. **Configure path**:
   ```bash
   export EARTHNOW_MODEL_TOPOGRAPHY_FILE=/path/to/model_topo_static.nc
   ```

4. **Verify**:
   - Check static file contains PHIS data
   - Run product to confirm topography is loaded correctly

## Performance

- Processing: ~3-6 seconds
- vs IDL: 2-3x faster
- Input: 1 file (vs 48 for 24-hour average version)

