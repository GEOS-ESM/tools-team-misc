"""
Data Readers Module

Provides utilities for reading climatology and topography data used in the
T2M anomaly product calculation.

See docs/T2M_ANOMALY.md for complete documentation.
"""

import numpy as np
from datetime import datetime
from pathlib import Path
from scipy import ndimage
from typing import Tuple, Optional, Union

from earthnow import paths
from earthnow.wxmaps_utils import parse_date_string
from earthnow.data_readers.merra2 import merra2

try:
    import netCDF4
except ImportError:
    netCDF4 = None


# Constants
GRAVITY = 9.80665  # m/s^2
LAPSE_RATE = 6.0   # °C/km (standard atmospheric lapse rate)


def read_climatology_3day_smooth(
    date: Union[str, datetime],
    target_lons: int,
    target_lats: int,
    climatology_file: str = None,
) -> np.ndarray:
    """
    Read climatology data with 3-day running mean smoothing.

    Reads climatology for the previous day, current day, and next day, then
    applies a 3-day running mean filter. Also performs linear interpolation
    within the current day based on hour/minute.

    Parameters
    ----------
    date : str or datetime
        Target date/time for climatology. String format: YYYYMMDD_HHz or YYYYMMDD_HHMMz
    target_lons : int
        Number of longitude points for regridding
    target_lats : int
        Number of latitude points for regridding
    climatology_file : str, optional
        Path to climatology NetCDF file

    Returns
    -------
    np.ndarray
        Climatology field interpolated to target time and resolution,
        shape (target_lats, target_lons) in Celsius

    Raises
    ------
    FileNotFoundError
        If climatology file does not exist
    ImportError
        If netCDF4 is not available
    RuntimeError
        If error occurs reading climatology data
    """
    # Convert string to datetime if needed
    if isinstance(date, str):
        date = parse_date_string(date)
    if netCDF4 is None:
        raise ImportError("netCDF4 is required for climatology reading")

    if climatology_file is None:
        climatology_file = str(paths.CLIMATOLOGY_FILE)

    if not Path(climatology_file).exists():
        raise FileNotFoundError(f"Climatology file not found: {climatology_file}")

    try:
        ds = netCDF4.Dataset(climatology_file, "r")

        # Get day of year (1-365, no leap years in climatology)
        doy = date.timetuple().tm_yday
        doy = min(doy, 365)  # Cap at 365 for edge cases

        # Get previous and next days (with bounds checking)
        doy_prev = max(1, doy - 1)
        doy_next = min(365, doy + 1)

        # Read 3 days of climatology
        # Note: doy is 1-indexed, but array indices are 0-indexed
        clim_prev = ds.variables["t2mmeanclim"][doy_prev - 1, :, :]
        clim_curr = ds.variables["t2mmeanclim"][doy - 1, :, :]
        clim_next = ds.variables["t2mmeanclim"][doy_next - 1, :, :]

        ds.close()

        # Transpose from (lon, lat) to (lat, lon) if needed
        if clim_prev.shape[0] > clim_prev.shape[1]:
            # Assuming lon > lat, so it's (lon, lat)
            clim_prev = clim_prev.T
            clim_curr = clim_curr.T
            clim_next = clim_next.T

        # Time interpolation within current day
        hour_frac = (date.hour + date.minute / 60.0) / 24.0
        clim_inst = clim_curr * hour_frac + clim_prev * (1.0 - hour_frac)

        # 3-day running mean for smoothing
        clim_smooth = (clim_prev + clim_inst + clim_next) / 3.0

        # Regrid to target resolution if needed
        if clim_smooth.shape != (target_lats, target_lons):
            clim_smooth = regrid_data(clim_smooth, target_lons, target_lats)

        return clim_smooth

    except netCDF4.error as e:
        raise RuntimeError(f"NetCDF error reading climatology: {e}")
    except Exception as e:
        raise RuntimeError(f"Error reading climatology: {e}")


def read_merra2_reference_topography(target_lons: int, target_lats: int) -> np.ndarray:
    """
    Read MERRA-2 reference topography (surface geopotential height).

    Uses a fixed reference date (2015-01-01) to provide a consistent, unchanging
    topography baseline for all anomaly calculations. This matches the approach
    in the original IDL script ploteic_tanom.pro.

    Parameters
    ----------
    target_lons : int
        Number of longitude points for target grid
    target_lats : int
        Number of latitude points for target grid

    Returns
    -------
    np.ndarray
        Reference topography in meters, shape (target_lats, target_lons)

    Notes
    -----
    Uses fixed date 2015-01-01 as a constant reference point. MERRA-2
    topography (PHIS) is static and doesn't change over time, so the
    specific date chosen is arbitrary.
    """
    try:
        # Use MERRA-2 reader with fixed reference date
        reader = merra2()
        
        # Read PHIS (surface geopotential) from fixed reference date
        # PHIS is in geopotential height, divide by g to get meters
        ref_date_fdate = datetime(2015, 1, 1)
        ref_date_pdate = datetime(2015, 1, 1)
        
        topo_gp, lats, lons, meta = reader.read_variable(
            ref_date_fdate, ref_date_pdate, variables=["PHIS"]
        )
        
        if topo_gp is None:
            raise ValueError("Unable to read PHIS from MERRA-2")
        
        # Convert geopotential to height in meters (PHIS / g)
        topo_m = topo_gp / GRAVITY
        
        # Regrid to target resolution if needed
        if topo_m.shape != (target_lats, target_lons):
            topo_m = regrid_data(topo_m, target_lons, target_lats)
        
        return topo_m
        
    except Exception as e:
        raise RuntimeError(f"Error reading MERRA-2 reference topography: {e}")


def regrid_data(
    data: np.ndarray,
    target_lons: int,
    target_lats: int,
    order: int = 1,
) -> np.ndarray:
    """
    Regrid data to target resolution using interpolation.

    Uses scipy's ndimage.zoom with configurable interpolation order.

    Parameters
    ----------
    data : np.ndarray
        Input data, shape (nlats, nlons)
    target_lons : int
        Target number of longitude points
    target_lats : int
        Target number of latitude points
    order : int, optional
        Interpolation order (0=nearest, 1=linear, 3=cubic). Default: 1 (linear)

    Returns
    -------
    np.ndarray
        Regridded data, shape (target_lats, target_lons)

    Raises
    ------
    ValueError
        If input data dimensions are invalid
    """
    if len(data.shape) != 2:
        raise ValueError(f"Expected 2D data, got shape {data.shape}")

    current_lats, current_lons = data.shape
    if current_lats == 0 or current_lons == 0:
        raise ValueError("Input data has zero dimension")

    zoom_factors = (target_lats / current_lats, target_lons / current_lons)
    regridded = ndimage.zoom(data, zoom_factors, order=order)

    return regridded


def calculate_temperature_anomaly(
    t2m_kelvin: np.ndarray,
    climatology_celsius: np.ndarray,
    model_elevation: np.ndarray,
    reference_elevation: Optional[np.ndarray] = None,
    lapse_rate: float = LAPSE_RATE,
) -> np.ndarray:
    """
    Calculate temperature anomaly with lapse rate correction.

    Computes anomaly accounting for elevation differences between model and
    reference topography using a specified lapse rate (typically 6°C/km).

    Parameters
    ----------
    t2m_kelvin : np.ndarray
        2-meter temperature in Kelvin, shape (nlats, nlons)
    climatology_celsius : np.ndarray
        Climatology in Celsius, shape (nlats, nlons)
    model_elevation : np.ndarray
        Model topography in meters, shape (nlats, nlons)
    reference_elevation : np.ndarray, optional
        Reference topography in meters, shape (nlats, nlons).
        If None, uses model_elevation (no lapse correction applied).
    lapse_rate : float, optional
        Atmospheric lapse rate in °C/km. Default: 6.0

    Returns
    -------
    np.ndarray
        Temperature anomaly in Celsius, shape (nlats, nlons)

    Notes
    -----
    The anomaly formula is:
        anomaly = T2M - (Climatology - lapse_rate * (model_elev - ref_elev) / 1000)

    The lapse rate correction accounts for elevation differences between the
    model and reference topography. This prevents spurious anomalies in
    mountainous regions where model topography differs from the reference
    (typically MERRA-2 at ~0.5° resolution).

    Example
    -------
    >>> t2m = 280.0  # 7°C in Kelvin
    >>> clim = 10.0  # 10°C climatology
    >>> model_elev = 1000  # 1000m model elevation
    >>> ref_elev = 0  # Sea level reference
    >>> anom = calculate_temperature_anomaly(t2m, clim, model_elev, ref_elev)
    >>> # anom ≈ -17°C (7°C - (10°C - 6°C)) = 7 - 4 = 3°C... wait
    >>> # Actually: 7 - (10 - 6*1) = 7 - 4 = 3°C anomaly
    """
    # Convert T2M from Kelvin to Celsius
    t2m_celsius = t2m_kelvin - 273.15

    # Use model elevation if reference not provided
    if reference_elevation is None:
        reference_elevation = model_elevation

    # Calculate lapse rate correction (°C/km * km = °C)
    elev_diff = model_elevation - reference_elevation  # meters
    lapse_correction = lapse_rate * (elev_diff / 1000.0)  # °C

    # Compute anomaly
    anomaly = t2m_celsius - (climatology_celsius - lapse_correction)

    return anomaly


def cap_extremes(
    data: np.ndarray,
    vmin: float,
    vmax: float,
    epsilon: float = 0.01,
) -> np.ndarray:
    """
    Cap extreme values to prevent color mapping artifacts.

    Parameters
    ----------
    data : np.ndarray
        Input data array
    vmin : float
        Minimum colorbar value
    vmax : float
        Maximum colorbar value
    epsilon : float, optional
        Small offset from extreme values. Default: 0.01

    Returns
    -------
    np.ndarray
        Data with extremes capped
    """
    return np.clip(data, vmin + epsilon, vmax - epsilon)
