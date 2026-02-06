"""
GenCast-GEOS-FP Data Reader for WxMaps

Reads GenCast ensemble forecast predictions trained on GEOS-FP data.
"""

import os
import glob
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from netCDF4 import Dataset
import numpy as np

from .registry import register


@register("gencast_geos_fp")
class GenCastGEOSFPReader:
    """
    Data reader for GenCast ensemble forecasts trained on GEOS-FP.
    
    GenCast produces probabilistic forecasts with multiple ensemble members.
    Output is at 1.0° resolution with 13 pressure levels.
    Forecast timesteps are every 12 hours from +12h to +240h (10 days).
    """
    
    def __init__(self,
                 exp_path="/discover/nobackup/projects/gmao/osse2/GenCast_FP",
                 exp_id="GenCast-f5421_fpp",
                 exp_res="100KM",
                 ensemble_member='mean'):
        """
        Initialize GenCast-GEOS-FP reader.
        
        Parameters
        ----------
        exp_path : str
            Base path to GenCast data
        exp_id : str
            Experiment ID (e.g., 'f5421_fpp')
        exp_res : str
            exp_res directory name (e.g., '100KM')
        ensemble_member : int or str
            Ensemble member to read (0-31, or 'mean' for ensemble mean)
        """
        self.name = "GenCast"
        self.exp_path = exp_path
        self.exp_id = exp_id
        self.exp_res = exp_res
        self.ensemble_member = ensemble_member
        self.fp_id = exp_id.removeprefix("GenCast-")
        
        self.data_dir = os.path.join(
            exp_path, self.fp_id, exp_res, "2-predictions"
        )
        
        # Available pressure levels (hPa)
        self.pressure_levels = np.array([50, 100, 150, 200, 250, 300, 400, 
                                         500, 600, 700, 850, 925, 1000])
        
        print("GenCast-GEOS-FP Reader initialized:")
        print(f"  Experiment ID : {exp_id}")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Ensemble member: {ensemble_member}")
    
    # ------------------------------------------------------------------
    # File path construction
    # ------------------------------------------------------------------
    
    def _build_filename(self, fdate: str) -> str:
        """
        Build GenCast filename from forecast initialization date.
        
        Format: gencast-dataset-prediction-geos_date-YYYY-MM-DDTHH_res-1.0_levels-13_steps-20.nc
        
        Parameters
        ----------
        fdate : str
            Forecast initialization date (e.g., '20260128_00z', '2026-01-28_00:00:00')
        
        Returns
        -------
        str
            Filename
        """
        from wxmaps_utils import parse_date_string
        
        fdate_clean = fdate.replace("z", "").replace("Z", "")
        dt = parse_date_string(fdate_clean)
        
        # Format: 2026-01-28T00
        date_str = dt.strftime("%Y-%m-%dT%H")
        
        return f"gencast-dataset-prediction-geos_date-{date_str}_res-1.0_levels-13_steps-20.nc"
    
    def get_file_path(self, fdate: str) -> str:
        """
        Get full path to GenCast file.
        
        Parameters
        ----------
        fdate : str
            Forecast initialization date
        
        Returns
        -------
        str
            Full path to file
        """
        filename = self._build_filename(fdate)
        path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"GenCast file not found: {path}")
        
        return path
    
    # ------------------------------------------------------------------
    # Time handling
    # ------------------------------------------------------------------
    
    def _pdate_to_time_index(self, fdate: str, pdate: str) -> int:
        """
        Convert valid date to time index in GenCast file.
        
        GenCast time dimension is every 12 hours starting at +12h.
        
        Parameters
        ----------
        fdate : str
            Forecast initialization date
        pdate : str
            Valid/prediction date
        
        Returns
        -------
        int
            Time index (0-19 for 20 timesteps)
        """
        from wxmaps_utils import parse_date_string
        
        fdate_clean = fdate.replace("z", "").replace("Z", "")
        pdate_clean = pdate.replace("z", "").replace("Z", "")
        
        fdate_dt = parse_date_string(fdate_clean)
        pdate_dt = parse_date_string(pdate_clean)
        
        # Calculate forecast hour
        forecast_hour = int((pdate_dt - fdate_dt).total_seconds() / 3600)
        
        # GenCast starts at +12h with 12h intervals
        # time[0] = +12h, time[1] = +24h, ..., time[19] = +240h
        if forecast_hour < 12 or forecast_hour > 240:
            raise ValueError(
                f"GenCast only has forecasts from +12h to +240h. "
                f"Requested: +{forecast_hour}h"
            )
        
        if forecast_hour % 12 != 0:
            raise ValueError(
                f"GenCast timesteps are every 12 hours. "
                f"Requested: +{forecast_hour}h (not divisible by 12)"
            )
        
        time_index = (forecast_hour - 12) // 12
        
        return time_index
    
    # ------------------------------------------------------------------
    # Variable name mapping
    # ------------------------------------------------------------------
    
    def _map_variable_name(self, varname: str) -> Tuple[str, Optional[int]]:
        """
        Map WxMaps variable name to GenCast variable name.
        
        Also extracts pressure level if specified (e.g., T500 → temperature, 500)
        
        Parameters
        ----------
        varname : str
            WxMaps variable name
        
        Returns
        -------
        tuple
            (gencast_varname, pressure_level)
            pressure_level is None for 2D variables
        """
        # Variable name mapping
        VAR_MAP = {
            # 2D variables
            "U10M": "10m_u_component_of_wind",
            "V10M": "10m_v_component_of_wind",
            "T2M": "2m_temperature",
            "SLP": "mean_sea_level_pressure",
            "MSLP": "mean_sea_level_pressure",
            "SST": "sea_surface_temperature",
            "PRECTOT": "total_precipitation_12hr",
            "PRECIP": "total_precipitation_12hr",
            
            # 3D variables (base names)
            "H": "geopotential",
            "HGT": "geopotential",
            "HEIGHT": "geopotential",
            "Q": "specific_humidity",
            "SPFH": "specific_humidity",
            "T": "temperature",
            "TEMP": "temperature",
            "U": "u_component_of_wind",
            "V": "v_component_of_wind",
            "OMEGA": "vertical_velocity",
            "W": "vertical_velocity",
        }
        
        # Check for level extraction (e.g., T500, H850)
        import re
        match = re.match(r'^([A-Z_]+?)(\d+)$', varname)
        if match:
            base_var = match.group(1)
            level = int(match.group(2))
            
            if base_var in VAR_MAP:
                return VAR_MAP[base_var], level
        
        # No level extraction, check direct mapping
        if varname in VAR_MAP:
            return VAR_MAP[varname], None
        
        # Return as-is (might be GenCast native name)
        return varname, None
    
    # ------------------------------------------------------------------
    # Main read function
    # ------------------------------------------------------------------
    
    def read_variable(self, fdate: str, pdate: str, variables, 
                     level=None, ensemble_member=None, **kwargs):
        """
        Read variable from GenCast forecast.
        
        Parameters
        ----------
        fdate : str
            Forecast initialization date
        pdate : str
            Valid/prediction date (must be +12h to +240h in 12h increments)
        variables : str or List[str]
            Variable name(s) to read (tries in order, returns first found)
        level : float, optional
            Pressure level (hPa) for 3D variables
        ensemble_member : int or str, optional
            Ensemble member to read (0-31, or 'mean')
            If None, uses value from __init__
        **kwargs
            Additional arguments (ignored, for compatibility with other readers)
         
        Returns
        -------
        tuple
            (data, lats, lons, metadata)
        """
        if isinstance(variables, str):
            variables = [variables]
        
        if ensemble_member is None:
            ensemble_member = self.ensemble_member

        # Check for accumulation variables
        for varname in variables:
            if varname in ["PRECACCUM", "SNOWACCUM"]:
                return self._compute_accumulation(fdate, pdate, varname, ensemble_member)

        # Check for vorticity computation (VORT500, VORT850, etc.)
        for varname in variables:
            if varname.startswith("VORT") and len(varname) > 4:
                try:
                    vort_level = float(varname[4:])
                    print(f"Detected computed variable: {varname} (vorticity at {vort_level} hPa)")
                    return self.compute_vorticity(fdate, pdate, level=vort_level, ensemble_member=ensemble_member)
                except ValueError:
                    # Not a valid VORT{level} format, continue
                    pass
 
        # Get file path
        path = self.get_file_path(fdate)
        
        # Get time index
        time_idx = self._pdate_to_time_index(fdate, pdate)
        
        # Try each variable in priority order
        for varname in variables:
            gencast_var, extracted_level = self._map_variable_name(varname)
            
            # Use extracted level if specified in variable name
            if extracted_level is not None:
                level = extracted_level
            
            try:
                with Dataset(path) as nc:
                    if gencast_var not in nc.variables:
                        continue
                    
                    var = nc.variables[gencast_var]
                    
                    # Read coordinates
                    lats = nc.variables['lat'][:]
                    lons = nc.variables['lon'][:]
                    
                    # Determine data shape and extract
                    # All variables have shape: (sample, batch, time, [level,] lat, lon)
                    
                    is_3d = 'level' in var.dimensions
                    
                    if ensemble_member == 'mean':
                        # Compute ensemble mean across all samples
                        if is_3d:
                            if level is None:
                                raise ValueError(
                                    f"{gencast_var} is 3D, must specify level"
                                )
                            
                            # Find level index
                            levels = nc.variables['level'][:]
                            level_idx = np.argmin(np.abs(levels - level))
                            actual_level = levels[level_idx]
                            
                            # Extract: mean over sample, batch=0, time=time_idx, level=level_idx
                            data = var[:, 0, time_idx, level_idx, :, :].mean(axis=0)
                            
                        else:
                            # 2D variable
                            # Extract: mean over sample, batch=0, time=time_idx
                            data = var[:, 0, time_idx, :, :].mean(axis=0)
                    
                    else:
                        # Single ensemble member
                        sample_idx = int(ensemble_member)
                        
                        if sample_idx < 0 or sample_idx >= var.shape[0]:
                            raise ValueError(
                                f"Ensemble member {sample_idx} out of range "
                                f"(0-{var.shape[0]-1})"
                            )
                        
                        if is_3d:
                            if level is None:
                                raise ValueError(
                                    f"{gencast_var} is 3D, must specify level"
                                )
                            
                            # Find level index
                            levels = nc.variables['level'][:]
                            level_idx = np.argmin(np.abs(levels - level))
                            actual_level = levels[level_idx]
                            
                            # Extract: sample=sample_idx, batch=0, time=time_idx, level=level_idx
                            data = var[sample_idx, 0, time_idx, level_idx, :, :]
                            
                        else:
                            # 2D variable
                            # Extract: sample=sample_idx, batch=0, time=time_idx
                            data = var[sample_idx, 0, time_idx, :, :]
                    
                    # Convert to numpy array (in case it's masked)
                    data = np.array(data)

                    # Special handling for geopotential -> height conversion
                    if gencast_var == "geopotential":
                        # Convert geopotential (m²/s²) to geopotential height (m)
                        g = 9.80665  # m/s²
                        data = data / g
                        print(f"  Converted geopotential to height (divided by g={g})")

                    # Build metadata
                    units = getattr(var, "units", "")
                    
                    # Update units if we converted geopotential
                    if gencast_var == "geopotential":
                        units = "m"  # Geopotential height in meters
                    
                    # Build metadata
                    metadata = {
                        "units": getattr(var, "units", ""),
                        "long_name": getattr(var, "long_name", gencast_var),
                        "file": os.path.basename(path),
                        "gencast_variable": gencast_var,
                        "ensemble_member": ensemble_member,
                        "time_index": time_idx,
                        "forecast_hour": (time_idx + 1) * 12,
                        "grid": "latlon_1.0deg",
                        "is_3d": is_3d,
                    }
                    
                    if is_3d:
                        metadata["pressure_level"] = float(actual_level)
                        metadata["available_levels"] = levels.tolist()
                    
                    print(f"Using file      : {os.path.basename(path)}")
                    print(f"Using variable  : {gencast_var}")
                    print(f"Ensemble member : {ensemble_member}")
                    print(f"Time index      : {time_idx} (+{metadata['forecast_hour']}h)")
                    if is_3d:
                        print(f"Pressure level  : {actual_level} hPa")
                    print(f"Data shape      : {data.shape}")
                    print(f"Data range      : [{np.nanmin(data):.2e}, {np.nanmax(data):.2e}]")
                    
                    return data, lats, lons, metadata
                    
            except Exception as e:
                print(f"Warning: Could not read {varname}: {e}")
                continue
        
        raise FileNotFoundError(
            f"No suitable GenCast variable found\n"
            f"  Requested: {variables}\n"
            f"  File: {path}"
        )
    
    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------

    def compute_vorticity(self, fdate: str, pdate: str, level: float = 500.0, 
                         ensemble_member=None):
        """
        Compute relative vorticity from U and V wind components.
        
        Vorticity = dv/dx - du/dy
        
        Parameters
        ----------
        fdate : str
            Forecast initialization date
        pdate : str
            Valid/prediction date
        level : float
            Pressure level in hPa (default: 500)
        ensemble_member : int or str, optional
            Ensemble member (0-31, or 'mean')
        
        Returns
        -------
        tuple
            (vorticity, lats, lons, metadata)
            - vorticity: numpy array of relative vorticity (s^-1)
            - lats: latitude array
            - lons: longitude array
            - metadata: dict with units, long_name, etc.
        """
        print(f"Computing vorticity at {level} hPa...")
        
        if ensemble_member is None:
            ensemble_member = self.ensemble_member
        
        # Read U and V components
        u_data, lats, lons, u_meta = self.read_variable(
            fdate=fdate,
            pdate=pdate,
            variables=["U"],
            level=level,
            ensemble_member=ensemble_member
        )
        
        v_data, _, _, v_meta = self.read_variable(
            fdate=fdate,
            pdate=pdate,
            variables=["V"],
            level=level,
            ensemble_member=ensemble_member
        )
        
        # Compute vorticity using centered finite differences
        vorticity = self._compute_vorticity_latlon(u_data, v_data, lats, lons)
        
        # Create metadata
        metadata = {
            "units": "s-1",
            "long_name": f"relative_vorticity_at_{level}hPa",
            "file": u_meta.get("file"),
            "ensemble_member": ensemble_member,
            "time_index": u_meta.get("time_index"),
            "forecast_hour": u_meta.get("forecast_hour"),
            "grid": "latlon_1.0deg",
            "is_3d": False,
            "computed_from": "U,V wind components",
            "level": level,
        }
        
        print(f"  Vorticity range: [{vorticity.min():.2e}, {vorticity.max():.2e}] s^-1")
        
        return vorticity, lats, lons, metadata
    
    
    def _compute_vorticity_latlon(self, u, v, lats, lons):
        """
        Compute relative vorticity on a lat-lon grid.
        
        Uses centered finite differences:
            ζ = dv/dx - du/dy
        
        Parameters
        ----------
        u : np.ndarray
            Zonal wind component (m/s), shape (nlat, nlon)
        v : np.ndarray
            Meridional wind component (m/s), shape (nlat, nlon)
        lats : np.ndarray
            Latitude array (degrees), shape (nlat,)
        lons : np.ndarray
            Longitude array (degrees), shape (nlon,)
        
        Returns
        -------
        np.ndarray
            Relative vorticity (s^-1), shape (nlat, nlon)
        """
        # Earth radius in meters
        R_earth = 6.371e6
        
        # Convert degrees to radians
        lats_rad = np.deg2rad(lats)
        lons_rad = np.deg2rad(lons)
        
        # Create 2D meshes
        lon_mesh, lat_mesh = np.meshgrid(lons_rad, lats_rad)
        
        # Compute grid spacing
        dlat = np.diff(lats_rad).mean()  # radians
        dlon = np.diff(lons_rad).mean()  # radians
        
        # Compute derivatives using centered differences
        # dv/dx (meridional derivative of v)
        dv_dx = np.zeros_like(v)
        dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dlon * R_earth * np.cos(lat_mesh[:, 1:-1]))
        dv_dx[:, 0] = (v[:, 1] - v[:, -1]) / (2 * dlon * R_earth * np.cos(lat_mesh[:, 0]))  # Periodic BC
        dv_dx[:, -1] = (v[:, 0] - v[:, -2]) / (2 * dlon * R_earth * np.cos(lat_mesh[:, -1]))  # Periodic BC
        
        # du/dy (zonal derivative of u)
        du_dy = np.zeros_like(u)
        du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dlat * R_earth)
        du_dy[0, :] = (u[1, :] - u[0, :]) / (dlat * R_earth)  # Forward difference at pole
        du_dy[-1, :] = (u[-1, :] - u[-2, :]) / (dlat * R_earth)  # Backward difference at pole
        
        # Relative vorticity
        vorticity = dv_dx - du_dy
        
        return vorticity

    def _compute_accumulation(self, fdate: str, pdate: str, varname: str, 
                             ensemble_member=None):
        """
        Compute accumulated precipitation from forecast start to valid time.
        
        GenCast provides 12-hourly accumulated precipitation in each timestep.
        Sum all timesteps from +12h to the requested time.
        
        Parameters
        ----------
        fdate : str
            Forecast initialization date
        pdate : str
            Valid/prediction date
        varname : str
            Accumulation variable ('PRECACCUM' or 'SNOWACCUM')
        ensemble_member : int or str, optional
            Ensemble member (0-31, or 'mean')
        
        Returns
        -------
        tuple
            (accumulated_data, lats, lons, metadata)
        """
        from wxmaps_utils import parse_date_string
        
        if varname == "SNOWACCUM":
            raise NotImplementedError(
                "SNOWACCUM not available in GenCast (no snow/rain partitioning)"
            )
        
        print(f"Computing precipitation accumulation from {fdate} to {pdate}")
        
        # Parse dates
        fdate_clean = fdate.replace("z", "").replace("Z", "")
        pdate_clean = pdate.replace("z", "").replace("Z", "")
        fdate_dt = parse_date_string(fdate_clean)
        pdate_dt = parse_date_string(pdate_clean)
        
        # Calculate forecast hours
        forecast_hour = int((pdate_dt - fdate_dt).total_seconds() / 3600)
        
        # Special case: if pdate == fdate, return zeros
        if forecast_hour <= 0:
            print(f"  pdate <= fdate: No accumulation period, returning zeros")
            
            # Get grid from file
            path = self.get_file_path(fdate)
            with Dataset(path) as nc:
                lats = nc.variables['lat'][:]
                lons = nc.variables['lon'][:]
            
            zeros = np.zeros((len(lats), len(lons)))
            
            metadata = {
                "units": "mm",
                "long_name": "total_precipitation_accumulated",
                "file": os.path.basename(path),
                "ensemble_member": ensemble_member,
                "accumulation_start": fdate,
                "accumulation_end": pdate,
                "num_timesteps": 0,
                "total_hours": 0.0,
                "grid": "latlon_1.0deg",
            }
            
            return zeros, lats, lons, metadata
        
        # GenCast timesteps are +12h, +24h, ..., +240h
        # Each timestep contains 12hr accumulated precip for that period
        # To get total accumulation to +48h: sum timesteps at +12h, +24h, +36h, +48h
        
        if forecast_hour > 240:
            raise ValueError(f"GenCast only has forecasts to +240h, requested +{forecast_hour}h")
        
        # Get file path
        path = self.get_file_path(fdate)
        
        # Determine which timesteps to sum
        # Round down to nearest 12h increment
        last_timestep_hour = (forecast_hour // 12) * 12
        if last_timestep_hour < 12:
            last_timestep_hour = 12
        
        timestep_hours = list(range(12, last_timestep_hour + 1, 12))
        num_timesteps = len(timestep_hours)
        
        print(f"  Summing {num_timesteps} 12-hourly periods: {timestep_hours}")
        
        accumulated = None
        lats = None
        lons = None
        
        with Dataset(path) as nc:
            lats = nc.variables['lat'][:]
            lons = nc.variables['lon'][:]
            
            var = nc.variables['total_precipitation_12hr']
            
            # Sum all timesteps
            for hour in timestep_hours:
                time_idx = (hour - 12) // 12
                
                if ensemble_member == 'mean':
                    # Ensemble mean: mean over sample dimension
                    # Shape: (sample, batch, time, lat, lon)
                    data = var[:, 0, time_idx, :, :].mean(axis=0)
                else:
                    # Single member
                    sample_idx = int(ensemble_member)
                    data = var[sample_idx, 0, time_idx, :, :]
                
                data = np.array(data)*39.3701*25.4
                
                if accumulated is None:
                    accumulated = data
                else:
                    accumulated += data
                
                print(f"    +{hour}h: increment range [{np.nanmin(data):.3f}, {np.nanmax(data):.3f}]")
        
        # GenCast precip is already in mm (or kg/m²)
        # No conversion needed
        
        metadata = {
            "units": "mm",
            "long_name": "total_precipitation_accumulated",
            "file": os.path.basename(path),
            "ensemble_member": ensemble_member,
            "accumulation_start": fdate,
            "accumulation_end": pdate,
            "num_timesteps": num_timesteps,
            "total_hours": float(last_timestep_hour),
            "grid": "latlon_1.0deg",
            "gencast_variable": "total_precipitation_12hr",
        }
        
        print(f"  Accumulation complete:")
        print(f"    Total period: {last_timestep_hour} hours ({num_timesteps} timesteps)")
        print(f"    Accumulated range: [{np.nanmin(accumulated):.3f}, {np.nanmax(accumulated):.3f}] mm")
        
        return accumulated, lats, lons, metadata
    
    def get_available_variables(self, fdate: str) -> List[str]:
        """
        Get list of available variables in file.
        
        Parameters
        ----------
        fdate : str
            Forecast initialization date
        
        Returns
        -------
        List[str]
            List of variable names
        """
        path = self.get_file_path(fdate)
        
        with Dataset(path) as nc:
            return [v for v in nc.variables.keys() 
                   if v not in ('lat', 'lon', 'level', 'time', 'sample', 'batch')]
    
    def find_available_times(self, fdate: str) -> List[datetime]:
        """
        Find available forecast valid times.
        
        GenCast produces forecasts every 12 hours from +12h to +240h.
        
        Parameters
        ----------
        fdate : str
            Forecast initialization date
        
        Returns
        -------
        List[datetime]
            List of valid times
        """
        from wxmaps_utils import parse_date_string
        
        fdate_clean = fdate.replace("z", "").replace("Z", "")
        fdate_dt = parse_date_string(fdate_clean)
        
        # Check if file exists
        try:
            path = self.get_file_path(fdate)
        except FileNotFoundError:
            return []
        
        # GenCast has 20 timesteps: +12h, +24h, ..., +240h
        times = []
        for hour in range(12, 252, 12):  # 12 to 240 inclusive
            times.append(fdate_dt + timedelta(hours=hour))
        
        return times
    
    def list_available_cycles(self) -> List[datetime]:
        """
        List available forecast initialization cycles.
        
        Returns
        -------
        List[datetime]
            List of available forecast dates
        """
        pattern = os.path.join(
            self.data_dir, 
            "gencast-dataset-prediction-geos_date-*_res-1.0_levels-13_steps-20.nc"
        )
        
        cycles = []
        for filepath in glob.glob(pattern):
            try:
                # Extract date from filename
                # Format: gencast-dataset-prediction-geos_date-2026-01-28T00_...
                basename = os.path.basename(filepath)
                date_part = basename.split("_date-")[1].split("_res")[0]
                # date_part is like "2026-01-28T00"
                dt = datetime.strptime(date_part, "%Y-%m-%dT%H")
                cycles.append(dt)
            except Exception as e:
                print(f"Warning: Could not parse filename {basename}: {e}")
                continue
        
        return sorted(cycles)


# ------------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------------
def create_reader(**kwargs):
    """
    Factory function to create GenCastGEOSFPReader instance.
    
    Usage:
        from gencast_geos_fp import create_reader
        reader = create_reader(exp_id="GenCast-f5421_fpp", ensemble_member=0)
    """
    return GenCastGEOSFPReader(**kwargs)
