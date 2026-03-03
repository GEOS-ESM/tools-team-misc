"""
GEOS Forward Processing Data Reader for WxMaps

Reads GEOS Forward Processing (FP) forecast data with cubed-sphere regridding support.
"""

import os
import glob
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
from netCDF4 import Dataset
import numpy as np

from .registry import register


@register("geos_forward_processing")
class GEOSForwardProcessingReader:
    """
    Data reader for GEOS Forward Processing forecasts.
    
    Automatically detects and handles cubed-sphere grids (c0720) with regridding.
    """
    
    # Collection definitions
    INST_COLLECTION = "inst_30mn_met_c0720sfc"  # High-res 30-minute instantaneous (cubed-sphere)
    TAVG_COLLECTIONS = [
        "tavg1_2d_flx_Nx",   # Hourly time-averaged fluxes (precip, radiation, etc.)
        "tavg1_2d_slv_Nx",   # Hourly time-averaged single-level diagnostics
    ]
    PRES_COLLECTION = "inst3_3d_asm_Np"          # 3-hourly 3D pressure level (lat-lon)
    
    def __init__(self,
                 base_path="/discover/nobackup/projects/gmao/gmao_ops/pub",
                 exp_id="f5295_fp",
                 collection=None,
                 target_resolution="0.25deg"):
        """
        Initialize GEOS Forward Processing reader.
        
        Parameters
        ----------
        base_path : str
            Base path to GEOS FP data
        exp_id : str
            Experiment ID (e.g., 'f5295_fp')
        collection : str, optional
            Override default collection selection
        target_resolution : str
            Target lat-lon resolution for cubed-sphere regridding
            Options: 
                "auto" - Match cubed-sphere resolution (c720 → 2880x1441)
                "0.25deg" - Force 1152x721
                "0.5deg" - Force 576x361
        """
        self.name = "GEOS-FP"
        self.base_path = base_path
        self.exp_id = exp_id
        self.collection = collection
        self.target_resolution = target_resolution
        
        self.forecast_base = os.path.join(base_path, exp_id, "forecast")
        
        # Cache for regridding
        self._target_lats = None
        self._target_lons = None
        
        print("GEOS Forward Processing Reader initialized:")
        print(f"  Experiment ID : {exp_id}")
        print(f"  Forecast base : {self.forecast_base}")
        print(f"  Target res    : {target_resolution}")
    
    # ------------------------------------------------------------------
    # Cubed-sphere grid handling
    # ------------------------------------------------------------------
    
    def _detect_cubed_sphere_resolution(self, nc):
        """
        Detect cubed-sphere resolution from NetCDF file dimensions.
        
        Parameters
        ----------
        nc : netCDF4.Dataset
            Open NetCDF file
        
        Returns
        -------
        int
            Cubed-sphere resolution (e.g., 720 for c720)
        """
        if "Xdim" in nc.dimensions:
            nx = nc.dimensions["Xdim"].size
            print(f"  Detected cubed-sphere resolution: c{nx}")
            return nx
        elif "nf" in nc.dimensions and "Ydim" in nc.dimensions:
            nx = nc.dimensions["Ydim"].size
            print(f"  Detected cubed-sphere resolution: c{nx}")
            return nx
        else:
            # Default fallback
            print(f"  Warning: Could not detect cubed-sphere resolution, assuming c720")
            return 720


    def _get_target_latlon_grid(self, cubed_sphere_nx=None):
        """
        Get target lat-lon grid based on resolution.
        
        For "auto" resolution, derives grid from cubed-sphere resolution:
            c{N} → (4*N) x (2*N + 1) lat-lon grid
            c720 → 2880 x 1441
            c360 → 1440 x 721
            c180 → 720 x 361
        
        Parameters
        ----------
        cubed_sphere_nx : int, optional
            Cubed-sphere resolution (e.g., 720 for c720)
            Used when target_resolution="auto"
        
        Returns
        -------
        tuple
            (lats, lons) - 1D arrays for target grid
        """
        if self._target_lats is not None and self._target_lons is not None:
            return self._target_lats, self._target_lons
        
        if self.target_resolution == "auto":
            # Derive from cubed-sphere resolution
            if cubed_sphere_nx is None:
                raise ValueError("Cannot use 'auto' resolution without cubed_sphere_nx parameter")
            
            ## Formula: c{N} → (4*N) x (2*N + 1)
            #nlon = 4 * cubed_sphere_nx
            #nlat = 2 * cubed_sphere_nx + 1
            print(f"Auto-detected target grid from c{cubed_sphere_nx}: {nlat} x {nlon}")

            # Force 1152 x 721 (same as c360)
            nlat, nlon = 721, 1152
            
        elif self.target_resolution == "0.25deg":
            # 1152 x 721 (same as c360)
            nlat, nlon = 721, 1152
            
        elif self.target_resolution == "0.5deg":
            # 576 x 361 (same as c180)
            nlat, nlon = 361, 576
            
        else:
            raise ValueError(f"Unknown target resolution: {self.target_resolution}")
        
        # Create lat-lon grids
        # Lat: -90 to +90 (nlat points)
        # Lon: -180 to +180 (nlon points, excluding endpoint for periodicity)
        self._target_lats = np.linspace(-90, 90, nlat)
        self._target_lons = np.linspace(-180, 180, nlon, endpoint=False)
        
        dlat = 180.0 / (nlat - 1)
        dlon = 360.0 / nlon
        
        print(f"Target grid: {nlat} x {nlon} (Δlat={dlat:.4f}°, Δlon={dlon:.4f}°)")
        
        return self._target_lats, self._target_lons


    def _regrid_cubed_sphere_to_latlon(self, data_tiles, cs_lons, cs_lats):
        """
        Regrid cubed-sphere data to lat-lon grid using nearest-neighbor.
        
        Parameters
        ----------
        data_tiles : np.ndarray
            Data on cubed-sphere grid, shape (6, ny, nx) for 6 tiles
        cs_lons : np.ndarray
            Longitude coordinates, shape (6, ny, nx)
        cs_lats : np.ndarray
            Latitude coordinates, shape (6, ny, nx)
        
        Returns
        -------
        np.ndarray
            Regridded data on lat-lon grid, shape (nlat, nlon)
        """
        print("Regridding cubed-sphere to lat-lon...")
        
        # Detect cubed-sphere resolution from data shape
        if data_tiles.ndim == 3 and data_tiles.shape[0] == 6:
            cubed_sphere_nx = data_tiles.shape[1]  # Should be same as shape[2]
            self._cubed_sphere_nx = cubed_sphere_nx
        else:
            raise ValueError(f"Unexpected data shape: {data_tiles.shape}. Expected (6, nx, nx)")
        
        # Get target grid (will auto-detect if resolution="auto")
        target_lats, target_lons = self._get_target_latlon_grid(cubed_sphere_nx=cubed_sphere_nx)
        
        # Handle longitude wrapping (convert > 180 to negative)
        cs_lons = np.where(cs_lons > 180, cs_lons - 360, cs_lons)
        
        # Flatten cubed-sphere data and coordinates
        cs_lats_flat = cs_lats.flatten()
        cs_lons_flat = cs_lons.flatten()
        data_flat = data_tiles.flatten()
        
        # Remove any missing/invalid data points
        valid_mask = np.isfinite(data_flat)
        cs_lats_flat = cs_lats_flat[valid_mask]
        cs_lons_flat = cs_lons_flat[valid_mask]
        data_flat = data_flat[valid_mask]
        
        # Create target grid meshes
        target_lon_mesh, target_lat_mesh = np.meshgrid(target_lons, target_lats)
        
        # Use scipy griddata for interpolation
        from scipy.interpolate import griddata
        
        print(f"  Interpolating {len(data_flat)} cubed-sphere points to {len(target_lats)}x{len(target_lons)} lat-lon grid...")
        
        regridded = griddata(
            points=(cs_lons_flat, cs_lats_flat),
            values=data_flat,
            xi=(target_lon_mesh, target_lat_mesh),
            method='nearest'  # Fast, can change to 'linear' or 'cubic'
        )
        
        print("  Regridding complete")
        
        return regridded
    
    def _is_cubed_sphere_collection(self, collection):
        """Check if collection uses cubed-sphere grid (has 'c0720' or 'c720' in name)"""
        return 'c0720' in collection.lower() or 'c720' in collection.lower()
    
    # ------------------------------------------------------------------
    # Collection logic
    # ------------------------------------------------------------------
    
    def _candidate_collections(self, var_type='inst') -> List[str]:
        """
        Return prioritized list of collections for variable type.
        
        Parameters
        ----------
        var_type : str
            Variable type: 'inst', 'tavg', 'accum', or 'pres'
        
        Returns
        -------
        List[str]
            Prioritized list of collection names to check
        """
        if var_type == 'inst':
            # Try high-res cubed-sphere first, then standard tavg
            return [self.INST_COLLECTION] + self.TAVG_COLLECTIONS
        elif var_type == 'tavg':
            # Try tavg only
            return self.TAVG_COLLECTIONS
        elif var_type == 'accum':
            # For accumulated fields, try both inst and tavg
            return [self.INST_COLLECTION] + self.TAVG_COLLECTIONS
        elif var_type == 'pres':
            # 3D pressure level data
            return [self.PRES_COLLECTION]
        else:
            raise ValueError(f"Unknown var_type: {var_type}")

    # ------------------------------------------------------------------
    # Path construction helpers
    # ------------------------------------------------------------------
    
    def _cycle_dir(self, fdate: str) -> str:
        """
        Convert forecast/cycle date to directory path.
        
        Parameters
        ----------
        fdate : str
            Forecast initialization date (e.g., '20260122_00z', '2026-01-22_00:00:00')
        
        Returns
        -------
        str
            Directory path relative to forecast_base (e.g., 'Y2026/M01/D22/H00')
        """
        from wxmaps_utils import parse_date_string
        
        fdate_clean = fdate.replace("z", "").replace("Z", "")
        dt = parse_date_string(fdate_clean)
        return f"Y{dt.year:04d}/M{dt.month:02d}/D{dt.day:02d}/H{dt.hour:02d}"
    
    def _build_filename(self, fdate: str, pdate: str, collection: str) -> str:
        """
        Build filename from forecast date, valid date, and collection.
        
        Format: GEOS.fp.fcst.{collection}.{cycle_YYYYMMDD_HH}+{valid_YYYYMMDD_HHMM}.V01.nc4
        
        For tavg collections, adjusts timestamp to account for centered averaging:
            - tavg1_* (1-hour average): timestamp is 30 min before period end
            - tavg3_* (3-hour average): timestamp is 90 min before period end
        
        Parameters
        ----------
        fdate : str
            Forecast initialization date
        pdate : str
            Valid/prediction date (period END time for tavg)
        collection : str
            Collection name
        
        Returns
        -------
        str
            Filename
        """
        from wxmaps_utils import parse_date_string
        from datetime import timedelta
        
        fdate_clean = fdate.replace("z", "").replace("Z", "")
        pdate_clean = pdate.replace("z", "").replace("Z", "")
        
        cycle_dt = parse_date_string(fdate_clean)
        valid_dt = parse_date_string(pdate_clean)
        
        # Adjust valid time for tavg collections (centered timestamps)
        if "tavg1" in collection:
            # 1-hour average: timestamp is 30 minutes before period end
            valid_dt = valid_dt - timedelta(minutes=30)
        elif "tavg3" in collection:
            # 3-hour average: timestamp is 90 minutes before period end
            valid_dt = valid_dt - timedelta(minutes=90)
        # Add more tavg intervals as needed (tavg6, tavg24, etc.)
        
        cycle_str = cycle_dt.strftime("%Y%m%d_%H")
        valid_str = valid_dt.strftime("%Y%m%d_%H%M")
        
        return f"GEOS.fp.fcst.{collection}.{cycle_str}+{valid_str}.V01.nc4"

    # ------------------------------------------------------------------
    # Core resolver (SINGLE SOURCE OF TRUTH)
    # ------------------------------------------------------------------
    
    def resolve_file(self, fdate: str, pdate: str, variables=None, var_type='inst'):
        """
        Resolve the best available GEOS FP file.
        
        Searches collections in priority order and returns first file
        containing requested variable(s).
        
        Parameters
        ----------
        fdate : str
            Forecast initialization date
        pdate : str
            Valid/prediction date
        variables : str or List[str], optional
            Variable name(s) to search for. If None, return first valid file.
        var_type : str
            Variable type: 'inst', 'tavg', 'accum', or 'pres'
        
        Returns
        -------
        tuple
            (path, collection, matched_variable)
            - path: Full path to file
            - collection: Collection name
            - matched_variable: First matching variable name, or None
        
        Raises
        ------
        FileNotFoundError
            If no suitable file is found
        """
        if isinstance(variables, str):
            variables = [variables]
        
        cycle_dir = self._cycle_dir(fdate)
        forecast_dir = os.path.join(self.forecast_base, cycle_dir)
        
        if not os.path.isdir(forecast_dir):
            raise FileNotFoundError(f"Forecast directory not found: {forecast_dir}")
        
        for collection in self._candidate_collections(var_type=var_type):
            filename = self._build_filename(fdate, pdate, collection)
            path = os.path.join(forecast_dir, filename)
            
            if not os.path.exists(path):
                continue
            
            # If no specific variables requested, return first valid file
            if variables is None:
                self.collection = collection
                return path, collection, None
            
            # Check if requested variable exists in file
            try:
                with Dataset(path) as nc:
                    for var in variables:
                        if var in nc.variables:
                            self.collection = collection
                            return path, collection, var
            except Exception as e:
                print(f"Warning: Could not read {path}: {e}")
                continue
        
        raise FileNotFoundError(
            f"No suitable GEOS FP file found for fdate={fdate}, pdate={pdate}\n"
            f"  Requested variables: {variables}\n"
            f"  Searched collections: {self._candidate_collections(var_type=var_type)}\n"
            f"  Forecast directory: {forecast_dir}"
        )
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    def get_file_path(self, fdate: str, pdate: str, var_type='inst'):
        """
        Get path to first available file for given dates.
        
        Parameters
        ----------
        fdate : str
            Forecast initialization date
        pdate : str
            Valid/prediction date
        var_type : str
            Variable type: 'inst', 'tavg', 'accum', or 'pres'
        
        Returns
        -------
        str
            Full path to file
        """
        path, _, _ = self.resolve_file(fdate, pdate, var_type=var_type)
        return path
    
    def read_variable(self, fdate: str, pdate: str, variables, var_type='inst', 
                     level=None):
        """
        Read the first available variable from a prioritized list.
        
        Automatically handles cubed-sphere to lat-lon regridding when needed.
        
        Special handling for computed/extracted variables:
            - VORT{level} (e.g., VORT500) - Computes vorticity from U,V at specified level
            - H{level} (e.g., H850) - Extracts geopotential height at specified level
            - T{level} (e.g., T500) - Extracts temperature at specified level
            - RH{level} (e.g., RH700) - Extracts relative humidity at specified level
            - PHIS, PS, SLP - 2D surface fields from pressure level files
            - RAIN - Liquid precipitation (PRECTOT - PRECSNO)
            - SNOW - Frozen precipitation (PRECSNO)
            - ICE - Ice pellets (returns zeros, not available in FP)
            - FRZR - Freezing rain (returns zeros, not available in FP)
            - Any 3D variable with {level} suffix extracts from pressure level file
        
        Parameters
        ----------
        fdate : str
            Forecast initialization date
        pdate : str
            Valid/prediction date
        variables : str or List[str]
            Variable name(s) to search for (prioritized)
            Special computed/extracted variables:
                - "VORT500", "VORT850", etc. - Relative vorticity at pressure level
                - "H850", "H500", etc. - Geopotential height at pressure level
                - "T500", "T850", etc. - Temperature at pressure level
                - "RH700", etc. - Relative humidity at pressure level
                - "PHIS" - Surface geopotential (constant field)
                - "PS" - Surface pressure
                - "SLP" - Sea level pressure
                - "RAIN" - Liquid precipitation (computed from PRECTOT - PRECSNO)
                - "SNOW" - Frozen precipitation (PRECSNO)
                - "ICE" - Ice pellets (returns zeros)
                - "FRZR" - Freezing rain (returns zeros)
        var_type : str
            Variable type: 'inst', 'tavg', 'accum', or 'pres'
            Note: Automatically set to 'pres' for extracted level variables
        level : float, optional
            Pressure level (hPa) to extract for 3D variables (var_type='pres')
            If None and data is 3D, returns all levels.
        
        Returns
        -------
        tuple
            (data, lats, lons, metadata)
            - data: numpy array of variable data
            - lats: latitude array
            - lons: longitude array
            - metadata: dict with units, long_name, collection, grid, file, etc.
        """
        if isinstance(variables, str):
            variables = [variables]
        
        # Check for special 2D surface fields that exist in pressure files
        SURFACE_FIELDS_IN_PRES = ["PHIS", "PS", "SLP"]
        
        # Precipitation type fields
        PRECIP_TYPE_FIELDS = {
            "RAIN": "liquid_precipitation",
            "SNOW": "frozen_precipitation", 
            "ICE": "ice_pellets",
            "FRZR": "freezing_rain"
        }
        
        for varname in variables:
            # Check for VORT{level} pattern (computed variable)
            if varname.startswith("VORT") and len(varname) > 4:
                try:
                    vort_level = float(varname[4:])
                    print(f"Detected computed variable: {varname} (vorticity at {vort_level} hPa)")
                    return self.compute_vorticity(fdate, pdate, level=vort_level)
                except ValueError:
                    pass
            
            # Check for precipitation type fields
            if varname in PRECIP_TYPE_FIELDS:
                print(f"Detected precipitation type variable: {varname}")
                try:
                    return self._get_precip_type(fdate, pdate, varname)
                except FileNotFoundError as e:
                    print(f"  Could not compute {varname}: {e}")
                    continue
            
            # Check if this is a surface field that exists in pressure files
            if varname in SURFACE_FIELDS_IN_PRES:
                print(f"Detected surface field in pressure file: {varname}")
                try:
                    return self._read_surface_field_from_pres(fdate, pdate, varname)
                except FileNotFoundError:
                    print(f"  Could not find {varname} in pressure file, trying other collections...")
                    continue
            
            # NEW: Check for pressure level extraction pattern
            # BUT only if the variable name doesn't exist as-is in the file
            extracted_var, extracted_level = self._parse_level_variable(varname)
            if extracted_var and extracted_level:
                # First, try to read the variable as-is (e.g., "H500" might exist directly)
                try:
                    print(f"Checking if {varname} exists as direct variable...")
                    path, collection, found_var = self.resolve_file(
                        fdate, pdate, variables=[varname], var_type=var_type
                    )
                    # If we got here, the variable exists as-is, so DON'T do level extraction
                    print(f"  Found {varname} directly in {collection}, using as-is")
                    # Break out of this check and continue to normal variable reading below
                except FileNotFoundError:
                    # Variable doesn't exist as-is, so try level extraction
                    print(f"Detected level extraction: {varname} → {extracted_var} at {extracted_level} hPa")
                    try:
                        return self.read_variable(
                            fdate=fdate,
                            pdate=pdate,
                            variables=[extracted_var],
                            var_type='pres',
                            level=extracted_level
                        )
                    except FileNotFoundError:
                        print(f"  Could not find {extracted_var} at {extracted_level} hPa, trying next variable...")
                        continue
        
        # Normal variable reading - only if we get here without returning
        # This means none of the special handlers matched
        path, collection, varname = self.resolve_file(
            fdate, pdate, variables, var_type=var_type
        )

        is_cubed_sphere = self._is_cubed_sphere_collection(collection)
        
        with Dataset(path) as nc:
            var = nc.variables[varname]
            data = var[:]
            
            # Read grid
            if is_cubed_sphere:
                # Cubed-sphere data - coordinates are in file
                if "lons" in nc.variables and "lats" in nc.variables:
                    cs_lons = nc.variables["lons"][:]  # (nf, Ydim, Xdim)
                    cs_lats = nc.variables["lats"][:]  # (nf, Ydim, Xdim)
                    grid = "cubed_sphere"
                    print(f"Detected cubed-sphere grid (c0720)")
                else:
                    raise ValueError(f"Cubed-sphere file missing lons/lats: {path}")
            elif "lat" in nc.variables and "lon" in nc.variables:
                lats = nc.variables["lat"][:]
                lons = nc.variables["lon"][:]
                grid = "latlon"
            else:
                raise ValueError(f"No recognizable grid found in {path}")
            
            # Handle dimensions
            is_3d = False
            pressure_levels = None
            
            if "lev" in var.dimensions:
                is_3d = True
                if "lev" in nc.variables:
                    pressure_levels = nc.variables["lev"][:]
            
            # Expected data shape for cubed-sphere: (time, nf, Ydim, Xdim)
            # Expected data shape for lat-lon: (time, lat, lon) or (time, lev, lat, lon)
            
            # Squeeze time dimension if present
            if "time" in var.dimensions:
                if data.shape[0] == 1:
                    data = data[0]  # Remove time dimension
            
            # Handle pressure level extraction for 3D data
            actual_level = None
            if is_3d and level is not None:
                if pressure_levels is None:
                    raise ValueError(f"Cannot extract level {level}: no pressure levels found")
                
                # Find nearest pressure level
                level_idx = np.argmin(np.abs(pressure_levels - level))
                actual_level = pressure_levels[level_idx]
                
                # Extract level (data is now [lev, lat, lon] after time squeeze)
                data = data[level_idx, :, :]
                
                print(f"Extracting level {level} hPa (nearest: {actual_level:.2f} hPa, index {level_idx})")
            
            # Handle cubed-sphere regridding
            if is_cubed_sphere:
                print(f"Data on cubed-sphere: shape = {data.shape}")
                
                # Data should be (nf, Ydim, Xdim) = (6, 720, 720) after time squeeze
                if data.ndim != 3 or data.shape[0] != 6:
                    raise ValueError(f"Unexpected cubed-sphere data shape: {data.shape}. Expected (6, nx, nx)")
                
                # Detect resolution
                cubed_sphere_nx = data.shape[1]
                
                # Regrid to lat-lon
                data = self._regrid_cubed_sphere_to_latlon(data, cs_lons, cs_lats)
                lats, lons = self._get_target_latlon_grid(cubed_sphere_nx=cubed_sphere_nx)
                
                if self.target_resolution == "auto":
                    grid = f"latlon_regridded_from_c{cubed_sphere_nx}_auto_{lats.shape[0]}x{lons.shape[0]}"
                else:
                    grid = f"latlon_regridded_from_c{cubed_sphere_nx}_{self.target_resolution}"
            
            metadata = {
                "units": getattr(var, "units", ""),
                "long_name": getattr(var, "long_name", varname),
                "collection": collection,
                "grid": grid,
                "file": os.path.basename(path),
                "is_3d": is_3d,
                "is_cubed_sphere": is_cubed_sphere,
            }
            
            if is_3d:
                metadata["pressure_levels"] = pressure_levels
                if actual_level is not None:
                    metadata["extracted_level"] = float(actual_level)
            
            print(f"Using file      : {os.path.basename(path)}")
            print(f"Using collection: {collection}")
            print(f"Using variable  : {varname}")
            print(f"Grid type       : {grid}")
            if is_3d and level is None:
                print(f"Data shape      : {data.shape} (3D: lev x lat x lon)")
            else:
                print(f"Data shape      : {data.shape} (2D: lat x lon)")
        
        return data, lats, lons, metadata
    
    
    def _get_precip_type(self, fdate: str, pdate: str, ptype: str):
        """
        Get precipitation by type.
        
        GEOS FP only provides:
            - PRECTOT: Total precipitation (liquid + frozen)
            - PRECSNO: Frozen precipitation (snow)
        
        Both are available in the inst_30mn_met_c0720sfc collection.
        
        Derived fields:
            - RAIN: Liquid precipitation = PRECTOT - PRECSNO
            - SNOW: Frozen precipitation = PRECSNO
            - ICE: Ice pellets = 0 (not available)
            - FRZR: Freezing rain = 0 (not available)
        
        Parameters
        ----------
        fdate : str
            Forecast initialization date
        pdate : str
            Valid/prediction date
        ptype : str
            Precipitation type: 'RAIN', 'SNOW', 'ICE', or 'FRZR'
        
        Returns
        -------
        tuple
            (data, lats, lons, metadata)
        """
        print("Starting precip type detection")
        if ptype == "RAIN":
            # RAIN = PRECTOT - PRECSNO
            print("Computing liquid precipitation (RAIN = PRECTOT - PRECSNO)...")
            
            # Read PRECTOT - available in inst collection
            prectot_data, lats, lons, meta_tot = self._read_precip_variable(
                fdate, pdate, ["PRECTOT", "PRECTOTLAND", "PRECIP"]
            )
            
            # Read PRECSNO - available in inst collection
            precsno_data, _, _, meta_sno = self._read_precip_variable(
                fdate, pdate, ["PRECSNO", "PRECSNOLAND", "SNOWFALL"]
            )
            
            # Compute liquid precipitation
            rain = prectot_data - precsno_data
            rain = np.maximum(rain, 0)  # Ensure non-negative
            
            metadata = {
                "units": meta_tot.get("units", "kg m-2 s-1"),
                "long_name": "liquid_precipitation_rate",
                "collection": meta_tot.get("collection"),
                "grid": meta_tot.get("grid"),
                "file": meta_tot.get("file"),
                "is_3d": False,
                "is_cubed_sphere": meta_tot.get("is_cubed_sphere"),
                "computed_from": "PRECTOT - PRECSNO",
            }
            
            print(f"  RAIN range: [{rain.min():.2e}, {rain.max():.2e}] {metadata['units']}")
            
            return rain, lats, lons, metadata
        
        elif ptype == "SNOW":
            # SNOW = PRECSNO
            print("Reading frozen precipitation (SNOW = PRECSNO)...")
            
            snow, lats, lons, meta = self._read_precip_variable(
                fdate, pdate, ["PRECSNO", "PRECSNOLAND", "SNOWFALL"]
            )
            
            # Update metadata
            meta["long_name"] = "frozen_precipitation_rate"
            meta["computed_from"] = "PRECSNO"
            
            return snow, lats, lons, meta
        
        elif ptype == "ICE":
            # ICE = 0 (not available in GEOS FP)
            print("Ice pellets not available in GEOS FP, returning zeros...")
            
            # Get grid from any available precip field
            prectot, lats, lons, meta_tot = self._read_precip_variable(
                fdate, pdate, ["PRECTOT", "PRECTOTLAND", "PRECIP"]
            )
            
            ice = np.zeros_like(prectot)
            
            metadata = {
                "units": meta_tot.get("units", "kg m-2 s-1"),
                "long_name": "ice_pellets_precipitation_rate",
                "collection": meta_tot.get("collection"),
                "grid": meta_tot.get("grid"),
                "file": meta_tot.get("file"),
                "is_3d": False,
                "is_cubed_sphere": meta_tot.get("is_cubed_sphere"),
                "computed_from": "zeros (not available in GEOS FP)",
            }
            
            print(f"  ICE: all zeros (not available)")
            
            return ice, lats, lons, metadata
        
        elif ptype == "FRZR":
            # FRZR = 0 (not available in GEOS FP)
            print("Freezing rain not available in GEOS FP, returning zeros...")
            
            # Get grid from any available precip field
            prectot, lats, lons, meta_tot = self._read_precip_variable(
                fdate, pdate, ["PRECTOT", "PRECTOTLAND", "PRECIP"]
            )
            
            frzr = np.zeros_like(prectot)
            
            metadata = {
                "units": meta_tot.get("units", "kg m-2 s-1"),
                "long_name": "freezing_rain_precipitation_rate",
                "collection": meta_tot.get("collection"),
                "grid": meta_tot.get("grid"),
                "file": meta_tot.get("file"),
                "is_3d": False,
                "is_cubed_sphere": meta_tot.get("is_cubed_sphere"),
                "computed_from": "zeros (not available in GEOS FP)",
            }
            
            print(f"  FRZR: all zeros (not available)")
            
            return frzr, lats, lons, metadata
        
        else:
            raise ValueError(f"Unknown precipitation type: {ptype}")
    
    
    def _read_precip_variable(self, fdate: str, pdate: str, variables: List[str]):
        """
        Read precipitation variable from inst collection (with cubed-sphere handling).
        
        This method directly reads PRECTOT/PRECSNO without recursing through
        the special variable handlers.
        
        Parameters
        ----------
        fdate : str
            Forecast initialization date
        pdate : str
            Valid/prediction date
        variables : List[str]
            List of variable names to try (e.g., ["PRECTOT", "PRECTOTLAND"])
        
        Returns
        -------
        tuple
            (data, lats, lons, metadata)
        """
        # Directly resolve file with actual variable names (not RAIN/SNOW/etc)
        path, collection, varname = self.resolve_file(
            fdate, pdate, variables=variables, var_type='inst'
        )
        
        is_cubed_sphere = self._is_cubed_sphere_collection(collection)
        
        with Dataset(path) as nc:
            if varname not in nc.variables:
                raise ValueError(f"Variable {varname} not found in {path}")
            
            var = nc.variables[varname]
            data = var[:]
            
            # Read grid
            if is_cubed_sphere:
                # Cubed-sphere data - coordinates are in file
                if "lons" in nc.variables and "lats" in nc.variables:
                    cs_lons = nc.variables["lons"][:]
                    cs_lats = nc.variables["lats"][:]
                    grid = "cubed_sphere"
                else:
                    raise ValueError(f"Cubed-sphere file missing lons/lats: {path}")
            elif "lat" in nc.variables and "lon" in nc.variables:
                lats = nc.variables["lat"][:]
                lons = nc.variables["lon"][:]
                grid = "latlon"
            else:
                raise ValueError(f"No recognizable grid found in {path}")
            
            # Squeeze time dimension if present
            if "time" in var.dimensions:
                if data.shape[0] == 1:
                    data = data[0]
            
            # Handle cubed-sphere regridding
            if is_cubed_sphere:
                print(f"  {varname} on cubed-sphere: shape = {data.shape}")
                
                if data.ndim != 3 or data.shape[0] != 6:
                    raise ValueError(f"Unexpected cubed-sphere data shape: {data.shape}")
                
                # Detect resolution
                cubed_sphere_nx = data.shape[1]
                
                # Regrid to lat-lon
                data = self._regrid_cubed_sphere_to_latlon(data, cs_lons, cs_lats)
                lats, lons = self._get_target_latlon_grid(cubed_sphere_nx=cubed_sphere_nx)
                
                if self.target_resolution == "auto":
                    grid = f"latlon_regridded_from_c{cubed_sphere_nx}_auto_{lats.shape[0]}x{lons.shape[0]}"
                else:
                    grid = f"latlon_regridded_from_c{cubed_sphere_nx}_{self.target_resolution}"
            
            metadata = {
                "units": getattr(var, "units", ""),
                "long_name": getattr(var, "long_name", varname),
                "collection": collection,
                "grid": grid,
                "file": os.path.basename(path),
                "is_3d": False,
                "is_cubed_sphere": is_cubed_sphere,
            }
            
            print(f"  Read {varname} from {os.path.basename(path)}")
            print(f"  Data range: [{data.min():.2e}, {data.max():.2e}] {metadata['units']}")
        
        return data, lats, lons, metadata

    def _parse_level_variable(self, varname: str):
        """
        Parse variable name with embedded pressure level.
        
        Extracts base variable name and pressure level from patterns like:
            H850 → ("H", 850.0)
            T500 → ("T", 500.0)
            RH700 → ("RH", 700.0)
            U250 → ("U", 250.0)
        
        Parameters
        ----------
        varname : str
            Variable name potentially with embedded level
        
        Returns
        -------
        tuple
            (base_varname, level) or (None, None) if no level pattern found
        """
        import re
        
        # Pattern: letters followed by digits at the end
        # Examples: H850, T500, RH700, OMEGA500
        match = re.match(r'^([A-Z_]+?)(\d+)$', varname)
        
        if match:
            base_var = match.group(1)
            level_str = match.group(2)
            
            # Common pressure levels: 1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 50, 10, 1
            # Level should be 1-4 digits and reasonable pressure value
            try:
                level = float(level_str)
                if 1 <= level <= 1050:  # Valid pressure range
                    return base_var, level
            except ValueError:
                pass
        
        return None, None

    def read_variable(self, fdate: str, pdate: str, variables, var_type='inst', 
                     level=None):
        """
        Read the first available variable from a prioritized list.

        Automatically handles cubed-sphere to lat-lon regridding when needed.
        
        Special handling for computed/extracted variables:
            - VORT{level} (e.g., VORT500) - Computes vorticity from U,V at specified level
            - H{level} (e.g., H850) - Extracts geopotential height at specified level
            - T{level} (e.g., T500) - Extracts temperature at specified level
            - RH{level} (e.g., RH700) - Extracts relative humidity at specified level
            - PHIS, PS, SLP - 2D surface fields from pressure level files
            - Any 3D variable with {level} suffix extracts from pressure level file
        
        Parameters
        ----------
        fdate : str
            Forecast initialization date
        pdate : str 
            Valid/prediction date
        variables : str or List[str]
            Variable name(s) to search for (prioritized)
            Special computed/extracted variables:
                - "VORT500", "VORT850", etc. - Relative vorticity at pressure level
                - "H850", "H500", etc. - Geopotential height at pressure level
                - "T500", "T850", etc. - Temperature at pressure level
                - "RH700", etc. - Relative humidity at pressure level
                - "PHIS" - Surface geopotential (constant field)
                - "PS" - Surface pressure
                - "SLP" - Sea level pressure
        var_type : str
            Variable type: 'inst', 'tavg', 'accum', or 'pres'
            Note: Automatically set to 'pres' for extracted level variables
        level : float, optional
            Pressure level (hPa) to extract for 3D variables (var_type='pres')
            If None and data is 3D, returns all levels.
        
        Returns
        -------
        tuple
            (data, lats, lons, metadata)
            - data: numpy array of variable data
            - lats: latitude array
            - lons: longitude array
            - metadata: dict with units, long_name, collection, grid, file, etc.
        """
        if isinstance(variables, str):
            variables = [variables]
        
        # Check for special 2D surface fields that exist in pressure files
        SURFACE_FIELDS_IN_PRES = ["PHIS", "PS", "SLP"]
        
        # Precipitation type fields
        PRECIP_TYPE_FIELDS = ["RAIN", "SNOW", "ICE", "FRZR"]

        # Check for accumulation variables
        ACCUMULATION_VARIABLES = {
            "PRECACCUM": ("PRECTOT", 1.0),      # Total precip accumulation, mm
            "SNOWACCUM": ("PRECSNO", 10.0),     # Snow depth accumulation (10:1 ratio), mm
            "RAINACCUM": ("PRECLSC", 1.0),      # Large-scale precip only, mm (if needed)
        }
        
        for varname in variables:
            # Check for VORT{level} pattern (computed variable)
            if varname.startswith("VORT") and len(varname) > 4:
                try:
                    vort_level = float(varname[4:])
                    print(f"Detected computed variable: {varname} (vorticity at {vort_level} hPa)")
                    return self.compute_vorticity(fdate, pdate, level=vort_level)
                except ValueError:
                    pass

            # Check for accumulated fields
            if varname in ACCUMULATION_VARIABLES:
                base_var, scale_factor = ACCUMULATION_VARIABLES[varname]
                print(f"Detected accumulation variable: {varname}")
                try:
                    return self._compute_accumulation(fdate, pdate, base_var, scale_factor)
                except FileNotFoundError as e:
                    print(f"  Could not compute {varname}: {e}")
                    continue

            # Check for precipitation type fields
            if varname in PRECIP_TYPE_FIELDS:
                print(f"Detected precipitation type variable: {varname}")
                try:
                    return self._get_precip_type(fdate, pdate, varname)
                except FileNotFoundError as e:
                    print(f"  Could not compute {varname}: {e}")
                    continue
            
            # Check if this is a surface field that exists in pressure files
            if varname in SURFACE_FIELDS_IN_PRES:
                print(f"Detected surface field in pressure file: {varname}")
                try:
                    return self._read_surface_field_from_pres(fdate, pdate, varname)
                except FileNotFoundError:
                    print(f"  Could not find {varname} in pressure file, trying other collections...")
                    continue
            
            # Check for pressure level extraction pattern
            # SKIP this logic if the variable name contains a level but might exist directly
            # (e.g., H500, H1000 exist as direct variables in c720)
            extracted_var, extracted_level = self._parse_level_variable(varname)
            if extracted_var and extracted_level:
                # Only do level extraction for variables that DON'T exist directly
                # Try normal reading first (will check inst collection)
                try:
                    # This will try inst collection first via _candidate_collections
                    path, collection, found_var = self.resolve_file(
                        fdate, pdate, variables=[varname], var_type=var_type
                    )
                    print(f"  Found {varname} directly in {collection}, not doing level extraction")
                    # Fall through to normal reading below
                    break  # Exit the for loop, use this variable
                except FileNotFoundError:
                    # Variable doesn't exist as-is, try level extraction from pressure file
                    print(f"  {varname} not found directly, trying level extraction: {extracted_var} at {extracted_level} hPa")
                    try:
                        return self.read_variable(
                            fdate=fdate,
                            pdate=pdate,
                            variables=[extracted_var],
                            var_type='pres',
                            level=extracted_level
                        )
                    except FileNotFoundError:
                        print(f"  Could not extract {extracted_var} at {extracted_level} hPa, trying next variable...")
                        continue
        
        # Normal variable reading - only if we get here without returning
        # This means none of the special handlers matched
        path, collection, varname = self.resolve_file(
            fdate, pdate, variables, var_type=var_type
        )

        is_cubed_sphere = self._is_cubed_sphere_collection(collection)
        
        with Dataset(path) as nc:
            var = nc.variables[varname]
            data = var[:]
            
            # Read grid
            if is_cubed_sphere:
                # Cubed-sphere data - coordinates are in file
                if "lons" in nc.variables and "lats" in nc.variables:
                    cs_lons = nc.variables["lons"][:]  # (nf, Ydim, Xdim)
                    cs_lats = nc.variables["lats"][:]  # (nf, Ydim, Xdim)
                    grid = "cubed_sphere"
                    print(f"Detected cubed-sphere grid (c0720)")
                else:
                    raise ValueError(f"Cubed-sphere file missing lons/lats: {path}")
            elif "lat" in nc.variables and "lon" in nc.variables:
                lats = nc.variables["lat"][:]
                lons = nc.variables["lon"][:]
                grid = "latlon"
            else:
                raise ValueError(f"No recognizable grid found in {path}")
            
            # Handle dimensions
            is_3d = False
            pressure_levels = None
            
            if "lev" in var.dimensions:
                is_3d = True
                if "lev" in nc.variables:
                    pressure_levels = nc.variables["lev"][:]
            
            # Expected data shape for cubed-sphere: (time, nf, Ydim, Xdim)
            # Expected data shape for lat-lon: (time, lat, lon) or (time, lev, lat, lon)
            
            # Squeeze time dimension if present
            if "time" in var.dimensions:
                if data.shape[0] == 1:
                    data = data[0]  # Remove time dimension
            
            # Handle pressure level extraction for 3D data
            actual_level = None
            if is_3d and level is not None:
                if pressure_levels is None:
                    raise ValueError(f"Cannot extract level {level}: no pressure levels found")
                
                # Find nearest pressure level
                level_idx = np.argmin(np.abs(pressure_levels - level))
                actual_level = pressure_levels[level_idx]
                
                # Extract level (data is now [lev, lat, lon] after time squeeze)
                data = data[level_idx, :, :]
                
                print(f"Extracting level {level} hPa (nearest: {actual_level:.2f} hPa, index {level_idx})")
            
            # Handle cubed-sphere regridding
            if is_cubed_sphere:
                print(f"Data on cubed-sphere: shape = {data.shape}")
                
                # Data should be (nf, Ydim, Xdim) = (6, 720, 720) after time squeeze
                if data.ndim != 3 or data.shape[0] != 6:
                    raise ValueError(f"Unexpected cubed-sphere data shape: {data.shape}. Expected (6, nx, nx)")
                
                # Detect resolution
                cubed_sphere_nx = data.shape[1]
                
                # Regrid to lat-lon
                data = self._regrid_cubed_sphere_to_latlon(data, cs_lons, cs_lats)
                lats, lons = self._get_target_latlon_grid(cubed_sphere_nx=cubed_sphere_nx)
                
                if self.target_resolution == "auto":
                    grid = f"latlon_regridded_from_c{cubed_sphere_nx}_auto_{lats.shape[0]}x{lons.shape[0]}"
                else:
                    grid = f"latlon_regridded_from_c{cubed_sphere_nx}_{self.target_resolution}"
            
            metadata = {
                "units": getattr(var, "units", ""),
                "long_name": getattr(var, "long_name", varname),
                "collection": collection,
                "grid": grid,
                "file": os.path.basename(path),
                "is_3d": is_3d,
                "is_cubed_sphere": is_cubed_sphere,
            }
            
            if is_3d:
                metadata["pressure_levels"] = pressure_levels
                if actual_level is not None:
                    metadata["extracted_level"] = float(actual_level)
            
            print(f"Using file      : {os.path.basename(path)}")
            print(f"Using collection: {collection}")
            print(f"Using variable  : {varname}")
            print(f"Grid type       : {grid}")
            if is_3d and level is None:
                print(f"Data shape      : {data.shape} (3D: lev x lat x lon)")
            else:
                print(f"Data shape      : {data.shape} (2D: lat x lon)")
        
        return data, lats, lons, metadata

    def _compute_accumulation(self, fdate: str, pdate: str, variable: str, scale_factor: float = 1.0):
        """
        Accumulate precipitation from forecast start to valid time.
        
        Uses tavg1_2d_flx_Nx collection which contains time-averaged fluxes.
        Accumulates by summing (flux_rate * time_interval) for all timesteps.
        
        Special case: If pdate == fdate (requesting accumulation at initialization time),
        returns zeros (no accumulation yet).
        
        Parameters
        ----------
        fdate : str
            Forecast initialization time (accumulation start)
        pdate : str
            Valid/end time for accumulation
        variable : str
            Base variable to accumulate (PRECTOT, PRECSNO, etc.)
        scale_factor : float
            Scaling factor (e.g., 10.0 for snow depth with 10:1 ratio)
        
        Returns
        -------
        tuple
            (accumulated_data, lats, lons, metadata)
            - accumulated_data: Total accumulation in mm
            - lats: Latitude array
            - lons: Longitude array
            - metadata: Dictionary with accumulation info
        """
        from wxmaps_utils import parse_date_string
        from datetime import timedelta
        
        print(f"Computing accumulation for {variable} from {fdate} to {pdate}")
        
        # Parse dates
        fdate_clean = fdate.replace("z", "").replace("Z", "")
        pdate_clean = pdate.replace("z", "").replace("Z", "")
        start_dt = parse_date_string(fdate_clean)
        end_dt = parse_date_string(pdate_clean)
        
        # Special case: If pdate == fdate, no accumulation yet
        if start_dt >= end_dt:
            print(f"  pdate <= fdate: No accumulation period, returning zeros")
            
            # Read one file to get grid dimensions
            # Try to read PHIS from inst collection to get grid
            try:
                _, lats, lons, meta = self.read_variable(
                    fdate=fdate,
                    pdate=fdate,
                    variables=["PHIS"],
                    var_type='inst'
                )
            except Exception as e:
                raise FileNotFoundError(
                    f"Cannot get grid dimensions for accumulation: {e}"
                )
            
            # Return zeros with correct shape
            zeros = np.zeros((len(lats), len(lons)))
            
            metadata = {
                "units": "mm",
                "long_name": f"{variable}_accumulated_{scale_factor}x",
                "collection": "tavg1_2d_flx_Nx",
                "grid": meta.get("grid", "unknown"),
                "accumulation_start": fdate,
                "accumulation_end": pdate,
                "num_timesteps": 0,
                "total_hours": 0.0,
                "scale_factor": scale_factor,
                "base_variable": variable,
            }
            
            print(f"  Accumulation complete: 0.0 mm (no time elapsed)")
            
            return zeros, lats, lons, metadata
        
        # Find all available times in tavg collection
        print(f"  Finding available timesteps in tavg1_2d_flx_Nx...")
        all_times = self.find_available_times(fdate, var_type='tavg')
        
        # Filter to times between start and end (inclusive)
        # Use start_dt < t to exclude the initialization time itself
        valid_times = [t for t in all_times if start_dt < t <= end_dt]
        
        if not valid_times:
            print(f"  No tavg files found in accumulation period, returning zeros")
            
            # Get grid from PHIS
            try:
                _, lats, lons, meta = self.read_variable(
                    fdate=fdate,
                    pdate=fdate,
                    variables=["PHIS"],
                    var_type='inst'
                )
            except Exception as e:
                raise FileNotFoundError(
                    f"Cannot get grid dimensions for accumulation: {e}"
                )
            
            zeros = np.zeros((len(lats), len(lons)))
            
            metadata = {
                "units": "mm",
                "long_name": f"{variable}_accumulated_{scale_factor}x",
                "collection": "tavg1_2d_flx_Nx",
                "grid": meta.get("grid", "unknown"),
                "accumulation_start": fdate,
                "accumulation_end": pdate,
                "num_timesteps": 0,
                "total_hours": (end_dt - start_dt).total_seconds() / 3600.0,
                "scale_factor": scale_factor,
                "base_variable": variable,
            }
            
            return zeros, lats, lons, metadata
        
        print(f"  Found {len(valid_times)} timesteps to accumulate")
        
        # Initialize accumulation
        accumulated = None
        lats = None
        lons = None
        total_hours = 0.0
        
        # Accumulate over all timesteps
        for i, time_dt in enumerate(valid_times):
            # Format time for read_variable
            time_str = time_dt.strftime("%Y%m%d_%H") + "z"
            
            print(f"  Reading timestep {i+1}/{len(valid_times)}: {time_str}")
            
            # Read instantaneous flux rate (kg m-2 s-1)
            try:
                data, lats, lons, meta = self.read_variable(
                    fdate=fdate,
                    pdate=time_str,
                    variables=[variable],
                    var_type='tavg'
                )
            except FileNotFoundError:
                print(f"    Warning: Could not read {variable} at {time_str}, skipping")
                continue
            
            # Calculate time interval
            if i == 0:
                # First timestep: interval from start to this time
                dt_hours = (time_dt - start_dt).total_seconds() / 3600.0
            else:
                # Subsequent timesteps: interval from previous time to this time
                dt_hours = (time_dt - valid_times[i-1]).total_seconds() / 3600.0
            # tavg1_2d_flx_Nx files contain time-averaged fluxes (kg m-2 s-1)
            # Convert to mm over the time interval:
            #   kg/m²/s * seconds * (1 mm / 1 kg/m²) = mm
            #dt_seconds = dt_hours * 3600.0
            dt_seconds = 3600.0
 
            # Accumulate: flux_rate (kg/m²/s) * time_interval (s) = kg/m² = mm
            increment_mm = data * dt_seconds * scale_factor
            
            if accumulated is None:
                accumulated = increment_mm
            else:
                accumulated += increment_mm
            
            total_hours += dt_hours
            
            print(f"    Δt = {dt_hours:.2f} hrs, increment range: [{increment_mm.min():.3f}, {increment_mm.max():.3f}] mm")
        
        if accumulated is None:
            raise FileNotFoundError(
                f"No valid data found for {variable} accumulation"
            )
        
        # Create metadata
        metadata = {
            "units": "mm",
            "long_name": f"{variable}_accumulated_{scale_factor}x",
            "collection": "tavg1_2d_flx_Nx",
            "grid": meta.get("grid", "unknown"),
            "accumulation_start": fdate,
            "accumulation_end": pdate,
            "num_timesteps": len(valid_times),
            "total_hours": total_hours,
            "scale_factor": scale_factor,
            "base_variable": variable,
        }
        
        print(f"  Accumulation complete:")
        print(f"    Total period: {total_hours:.2f} hours")
        print(f"    Accumulated range: [{accumulated.min():.3f}, {accumulated.max():.3f}] mm")
        if scale_factor != 1.0:
            print(f"    (with {scale_factor}x scaling factor)")
        
        return accumulated, lats, lons, metadata

    def _read_surface_field_from_pres(self, fdate: str, pdate: str, varname: str):
        """
        Read 2D surface field from pressure level file.
        
        Fields like PHIS, PS, SLP exist in pressure level files but are 2D
        (no level dimension).
        
        Parameters
        ----------
        fdate : str
            Forecast initialization date
        pdate : str
            Valid/prediction date
        varname : str
            Variable name (e.g., 'PHIS', 'PS', 'SLP')
        
        Returns
        -------
        tuple
            (data, lats, lons, metadata)
        """
        # Resolve pressure level file
        path, collection, found_var = self.resolve_file(
            fdate, pdate, variables=[varname], var_type='pres'
        )
        
        with Dataset(path) as nc:
            if varname not in nc.variables:
                raise ValueError(f"Variable {varname} not found in {path}")
            
            var = nc.variables[varname]
            data = var[:]
            
            # Read grid (pressure files are lat-lon, not cubed-sphere)
            if "lat" in nc.variables and "lon" in nc.variables:
                lats = nc.variables["lat"][:]
                lons = nc.variables["lon"][:]
                grid = "latlon"
            else:
                raise ValueError(f"No lat/lon coordinates found in {path}")
            
            # Handle dimensions
            # PHIS is typically (time, lat, lon) or just (lat, lon)
            # PS/SLP are (time, lat, lon)
            
            # Squeeze time dimension if present
            if "time" in var.dimensions:
                if data.ndim >= 3 and data.shape[0] == 1:
                    data = data[0]  # Remove time dimension
            
            metadata = {
                "units": getattr(var, "units", ""),
                "long_name": getattr(var, "long_name", varname),
                "collection": collection,
                "grid": grid,
                "file": os.path.basename(path),
                "is_3d": False,
                "is_cubed_sphere": False,
                "is_surface_field": True,
            }
            
            print(f"Using file      : {os.path.basename(path)}")
            print(f"Using collection: {collection}")
            print(f"Using variable  : {varname} (2D surface field)")
            print(f"Grid type       : {grid}")
            print(f"Data shape      : {data.shape} (2D: lat x lon)")
        
        return data, lats, lons, metadata

    def _expand_variable_aliases(self, variables):
        """
        Expand variable name aliases to include common alternatives.
        
        For example:
            H → [H, HGT, HEIGHT, Z, HGHT]
            PRECTOT → [PRECTOT, PRECTOTLAND, PRECIP, TPRECIP]
        
        Parameters
        ----------
        variables : str or List[str]
            Variable name(s)
        
        Returns
        -------
        List[str]
            Expanded list of variable names to try
        """
        if isinstance(variables, str):
            variables = [variables]
        
        # Variable alias mapping
        ALIASES = {
            "H": ["H", "HGT", "HEIGHT", "Z", "HGHT"],
            "T": ["T", "TEMP", "TEMPERATURE"],
            "RH": ["RH", "RHUM", "RELHUM"],
            "U": ["U", "UWND", "UWIND"],
            "V": ["V", "VWND", "VWIND"],
            "OMEGA": ["OMEGA", "W", "VVEL"],
            "Q": ["Q", "QV", "SPFH", "SPHUM"],
            "PHIS": ["PHIS", "GEOPOT", "GEOPOTENTIAL"],
            "PS": ["PS", "PSFC", "SURFPRES"],
            "SLP": ["SLP", "MSLP", "PRMSL"],
            "PRECTOT": ["PRECTOT", "PRECTOTLAND", "PRECIP", "TPRECIP"],
            "PRECSNO": ["PRECSNO", "PRECSNOLAND", "SNOWFALL", "SNWFALL"],
            "RAIN": ["RAIN", "RAINRATE", "PRECRAIN"],
            "SNOW": ["SNOW", "SNOWRATE", "PRECSNO"],
        }
        
        expanded = []
        for var in variables:
            # Check if this is a level-extracted variable (e.g., H850)
            base_var, level = self._parse_level_variable(var)
            
            if base_var and level:
                # Expand base variable with aliases, then add level back
                if base_var in ALIASES:
                    for alias in ALIASES[base_var]:
                        expanded.append(f"{alias}{int(level)}")
                else:
                    expanded.append(var)
            else:
                # No level extraction, just expand base variable
                if var in ALIASES:
                    expanded.extend(ALIASES[var])
                else:
                    expanded.append(var)
        
        return expanded

    def get_available_variables(self, fdate: str, pdate: str, var_type='inst') -> List[str]:
        """
        Return sorted list of variables in resolved file.
        
        Parameters
        ----------
        fdate : str
            Forecast initialization date
        pdate : str
            Valid/prediction date
        var_type : str
            Variable type: 'inst', 'tavg', 'accum', or 'pres'
        
        Returns
        -------
        List[str]
            Sorted list of available variable names (excluding coordinates)
        """
        path, _, _ = self.resolve_file(fdate, pdate, var_type=var_type)
        
        with Dataset(path) as nc:
            return sorted(
                v for v in nc.variables
                if v not in ("lat", "lon", "lats", "lons", "time", "lev", 
                            "Xdim", "Ydim", "XCdim", "YCdim", "nf", "ncontact",
                            "contacts", "anchor", "orientationStrLen")
            )
    
    def get_pressure_levels(self, fdate: str, pdate: str) -> Optional[np.ndarray]:
        """
        Get available pressure levels from 3D pressure level file.
        
        Parameters
        ----------
        fdate : str
            Forecast initialization date
        pdate : str
            Valid/prediction date
        
        Returns
        -------
        np.ndarray or None
            Array of pressure levels in hPa, or None if not a 3D file
        """
        path, _, _ = self.resolve_file(fdate, pdate, var_type='pres')
        
        with Dataset(path) as nc:
            if "lev" in nc.variables:
                return nc.variables["lev"][:]
            return None

    def compute_vorticity(self, fdate: str, pdate: str, level: float = 500.0):
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
        
        # Read U and V components
        u_data, lats, lons, u_meta = self.read_variable(
            fdate=fdate,
            pdate=pdate,
            variables=["U"],
            var_type='pres',
            level=level
        )
        
        v_data, _, _, v_meta = self.read_variable(
            fdate=fdate,
            pdate=pdate,
            variables=["V"],
            var_type='pres',
            level=level
        )
        
        # Compute vorticity using centered finite differences
        vorticity = self._compute_vorticity_latlon(u_data, v_data, lats, lons)
        
        # Create metadata
        metadata = {
            "units": "s-1",
            "long_name": f"relative_vorticity_at_{level}hPa",
            "collection": u_meta.get("collection"),
            "grid": u_meta.get("grid"),
            "file": u_meta.get("file"),
            "is_3d": False,
            "is_cubed_sphere": u_meta.get("is_cubed_sphere"),
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
    
    
    def find_available_times(self, fdate: str, var_type='inst') -> List[datetime]:
        """
        Find available valid times for the given forecast initialization.
        
        For tavg collections, adjusts timestamps back to period END times
        (files use centered timestamps).
        
        For var_type='tavg', searches tavg1 (hourly) collections first and only
        those if available, to avoid mixing with 3-hourly data.
        
        Parameters
        ----------
        fdate : str
            Forecast initialization date
        var_type : str
            Variable type: 'inst', 'tavg', 'accum', or 'pres'
        
        Returns
        -------
        List[datetime]
            Sorted list of available valid times (period END times for tavg)
        """
        from datetime import timedelta
        
        cycle_dir = self._cycle_dir(fdate)
        forecast_dir = os.path.join(self.forecast_base, cycle_dir)
        
        if not os.path.isdir(forecast_dir):
            raise FileNotFoundError(f"Forecast directory not found: {forecast_dir}")
        
        times = []
        
        candidate_collections = self._candidate_collections(var_type=var_type)
        
        # For tavg, try hourly collections first
        if var_type == 'tavg':
            hourly_collections = [c for c in candidate_collections if 'tavg1' in c]
            other_collections = [c for c in candidate_collections if 'tavg1' not in c]
            
            # Try hourly first
            for collection in hourly_collections:
                times_from_collection = self._get_times_from_collection(
                    forecast_dir, collection
                )
                if times_from_collection:
                    times.extend(times_from_collection)
            
            # Only try other collections if no hourly data found
            if not times:
                for collection in other_collections:
                    times_from_collection = self._get_times_from_collection(
                        forecast_dir, collection
                    )
                    times.extend(times_from_collection)
        else:
            # For non-tavg, use all candidate collections
            for collection in candidate_collections:
                times_from_collection = self._get_times_from_collection(
                    forecast_dir, collection
                )
                times.extend(times_from_collection)
        
        if not times:
            raise FileNotFoundError(
                f"No model output found for fdate={fdate}, var_type={var_type}\n"
                f"  Forecast directory: {forecast_dir}\n"
                f"  Collections searched: {candidate_collections}"
            )
        
        return sorted(set(times))
    
    
    def _get_times_from_collection(self, forecast_dir: str, collection: str) -> List[datetime]:
        """
        Extract valid times from files in a specific collection.
        
        Parameters
        ----------
        forecast_dir : str
            Directory containing forecast files
        collection : str
            Collection name
        
        Returns
        -------
        List[datetime]
            List of valid times found in this collection
        """
        from datetime import timedelta
        
        pattern = os.path.join(
            forecast_dir, f"GEOS.fp.fcst.{collection}.*.V01.nc4"
        )
        
        # Determine time offset for this collection
        time_offset = timedelta(0)
        if "tavg1" in collection:
            time_offset = timedelta(minutes=30)
        elif "tavg3" in collection:
            time_offset = timedelta(minutes=90)
        
        times = []
        
        for filepath in glob.glob(pattern):
            try:
                basename = os.path.basename(filepath)
                
                # Find the date part (contains '+')
                date_part = None
                for part in basename.split("."):
                    if '+' in part:
                        date_part = part
                        break
                
                if date_part is None:
                    continue
                
                date_components = date_part.split("+")
                if len(date_components) != 2:
                    continue
                
                valid_str = date_components[1]  # e.g., "20260122_0530"
                
                # Parse valid time (this is the CENTERED time in the file)
                file_time = datetime.strptime(valid_str, "%Y%m%d_%H%M")
                
                # Adjust to period END time for tavg collections
                valid_time = file_time + time_offset
                
                times.append(valid_time)
                
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse filename {basename}: {e}")
                continue
        
        if times:
            print(f"  Found {len(times)} times in {collection}")
        
        return times

    def list_available_cycles(self, start_date: str = None, end_date: str = None) -> List[datetime]:
        """
        List available forecast initialization cycles.
        
        Parameters
        ----------
        start_date : str, optional
            Start date for search (default: search all)
            Accepts formats: 'YYYY-MM-DD', 'YYYYMMDD_HHz', '2026-01-20_00:00:00'
        end_date : str, optional
            End date for search (default: search all)
            Accepts formats: 'YYYY-MM-DD', 'YYYYMMDD_HHz', '2026-01-20_00:00:00'
        
        Returns
        -------
        List[datetime]
            Sorted list of available forecast initialization times
        """
        cycles = []
        
        # Find all year directories
        year_pattern = os.path.join(self.forecast_base, "Y????")
        for year_dir in sorted(glob.glob(year_pattern)):
            year = int(os.path.basename(year_dir)[1:])
            
            # Find month directories
            month_pattern = os.path.join(year_dir, "M??")
            for month_dir in sorted(glob.glob(month_pattern)):
                month = int(os.path.basename(month_dir)[1:])
                
                # Find day directories
                day_pattern = os.path.join(month_dir, "D??")
                for day_dir in sorted(glob.glob(day_pattern)):
                    day = int(os.path.basename(day_dir)[1:])
                    
                    # Find hour directories
                    hour_pattern = os.path.join(day_dir, "H??")
                    for hour_dir in sorted(glob.glob(hour_pattern)):
                        hour = int(os.path.basename(hour_dir)[1:])
                        
                        # Check if directory has files
                        if glob.glob(os.path.join(hour_dir, "GEOS.fp.fcst.*.nc4")):
                            cycles.append(datetime(year, month, day, hour))
        
        # Filter by date range if specified
        if start_date:
            start_dt = self._parse_date_flexible(start_date)
            cycles = [c for c in cycles if c >= start_dt]
        
        if end_date:
            end_dt = self._parse_date_flexible(end_date)
            cycles = [c for c in cycles if c <= end_dt]
        
        return sorted(cycles)
    
    
    def _parse_date_flexible(self, date_str: str) -> datetime:
        """
        Parse date string with flexible format support.
        
        Parameters
        ----------
        date_str : str
            Date string in various formats
        
        Returns
        -------
        datetime
            Parsed datetime object
        """
        # Try simple YYYY-MM-DD format first
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            pass
        
        # Try YYYYMMDD format
        try:
            return datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            pass
        
        # Try YYYY-MM-DD HH:MM:SS format
        try:
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
        
        # Fall back to wxmaps_utils parser
        try:
            from wxmaps_utils import parse_date_string
            return parse_date_string(date_str)
        except Exception:
            pass
        
        # If all else fails, raise error
        raise ValueError(
            f"Unable to parse date string: {date_str}. "
            f"Supported formats: 'YYYY-MM-DD', 'YYYYMMDD', 'YYYY-MM-DD HH:MM:SS', 'YYYYMMDD_HHz'"
        )

# ------------------------------------------------------------------
# Convenience function for module-level import
# ------------------------------------------------------------------
def create_reader(**kwargs):
    """
    Factory function to create GEOSForwardProcessingReader instance.
    
    Usage:
        from geos_forward_processing import create_reader
        reader = create_reader(exp_id="f5295_fp")
    """
    return GEOSForwardProcessingReader(**kwargs)
