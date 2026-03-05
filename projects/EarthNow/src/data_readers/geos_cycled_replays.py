"""
GEOS Model Cycled Replay Data Reader
Unified, scratch-aware reader for WxMaps
"""

import os
import glob
import numpy as np
from netCDF4 import Dataset
from datetime import datetime
from typing import Tuple, List, Dict, Optional
from functools import lru_cache
from .registry import register

@register("geos_cycled_replays")
class GEOSDataReader:
    """Reader for GEOS model output"""

    LCC_INST_COLLECTION = "hwt_15mn_slv_LCC"
    LCC_ACCUM_COLLECTION = "hwt_01hr_acc_LCC"
    LCC_PRES_COLLECTION = "hwt_15mn_prs_LCC"
    LCC_TAVG_COLLECTION = "hwt_01hr_slv_LCC"

    INST_COLLECTION  = "inst1_2d_asm_Nx"
    TAVG_COLLECTION = "tavg1_2d_flx_Nx"
    ACCUM_COLLECTION = "tavg1_2d_flx_Nx"
    PRES_COLLECTION = "geosgcm_fcst"

    LCC_GRID_FILE = (
        "/discover/nobackup/projects/gmao/osse2/"
        "stage/BCS_FILES/lambert_grid.nc4"
    )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(self,
                 exp_path="/discover/nobackup/projects/gmao/osse2/HWT",
                 exp_res="CONUS02KM",
                 exp_id="Feature-c2160_L137",
                 collection=None,
                 map_type=None):

        self.name = "GEOS-Cycled_Replays"
        self.exp_path  = exp_path
        self.exp_res   = exp_res
        self.exp_id    = exp_id
        self.map_type  = map_type
        self.collection = collection

        self.forecast_base = os.path.join(
            exp_path, exp_res, exp_id, "forecasts"
        )

        print("GEOS Reader initialized:")
        print(f"  Experiment    : {exp_res}/{exp_id}")
        print(f"  Forecast base : {self.forecast_base}")

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------
    @lru_cache(maxsize=1)
    def _load_lcc_latlon(self):
        with Dataset(self.LCC_GRID_FILE) as nc:
            return nc.variables["lats"][:], nc.variables["lons"][:]

    # ------------------------------------------------------------------
    # Collection logic
    # ------------------------------------------------------------------
    def _candidate_collections(self, var_type='inst') -> List[str]:
        if var_type == 'inst':
            # 2D inst
            if self.exp_res == "CONUS02KM" and self.map_type == "conus":
                return [self.LCC_INST_COLLECTION, self.INST_COLLECTION]
            return [self.INST_COLLECTION]
        elif var_type == 'tavg':
            # 2D tavg
            if self.exp_res == "CONUS02KM" and self.map_type == "conus":
                return [self.LCC_TAVG_COLLECTION, self.TAVG_COLLECTION]
            return [self.TAVG_COLLECTION]
        elif var_type == 'accum':
            # 2D accum
            if self.exp_res == "CONUS02KM" and self.map_type == "conus":
                return [self.LCC_ACCUM_COLLECTION, self.ACCUM_COLLECTION]
            return [self.TAVG_COLLECTION]
        elif var_type == 'pres':
            # 3D pressure
            if self.exp_res == "CONUS02KM" and self.map_type == "conus":
                return [self.LCC_PRES_COLLECTION, self.PRES_COLLECTION]
            return [self.PRES_COLLECTION]
        else:
            raise ValueError(f"Unknown var_type in _candidate_collections: {var_type}")

    # ------------------------------------------------------------------
    # Forecast directory helpers
    # ------------------------------------------------------------------
    def _cycle_dir(self, fdate: str) -> str:
        fdate_clean = fdate.replace("z", "").replace("Z", "")
        return f"CYCLED_REPLAY_P10800_C21600_T21600_{fdate_clean}z"

    def _forecast_dirs(self, fdate: str):
        """
        Yield forecast directories in priority order:
        1) cycle dir
        2) cycle dir / scratch
        """
        base = os.path.join(self.forecast_base, self._cycle_dir(fdate))
        yield base
        yield os.path.join(base, "scratch")

    # ------------------------------------------------------------------
    # Core resolver (SINGLE SOURCE OF TRUTH)
    # ------------------------------------------------------------------
    def resolve_file(self, fdate: str, pdate: str, variables=None, var_type='inst', raise_on_missing=False):
        """
        Resolve the best available GEOS file.

        Args:
            fdate: Forecast date
            pdate: Product date
            variables: Variable(s) to look for
            var_type: Type of variable ('inst', 'tavg', etc.)
            raise_on_missing: If True, raise FileNotFoundError. If False, return (None, None, None)

        Returns:
            (path, collection, matched_variable) or (None, None, None) if not found
        """
        from wxmaps_utils import parse_date_string

        if isinstance(variables, str):
            variables = [variables]

        pdate_dt = parse_date_string(pdate)
        timestamp = pdate_dt.strftime("%Y%m%d_%H%M")

        for forecast_dir in self._forecast_dirs(fdate):
            if not os.path.isdir(forecast_dir):
                continue

            for collection in self._candidate_collections(var_type=var_type):
                filename = f"GEOS.{collection}.{timestamp}z.nc4"
                path = os.path.join(forecast_dir, filename)

                if not os.path.exists(path):
                    continue

                if variables is None:
                    self.collection = collection
                    return path, collection, None

                try:
                    with Dataset(path) as nc:
                        for var in variables:
                            if var in nc.variables:
                                self.collection = collection
                                return path, collection, var
                except Exception:
                    continue

        # File not found - handle gracefully
        error_msg = (
            f"No suitable GEOS file found for {pdate} "
            f"(vars={variables}, collections={self._candidate_collections(var_type=var_type)})"
        )
    
        if raise_on_missing:
            raise FileNotFoundError(error_msg)
        else:
            # Log warning but don't crash
            print(f"WARNING: {error_msg}")
            return None, None, None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_file_path(self, fdate: str, pdate: str, raise_on_missing=False) -> Optional[str]:
        """
        Get the file path for a given date.
        
        Returns:
            str: File path, or None if not found and raise_on_missing=False
        """
        path, _, _ = self.resolve_file(fdate, pdate, raise_on_missing=raise_on_missing)
        return path

    def read_variable(self, fdate: str, pdate: str, variables, var_type='inst', raise_on_missing=False):
        """
        Read the first available variable from a prioritized list.
        
        Returns:
            (data, lats, lons, metadata) or (None, None, None, None) if not found and raise_on_missing=False
        """
        if isinstance(variables, str):
            variables = [variables]

        path, collection, varname = self.resolve_file(
            fdate, pdate, variables, var_type=var_type, raise_on_missing=raise_on_missing
        )
        
        # Handle missing file gracefully
        if path is None:
            return None, None, None, None

        with Dataset(path) as nc:
            var = nc.variables[varname]
            data = var[:]

            if data.ndim >= 3:
                data = data[0]

            if "lat" in nc.variables and "lon" in nc.variables:
                lats = nc.variables["lat"][:]
                lons = nc.variables["lon"][:]
                grid = "latlon"
            else:
                lats, lons = self._load_lcc_latlon()
                grid = "lcc"

            metadata = {
                "units": getattr(var, "units", ""),
                "long_name": getattr(var, "long_name", varname),
                "collection": collection,
                "grid": grid,
                "file": os.path.basename(path),
            }

            print(f"Using file      : {path}")
            print(f"Using collection: {collection}")
            print(f"Using variable  : {varname}")

            return data, lats, lons, metadata

    def read_multiple_variables(self, fdate: str, pdate: str,
                                variables: List[str], raise_on_missing=False) -> Dict[str, Tuple]:
        """
        Read multiple variables from the same resolved file.
        
        Returns:
            Dict of {varname: (data, lats, lons, metadata)} or empty dict if not found
        """
        path, collection, _ = self.resolve_file(fdate, pdate, raise_on_missing=raise_on_missing)
        
        # Handle missing file gracefully
        if path is None:
            return {}

        results = {}

        with Dataset(path) as nc:
            if "lat" in nc.variables and "lon" in nc.variables:
                lats = nc.variables["lat"][:]
                lons = nc.variables["lon"][:]
            else:
                lats, lons = self._load_lcc_latlon()

            for varname in variables:
                if varname not in nc.variables:
                    continue

                var = nc.variables[varname]
                data = var[:]

                if data.ndim >= 3:
                    data = data[0]

                results[varname] = (
                    data,
                    lats,
                    lons,
                    {
                        "units": getattr(var, "units", ""),
                        "long_name": getattr(var, "long_name", varname),
                        "collection": collection,
                    },
                )

        return results

    def get_available_variables(self, fdate: str, pdate: str, raise_on_missing=False) -> List[str]:
        """
        Return sorted list of variables in resolved file.
        
        Returns:
            List of variable names, or empty list if file not found and raise_on_missing=False
        """
        path, _, _ = self.resolve_file(fdate, pdate, raise_on_missing=raise_on_missing)
        
        if path is None:
            return []

        with Dataset(path) as nc:
            return sorted(
                v for v in nc.variables
                if v not in ("lat", "lon", "time", "lev")
            )

    def find_available_times(self, fdate: str) -> List[datetime]:
        """
        Find available times for the preferred collection.
        """
        times = []

        for forecast_dir in self._forecast_dirs(fdate):
            if not os.path.isdir(forecast_dir):
                continue

            for collection in self._candidate_collections():
                pattern = os.path.join(
                    forecast_dir, f"GEOS.{collection}.*.nc4"
                )
                for f in glob.glob(pattern):
                    try:
                        t = os.path.basename(f).split(".")[2].replace("z", "")
                        times.append(datetime.strptime(t, "%Y%m%d_%H%M"))
                    except Exception:
                        continue

            if times:
                break

        if not times:
            raise FileNotFoundError(f"No model output found for {fdate}")

        return sorted(set(times))
