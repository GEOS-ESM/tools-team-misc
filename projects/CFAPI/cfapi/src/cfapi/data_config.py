# core.py
import os
import json
import time
import math
import datetime as dt
import functools as ft
import dataclasses as dc
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
import xarray as xr

# External, project-local dependencies
from .api_config import versionConfig, ROOT
from .validator import CliConfig
from .constants import CACHE_ROOT, BadRequest, CliConfig, dataIndex
MISSING = 1.0e15
# ----------------------------
# Config loading / management
# ----------------------------

logger = logging.getLogger('cfapi.core')
# ----------------------------
# Validation helpers
# ----------------------------

def format_fill_value_scientific(obj, fill_value: float=1.0e15, fmt: str="%.1e"):
    if isinstance(obj, float):
        try:
            if math.isclose(obj, fill_value, rel_tol=1.0e-6):
                return fmt % fill_value   # or fmt % obj
            else:
                return obj
        except TypeError:
            logger.warning(f'obj: {obj}, fill_value: {fill_value}')
            return obj
    elif isinstance(obj, dict):
        return {k: format_fill_value_scientific(v, fill_value, fmt) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [format_fill_value_scientific(v,fill_value, fmt) for v in obj]
    return obj

# ----------------------------
# File discovery
# ----------------------------

class fileHelper(versionConfig): 
    def __init__(
            self, 
            cfg: CliConfig,
            config_data: Optional[dict] = None,
            config_params: Optional[dict] = None,
            config_dir: Optional[Path|str]=None,

            ):
        super().__init__(version=cfg.version, config_dir=config_dir,
                         config_params=config_params, config_data=config_data)
        self.cfg = cfg
        self.stream_name: str = self.get_stream_name(self.cfg.datakey)
        self.latest_time: dt.datetime = self.get_latest()
        self.field_keys, self.product_keys = self.get_keys()

    def get_stream_name(self, datakey: str):
        
        stream_name: str = self.data.get('data',{}).get(datakey)
        return stream_name

    def _get_dset_path(self, file: Path) -> Path:
        with open(file, "rt") as tmp_f:
            txt = "dset"
            for line in tmp_f:
                if txt in line.lower():
                    f_path = Path(line.split(" ",1)[-1]).parent
        return f_path.resolve()

    def _get_dt(self, s: Optional[str|dt.datetime], default: Optional[dt.datetime]=None) -> Optional[dt.datetime]:
        if not s or s == 'latest':
            return default
        elif isinstance(s, dt.datetime):
            return s
        try:
            return dt.datetime.strptime(s,'%Y%m%d') 
        except Exception:
            logger.exception(f'Unable to convert date input: {s}')
            raise BadRequest(f'Invalid date input: {s}. Allowed format: YYYYMMDD')
        
    def get_latest(self) -> dt.datetime:
        ctrl_key = self.cfg.datakey.replace('v72','slv')
        ctrl = self.data.get('data',{}).get(ctrl_key)
        latest_file = self.latest_dir / f"{ctrl}.latest"
        f_path = self._get_dset_path(latest_file)
        rel_path = Path(f_path).relative_to(self.base_dir)
        path_template: str = self.dataset.get('fcast')
        latest = dt.datetime.strptime(str(rel_path), path_template)
        return latest
    
    def get_date_range(self) -> Tuple[dt.datetime, dt.datetime]:
        date_limits: List = self.data.get('date_limits',{}).get(self.cfg.collection, [None,None])
        min_date, max_date = date_limits 
        min_date = dt.datetime.strptime(str(min_date),'%Y%m%d') if min_date else None
        max_date = dt.datetime.strptime(str(max_date),'%Y%m%d') if max_date else self.latest_time
        return min_date, max_date
    
    def dir_for(self, when: dt.datetime) -> Path:
        """Absolute directory for this collection/time."""
        path_template_str = self.dataset.get(self.cfg.collection,'')
        return self.base_dir / when.strftime(path_template_str)

    def glob_for(self, when: dt.datetime) -> List[str]:
        """Return (files, wildcard_path) for this time slice."""
        d = self.dir_for(when)
        
        files = sorted(d.glob(f"*{self.stream_name}*"))
        flist = [str(file) for file in files]
        if not flist:
            logger.info(str(d / f'*{self.stream_name}*'))
            raise BadRequest("No data for this period or bad request (empty file list).")            
        return flist
    
    def get_file_list(self, 
            data_limit_override: bool=False,
            datakey: Optional[str]=None) -> List[str]:
        """ Universesal entrance for file list functions"""
        datakey = datakey or self.cfg.datakey
        time_period = self.cfg.days
        temp = self.stream_name
        self.stream_name = self.get_stream_name(datakey)

        if self.cfg.collection == 'fcast':
            files = self.get_fcst_file_list()
        else:
            files = self.get_assim_file_list(data_limit_override, time_period)

        if temp != self.stream_name:
            self.stream_name = temp
        return files

    def get_fcst_file_list(self) -> List[str]:
        _, max_date = self.get_date_range()
        start = self._get_dt(self.cfg.start, max_date)
        return self.glob_for(start)
    
    def get_assim_file_list(self,
            data_limit_override: bool=False,
            time_period: int=1) -> List[str]:
        period = dt.timedelta(time_period + 1) #default to 1 day (capture 2 days to grab last 24 hours)
        min_date, max_date = self.get_date_range()
        time_flag = 0
        end = self._get_dt(self.cfg.end)
        start = self._get_dt(self.cfg.start)
        if not end and not start:
            end = max_date
            start = end - period
            time_flag = 'end'
        elif start and not end:
            end = min(max_date, start + period)
            time_flag = 'start'
        elif end and not start:
            start = max(min_date,end - period)
            time_flag = 'end'
        else:
            time_flag = 'start'
            del_time = (end - start)
            time_period = del_time.days
            if del_time > dt.timedelta(32) and "aqc" not in self.stream_name and not data_limit_override:
                raise BadRequest(f"Requested time period ({del_time.days} days) too long. Period should be < 30 days")
                # return [], 'Over data limit'
            if del_time > dt.timedelta(92):
                raise BadRequest(f"Requested time period ({del_time.days} days) too long. Period should be < 90 days")
        cur = start
        file_list: List[str] = []
        while cur <= end:
            file_list += self.glob_for(cur)
            cur += dt.timedelta(days=1)
        return self.get_index_by_time(file_list, time_period, time_flag)
        
    def get_index_by_time(self, file_list, time_period, time_flag):
        dt_list = [dt.datetime.strptime(f.rsplit('.',3)[-3],'%Y%m%d_%H%Mz') for f in file_list]
        if time_flag=='end':
            flist_out = [f for f,fdt in zip(file_list,dt_list) if (dt_list[-1] - fdt) <= dt.timedelta(days=time_period)]
        elif time_flag=='start':
            flist_out = [f for f,fdt in zip(file_list,dt_list) if (fdt - dt_list[0]) <= dt.timedelta(days=time_period)]
        else:
            flist_out = file_list
        return flist_out

    def get_keys(self, datakey: Optional[str]=None, products: Optional[List]=None) -> Tuple[dict, dict]:
        products = products or self.cfg.products
        datakey = datakey or self.cfg.datakey
        ds_fields = self.params.get(datakey,{})
        field_info = self.params.get('fields',{})
        field_keys: List[str] = []
        for p in products:
            field_keys += ds_fields.get(p,{}).get('fields',[p])
            
        product_keys = [{'key': k, **field_info.get(k)} for k in field_keys] 
        return field_keys, product_keys

    def get_units(self, product_keys: Optional[List]=None):
        product_keys = product_keys or self.product_keys
        return {d.get('key'):d.get('units') for d in product_keys}




def _nearest_index(arr: np.ndarray, value: float) -> int:
    # Works great for moderate sizes and irregular spacing
    return int(np.abs(arr - value).argmin())

def _get_init_points(file: str|Path, 
                     keys: List[str], 
                     lat: float, lon: float, 
                     decode_times: bool=True, 
                     engine:str="netcdf4") -> dataIndex:
    with xr.open_dataset(file, decode_times=decode_times, engine=engine) as ds0:
        nlevs = len(ds0.lev) if ds0.lev.ndim else 0
        ilon = _nearest_index(ds0.lon.values, lon)
        ilat = _nearest_index(ds0.lat.values, lat)
        near_lon = float(ds0.lon.values[ilon])
        near_lat = float(ds0.lat.values[ilat])
        t0 = ds0.time.values[0].astype('datetime64[us]').astype(dt.datetime)
        longnames = {d.get('key'):ds0[d.get('var')].long_name for d in keys}
        return dataIndex(nlevs=nlevs,lon=near_lon,
                         lat=near_lat,
                         t0=t0,longnames=longnames)
    

class cacheHelper:
    def __init__(self, cfg: CliConfig, d0: dataIndex, **kwargs):
        cache_root = kwargs.get('cache_root')
        self.cache_root: Path = Path(cache_root or CACHE_ROOT)
        fill_value = kwargs.get('fill_value')
        self.fill_value = fill_value or MISSING
        self.cfg = cfg
        self.d0 = d0
        self.use_cache = kwargs.get('use_cache',True)
        self.refresh = kwargs.get('refresh',False)
        self.path = None#self._get_cache_path()
        self.time_str = "%Y-%m-%dT%H:%M:%S"
        self.products = self.cfg.products

    def img_cache(self, 
                  plot_type: Literal['sfc','pres','assim'] = 'sfc',
                  product: Literal['NO2', 'O3', 'PM25'] = 'NO2'
                  ):
        file_time = self.d0.t0.strftime('%Y%m%dT%H%M')
        img_dir = self.cache_root / 'plots' / file_time 
        img = img_dir / f"cf_{plot_type}_{product}_{self.d0.lat}_{self.d0.lon}.png"

    def _get_cache_path(self) -> Path:
        p = '+'.join(self.cfg.products)
        file_time = self.d0.t0.strftime('%Y%m%dT%H%M')
        cache_dir = self.cache_root / 'file_cache' / self.cfg.version / self.cfg.collection / self.cfg.datakey / file_time / p
        if self.cfg.collection == 'assim' and self.cfg.days>1:
            fn = f"data_{self.d0.lat}_{self.d0.lon}_{self.cfg.days}days.json"
        else:
            fn = f"data_{self.d0.lat}_{self.d0.lon}.json"
        cache_path = cache_dir / fn
        return cache_path
    
    def base_schema(self, times) -> dict:
        schema = {
            "version": self.cfg.version,
            "lat": float(self.cfg.lat),
            "lon": float(self.cfg.lon),
            "near_lat": float(self.d0.lat),
            "near_lon": float(self.d0.lon),
            "type": "object",
            "missing value": MISSING,
            }
        if self.cfg.collection == 'fcast':
            schema.update({
                  "description": f"5-day forecast of hourly values of ", 
                  "forecast initialization time": self.d0.t0.strftime(self.time_str)})
        else:
            schema.update({
                    "description": f"Replay of ", 
                    "time range": f"{times[0]} to {times[-1]}"})
        return schema
    
    def dataset_schema(self, 
                           levs: List, 
                           helper: fileHelper, 
                           datakey: Optional[str]=None, 
                           products: Optional[List]=None):
        if datakey:
            field_keys, product_keys = helper.get_keys(datakey, products)
        else:
            datakey = self.cfg.datakey
            field_keys = helper.field_keys
            product_keys = helper.product_keys

        if 'slv' in datakey:
            nlevs = "Surface Level"
        else:
            nlevs = datakey[-2:]
        return {
            "product": field_keys,
            "dataset": helper.get_stream_name(datakey),
            "lev": nlevs,
            "levs": levs,
            "units": helper.get_units(product_keys)
        }   

    def get_schema(self, times: List,
                           levs: List, 
                           helper: fileHelper, 
                           datakey: Optional[str]=None, 
                           products: Optional[List]=None):
        base_schema = self.base_schema(times)
        ds_schema = self.dataset_schema(levs, helper,datakey,products)
        base_schema['description'] += ', '.join(self.cfg.products)
        return {
            "longname": self.d0.longnames,
            **ds_schema,
            **base_schema
        }   

    def check_cache(self) -> dict:
        out = {}
        if not self.use_cache or self.refresh:
            return out 
        if not self.path:
            self.path = self._get_cache_path()
        try: 
            if self.path.exists():
                
                with self.path.open("r") as fp:
                    out = json.load(fp)
                return out
            else:
                return out
        except Exception:
            logger.exception(f'File {str(self.path)} could not be read!')
            out = {}
        return out
        
    def save(self, result: dict):
        if not self.path:
            self.path = self._get_cache_path()
        if self.use_cache:
            self.path.parent.mkdir(parents=True, exist_ok=True,mode=0o775)
            dt_var = 'forecast initialization time'
            init_time = result.get(dt_var)
            if init_time and isinstance(init_time,dt.datetime):
               result[dt_var] = init_time.strftime("%Y-%m-%dT%H:%M:%S")
            result = format_fill_value_scientific(result, self.fill_value, fmt='%.1e')
            try:
                with self.path.open("w") as fp:
                    json.dump(result, fp)
                os.chmod(self.path, 0o755) 
            except Exception:
                logger.exception(f'File {str(self.path)} not saved')       

