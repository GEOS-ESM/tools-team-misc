# core.py
import os
import json
import time
import datetime as dt
import functools as ft
import dataclasses as dc
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
from typing import (Any, Dict, List, Literal, 
                    Optional, Tuple, Union, Sequence)
import logging

import numpy as np
import pandas as pd
import xarray as xr

# External, project-local dependencies
from .validator import CliConfig
from .validator import coerce_inputs, validate_and_build_config
from .data_config import fileHelper, cacheHelper, MISSING
from .constants import dataIndex, BadRequest
# ----------------------------
# Config loading / management
# ----------------------------

logger = logging.getLogger('cfapi')

# ----------------------------
# Core compute helpers
# ----------------------------

def _nearest_index(arr: np.ndarray, value: float) -> int:
    # Works great for moderate sizes and irregular spacing
    return int(np.abs(arr - value).argmin())

def _get_init_points(file: str|Path, 
                     keys: List[str], 
                     lat: float, lon: float, 
                     decode_times: bool=True, 
                     engine:Optional[str]=None) -> dataIndex:
    engine = engine or "netcdf4"
    with xr.open_dataset(file, decode_times=decode_times, engine=engine) as ds0:
        nlevs = len(ds0.lev) if ds0.lev.ndim else 0
        ilon = _nearest_index(ds0.lon.values, lon)
        ilat = _nearest_index(ds0.lat.values, lat)
        near_lon = float(ds0.lon.values[ilon])
        near_lat = float(ds0.lat.values[ilat])
        t0 = ds0.time.values[0].astype('datetime64[us]').astype(dt.datetime)
        longnames = {d.get('key'):ds0[d.get('var')].long_name for d in keys}
        return dataIndex(nlevs=nlevs,lon=near_lon,ilon=ilon,
                         lat=near_lat,ilat=ilat,
                         t0=t0,longnames=longnames)

def get_interp_dat(
        data_file: str, 
        near_lat: float, 
        near_lon: float, 
        keys: List[str],
        engine: Optional[str]=None,
        ):
    engine = engine or "netcdf4"
    with xr.open_dataset(data_file, decode_times = True, engine = engine) as ds:
        data = ds[keys]
        data = data.sel(lat = near_lat, lon=near_lon)
        df = data.to_dataframe()
        df.drop(['lat','lon'],axis=1, inplace=True)
        # df = data
    return df

def pool_get_interp_dat(files, di: dataIndex, keys, mp_pool=None, *, engine="netcdf4"):
    pool = mp_pool or mp.Pool()
    keys_in = [d.get('var') for d in keys]
    target = ft.partial(get_interp_dat, keys=keys_in, near_lat=di.lat, near_lon=di.lon,engine=engine)
    df_parts = pool.map(target, files)
    if mp_pool is None:
        pool.close()
        pool.join()
    df = pd.concat(df_parts)
    return df

def safe_pool_interp_data(files, d0, keys, pool, *, engine="netcdf4"):
    keys_in = [d.get('var') for d in keys]
    target = ft.partial(get_interp_dat, keys=keys_in, near_lat=d0.lat, near_lon=d0.lon, engine=engine)
    # Reuse the provided pool; if None, create/close one just for this call
    own = pool is None
    pool = pool or mp.Pool()
    try:
        chsz = max(1, len(files) // (getattr(pool, "_processes", 1) * 4 or 1))
        df_parts = pool.map(target, files, chunksize=chsz)
    finally:
        if own:
            pool.close()
            pool.join()
    df = pd.concat(df_parts)
    return df

def get_data(
        files: list, 
        di: dataIndex, 
        keys: list, 
        mp_pool=None, 
        *, 
        engine:str="netcdf4",
        fill_value: Optional[str|float]=None):
    df = safe_pool_interp_data(files,di,keys,mp_pool,engine=engine)
    levs = [float(v) for v in _get_coords(df,"lev")]
    if levs:
        df = df.reorder_levels(["time", "lev"]).sort_index()
    else:
        df.sort_index(inplace=True)
    time_arr = [str(t).replace(" ", "T") for t in _get_coords(df,"time")]
    interp_data = df2dict(df, keys,fill_value)
    return SimpleNamespace(data = interp_data, times = time_arr, levs = levs)

def _get_coords(df, coord):
    try:
        return list(df.index.get_level_values(coord).unique())
    except Exception:
        return []

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def format_scientific(obj, fmt="%.1e"):
    if isinstance(obj, float):
        return fmt % obj
    elif isinstance(obj, dict):
        return {k: format_scientific(v, fmt) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [format_scientific(v, fmt) for v in obj]
    return obj

def unit_conv(df: pd.DataFrame, products: str|List[str]) -> pd.DataFrame:
    for d in products:
        p = d.get('key')
        s_factor = d.get('scaling','1')
        if p not in df.columns:
            var = d.get('var')
            df[p] = df[var].copy(deep=True)
        if s_factor in ['ppbv','ppb','1e9','1.0e9']:
            df[p] = df[p] * 1.0e9
        elif is_number(s_factor):
            df[p] = df[p] * float(s_factor)
        elif s_factor in ['KtoF']:
            df[p] = (df[p]-273.15) * 9.0/5.0 + 32 
        elif s_factor in ['KtoC']:
            df[p] = (df[p]-273.15)
    return df 
    

def df2dict(
        df: pd.DataFrame, 
        products: List[str], 
        fill_value: Optional[str|float]=None) -> Dict[str, Dict[Any, List[float]]]:
    """
    Make a p23/v72 dataframe into nested dict: {product: {lev: [values...]}}.
    or flattens slv dataframes to {product: [values...], ...} 
    where the list index corresponds to time
    """
    out: Dict[str, Dict[Any, List[float]]] = {}
    fill_value = fill_value or MISSING
    df = unit_conv(df,products)
    levs = _get_coords(df, "lev")

    if len(levs) > 1:
        base_tbl = pd.pivot_table(
            df,
            index="time",
            columns="lev",
            dropna=False,
        )
    else:
        base_tbl = df
    for d in products:
        key = d.get('key')
        if len(levs) > 1:
            # tbl = base_tbl[key].reindex(columns=levs, fill_value=fill_value)
            tbl = base_tbl[key].reindex(columns=levs)
        else:
            # tbl = base_tbl[[key]].fillna(fill_value)
            tbl = base_tbl[[key]].fillna(fill_value)

        tbl = tbl.reset_index(drop=False).drop(columns=["time"])
        # tbl = tbl.fillna(fill_value)
        out[key] = tbl.to_dict(orient="list")
        if key in tbl.columns:
            out[key] = out[key].get(key)
    return out

def parse_time(t,dt_str: Optional[str]=None):
    dt_str = dt_str or '%Y-%m-%dT%H:%M:%S'
    if isinstance(t, str):
        try:
            return dt.datetime.strptime(t, dt_str)
        except ValueError:
            return dt.datetime.fromisoformat(t)
    return t

def dict2df(
    data: Dict[str, Any],
    times: Sequence,
    products: Optional[List[dict]] = None,
    set_index: bool = True,
) -> pd.DataFrame:
    """
    Reverse df2dict, given:
      - `data` in the structure produced by df2dict:
          * p23/l72: data[product] -> {lev: [values...]}
          * slv:     data[product] -> [values...]
      - `times`: sequence of time values (same length as value lists)
      - `products` (optional): list of dicts with 'key' entries,
        to enforce a specific product order.

    Returns:
      - p23/l72: DataFrame with MultiIndex (time, lev)
      - slv:     DataFrame with Index = time
    """
    times = [parse_time(t) for t in times]

    if not data:
        return pd.DataFrame()

    # Decide order of products
    if products is not None:
        prod_keys = [k for k in data.keys() if k.lower() in [p.lower() for p in products]]
    else:
        prod_keys = list(data.keys())

    # Peek at the first product to see if we're p23/l72 or slv
    first_vals = data[prod_keys[0]]

    # -------------------------
    # Multi-level p23/l72 case
    # -------------------------
    if isinstance(first_vals, dict):
        # Each product: vals = {lev: [values_over_time]}
        series_list = []
        for key in prod_keys:
            lev_map = data[key]  # {lev: [ ... ]}
            # Wide table: index=time, columns=lev
            wide = pd.DataFrame(lev_map, index=times)
            # Stack to MultiIndex: (time, lev)
            stacked = wide.stack().rename(key)  # Series with MultiIndex
            series_list.append(stacked)

        # Concatenate all products along columns
        df = pd.concat(series_list, axis=1)
        df.index.set_names(["time", "lev"], inplace=True)

        if not set_index:
            df = df.reset_index()

        return df

    # -------------
    # slv case
    # -------------
    else:
        dframe = {
            key: data[key]
            for key in prod_keys
        }
        df = pd.DataFrame(dframe, index=times)
        df.index.name = "time"

        if not set_index:
            df = df.reset_index()

        return df

def _add_response_time(result: dict, t0: time.time) -> dict:
    result["schema"]["response time"] = f"{time.time() - t0:.3f}"
    return result
# ----------------------------
# Public APIs (former routes)
# ----------------------------

def build_response_raw(
    **kwargs
) -> Dict[str, Any]:
    logger_in = kwargs.get('logger', logger) 
    start_date = kwargs.get('start_date')
    end_date = kwargs.get('end_date')
    raw = dict(
        version = kwargs.get('version'),
        collection = kwargs.get('collection'),
        dataset = kwargs.get('dataset'),
        level = kwargs.get('level'),
        product = kwargs.get('product'),
        products = kwargs.get('products'),
        lat = kwargs.get('lat'),
        lon = kwargs.get('lon'),
        start = kwargs.get('start',start_date),
        end = kwargs.get('end',end_date),
        days = kwargs.get('days'),
    )
    ns = coerce_inputs(raw)                     # same converters as CLI
    logger_in.info(ns)
    cfg = validate_and_build_config(ns, None)   # same cross-field rules

    # Now call your core with canonicalized inputs
    out = build_response_cfg(
        cfg = cfg,
        **kwargs
    )
    return out

def build_response_cfg(
    cfg: CliConfig,
    **kwargs
) -> Dict[str, Any]:
    """
    Former `/cfapi/<...>/get_dat` logic, minus Flask.

    Returns a dict with 'schema', 'values', and 'time'.
    Raises BadRequest on invalid inputs; returns a dict with a 'schema' key otherwise.
    """
    t0 = time.time()
    fill_value = kwargs.get('fill_value', MISSING)

    helper = fileHelper(cfg, kwargs.get('config_data'), kwargs.get('config_params'))
    # File list
    file_list = helper.get_file_list(kwargs.get('data_limit_override',False))

    d0 = _get_init_points(file_list[0], helper.product_keys, cfg.lat, cfg.lon, engine=kwargs.get('netcdf_engine'))
    # Optional cache path
    cache = cacheHelper(cfg, d0, **kwargs)

    data_out = cache.check_cache()
    
    if data_out:
        data_out = _add_response_time(data_out, t0)
        return data_out
    
    output = get_data(file_list, d0, helper.product_keys, kwargs.get('mp_pool'), fill_value=fill_value)

    schema = cache.get_schema(output.times, output.levs, helper)

    result = {"schema": schema, "values": output.data, "time": output.times}
    result = _add_response_time(result, t0)

    # Write cache if requested
    cache.save(result)

    return result

def get_multifile_vars(
    cfgs: List[CliConfig],
    **kwargs
) -> Dict[str, Any]:
    t0 = time.time()
    fill_value = kwargs.get('fill_value', MISSING)
    file_loc = kwargs.get('file_loc',False)
    helper = fileHelper(cfgs[0], kwargs.get('config_data'), kwargs.get('config_params'))
    # File list
    fl_args = dict(
            data_limit_override = kwargs.get('data_limit_override',False))
    cache_args = dict(
        use_cache = kwargs.get('use_cache', True),
        refresh = kwargs.get('refresh',False),
        cache_root = kwargs.get('cache_root'),
        fill_value = kwargs.get('fill_value'),
    )
    init_args = dict(
                     lat = float(cfgs[0].lat), 
                     lon = float(cfgs[0].lon), 
                     engine=kwargs.get('netcdf_engine', 'netcdf4')
    )

    file_list = helper.get_file_list(**fl_args)

    d0 = _get_init_points(file_list[0], helper.product_keys, **init_args)

    data_all = []
    with mp.Pool(processes=8) as pool:
        data_out = []
        for cfg in cfgs:
            _, keys = helper.get_keys(cfg.datakey, cfg.products)
            file_list = helper.get_file_list(datakey=cfg.datakey, **fl_args)
            d0 = _get_init_points(file_list[0], keys, **init_args)

            cache = cacheHelper(cfg, d0, **cache_args)
            data_out = cache.check_cache()
            if data_out:
                data_all.append(data_out)
                continue

            output = get_data(file_list, d0, keys, pool, engine="netcdf4", fill_value=fill_value)
            schema = cache.get_schema(output.times, output.levs, helper, cfg.datakey, cfg.products)

            result = {"schema": schema, "values": output.data, "time": output.times}
            result = _add_response_time(result, t0)
            # Write cache if requested
            cache.save(result)
            fn = cache._get_cache_path()
            if file_loc and fn.is_file():
                data_all.append({'file':fn, **result})
            else:
                data_all.append(result)

    return data_all

def get_root_schema(schema_file: Union[str, Path]=None) -> dict:
    from cfapi.config import loader
    schema = loader.load_yaml('cfapi_root_schema.yml')
    return schema
