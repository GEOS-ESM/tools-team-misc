import os
import sys
import requests
import datetime as dt
from pathlib import Path
from types import SimpleNamespace
import logging
from typing import (
    List,
    Generator,
    Tuple,
    Dict,
    Optional,
    Any,
    Mapping,
    Iterable,
    Literal,
)

import numpy as np

from cfapi import validator
from cfapi.validator import CliConfig
from cfapi import core

logger = logging.getLogger("cf_map.data_prepper")
__all__ = ["CFDatagramPrep", "get_data", "api_call", "calc_wspd", "get_wbackup"]


def get_data(groups_in: List[dict], raw_in: dict, dry: bool = False) -> Dict[str, Any]:
    def group_datakey(groups_in):
        ds_group = {}
        for group in groups_in:
            datakey = group.get("datakey")
            if datakey not in ds_group:
                ds_group[datakey] = list(group.get("products"))
            else:
                ds_group[datakey] += group.get("products")
        return [{"datakey": k, "products": v} for k, v in ds_group.items()]

    def build_cfgs(groups: List[dict], raw_in: dict) -> List[CliConfig]:
        cfgs = []
        for group in groups:
            raw = {
                "datakey": group.get("datakey"),
                "products": group.get("products"),
                **raw_in,
            }
            cfg = validator.coerce_config(raw)
            cfgs.append(cfg)
        return cfgs

    groups = group_datakey(groups_in)
    cfgs = build_cfgs(groups, raw_in)
    if dry:
        results = [{"cfg": cfg} for cfg in cfgs]
    else:
        if cfgs[0].version == "v2":
            results = core.get_multifile_vars(cfgs, file_loc=True)
        if cfgs[0].version == "v1":
            results = []
            for cfg in cfgs:
                res = api_call(cfg)
                results.append(res)
    output = {}
    for group, result in zip(groups, results):
        output[group.get("datakey")] = result

    return output


def api_call(cfg: CliConfig):
    params = {
        "dataset": cfg.dataset,
        "level": cfg.level,
        "products": cfg.products,
        "lat": cfg.lat,
        "lon": cfg.lon,
        "start_date": cfg.start,
        "days": cfg.days,
        "version": cfg.version,
    }

    response = requests.get(
        f"https://fluid.nccs.nasa.gov/cf/api/{cfg.collection}", params=params
    )
    data = response.json()
    return data


def calc_winddir(U: List[float], V: List[float]) -> List[float]:
    angle_radians = np.arctan2(V, U)
    angle_degrees = np.degrees(angle_radians)
    return (angle_degrees + 180) % 360


def calc_wspd(U: List[float], V: List[float]) -> List[float]:
    return np.sqrt([(u_val**2 + v_val**2) for u_val, v_val in zip(U, V)])


def time2dt(times: List[dt.datetime]) -> List[str]:
    return [dt.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S") for t in times]


def get_wbackup(d: dict, keys: List[str], default: Optional[Any] = None):
    var = next((d.get(k) for k in keys if k in d.keys()), default)
    return var


def dt2str(t: Optional[dt.datetime | str]) -> str | None:
    if isinstance(t, (dt.datetime, dt.date)):
        return t.strftime("%Y%m%d")
    elif "-" in t:
        return t.replace("-", "")
    else:
        return t


class CFDatagramPrep:
    def __init__(
        self,
        lat: str | float,
        lon: str | float,
        plot_type: Optional[str] = None,
        product: Optional[str] = None,
        **kwargs,
    ):
        self.product = (product or "no2").lower()
        plot_type = plot_type or "pres"
        version = kwargs.pop("version", "v2")
        data = self.get_plot_vars(lat, lon, plot_type, version, **kwargs)
        schema = data.get("met_slv").get("schema", {})
        self.lat = schema.get("near_lat")
        self.lon = schema.get("near_lon")
        self.met = self.prep_met(data.get("met_slv"))
        self.sfc = self.prep_sfc(get_wbackup(data, ["chm_slv", "aqc_slv"], {}))
        self.sfc_pm25 = self.prep_sfc_pm25(data.get("chm_slv"))
        self.times = self.met.time
        self.data = data
        if data.get("chm_p23"):
            if not kwargs.get("dry"):
                self.contour = self.prep_contour_data(data.get("chm_p23"))

    def __repr__(self):
        """
        Return a concise summary of the CFDatagramPrep object.
        Example:
        <CFDatagramPrep plot='replay' version='v2' lat=38.9 lon=-77.0 range=2025-10-01→2025-10-31>
        <CFDatagramPrep plot='pres' version='v2' lat=38.9 lon=-77.0 fcst_init=2025-10-01>
        """
        # Try to extract plot type and version from data
        # (they’re embedded inside the data tree rather than stored directly)
        version = None
        for v in ("met_slv", "chm_slv", "aqc_slv", "chm_p23"):
            if v in getattr(self, "data", {}):
                d = self.data[v]
                if isinstance(d, dict) and "schema" in d and "version" in d["schema"]:
                    version = d["schema"]["version"]
                    break
        version = version or "v2"

        # Infer plot type
        if "aqc_slv" in self.data:
            plot_type = "replay"
        elif "chm_p23" in self.data:
            plot_type = "pres"
        else:
            plot_type = "sfc"

        # Format times
        if hasattr(self, "times") and self.times:
            t0 = self.times[0]
            t1 = self.times[-1]
            if plot_type == "replay":
                time_str = f"range={t0:%Y-%m-%d}→{t1:%Y-%m-%d}"
            else:
                time_str = f"fcst_init={t0:%Y-%m-%d}"
        else:
            time_str = "time=?"

        return (
            f"<CFDatagramPrep plot='{plot_type}' version='{version}' "
            f"lat={self.lat:.2f} lon={self.lon:.2f} {time_str}>"
        )

    def get_plot_vars(
        self,
        lat: str | float,
        lon: str | float,
        plot_type: Literal["pres", "sfc", "replay"],
        version: str,
        **kwargs,
    ) -> List[dict]:
        groups = [dict(datakey="met_slv", products=["MET"])]
        if plot_type == "replay":
            raw_in = dict(
                version=version,
                collection="assim",
                lat=lat,
                lon=lon,
            )
            start = kwargs.get("start", kwargs.get("start_date"))
            end = kwargs.get("end", kwargs.get("end_date"))
            num_days = kwargs.get("num_days", 30)
            if start:
                if end:
                    raw_in.update(dict(start=dt2str(start), end=dt2str(end)))
                else:
                    raw_in.update(dict(start=dt2str(start), num_days=num_days))
            else:
                raw_in.update(dict(num_days=num_days))

            groups += [
                dict(datakey="aqc_slv", products=["NO2", "O3", "SO2", "PM25", "CO"])
            ]
        else:
            groups += [
                dict(
                    datakey="chm_slv",
                    products=["NO2", "O3", "SO2", "PM25", "CO", "PM25_RH35"],
                )
            ]

            if plot_type == "pres":
                groups += [
                    dict(datakey="chm_p23", products=["NO2", "O3", "SO2", "PM25", "CO"])
                ]
            start = kwargs.get("start", kwargs.get("fcst", "latest"))
            raw_in = dict(
                version=version,
                collection="fcast",
                lat=lat,
                lon=lon,
                start=start,
                end=None,
            )
        dry = kwargs.get("dry", False)
        return get_data(groups, raw_in, dry)

    def prep_met(self, data: dict) -> SimpleNamespace:
        vals = data.get("values", {})
        return SimpleNamespace(
            wspd=calc_wspd(
                get_wbackup(vals, ["U10M", "U"], []),
                get_wbackup(vals, ["V10M", "V"], []),
            ),
            winddir=calc_winddir(
                get_wbackup(vals, ["U10M", "U"], []),
                get_wbackup(vals, ["V10M", "V"], []),
            ),
            temp=get_wbackup(vals, ["T2M_F", "T_F", "T10M_F"], []),
            cldtt=vals.get("CLDTT", []),
            precip=vals.get("TPREC_mm", []),
            time=time2dt(data.get("time", [])),
            file=data.get("file"),
        )

    def prep_sfc(self, data: dict) -> SimpleNamespace:
        vals = data.get("values", {})
        return SimpleNamespace(
            o3=vals.get("O3"),
            so2=vals.get("SO2"),
            co=vals.get("CO"),
            no2=vals.get("NO2"),
            pm25=get_wbackup(vals, ["PM25_RH35", "PM25_RH35_GCC"], []),
            time=time2dt(data.get("time", [])),
            file=data.get("file"),
        )

    def prep_sfc_pm25(self, data: dict) -> SimpleNamespace:
        if not data:
            return None
        vals = data.get("values", {})
        return SimpleNamespace(
            ni=get_wbackup(vals, ["PM25nit_RH35", "PM25ni_RH35_GCC"], []),
            ss=get_wbackup(vals, ["PM25ss_RH35", "PM25ss_RH35_GCC"], []),
            du=get_wbackup(vals, ["PM25du_RH35", "PM25du_RH35_GCC"], []),
            oc=get_wbackup(vals, ["PM25oc_RH35", "PM25oc_RH35_GCC"], []),
            bc=get_wbackup(vals, ["PM25bc_RH35", "PM25bc_RH35_GCC"], []),
            su=get_wbackup(vals, ["PM25su_RH35", "PM25su_RH35_GCC"], []),
            soa=get_wbackup(vals, ["PM25soa_RH35", "PM25soa_RH35_GCC"], []),
            nh4=get_wbackup(vals, ["PM25nh4_RH35", "PM25nh4_RH35_GCC"], []),
            time=time2dt(data.get("time", [])),
            file=data.get("file"),
        )

    def prep_contour_data(self, data: dict, min_P: float = 500.0) -> SimpleNamespace:
        def get_2d_array(vals, lev_keys, keys, default=[]):
            data = get_wbackup(vals, keys, default)
            return np.array(
                [data.get(lev_keys.get(k)) for k in sorted(lev_keys.keys())]
            )

        vals = data.get("values", {})
        data0 = next((v for k, v in vals.items()), {})
        levs, keys = zip(*((float(k), k) for k in data0.keys() if float(k) >= min_P))
        key_dict = {float(k): k for k in data0.keys() if float(k) >= min_P}
        return SimpleNamespace(
            levs=[k for k in sorted(key_dict.keys())],
            no2=get_2d_array(vals, key_dict, ["NO2"]),
            so2=get_2d_array(vals, key_dict, ["SO2"]),
            o3=get_2d_array(vals, key_dict, ["O3"]),
            co=get_2d_array(vals, key_dict, ["CO"]),
            pm25=get_2d_array(vals, key_dict, ["PM25_RH35", "PM25_RH35_GCC", "PM25"]),
            time=time2dt(data.get("time", [])),
            file=data.get("file"),
        )
