import sys
import argparse
import datetime as dt
from pathlib import Path
from types import SimpleNamespace
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
import yaml
import logging

from cfapi.aliases import make_aliases_from_dict, parse_list_value
from cfapi.data_config import fileHelper
from cfapi.constants import get_cache_dir
from cfapi import validator as v
from cfapi.core import dict2df

logger = logging.getLogger("cf_map.local_config")
ROOT = Path(__file__).parents[2]
CACHE = get_cache_dir(ROOT)
IMG_OUTPUT_DIR = CACHE / "plots"
FILE_OUTPUT_DIR = CACHE / "file_downloads"
file_template = "cf{version}_{type}_{product}_{lat}_{lon}.{ext}"


# --------------------
# Types & validators
# --------------------
def lat_type(s: str) -> float:
    try:
        val = float(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Latitude must be a float: {s}") from e
    if not (-90.0 <= val <= 90.0):
        raise argparse.ArgumentTypeError(f"Latitude out of range [-90, 90]: {val}")
    return val


def lon_type(s: str) -> float:
    try:
        val = float(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Longitude must be a float: {s}") from e
    if 180.0 < val < 360.0:
        val -= 360
    elif not (-180.0 <= val <= 180.0):
        raise argparse.ArgumentTypeError(f"Longitude out of range [-180, 180]: {val}")
    return val


def num_days_type(s: str) -> float:
    try:
        val = float(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Number of days (days) must be a float or integer: {s}"
        ) from e
    if val > 31:
        raise argparse.ArgumentTypeError(
            f"Number of days (days) is limited to 1 month (31 days): {val}"
        )
    return val


def date_type_allow_latest(s: str) -> Optional[dt.datetime]:
    """Accept 'latest' or YYYYMMDD."""
    if s.lower() == "latest":
        return None  # None means "latest"
    try:
        return dt.datetime.strptime(s, "%Y%m%d")
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Date must be 'latest' or YYYYMMDD (e.g., 20250131); got: {s}"
        ) from e


def path_type(s: str) -> Path:
    try:
        val = Path(s)
    except Exception:
        raise argparse.ArgumentTypeError(f"Input {s} must be a path-like string")
    return val


def directory_type(s: str) -> Path:
    val = path_type(s)
    val.mkdir(parents=True, exist_ok=True, mode=0o775)
    return val


def plot_file_type(s: str) -> Path:
    val = path_type(s)
    if val.suffix.lower() not in ["png", "jpg", "jpeg"]:
        raise argparse.ArgumentTypeError(
            f"Output file {s} must have a figure extension"
        )
    val.parent.mkdir(parents=True, exist_ok=True, mode=0o775)
    return val


prod_dict = {
    "PM25": ["pm2.5", "pm_25"],
    "NO2": ["no2"],
    "O3": ["o3", "ozone"],
    "SO2": ["so2"],
    "CO": ["co"],
}
DATASETS = make_aliases_from_dict(
    {
        "aqc_slv": ["aqc_v1", "aqc_x1"],
        "chm_slv": ["chm_v1", "chm_x1"],
        "met_slv": ["met_v1", "met_x1"],
        "chm_p23": [],
        "chm_v72": [],
    }
)

COLLECTIONS = make_aliases_from_dict(
    {"fcast": ["fcst", "forecast"], "assim": ["ana", "assimilation", "replay", "rply"]}
)

VERSIONS = make_aliases_from_dict({"v1": ["1"], "v2": ["2"]})
PRODUCTS = make_aliases_from_dict(prod_dict)

FILE_PRODUCTS = make_aliases_from_dict({"MET": ["met"], **prod_dict})
PLOTTYPES = make_aliases_from_dict(
    {
        "pres": ["pressure", "contour", "datagram", "pres", "pres_plot"],
        "replay": [
            "assim",
            "historical",
            "hist",
            "rply",
            "replay",
            "assimilation",
            "ana",
            "analysis",
        ],
        "sfc": ["surface", "sfc", "surf", "surf_plot"],
    },
    info={
        "pres": "Forecast datagram plot, with cloud panel, pressure contour panel, and meteorology panel",
        "replay": "Assimilation data, with cloud panel, surface value panel, and meterology panel",
        "sfc": "Forecast data, with cloud panel, surface values panel, and meteorology panel",
    },
)

FORMATS = make_aliases_from_dict(
    {
        "xlsx": ["excel"],
        "txt": ["ascii", "csv"],
        "json": [],
    }
)


def parse_products_value(value: Any) -> List[str]:
    return parse_list_value(value, FILE_PRODUCTS)


class SplitProducts(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        canon_list = parse_products_value(values)
        setattr(namespace, self.dest, canon_list)


class Coercer:
    def __init__(self, converter, default):
        self.converter = converter
        self.default = default


CONVERTERS = {
    "plot": {
        "version": Coercer(VERSIONS, "v2"),
        "product": Coercer(PRODUCTS, "no2"),
        "plot_type": Coercer(PLOTTYPES, "surf"),
        "lat": Coercer(lat_type, 38.9),
        "lon": Coercer(lon_type, -77.0),
        "start": Coercer(date_type_allow_latest, None),
        "end": Coercer(date_type_allow_latest, None),
        "num_days": Coercer(num_days_type, None),
        "output_dir": Coercer(path_type, None),
    },
    "file": {
        "version": Coercer(VERSIONS, "v2"),
        "product": Coercer(parse_products_value, ["no2"]),
        "format": Coercer(FORMATS, "json"),
        "dataset": Coercer(DATASETS, "chm_slv"),
        "collection": Coercer(COLLECTIONS, "fcast"),
        "lat": Coercer(lat_type, 38.9),
        "lon": Coercer(lon_type, -77.0),
        "start": Coercer(date_type_allow_latest, None),
        "end": Coercer(date_type_allow_latest, None),
        "num_days": Coercer(num_days_type, None),
        "output_dir": Coercer(path_type, None),
    },
}


def coerce_inputs(
    raw: Mapping[str, Any], type: Optional[Literal["plot", "file"]] = None
) -> argparse.Namespace:
    ns = argparse.Namespace()
    if not type:
        if "format" in raw:
            type = "file"
        else:
            type = "plot"
    CONV = CONVERTERS.get(type)
    for key, conv in CONV.items():
        if key in raw and raw[key] is not None:
            setattr(ns, key, conv.converter(raw[key]))
        else:
            setattr(ns, key, conv.default)

    return ns


def get_date_range(version: Optional[str] = None):
    version = version or "v2"
    cfg = v.coerce_config({"version": version, "collection": "assim"})
    helper = fileHelper(cfg)
    min_date, max_date = helper.get_date_range()
    return min_date, max_date


def load_fluid_cfg():
    from cf_map.config import loader

    cfg = loader.load_yaml("cfmap_fluid.yml")
    return cfg
