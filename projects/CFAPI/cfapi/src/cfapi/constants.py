# core.py
import dataclasses as dc
import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import logging

# ----------------------------
# Config loading / management
# ----------------------------

ROOT = Path(__file__).parents[2]


def get_cache_dir(root: Path | str = ROOT) -> Path:
    root = Path(root)
    if any([d in str(root.parent) for d in ["fluiddev", "fluidprod"]]):
        return root.parents[1] / "static" / "plots" / "cf_map_grams"
    else:
        return ROOT


CACHE_ROOT = get_cache_dir()
CACHE = CACHE_ROOT / "file_cache"

logger = logging.getLogger("cfapi")


# --------------------
# Error Classes
# --------------------
class BadRequest(ValueError):
    """Raised when inputs fail validation (replaces flask.abort(404))."""


class ArgumentTypeError(ValueError):
    """Raised when inputs fail validation (replaces flask.abort(404))."""


# --------------------
# Containers
# --------------------


@dc.dataclass(frozen=True)
class CliConfig:
    version: str  # 'v1' | 'v2'
    collection: str  # 'fcast' | 'assim'
    dataset: str  # 'chm' | 'aqc' | 'met'
    level: str  # normalized: 'slv' | 'p23' | 'v72'
    product: str  # e.g. 'NO2','O3','PM25','HCHO','CO','SO2','MET'
    lat: float
    lon: float
    start: Optional[str]  # None => latest
    end: Optional[str]  # None => latest / open-ended (see semantics)
    products: Optional[List[str]]
    datakey: Optional[str]
    days: Optional[int] = 1


@dc.dataclass(frozen=True)
class dataIndex:
    lon: float
    ilon: int
    lat: float
    ilat: int
    t0: dt.datetime
    nlevs: int
    longnames: Dict[str, str]


@dc.dataclass(frozen=True)
class configSettings:
    data_limit_override: bool = False
    config_data: Dict[str, Any] = None
    config_params: Dict[str, Any] = None
    cache_root: Optional[Union[str, Path]] = None
    use_cache: bool = True
    netcdf_engine: str = "netcdf4"
    mp_pool: Optional[Any] = None
