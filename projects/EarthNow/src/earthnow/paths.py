"""
Central path configuration for EarthNow / WxMaps.

All hardcoded filesystem paths live here. Override the base roots via
environment variables to adapt to a different HPC cluster or local install:

    EARTHNOW_G6DEV_PUB   — shared g6dev published data root
    EARTHNOW_OSSE2_ROOT  — OSSE2 project data root
    EARTHNOW_GMAO_OPS    — GMAO ops published data root
    EARTHNOW_CITIES_DIR  — city data files directory
"""

import os
from pathlib import Path
from dotenv import load_dotenv

CONFIG_DIR = Path(__file__).resolve().parent
load_dotenv(CONFIG_DIR / ".env")


def env_check(name: str, default: str | Path) -> str:
    return str(os.environ.get(name, default))

def env_path(name: str, default: str | Path) -> Path:
    return Path(env_check(name, default))

# ---------------------------------------------------------------------------
# DATA URIS
# ---------------------------------------------------------------------------

MERRA2_URI = env_path(
    "EARTHNOW_MERRA2_URI",
    "/discover/nobackup/projects/gmao/merra2/data/pub/products/MERRA2_all/Y%Y/M%m/MERRA2.$collection.%Y%m%d.nc4",
)

CONUS2KMRP_URI = env_path(
    "EARTHNOW_CONUS2KMRP_URI",
    "/discover/nobackup/projects/gmao/osse2/HWT/CONUS02KM/Feature-c2160_L137/holding/$collection/%Y%m/Feature-c2160_L137.$collection.%Y%m%d_%H%Mz.nc4",
)

CONUS2KMFC_URI = env_path(
    "EARTHNOW_CONUS2KMFC_URI",
    "/discover/nobackup/projects/gmao/osse2/HWT/CONUS02KM/Feature-c2160_L137/forecasts/CYCLED_REPLAY_P10800_C21600_T21600_%%Y%%m%%d_%%Hz/GEOS.$collection.%Y%m%d_%H%Mz.nc4",
)

GEOS_FP_URI = env_path(
    "EARTHNOW_GEOS_FP_URI", "/discover/nobackup/projects/gmao/gmao_ops/pub"
)


# ---------------------------------------------------------------------------
# Base roots
# ---------------------------------------------------------------------------

_G6DEV_PUB = env_path(
    "EARTHNOW_G6DEV_PUB", "/discover/nobackup/projects/gmao/g6dev/pub"
)
_OSSE2_ROOT = env_path("EARTHNOW_OSSE2_ROOT", "/discover/nobackup/projects/gmao/osse2")
_GMAO_OPS = env_path(
    "EARTHNOW_GMAO_OPS", "/discover/nobackup/projects/gmao/gmao_ops/pub"
)
_CITIES_DIR = env_path("EARTHNOW_CITIES_DIR", "/home/wputman/IDL_BASE/CITIES")

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------
COLORTABLE_DIR = env_path("EARTHNOW_COLORTABLE_DIR", _G6DEV_PUB / "ColorTables")
COLORBAR_OUT_DIR = env_path(
    "EARTHNOW_COLORBAR_OUT_DIR", _G6DEV_PUB / "WxMaps" / "ColorBars"
)
BASE_IMAGE_DIR = env_path("EARTHNOW_BASE_IMAGE_DIR", _G6DEV_PUB / "BMNG")
GSHHS_DIR = env_path("EARTHNOW_GSHHS_DIR", _OSSE2_ROOT / "GSHHG" / "v2.3.7")
NWS_SHAPEFILE_DIR = env_path(
    "EARTHNOW_NWS_SHAPEFILE_DIR", _OSSE2_ROOT / "TSE_staging" / "SHAPE_FILES" / "ALL"
)
LCC_GRID_FILE = env_path(
    "EARTHNOW_LCC_GRID_FILE", _OSSE2_ROOT / "stage" / "BCS_FILES" / "lambert_grid.nc4"
)

# Experiment / data reader defaults
DEFAULT_HWT_EXP_PATH = env_path("EARTHNOW_DEFAULT_HWT_EXP_PATH", _OSSE2_ROOT / "HWT")
DEFAULT_GENCAST_EXP_PATH = env_path(
    "EARTHNOW_DEFAULT_GENCAST_EXP_PATH", _OSSE2_ROOT / "GenCast_FP"
)
DEFAULT_GEOS_FP_BASE = env_path("EARTHNOW_DEFAULT_GEOS_FP_BASE", _GMAO_OPS)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def colortable(filename: str) -> Path:
    return COLORTABLE_DIR / filename


def colorbar_output(filename: str) -> Path:
    return COLORBAR_OUT_DIR / filename


def city_file(filename: str) -> Path:
    return _CITIES_DIR / filename
