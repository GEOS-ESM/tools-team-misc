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

# ---------------------------------------------------------------------------
# Base roots
# ---------------------------------------------------------------------------

_G6DEV_PUB  = Path(os.environ.get("EARTHNOW_G6DEV_PUB",
                                   "/discover/nobackup/projects/gmao/g6dev/pub"))
_OSSE2_ROOT = Path(os.environ.get("EARTHNOW_OSSE2_ROOT",
                                   "/discover/nobackup/projects/gmao/osse2"))
_GMAO_OPS   = Path(os.environ.get("EARTHNOW_GMAO_OPS",
                                   "/discover/nobackup/projects/gmao/gmao_ops/pub"))
_CITIES_DIR = Path(os.environ.get("EARTHNOW_CITIES_DIR",
                                   "/home/wputman/IDL_BASE/CITIES"))

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------
COLORTABLE_DIR    = _G6DEV_PUB / "ColorTables"
COLORBAR_OUT_DIR  = _G6DEV_PUB / "WxMaps" / "ColorBars"
BASE_IMAGE_DIR    = _G6DEV_PUB / "BMNG"
GSHHS_DIR         = _OSSE2_ROOT / "GSHHG" / "v2.3.7"
NWS_SHAPEFILE_DIR = _OSSE2_ROOT / "TSE_staging" / "SHAPE_FILES" / "ALL"
LCC_GRID_FILE     = _OSSE2_ROOT / "stage" / "BCS_FILES" / "lambert_grid.nc4"

# Experiment / data reader defaults
DEFAULT_HWT_EXP_PATH     = _OSSE2_ROOT / "HWT"
DEFAULT_GENCAST_EXP_PATH = _OSSE2_ROOT / "GenCast_FP"
DEFAULT_GEOS_FP_BASE     = _GMAO_OPS

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def colortable(filename: str) -> Path:
    return COLORTABLE_DIR / filename


def colorbar_output(filename: str) -> Path:
    return COLORBAR_OUT_DIR / filename


def city_file(filename: str) -> Path:
    return _CITIES_DIR / filename
