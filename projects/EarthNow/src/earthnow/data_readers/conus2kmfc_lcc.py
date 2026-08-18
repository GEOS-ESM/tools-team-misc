from types import SimpleNamespace
from .dataservice import DataService
from .registry import register
from .variables import VARIABLE_REGISTRY

CONUS2KMFC_LCC_URI = "/discover/nobackup/projects/gmao/osse2/HWT/CONUS02KM/Feature-c2160_L137/forecasts/CYCLED_REPLAY_P10800_C21600_T21600_%%Y%%m%%d_%%Hz/GEOS.$collection.%Y%m%d_%H%Mz.nc4"

CONUS2KMFC_LCC_COORDS = (
    "/discover/nobackup/projects/gmao/osse2/stage/BCS_FILES/lambert_grid.nc4"
)

CONUS2KMFC_LCC_VARS = VARIABLE_REGISTRY.resolve_many(
    {
        "VORT500": "hwt_15mn_slv_LCC",
        "H500": "hwt_15mn_slv_LCC",
        "T2M": "hwt_15mn_slv_LCC",
        "UH25": "hwt_15mn_slv_LCC",
        "CAPE": "hwt_15mn_slv_LCC",
        "RAIN": "hwt_15mn_slv_LCC",
        "SNOW": "hwt_15mn_slv_LCC",
        "ICE": "hwt_15mn_slv_LCC",
        "DBZ_MAX": "hwt_15mn_slv_LCC",
        "U10M": "hwt_15mn_slv_LCC",
        "V10M": "hwt_15mn_slv_LCC",
        "U250": "hwt_15mn_slv_LCC",
        "V250": "hwt_15mn_slv_LCC",
        "SLP": "hwt_15mn_slv_LCC",
        "SNOWACCUM": "hwt_01hr_acc_LCC",
        "PRECACCUM": "hwt_01hr_acc_LCC",
        "NIEXTTAU": "hwt_15mn_slv_LCC",
        "SUEXTTAU": "hwt_15mn_slv_LCC",
        "DUEXTTAU": "hwt_15mn_slv_LCC",
        "SSEXTTAU": "hwt_15mn_slv_LCC",
        "OCEXTTAU": "hwt_15mn_slv_LCC",
        "BREXTTAU": "hwt_15mn_slv_LCC",
        "BCEXTTAU": "hwt_15mn_slv_LCC",
        "TBRB06RG": "hwt_15mn_slv_LCC",
        "SWTDN": "hwt_15mn_slv_LCC",
        "OSRB11RG": "hwt_15mn_slv_LCC",
        "OSRB10RG": "hwt_15mn_slv_LCC",
        "OSRB09RG": "hwt_15mn_slv_LCC",
        "HGT": "hwt_15mn_prs_LCC",
        "HGT_SFC": "hwt_15mn_slv_LCC",
    }
)

CONUS2KMFC_LCC = SimpleNamespace(
    uri=CONUS2KMFC_LCC_URI,
    description="GEOS_based_on_Feature-c2160_L137",
    type="forecast",
    title="CONUS 2KM Forecast",
    grid="lcc",
    vars=CONUS2KMFC_LCC_VARS,
    coords=CONUS2KMFC_LCC_COORDS,
)


@register("CONUS2KMFC_LCC")
class conus2kmfc_lcc(DataService):
    def __init__(self, **kwargs):
        super().__init__(CONUS2KMFC_LCC)
