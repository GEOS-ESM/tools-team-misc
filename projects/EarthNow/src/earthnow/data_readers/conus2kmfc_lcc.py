from types import SimpleNamespace
from .dataservice import DataService
from .registry import register

CONUS2KMFC_LCC_URI = "/discover/nobackup/projects/gmao/osse2/HWT/CONUS02KM/Feature-c2160_L137/forecasts/CYCLED_REPLAY_P10800_C21600_T21600_%%Y%%m%%d_%%Hz/GEOS.$collection.%Y%m%d_%H%Mz.nc4"

CONUS2KMFC_LCC_COORDS = (
    "/discover/nobackup/projects/gmao/osse2/stage/BCS_FILES/lambert_grid.nc4"
)

CONUS2KMFC_LCC_VARS = dict(
    VORT500="VORT500.hwt_15mn_slv_LCC",
    H500="H500.hwt_15mn_slv_LCC",
    T2M="TMP_2M.hwt_15mn_slv_LCC",
    HGT_SFC="HGT_SFC.hwt_15mn_slv_LCC",  # surface geopotential height
    REFL_MAX="REFC.hwt_15mn_slv_LCC",  # maximum composite radar reflectivity
    SNOW="SNOW.hwt_15mn_slv_LCC",
    RAIN="RAIN.hwt_15mn_slv_LCC",
    ICE="ICE.hwt_15mn_slv_LCC",
    EDGE_HGT="HGT.hwt_15mn_prs_LCC",  # layer edge height (NOT geopotential height)
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
