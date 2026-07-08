from types import SimpleNamespace
from .dataservice import DataService
from .registry import register

CONUS2KMRP_LCC_URI = "/discover/nobackup/projects/gmao/osse2/HWT/CONUS02KM/Feature-c2160_L137/holding/$collection/%Y%m/Feature-c2160_L137.$collection.%Y%m%d_%H%Mz.nc4"

CONUS2KMRP_LCC_COORDS = (
    "/discover/nobackup/projects/gmao/osse2/stage/BCS_FILES/lambert_grid.nc4"
)

CONUS2KMRP_LCC_VARS = dict(
    VORT500="VORT500.hwt_30mn_slv_LCC",
    H500="H500.hwt_30mn_slv_LCC",
    T2M="T2M.hwt_30mn_slv_LCC",
    REFC="REFC.hwt_30mn_slv_LCC",
    HGT_SFC="HGT_SFC.hwt_30mn_slv_LCC",
    SNOW="SNOW.hwt_30mn_slv_LCC",
    ICE="ICE.hwt_30mn_slv_LCC",
    RAIN="RAIN.hwt_30mn_slv_LCC",
)

CONUS2KMRP_LCC = SimpleNamespace(
    uri=CONUS2KMRP_LCC_URI,
    description="CONUS02km_137L_replay_to_GEOS-FP",
    type="analysis",
    title="CONUS 2KM Replay",
    grid="lcc",
    vars=CONUS2KMRP_LCC_VARS,
    coords=CONUS2KMRP_LCC_COORDS,
)


@register("CONUS2KMRP_LCC")
class conus2kmrp_lcc(DataService):
    def __init__(self, **kwargs):
        super().__init__(CONUS2KMRP_LCC)
