from types import SimpleNamespace
from .dataservice import DataService
from .registry import register

from earthnow import paths

CONUS2KMRP_LCC_VARS = dict(
    VORT500="VORT500.hwt_30mn_slv_LCC",
    H500="H500.hwt_30mn_slv_LCC",
    T2M="T2M.hwt_30mn_slv_LCC",
)

CONUS2KMRP_LCC = SimpleNamespace(
    uri=paths.CONUS2KMRP_URI,
    description="CONUS02km_137L_replay_to_GEOS-FP",
    type="analysis",
    title="CONUS 2KM Replay",
    grid="lcc",
    vars=CONUS2KMRP_LCC_VARS,
    coords=(paths.LCC_GRID_FILE),
)


@register("CONUS2KMRP_LCC")
class conus2kmrp_lcc(DataService):
    def __init__(self, **kwargs):
        super().__init__(CONUS2KMRP_LCC)
