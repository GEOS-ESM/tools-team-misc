from types import SimpleNamespace
from .dataservice import DataService
from .registry import register

from earthnow import paths

CONUS2KMFC_LCC_VARS = dict(
    VORT500="VORT500.hwt_15mn_slv_LCC",
    H500="H500.hwt_15mn_slv_LCC",
    TMP_2M="TMP_2M.hwt_15mn_slv_LCC",
)

CONUS2KMFC_LCC = SimpleNamespace(
    uri=paths.CONUS2KMFC_URI,
    description="GEOS_based_on_Feature-c2160_L137",
    type="forecast",
    title="CONUS 2KM Forecast",
    grid="lcc",
    vars=CONUS2KMFC_LCC_VARS,
    coords=(paths.LCC_GRID_FILE),
)


@register("CONUS2KMFC_LCC")
class conus2kmfc_lcc(DataService):
    def __init__(self, **kwargs):
        super().__init__(CONUS2KMFC_LCC)
