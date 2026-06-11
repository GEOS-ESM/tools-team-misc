from types import SimpleNamespace
from .dataservice import DataService
from .registry import register

from earthnow import paths

CONUS2KMFC_VARS = dict(
    VORT500="VORT500.inst1_2d_asm_Nx",
    H500="H500.inst1_2d_asm_Nx",
    T2M="T2M.inst1_2d_asm_Nx",
)

CONUS2KMFC = SimpleNamespace(
    uri=paths.CONUS2KMFC_URI,
    description="GEOS_based_on_Feature-c2160_L137",
    type="forecast",
    title="CONUS 2KM Forecast",
    grid="latlon",
    vars=CONUS2KMFC_VARS,
)


@register("CONUS2KMFC")
class conus2kmfc(DataService):
    def __init__(self, **kwargs):
        super().__init__(CONUS2KMFC)
