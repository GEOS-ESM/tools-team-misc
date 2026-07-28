from types import SimpleNamespace
from .dataservice import DataService
from .registry import register
from .variables import VARIABLE_REGISTRY

CONUS2KMFC_URI = "/discover/nobackup/projects/gmao/osse2/HWT/CONUS02KM/Feature-c2160_L137/forecasts/CYCLED_REPLAY_P10800_C21600_T21600_%%Y%%m%%d_%%Hz/GEOS.$collection.%Y%m%d_%H%Mz.nc4"

CONUS2KMFC_VARS = VARIABLE_REGISTRY.resolve_many(
    {
        "VORT500": "inst1_2d_asm_Nx",
        "H500": "inst1_2d_asm_Nx",
        "T2M": "inst1_2d_asm_Nx",
        "DBZ_MAX": "inst1_2d_asm_Nx",
        "UH25": "inst1_2d_asm_Nx",
        "CAPE": "inst1_2d_asm_Nx",
        "NIEXTTAU": "inst1_2d_asm_Nx",
        "SUEXTTAU": "inst1_2d_asm_Nx",
        "DUEXTTAU": "inst1_2d_asm_Nx",
        "SSEXTTAU": "inst1_2d_asm_Nx",
    }
)

CONUS2KMFC = SimpleNamespace(
    uri=CONUS2KMFC_URI,
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
