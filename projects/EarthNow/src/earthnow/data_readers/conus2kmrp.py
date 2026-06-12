from types import SimpleNamespace
from .dataservice import DataService
from .registry import register

# from earthnow import paths

CONUS2KMRP_URI = "/discover/nobackup/projects/gmao/osse2/HWT/CONUS02KM/Feature-c2160_L137/holding/$collection/%Y%m/Feature-c2160_L137.$collection.%Y%m%d_%H%Mz.nc4"


CONUS2KMRP_VARS = dict(
    VORT500="VORT500.inst1_2d_asm_Nx",
    H500="H500.inst1_2d_asm_Nx",
    T2M="T2M.inst1_2d_asm_Nx",
)

CONUS2KMRP = SimpleNamespace(
    uri=CONUS2KMRP_URI,
    description="CONUS02km_137L_replay_to_GEOS-FP",
    type="analysis",
    title="CONUS 2KM Replay",
    grid="latlon",
    vars=CONUS2KMRP_VARS,
)


@register("CONUS2KMRP")
class conus2kmrp(DataService):
    def __init__(self, **kwargs):
        super().__init__(CONUS2KMRP)
