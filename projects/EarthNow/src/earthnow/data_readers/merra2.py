from types import SimpleNamespace
from .dataservice import DataService
from .registry import register

# from earthnow import paths

MERRA2_URI = "/discover/nobackup/projects/gmao/merra2/data/pub/products/MERRA2_all/Y%Y/M%m/MERRA2.$collection.%Y%m%d.nc4"

MERRA2_VARS = dict(
    VORT500="none",
    H500="H500.tavg1_2d_slv_Nx",
    T2M="T2M.inst1_2d_asm_Nx",
    SLP="SLP.tavg1_2d_slv_Nx",
    U250="U250.tavg1_2d_slv_Nx",
    V250="V250.tavg1_2d_slv_Nx",
    U10M="U10M.tavg1_2d_slv_Nx",
    V10M="V10M.tavg1_2d_slv_Nx",
)

MERRA2 = SimpleNamespace(
    uri=MERRA2_URI,
    description="MERRA2 Analysis",
    type="analysis",
    title="MERRA2 Analysis",
    grid="latlon",
    vars=MERRA2_VARS,
)


@register("MERRA2")
class merra2(DataService):
    def __init__(self, **kwargs):
        super().__init__(MERRA2)
