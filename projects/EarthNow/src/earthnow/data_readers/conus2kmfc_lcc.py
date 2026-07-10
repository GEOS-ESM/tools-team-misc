from types import SimpleNamespace
from .dataservice import DataService, VARIABLE_MAPPING
from .registry import register

CONUS2KMFC_LCC_URI = "/discover/nobackup/projects/gmao/osse2/HWT/CONUS02KM/Feature-c2160_L137/forecasts/CYCLED_REPLAY_P10800_C21600_T21600_%%Y%%m%%d_%%Hz/GEOS.$collection.%Y%m%d_%H%Mz.nc4"

CONUS2KMFC_LCC_COORDS = (
    "/discover/nobackup/projects/gmao/osse2/stage/BCS_FILES/lambert_grid.nc4"
)

# keys in this dictionary must match keys in VARIABLE_MAPPING
# but keep the collection linked to the stream by defining them here.
# Defining them in VARIABLE_MAPPING as well enforces that you use a
# collection name that exists.
CONUS2KMFC_LCC_COLLECTIONS = dict(
    VORT500="hwt_15mn_slv_LCC",  # vorticity_at_500_hPa (s-1)
    H500="hwt_15mn_slv_LCC",  # height_at_500_hPa (m)
    T2M="hwt_15mn_slv_LCC",  # 2-meter_air_temperature (K)
    HGT_SFC="hwt_15mn_slv_LCC",  # surface geopotential height (m+2 s-2)
    REFL_MAX="hwt_15mn_slv_LCC",  # Maximum_composite_radar_reflectivity (dBZ)
    SNOW="hwt_15mn_slv_LCC",  # snowfall (kg m-2 s-1)
    RAIN="hwt_15mn_slv_LCC",  # rainfall (kg m-2 s-1)
    ICE="hwt_15mn_slv_LCC",  # icefall (km m-2 s-1)
    EDGE_HGT="hwt_15mn_prs_LCC",  # layer edge_heights (m) (NOT geopotential height)
)

# Create a dictionary including ONLY the variable names
# relevant to this stream
CONUS2KMFC_LCC_VARS = {}
for var, coll in CONUS2KMFC_LCC_COLLECTIONS:
    CONUS2KMFC_LCC_VARS[var] = VARIABLE_MAPPING[var][collection] + "." + collection


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
