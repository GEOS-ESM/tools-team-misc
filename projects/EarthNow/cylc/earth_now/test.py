from registry import PRODUCTS
from products import *

print(PRODUCTS)

p = max_reflectivity(data_reader="geos-fp", fdate="20250705_00z", pdate="20250708_00z")
p.exe()

p = sea_level_pressure(
    data_reader="geos-fp", fdate="20250705_00z", pdate="20250708_00z"
)
p.exe()

p = vorticity_heights_500mb(
    data_reader="geos-fp", fdate="20250705_00z", pdate="20250708_00z"
)
p.exe()
