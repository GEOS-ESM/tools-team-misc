import sys
import datetime as dt
from obs_coverage.observations.registry import OBS_TYPES
from obs_coverage.observations.obs_writer import write_fluid_json

obtype = OBS_TYPES["viirs_snpp"]

obs = obtype()
obstot = obtype()
for filename in sys.argv[1:]:
    obs.read(filename)
    obstot += obs
    print(obstot.lons.size)

for tbin in obstot.bin(6):
    obs = obstot.get(tbin)
    obs.thin()
    write_fluid_json(tbin, obs, '%Y%m%d%H')
