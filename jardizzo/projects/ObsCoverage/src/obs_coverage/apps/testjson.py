import sys
from obs_coverage.observations.registry import OBS_TYPES
from obs_coverage.observations.obs_writer import ObsWriter

obtype = OBS_TYPES["viirs_snpp"]

obs = obtype()
obstot = obtype()
for filename in sys.argv[1:]:
    obs.read(filename)
    obstot += obs

print(obstot.lats.size)
obstot.thin(2.0, 2.0)

writer = ObsWriter(obstot, 6)
writer.write_fluid_json('%Y%m%d%H')
