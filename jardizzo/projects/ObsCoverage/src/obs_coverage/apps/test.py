import sys
from obs_coverage.obs_readers.registry import OBS_READERS

reader = OBS_READERS['viirs_snpp']
fh = reader(sys.argv[1])

fh.get_obs()
