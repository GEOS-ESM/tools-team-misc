import sys
from obs_coverage.obs_readers.registry import OBS_READERS

def write_json(lats, lons, times):

    observations = {}

    for i in sort_indices:
         dattim = self._time[i].strftime('%Y%m%d%H')
         bin = observations.get(dattim, [])
         bin.append(f'{self._lons[i]:.2f} {self._lats[i]:.2f}')
         observations[dattim] = bin

    for time,obs in observations.items():
        with open(time+'.json', 'a') as f:
            json.dump(obs, f)


observations = {}
reader = OBS_READERS["viirs_snpp"]

for filename in sys.argv[1:]:
    fh = reader(filename)
    fh.get_obs()
    fh.thin_obs(1.0, 1.0)

    sort_indices = self._time.argsort()
    for i in sort_indices:
         dattim = fh._time[i].strftime('%Y%m%d%H')
         bin = observations.get(dattim, [])
         bin.append(f'{fh._lons[i]:.2f} {fh._lats[i]:.2f}')
         observations[dattim] = bin




    write_json(fh._lats, fh._lons, fh._time)
