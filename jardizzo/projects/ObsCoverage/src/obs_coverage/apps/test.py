import sys
import json
from obs_coverage.obs_readers.registry import OBS_READERS

observations = {}
reader = OBS_READERS["viirs_snpp"]

for filename in sys.argv[1:]:
    fh = reader(filename)
    fh.get_obs()
    fh.thin_obs(4.0, 4.0)

    sort_indices = fh._time.argsort()
    for i in sort_indices:
         dattim = fh._time[i].strftime('%Y%m%d%H')
         bin = observations.get(dattim, [])
         bin.append(f'{fh._lons[i]:.2f} {fh._lats[i]:.2f}')
         observations[dattim] = bin

for dattim, obs in sorted(observations.items()):
    print(dattim)
    with open(dattim+'.json', 'w') as f:
                json.dump(obs, f)

    def write_csv(self):

        sort_indices = self._time.argsort()

        times = self._time.round('6h')
        delta = (self._time - times) / np.timedelta64(1, 'h')

        for i in sort_indices:
             obtime = f'{delta[i]:.1f}'
             obtime = float(obtime)
             dattim = times[i].strftime('%Y%m%d%H')
             rec = f'{self.obs_category},{self.legend_category},{self.isis},{dattim},{obtime:.1f},{self._lons[i]:.1f},{self._lats[i]:.1f}'
             print(rec)

