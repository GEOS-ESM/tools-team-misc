class obs_writer(object):

    def __init__(self, obs):

        self.obs = obs
        self.bins = {}

    def bin_obs(self, spacing)

        obs = self.obs
        self.bins = {}
        sort_indices = obs.time.argsort()

        for i in sort_indices:
             dattim = obs.time[i].strftime('%Y%m%d%H')
             tbin = obs.time[i].round(spacing)
             delta = (self.time[i] - tbin) / np.timedelta64(1, spacing)
             bin = self.bins.get(tbin, [])
             bin.append((obs.lons[i],obs.lats[i],obs.time[i],delta))
             self.bins[tbin] = bin

    def write_json(self, fname):

        for tbin, obs in sorted(observations.items()):
            fname = tbin.strftime(fname) + '.json'
            with open(fname, 'w') as f:
                json.dump(obs, f)

    def write_csv(self):

        sort_indices = self._time.argsort()

        for i in sort_indices:
             obtime = f'{delta[i]:.1f}'
             obtime = float(obtime)
             dattim = times[i].strftime('%Y%m%d%H')
             rec = f'{self.obs_category},{self.legend_category},{self.isis},{dattim},{obtime:.1f},{self._lons[i]:.1f},{self._lats[i]:.1f}'
             print(rec)

