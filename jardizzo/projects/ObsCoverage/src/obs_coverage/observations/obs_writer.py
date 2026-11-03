import csv
import json
import numpy as np


class ObsWriter(object):

    def __init__(self, obs, inchrs=6):

        self.obs = obs
        self.bin_obs(inchrs)

    def bin_obs(self, inchrs):

        obs = self.obs

        self.bins = {}
        sort_indices = obs.times.argsort()

        for i in sort_indices:
            hours_in_secs = inchrs * 3600
            seconds = obs.times[i].astype(int)
            rounded_seconds = np.round(seconds / hours_in_secs) * hours_in_secs
            tbin = rounded_seconds.astype('datetime64[s]')
            bin = self.bins.get(tbin, [])
            bin.append(i)
            self.bins[tbin] = bin

    def write_fluid_json(self, fname):

        obs = self.obs

        for tbin, indices in sorted(self.bins.items()):
            oname = tbin.astype(object).strftime(fname) + ".json"
            buf = [f"{obs.lons[i]:.2f} {obs.lats[i]:.2f}" for i in indices]
            with open(oname, "w") as f:
                json.dump(buf, f)

    def write_csv(self, fname):

        obs = self.obs
        header = "obs_category,legend_category,isis,idate,obstime,lon,lat"
        header = header.split(",")

        for tbin, indices in sorted(self.bins.items()):
            oname = tbin.astype(object).strftime(fname) + ".csv"
            dattim = tbin.astype(object).strftime("%Y%m%d%H")
            buf = [
                (
                    obs.obs_category,
                    obs.legend_category,
                    obs.isis,
                    dattim,
                    f"{(obs.times[i] - tbin) / np.timedelta64(1, "h"):.1f}",
                    f"{obs.lons[i]:.1f}",
                    f"{obs.lats[i]:.1f}",
                )
                for i in indices
            ]

            with open(oname, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(buf)
