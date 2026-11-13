import copy
import h3
import pandas as pd
import numpy as np
import xarray as xr
import pandas as pd
from scipy.spatial import cKDTree

from obs_coverage.methods.registry import OBS_METHODS


class Observation(object):

    def __init__(self, **kwargs):

        self.quiet = kwargs.get("quiet", False)
        self.debug = kwargs.get("debug", False)
        self.set_grid_method("fastest", (2.0, 2.0))

        self.bins = {}
        self.lons = np.array([])
        self.lats = np.array([])
        self.times = np.array([], dtype="datetime64[s]")

    def open(self, filename):

        fh = xr.open_dataset(filename)

        if not self.quiet:
            print(f"Reading: {filename}")

        return fh

    def set_grid_method(self, method, resolution):

        self.grid_map = OBS_METHODS[method]
        self.resolution = resolution

    def thin(self):

        nobs = len(self)

        mapper = self.grid_map(self.lons, self.lats, self.resolution)
        self.lons = self.lons[mapper]
        self.lats = self.lats[mapper]
        self.times = self.times[mapper]

        n = len(self)

        if not self.quiet:
            print(f"{nobs} observations thinned down to {n}")

    def bin(self, inchrs):

        hours_in_secs = inchrs * 3600
        sort_indices = self.times.argsort()
        self.lats = self.lats[sort_indices]
        self.lons = self.lons[sort_indices]
        self.times = self.times[sort_indices]

        tbins = (
            np.round(self.times.astype(int) / hours_in_secs) * hours_in_secs
        ).astype("datetime64[s]")

        sorted_idx = np.argsort(tbins, kind="stable")
        vals, first_positions = np.unique(tbins, return_index=True)
        grouped_indices = np.split(sorted_idx, first_positions[1:])
        self.bins = dict(zip(vals, grouped_indices))

        return sorted(self.bins)

    def get(self, time_dt):

        obs = copy.copy(self)
        tbin = np.datetime64(time_dt)
        indices = self.bins.get(tbin, [])
        obs.lats = self.lats[indices]
        obs.lons = self.lons[indices]
        obs.times = self.times[indices]

        return obs

    def purge(self, flush=False):

        if flush:
            self.bins = {}
            self.lats = type(self.lats)()
            self.lons = type(self.lons)()
            self.times = type(self.times)()

            return

        if not self.bins:
            return

        tbin = sorted(self.bins)[-2]  # Preserve the last bin

        i = self.bins[tbin][-1]
        self.lats = self.lats[i + 1 :]
        self.lons = self.lons[i + 1 :]
        self.times = self.times[i + 1 :]

        self.bins = {}

    def __iadd__(self, other):

        self.lats = np.concatenate((self.lats, other.lats))
        self.lons = np.concatenate((self.lons, other.lons))
        self.times = np.concatenate((self.times, other.times))

        return self

    def __add__(self, other):

        result = copy.deepcopy(self)
        result += other
        return result

    def __len__(self):

        return self.times.size
