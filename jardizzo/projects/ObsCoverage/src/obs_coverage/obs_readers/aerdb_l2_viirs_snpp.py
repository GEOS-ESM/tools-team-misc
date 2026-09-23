import pandas as pd
import numpy as np
from datetime import datetime

from .obs_reader import ObsReader
from .registry import register


@register("viirs_snpp")
class AERDB_L2_VIIRS_SNPP(ObsReader):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.obs_category = 'Polar Orbiting'
        self.legend_category = 'VIIRS SNPP'
        self.isis = 'none'

    def get_obs(self):

        base_epoch = datetime(1993, 1, 1, 0, 0, 0)

        self.lats = self.fh["Latitude"].data.flatten()
        self.lons = self.fh["Longitude"].data.flatten()
        self.time = self.fh["Scan_Start_Time"].data.flatten()
        self.time = pd.to_timedelta(self.time, unit='s') + base_epoch

    def thin_obs(self, loninc=1.0, latinc=1.0):

        nobs = self.lons.size

        mapper = self.grid_map(loninc, latinc, self.lons, self.lats)
        self._lons = self.lons[mapper]
        self._lats = self.lats[mapper]
        self._time = self.time[mapper]

        n = self._lons.size

        print(f'{nobs} observations thinned down to {n}')
