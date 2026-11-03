import pandas as pd
import numpy as np
from datetime import datetime

from .observation import Observation
from .registry import register


@register("viirs_snpp")
class AERDB_L2_VIIRS_SNPP(Observation):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.obs_category = 'Polar Orbiting'
        self.legend_category = 'VIIRS SNPP'
        self.isis = 'none'

    def read(self, filename):

        fh = self.open(filename)

      # base_epoch = datetime(1993, 1, 1, 0, 0, 0)
        base_epoch = np.datetime64('1993-01-01T00:00:00')

        self.lats = fh["Latitude"].data.flatten()
        self.lons = fh["Longitude"].data.flatten()
        times = fh["Scan_Start_Time"].data.flatten()
        self.times = base_epoch + times.astype('timedelta64[s]')
      # self.times = pd.to_timedelta(times, unit='s') + base_epoch

        print(self.times)

        fh.close()
