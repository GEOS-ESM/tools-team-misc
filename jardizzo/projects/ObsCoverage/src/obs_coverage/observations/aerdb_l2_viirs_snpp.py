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

        self.set_grid("h3", 3)

    def read(self, filename):

        fh = self.open(filename)

      # base_epoch = datetime(1993, 1, 1, 0, 0, 0)
        base_epoch = np.datetime64('1993-01-01T00:00:00')

        lats = fh["Latitude"].data.flatten()
        lons = fh["Longitude"].data.flatten()
        times = fh["Scan_Start_Time"].data.flatten()
        aod_land = fh["Aerosol_Optical_Thickness_550_Land_Best_Estimate"].data.flatten()
        aod_ocean = fh["Aerosol_Optical_Thickness_550_Ocean_Best_Estimate"].data.flatten()

        valid_land = (aod_land != -999.0) & (~np.isnan(aod_land))
        valid_ocean = (aod_ocean != -999.0) & (~np.isnan(aod_ocean))

        global_valid_mask = (valid_land | valid_ocean)

        self.lats = lats[global_valid_mask]
        self.lons = lons[global_valid_mask]
        self.times = times[global_valid_mask]
        self.times = base_epoch + self.times.astype('timedelta64[s]')
      # self.times = pd.to_timedelta(times, unit='s') + base_epoch

        fh.close()
