import pandas as pd
import numpy as np
from datetime import datetime
from pyhdf.SD import SD, SDC

from .observation import Observation
from .registry import register


@register("AQUA_MYD04_L2")
class AQUA_MYD04_L2(Observation):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.obs_category = "Polar Orbiting"
        self.legend_category = "Aqua MODIS"
        self.isis = "none"

        self.set_grid_method("h3", 3)

    def read(self, filename):

        hdf = SD(filename, SDC.READ)

        base_epoch = np.datetime64("1993-01-01T00:00:00")

        lats = hdf.select("Latitude")[:].flatten()
        lons = hdf.select("Longitude")[:].flatten()
        times = hdf.select("Scan_Start_Time")[:].flatten()
        aod = hdf.select("AOD_550_Dark_Target_Deep_Blue_Combined")[:].flatten().astype(float)
        aod[aod == -9999] = np.nan
        valid_default = ~np.isnan(aod)

        self.lats = lats[valid_default]
        self.lons = lons[valid_default]
        self.times = times[valid_default]
        self.times = base_epoch + self.times.astype("timedelta64[s]")
