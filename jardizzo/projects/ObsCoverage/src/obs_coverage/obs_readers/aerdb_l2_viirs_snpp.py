from .obs_reader import ObsReader
from .registry import register

@register("viirs_snpp")
class AERDB_L2_VIIRS_SNPP(ObsReader):

    def get_obs(self):

        self.lats = self.fh['Latitude'].data
        self.lons = self.fh['Longitude'].data
        self.time = self.fh['Scan_Start_Time'].data

        print(self.lats)
