import numpy as np
import xarray as xr


class ObsReader(object):

    def __init__(self, filename, **kwargs):

        self.quiet = kwargs.get("quiet", False)
        self.debug = kwargs.get("debug", False)

        self.fh = xr.open_dataset(filename)

        if not self.quiet:
            print(f"Reading: {filename}")

    def grid_map(self, loninc, latinc, lons, lats):

        nobs = lons.size

        lon_start, lon_end, lon_step = -180.0, 180.0, loninc
        lat_start, lat_end, lat_step = -90.0, 90.0, latinc

        grid_lon = np.arange(lon_start, lon_end + lon_step, lon_step)
        grid_lat = np.arange(lat_start, lat_end + lat_step, lat_step)

        idim, jdim = grid_lon.size, grid_lat.size

        grid_size = idim * jdim

        i_indices = np.round((lons - lon_start) / lon_step).astype(int)
        j_indices = np.round((lats - lat_start) / lat_step).astype(int)
        k_indices = j_indices * idim + i_indices

        mapper = np.ma.masked_all(grid_size, dtype=int)
        mapper[k_indices] = np.arange(nobs, dtype=int)

        return mapper.compressed()

    def write_cvs(self):

        sort_indices = self._time.argsort()

        times = self._time.round('6h')
        delta = (self._time - times) / np.timedelta64(1, 'h')

        for i in sort_indices:
             rec = f'{self.obs_category},{self.legend_category},{self.isis},{times[i]},{delta[i]:.1f},{self._lons[i]:.1f},{self._lats[i]:.1f}'
             print(rec)
