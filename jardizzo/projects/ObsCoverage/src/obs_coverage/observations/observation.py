import copy
import h3
import pandas as pd
import numpy as np
import xarray as xr
import pandas as pd
from scipy.spatial import cKDTree


class Observation(object):

    def __init__(self, **kwargs):

        self.quiet = kwargs.get("quiet", False)
        self.debug = kwargs.get("debug", False)
        self.grid_methods = {
            "fastest": self.grid_map_fastest,
            "closest": self.grid_map_closest,
            "h3": self.grid_map_h3,
        }

        self.set_grid('fastest', (2.0, 2.0))

        self.bins = {}
        self.lons = np.array([])
        self.lats = np.array([])
        self.times = np.array([], dtype='datetime64[s]')

    def open(self, filename):

        fh = xr.open_dataset(filename)

        if not self.quiet:
            print(f"Reading: {filename}")

        return fh

    def set_grid(self, method, resolution):

        self.grid_map = self.grid_methods[method]
        self.resolution = resolution

    def thin(self):

        nobs = len(self)

        mapper = self.grid_map(self.lons, self.lats, self.resolution)
        self.lons = self.lons[mapper]
        self.lats = self.lats[mapper]
        self.times = self.times[mapper]

        n = len(self)

        if not self.quiet:
            print(f'{nobs} observations thinned down to {n}')

    def bin(self, inchrs):

        hours_in_secs = inchrs * 3600
        sort_indices = self.times.argsort()
        self.lats = self.lats[sort_indices]
        self.lons = self.lons[sort_indices]
        self.times = self.times[sort_indices]

        tbins = (np.round(self.times.astype(int) / hours_in_secs) * hours_in_secs).astype('datetime64[s]')

        sorted_idx = np.argsort(tbins, kind='stable')
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

        tbin = sorted(self.bins)[-2] # Preserve the last bin

        i = self.bins[tbin][-1]
        self.lats = self.lats[i+1:]
        self.lons = self.lons[i+1:]
        self.times = self.times[i+1:]

        self.bins = {}
      
    def grid_map_fastest(self, lons, lats, resolution):

        nobs = lons.size
        loninc, latinc = resolution

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

    def grid_map_closest(self, lons, lats, resolution):

        loninc, latinc = resolution

        obs_df = pd.DataFrame({"lat": lats, "lon": lons})

        # Define your target grid resolution and boundaries
        lon_start, lon_end, lon_step = -180.0, 180.0, loninc
        lat_start, lat_end, lat_step = -90.0, 90.0, latinc

        lon_grid = np.arange(lon_start, lon_end + lon_step, lon_step)
        lat_grid = np.arange(lat_start, lat_end + lat_step, lat_step)

        # Generate the grid cell centers
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        grid_centers = np.vstack([lat_mesh.ravel(), lon_mesh.ravel()]).T

        # Build a KDTree of the observation locations
        obs_coords = obs_df[["lat", "lon"]].values
        tree = cKDTree(grid_centers)

        # For every observation, find the nearest grid index
        distances, grid_indices = tree.query(obs_coords, k=1)

        # Sort observations by grid index, then by distance (ascending)
        # lexsort sorts by the last sequence first: grid_indices, then breaks
        # ties using distances
        sort_order = np.lexsort((distances, grid_indices))

        sorted_grid_indices = grid_indices[sort_order]
        sorted_obs_indices = sort_order

        #  Drop duplicates, keeping only the entry with the minimum distance
        _, unique_positions = np.unique(sorted_grid_indices, return_index=True)
        best_obs_indices = sorted_obs_indices[unique_positions]

        return best_obs_indices

    def grid_map_h3(self, lons, lats, resolution):

        df = pd.DataFrame({'latitude': lats, 'longitude': lons})

        # Map lat/lon to H3 cell IDs
        df["h3_cell"] = df.apply(
        lambda row: h3.geo_to_h3(row["latitude"], row["longitude"], resolution),
            axis=1,
        )

        # Thin the data by taking the first observation in each hexagon
        indices_array = df.drop_duplicates(subset=["h3_cell"]).index.to_numpy()

        return indices_array

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
