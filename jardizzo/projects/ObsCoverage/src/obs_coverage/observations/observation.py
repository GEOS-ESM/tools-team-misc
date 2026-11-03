import copy
import pandas as pd
import numpy as np
import xarray as xr
import pandas as pd
from scipy.spatial import cKDTree


class Observation(object):

    def __init__(self, filename=None, mapping="closest", **kwargs):

        self.quiet = kwargs.get("quiet", False)
        self.debug = kwargs.get("debug", False)
        self.mapping = {
            "fastest": self.grid_map_fastest,
            "closest": self.grid_map_closest,
        }
        self.grid_map = self.mapping.get(mapping)

        self.lons = np.array([])
        self.lats = np.array([])
        self.times = np.array([], dtype='datetime64[s]')

    def open(self, filename):

        fh = xr.open_dataset(filename)

        if not self.quiet:
            print(f"Reading: {filename}")

        return fh

    def set_mapping(self, mapping):

        self.grid_map = self.mapping.get(mapping, self.grid_map_fastest)

    def thin(self, loninc=1.0, latinc=1.0):

        nobs = self.lons.size

        mapper = self.grid_map(loninc, latinc, self.lons, self.lats)
        self.lons = self.lons[mapper]
        self.lats = self.lats[mapper]
        self.times = self.times[mapper]

        n = self.lons.size

        if not self.quiet:
            print(f'{nobs} observations thinned down to {n}')

    def grid_map_fastest(self, loninc, latinc, lons, lats):

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

    def grid_map_closest(self, loninc, latinc, lons, lats):

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

    def __iadd__(self, other):

        self.lats = np.concatenate((self.lats, other.lats))
        self.lons = np.concatenate((self.lons, other.lons))
        self.times = np.concatenate((self.times, other.times))

        print(self.lats.size, other.lats.size)

        return self

    def __add__(self, other):

        result = copy.deepcopy(self)
        result += other
        return result
