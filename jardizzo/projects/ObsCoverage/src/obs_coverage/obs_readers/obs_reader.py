import json
import numpy as np
import xarray as xr
import pandas as pd
from scipy.spatial import cKDTree


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

    def write_csv(self):

        sort_indices = self._time.argsort()

        times = self._time.round('6h')
        delta = (self._time - times) / np.timedelta64(1, 'h')

        for i in sort_indices:
             obtime = f'{delta[i]:.1f}'
             obtime = float(obtime)
             dattim = times[i].strftime('%Y%m%d%H')
             rec = f'{self.obs_category},{self.legend_category},{self.isis},{dattim},{obtime:.1f},{self._lons[i]:.1f},{self._lats[i]:.1f}'
             print(rec)

    def write_json(self):

        observations = {}
        sort_indices = self._time.argsort()
        
        for i in sort_indices:
             dattim = self._time[i].strftime('%Y%m%d%H')
             bin = observations.get(dattim, [])
             bin.append(f'{self._lons[i]:.2f} {self._lats[i]:.2f}')
             observations[dattim] = bin

        for time,obs in observations.items():
            with open(time+'.json', 'a') as f:
                json.dump(obs, f)

    def grid_map_new(self, loninc, latinc, lons, lats):

        obs_df = pd.DataFrame({
            'lat': lats,
            'lon': lons
        })

        # Define your target grid resolution and boundaries
        lon_start, lon_end, lon_step = -180.0, 180.0, loninc
        lat_start, lat_end, lat_step = -90.0, 90.0, latinc

        lon_grid = np.arange(lon_start, lon_end + lon_step, lon_step)
        lat_grid = np.arange(lat_start, lat_end + lat_step, lat_step)

        # Generate the grid cell centers
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        grid_centers = np.vstack([lat_mesh.ravel(), lon_mesh.ravel()]).T

        # Build a KDTree of the observation locations
        obs_coords = obs_df[['lat', 'lon']].values
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
