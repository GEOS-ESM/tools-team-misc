import h3
import pandas as pd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .registry import register


@register("fastest")
def grid_map_fastest(lons, lats, resolution):

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


@register("closest")
def grid_map_closest(lons, lats, resolution):

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


@register("h3")
def grid_map_h3(lons, lats, resolution):

    df = pd.DataFrame({"latitude": lats, "longitude": lons})

    # Map lat/lon to H3 cell IDs
    df["h3_cell"] = df.apply(
        lambda row: h3.geo_to_h3(row["latitude"], row["longitude"], resolution),
        axis=1,
    )

    # Thin the data by taking the first observation in each hexagon
    indices_array = df.drop_duplicates(subset=["h3_cell"]).index.to_numpy()

    return indices_array
