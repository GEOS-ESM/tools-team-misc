import xarray as xr

# Open the curvilinear grid file
curvilinear_ds = xr.open_dataset('uiowa_asiaaq_fcst_2024020800_000.nc')

# Open the grid constants file
grid_constants_ds = xr.open_dataset('uiowa_asiaaq_fcst_constants.nc')

# Extract latitude and longitude arrays from the grid constants
lat = grid_constants_ds['lat'].values
lon = grid_constants_ds['lon'].values

# Assign latitude and longitude coordinates to the curvilinear dataset
curvilinear_ds['lat'] = (('lat', 'lon'), lat)
curvilinear_ds['lon'] = (('lat', 'lon'), lon)

# Save the updated dataset
curvilinear_ds.to_netcdf('curvilinear_with_coords.nc')

