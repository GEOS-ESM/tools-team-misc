import xarray as xr
import matplotlib.pyplot as plt

f = 'https://opendap.nccs.nasa.gov/dods/merra2_gmi/inst1_2d_met_Nx'
ds = xr.open_dataset(f, decode_times=False)

slp = ds.slp.isel(time=0)
slp.plot(aspect=2, size=8)
plt.show()
