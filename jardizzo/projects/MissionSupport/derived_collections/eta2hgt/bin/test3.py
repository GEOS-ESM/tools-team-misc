import netCDF4 as nc

URL =  'https://opendap.nccs.nasa.gov/dods/gmao/geos-cf/fcast/aqc_tavg_1hr_g1440x721_v1.latest'
f = nc.Dataset(URL, 'r')
