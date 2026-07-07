import xarray as xr
import numpy as np
import os
from optparse import OptionParser

def append_density(ds):
    # Assuming you have a function to calculate air density and append it to the dataset
    # You can replace this with your actual function
    pass

def append_lml(nc, zlib=False):
    """
    Append vars at lowest model level to the oroginal file.
    """
    for v in ['bc_tot', 'dst_tot', 'pom_tot', 'pm25']:
        var = nc.variables[v]
        this = nc.createVariable(v+'_lml','f4',('Time', 'south_north', 'west_east'),zlib=zlib)
        this.description = var.description+' at lowest model level'
        this.missing_value = UNDEF # out of domain
        this.units = var.units
        this[0,:,:] = var[0,0,:,:]


def append_vars(ds):
    # Assuming you have a function to append other variables
    # You can replace this with your actual function
    p = ncin.variables['level'][:] # pressure levels, top to bottom in Pa
    nz, ny, nx = rho.shape
    dp = npy.zeros((nz,ny,nx))
    dp[0,:,:]=(p[0]+p[1])/2.
    for k in range(1,nz-1):
        dp[k,:,:] = (p[k+1]-p[k-1])/2.
    dp[nz-1,:,:] = ps - (p[nz-1]+p[nz-2])/2.

    pass

def main():
    # Parse command line options
    parser = OptionParser(usage="Usage: %prog [OPTIONS] input_file output_file",
                          version='1.0.0' )

    parser.add_option("-V", "--vars", dest="Vars", default=None,
                      help="Variables to sample (default=All)")

    parser.add_option("-f", "--format", dest="format", default='NETCDF4',
                      help="Output file format: one of NETCDF4, NETCDF4_CLASSIC, NETCDF3_CLASSIC or NETCDF3_64BIT (default=NETCDF4)")

    parser.add_option("-v", "--verbose", action="store_true", dest="verbose",
                      help="Verbose mode.")

    parser.add_option("-n", "--dryrun", action="store_true", dest="dryrun",
                      help="Dry-run mode: fill variables with zeros.")

    (options, args) = parser.parse_args()

    if len(args) == 2:
        input_File, output_File = args
    else:
        parser.error("must have 2 arguments: input_file output_file")

    # Open the input dataset
    ds = xr.open_dataset(input_File)

    # Define the target pressure levels
    target_pressure_levels = [975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100]

    # Interpolate the dataset to the target pressure levels
    ds_interpolated = ds.interp(lev=target_pressure_levels, method='linear', kwargs={'fill_value': 'extrapolate'})

    # Provide latitude and longitude details
    ds_interpolated['lat'] = ds['lat']
    ds_interpolated['lon'] = ds['lon']
    # Save the interpolated dataset to a new NetCDF file
    ds_interpolated.to_netcdf(output_File, format=options.format)

    # Close the datasets
    ds.close()
    ds_interpolated.close()

if __name__ == "__main__":
    main()

