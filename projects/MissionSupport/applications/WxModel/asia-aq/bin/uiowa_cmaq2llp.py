#!/usr/bin/env python
#
# Convert WRFchem output to COARDS compliant file on a regular lat-lon-pressure grid.
#
# Arlindo da Silva, March 2019.

from optparse   import OptionParser
from datetime   import datetime, timedelta
from netCDF4    import Dataset
import numpy as npy
import os

from curv2llp   import *

class myCurv2LLP(Curv2LLP):
    
    def get_cP(self,n,ncin):
        """
        Given time level "n", return 3D pressure for this time.
        """
        pres = ncin.variables['lev']  # hPa
        press = npy.tile(pres[:, npy.newaxis, npy.newaxis], (1, 5, 5))
        #press = npy.broadcast_to(pres, (5, 3))
        print(press)
        return press[:,:,:]

def getTyme_cmaq(cmaq_File):
    date_str = str(cmaq_File)[-17:-13]+str(cmaq_File)[-13:-11]+str(cmaq_File)[-11:-9]
    print(date_str)
    hour_str = str(cmaq_File)[-6:-3]
    print(hour_str)
    hour = int(hour_str)
    print(hour)

    date = datetime.strptime(date_str, "%Y%m%d")
    print(date)
    if hour >= 24:
        ndays = hour // 24
        date += timedelta(days=ndays)
        new_hour = hour - 24*ndays
    else:
        new_hour = hour

    print(new_hour)
    new_date_str = date.strftime("%Y%m%d")
    print(new_date_str)
    new_hour_str = '0'+str(new_hour).zfill(2)
    tyme = datetime(int(new_date_str[0:4]),
                int(new_date_str[4:6]),
                int(new_date_str[6:8]),
                int(new_hour_str))
    print(tyme)
    return npy.array([tyme,])

if __name__ == "__main__":

    format = 'NETCDF4'
    algo = 'linear'
    # rP = '925,850,700,600'
    rP = '1000, 925, 850, 700, 600'

#   Parse command line options
#   --------------------------
    parser = OptionParser(usage="Usage: %prog [OPTIONS] wrf_File",
                          version='1.0.0' )

    parser.add_option("-o", "--output", dest="outFile", default=None,
              help="Output NetCDF file (default: same was input with wrfout replaced with wrfllp)")

    parser.add_option("-a", "--algorithm", dest="algo", default=algo,
              help="Interpolation algorithm, one of linear, cubic (default=%s)"\
                          %algo)

    parser.add_option("-V", "--vars", dest="Vars", default=None,
              help="Variables to sample (default=All)")

    parser.add_option("-x", "--exclude", dest="ignore", default=None,
              help="Variables to ignore (default=None)")
    
    parser.add_option("-f", "--format", dest="format", default=format,
              help="Output file format: one of NETCDF4, NETCDF4_CLASSIC, NETCDF3_CLASSIC or NETCDF3_64BIT (default=%s)"%format )

    parser.add_option("-p", "--levels", dest="rP", default=rP,
              help="Levels to sample (default=%s)"%rP)

    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose",
                      help="Verbose mode.")

    parser.add_option("-n", "--dryrun",
                      action="store_true", dest="dryrun",
                      help="Dry-run mode: fill variables with zeros.")


    (options, args) = parser.parse_args()
    
    if len(args) == 2 :
        cord_File = args[0]
        wrf_File = args[1]
    else:
        parser.error("must have 2 argument: cord_File, wrf_File")

    if options.Vars is not None:
        options.Vars = options.Vars.split(',')

    if options.outFile is None:
        options.outFile = wrf_File.replace('wrfout_','wrfllp_')
    if options.outFile == wrf_File:
        options.outFile = 'wrfllp.nc4' # do not overwite input
        
    options.rP = [ float(p) for p in options.rP.split(',') ]
    
    if options.ignore is not None:
        options.ignore = options.ignore.split(',')
    else:
        options.ignore = []
        
    options.ignore += ['XLONG', 'XLAT', 'p', 'time','lon','lat'] # these are coordinates

    # Create consistent file name extension
    # -------------------------------------
    name, ext = os.path.splitext(options.outFile)
    if 'NETCDF4' in options.format:
        options.outFile = name + '.nc4'
    elif 'NETCDF3' in options.format:
        options.outFile = name + '.nc'
    else:
        raise ValueError, 'invalid extension <%s>'%ext

    # copy original file 
    # ------------------
    os.system("cp "+wrf_File+" "+wrf_File+".interm.nc")
    # Open the input file
    # -------------------
    nc_cord = Dataset( cord_File)
    nc = Dataset( wrf_File+".interm.nc", 'a', format=options.format)
    # Time range
    # ----------
    tyme = getTyme_cmaq(wrf_File)    
    
    # Instantiate regridding class
    # ----------------------------
    cLon = npy.squeeze(nc_cord.variables['lon'][:,:])
    cLat = npy.squeeze(nc_cord.variables['lat'][:,:])
    r = myCurv2LLP(cLon,cLat,options.rP)
    
    # Write output file
    # -----------------
    r.writeNC (tyme, options, nc, zlib=False )
   
    os.system("rm "+wrf_File+".interm.nc")
