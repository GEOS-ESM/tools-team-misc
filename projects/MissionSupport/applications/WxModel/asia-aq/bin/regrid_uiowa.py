#!/usr/bin/env python
#
# Convert WRFchem output to COARDS compliant file on a regular lat-lon-pressure grid.
#
# Added lines for UIOWA CMAQ outputs, Feb. 2024.
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
        press = ncin.variables['p'][n,:,:,:]/100.0  # hPa
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
    # Append vars
    # -----------
#    append_lml(nc,zlib=False)

    # Time range
    # ----------
#    tyme = getTyme(nc)
    tyme = getTyme_cmaq(wrf_File)
   
   
    # Instantiate regridding class
    # ----------------------------
    cLon = npy.squeeze(nc_cord.variables['lon'][:,:])
    cLat = npy.squeeze(nc_cord.variables['lat'][:,:])
    r = myCurv2LLP(cLon,cLat,options.rP)
   
    # Write output file
    # -----------------
    r.writeNC ( tyme, options, nc, zlib=False )
   
    os.system("rm "+wrf_File+".interm.nc")

