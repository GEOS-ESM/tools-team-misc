#! /usr/bin/env python

import os
import sys
import numpy as np
from netCDF4 import Dataset

import interface
import geos.netcdf as gnc
import geos.aerosol as ga

request = interface.parse_args(sys.argv[1:])

in_file  = request['iname']
out_file = request['oname']
vars     = request['vars'].split(',')

fh_in  = Dataset(in_file,  mode='r')
fh_out = gnc.Dataset(out_file)

# Write global attributes
# =======================

fh_out.write_global_attr(fh_in.__dict__,
                         Title='My Title',
                         History='File written by aerosol.py',
                         Filename=os.path.basename(out_file))

# Write dimensions
# ================

dims = ('time', 'lev', 'lat', 'lon')

time = fh_in.variables['time']
lat  = fh_in.variables['lat']
lon  = fh_in.variables['lon']
lev  = fh_in.variables['lev']

fh_out.write_var('time',    time[:],  ('time',), time.__dict__)
fh_out.write_var('lev',     lev[:],   ('lev',),  lev.__dict__)
fh_out.write_var('lat',     lat[:],   ('lat',),  lat.__dict__)
fh_out.write_var('lon',     lon[:],   ('lon',),  lon.__dict__)

# Write variables
# ===============

aer = ga.Aerosol(fh_in)

for name in vars:
    print(name)
    var, attr = aer.createVariable(name)  
    fh_out.write_var(name, var, dims, attr)

fh_out.close()
