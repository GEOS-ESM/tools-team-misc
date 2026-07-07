#! /usr/bin/env python

import os
import sys
import numpy as np
from netCDF4 import Dataset

import interface
import geos.netcdf as gnc
import geos.eta2hgt as gh

request = interface.parse_args(sys.argv[1:])
strict  = request.get('strict', False)
feet    = request.get('feet'  , False)
ground  = request.get('ground', False)
alt     = request.get('alt', False)


in_file  = request['iname']
hgt_file = request['hname']
out_file = request['oname']
vars     = request['vars'].split(',')
levels   = [float(lev) for lev in request['levels'].split(',')]

fh_in  = Dataset(in_file,  mode='r')
fh_hgt = Dataset(hgt_file, mode='r')
fh_out = gnc.Dataset(out_file)

# Write global attributes
# =======================

fh_out.write_global_attr(fh_in.__dict__,
                         Title='My Title',
                         History='File written by eta2hgt.py',
                         Filename=os.path.basename(out_file))

# Write dimensions
# ================

dims = ('time', 'lev', 'lat', 'lon')

time = fh_in.variables['time']
lat  = fh_in.variables['lat']
lon  = fh_in.variables['lon']
lev  = fh_in.variables['lev']

tyme = np.zeros( (1,), dtype=np.int32)
levs = np.asarray(levels, dtype=np.float32)

fh_out.write_var('time',    tyme,     ('time',), time.__dict__)

units = 'm'
if feet: units = 'ft'

fh_out.write_var('lev',     levs[:],  ('lev',),  lev.__dict__,
                 positive='up',
                 units=units,
                 coordinate='height',
                 standard_name='constant_height')

fh_out.write_var('lat',     lat[:],   ('lat',),  lat.__dict__)
fh_out.write_var('lon',     lon[:],   ('lon',),  lon.__dict__)

# Write variables
# ===============

e2h = gh.ETA2HGT(fh_in, fh_hgt, levels, strict=strict, feet=feet,
                                               ground=ground, alt=alt)

for name in vars:
    print(name)
    var, attr = e2h.createVariable(name)  
    fh_out.write_var(name, var, dims, attr)

fh_out.close()
