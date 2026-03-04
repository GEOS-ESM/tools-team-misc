#! /usr/bin/env python

import os
import sys
import numpy as np
from netCDF4 import Dataset

in_file  = sys.argv[1]
out_file = sys.argv[2]

fh_in  = Dataset(in_file,  mode='r')
fh_out = Dataset(out_file, "w", format="NETCDF4")

fh_out.setncatts(fh_in.__dict__)

for name in fh_in.dimensions:

    variable = fh_in.variables[name]
    fh_out.createDimension(name, variable.size)
    v = fh_out.createVariable(name, variable.datatype, variable.dimensions)
    v.setncatts(variable.__dict__)
    v[:] = variable[:]

for name, variable in fh_in.variables.iteritems():

    if name in fh_in.dimensions: continue

    v = fh_out.createVariable(name, variable.datatype, variable.dimensions)
    v.setncatts(variable.__dict__)

    v[:] = 0.0

fh_out.close()
