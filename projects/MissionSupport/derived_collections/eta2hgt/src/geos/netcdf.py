import os
import collections
from netCDF4 import Dataset as dset

class Dataset(object):

    def __init__(self, fname, **kwargs):

        self.fname = fname
        self.name  = os.path.basename(fname)
        self.path  = os.path.dirname(fname)

        try:
            os.makedirs(self.path, 0o755)
        except:
            pass

        self.fh = dset(fname, "w", format="NETCDF4")

        return

    def write_global_attr(self, attr, **kwargs):

        attr = collections.OrderedDict(attr)
        attr.update(list(kwargs.items()))
        self.fh.setncatts(attr)

    def write_var(self, name, var, dims, attr, **kwargs):

        attr = collections.OrderedDict(attr)
        attr.update(list(kwargs.items()))

        if dims and len(dims) == 1:
            self.fh.createDimension(name , var.size)

        vh = self.fh.createVariable(name, var.dtype, dims)
        vh.setncatts(attr)

        vh[:] = var


    def close(self): self.fh.close()
