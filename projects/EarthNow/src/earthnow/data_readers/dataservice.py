import os
import datetime as dt
from netCDF4 import Dataset

import logging

logger = logging.getLogger(__name__)


class SimpleNetCDFDataService(object):

    def __init__(self, stream):

        self.nc = None
        self.filename = ""
        self.stream = stream

    def open(self, collection, pdate, fdate=None):

        time_dt = dt.datetime.strptime(pdate, "%Y%m%d_%Hz")
        fname = time_dt.strftime(self.stream.uri)

        if fdate:
            fcst_dt = dt.datetime.strptime(fdate, "%Y%m%d_%Hz")
            fname = fcst_dt.strftime(fname)

        fname = fname.replace("$collection", collection)

        if fname == self.filename:
            return fname

        self.nc = Dataset(fname)
        self.filename = fname

        return fname

    def get_coords(self):

        if self.stream.grid != "lcc":
            lats = self.nc.variables["lat"][:]
            lons = self.nc.variables["lon"][:]
            return lats, lons
        else:
            with Dataset(self.stream.coords) as nc:
                return nc.variables["lats"][:], nc.variables["lons"][:]

    def get_var(self, varname, level=None):
        var = self.nc.variables[varname]
        return var

    def read_variable(self, fdate, pdate, variables):

        if isinstance(variables, str):
            variables = [variables]

        for name in variables:

            if name not in self.stream.vars:
                logger.debug(f"{name} not in {self.stream} variable list.")
                continue

            expr = self.stream.vars[name]
            varname, collection = expr.split(".")
            filename = self.open(collection, pdate, fdate)

            var = self.get_var(varname)
            lats, lons = self.get_coords()

            metadata = {
                "units": getattr(var, "units", ""),
                "long_name": getattr(var, "long_name", varname),
                "collection": collection,
                "grid": self.stream.grid,
                "file": os.path.basename(filename),
            }

            logger.info(f"Using file      : {filename}")
            logger.info(f"Using collection: {collection}")
            logger.info(f"Using variable  : {varname}")

            data = var[:]
            if data.ndim >= 3:
                data = data[0]

            return data, lats, lons, metadata

        raise ValueError(
            f"None of the specified variables were found. Searched for: {variables}"
        )

    def resolve_file(self, fdate, pdate, **kwargs):
        return "none", "none", "none"


DataService = SimpleNetCDFDataService
