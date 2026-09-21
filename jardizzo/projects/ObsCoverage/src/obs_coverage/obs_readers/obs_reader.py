import xarray as xr


class ObsReader(object):

    def __init__(self, filename, **kwargs):

        self.quiet = kwargs.get("quiet", False)
        self.debug = kwargs.get("debug", False)

        self.fh = xr.open_dataset(filename)

        if not self.quiet:
            print(f"Reading: {filename}")
