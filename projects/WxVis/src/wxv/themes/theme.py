from types import SimpleNamespace
from wxv.themes.registry import register
from wxv.myutils import dict_merge, str_replace
from wxv.conditional import Conditional


class theme(object):
    """
    Base class for WxVis themes.

    All commonly used methods for managing themes are include in
    this class.

    Parameters
    ----------
    define_plots : method
        Defines plots and layers.
    add_plots : method
        Creates a plot entry in the registry. This method will merge
        attribute settings if a plot already exists.
    add_layer : method
        Creates a layer entry in the registry. This method will merge
        attribute settings if a layer already exists.
    merge : method
        Merges one or more themes into one theme as a composite of all
        encountered definitions.
    expand : method
        Performs variables interpolation based on a request. Request parameters
        are used to resolve conditionals.
    attributes : SimpleNamespace
        Class attributes for plot and layer definitions.
    plots : dict
        Plot registry containing definitions by plot name.
    layers : dict
        Layer registry containing definitions by layer name.

    """

    def __init__(self, *args, **kwargs):

        self.attributes = SimpleNamespace()
        self.plots = {}
        self.layers = {}

    def define_plots(self):
        pass

    def add_plot(self, name, **kwargs):
        """
        Creates a plot entry in the registry.

        Binds attributes with a plot name and registers the entry.
        This method will merge attribute settings if a plot already exists.

        Parameters
        ----------
        name : string
            Name of plot. This is the registered name.
        kwargs : dict
            Attributes that define the plot.

        See Also
        --------
            self.plots : plot registry

        Returns
        -------
        None
            No return value

        """

        p = self.plots.get(name, {})
        dict_merge(p, kwargs)
        self.plots[name] = p

    def add_layer(self, name, **kwargs):
        """
        Creates a layer entry in the registry. 

        Binds attributes with a layer name and registers the entry.
        This method will merge attribute settings if a layer already exists.

        Parameters
        ----------
        name : string
            Name of layer. This is the registered name.
        kwargs : dict
            Attributes that define the layer.

        See Also
        --------
            self.layers : layer registry

        Returns
        -------
        None
            No return value

        """

        layer = self.layers.get(name, {})
        dict_merge(layer, kwargs)
        self.layers[name] = layer

    def merge(self, *args):

        for theme_obj in args:

            for name, attr in theme_obj.plots.items():
                self.add_plot(name, **attr)

            for name, attr in theme_obj.layers.items():
                self.add_layer(name, **attr)

    def expand(self, attributes, request):

        defs = {k: str(v) for k, v in request.items()}
        d = dict(attributes)

        for k, v in attributes.items():
            if isinstance(v, Conditional):
                result = v.get(request)
                defs[k] = str(result)
                d[k] = result
            else:
                defs[k] = str(v)

        for k, v in d.items():
            if isinstance(v, str):
                d[k] = str_replace(v, **defs)

        return d

    def __str__(self):

        return str(self.plots)
