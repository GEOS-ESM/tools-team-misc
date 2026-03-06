from .registry import register
from wxv.myutils import dict_merge, str_replace
from wxv.conditional import Conditional

class theme(object):

    def __init__(self, *args, **kwargs):

        self.plots = {}
        self.layers = {}

    def add_plot(self, name, **kwargs):

        p = self.plots.get(name, {})
        dict_merge(p, kwargs)
        self.plots[name] = p

    def add_layer(self, name, **kwargs):

        layer = self.layers.get(name, {})
        dict_merge(layer, kwargs)
        self.layers[name] = layer

    def expand(self, attributes, request):

        defs = { k:str(v) for k,v in request.items() }
        d = dict(attributes)

        for k,v in attributes.items():
            if isinstance(v, Conditional):
                result = v.get(request)
                defs[k] = str(result)
                d[k] = result
            else:
                defs[k] = str(v)

        for k,v in d.items():
            if isinstance(v, str):
                d[k] = str_replace(v, **defs)

        return d

    def __str__(self):

        return str(self.plots)
