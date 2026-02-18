from .registry import register
from myutils import dict_merge
from conditional import Conditional

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

    def expand(self, name, request):

        for k,v in self.layers[name].items():

            if isinstance(v, Conditional):
                result = v.get(request)
                print(f'{k} is a conditional with value {result}')

    def __str__(self):

        return str(self.plots)
