from importlib.resources import files
from pathlib import Path
from typing import Optional
from types import SimpleNamespace

import yaml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def load_yaml(name):
    path = files("cf_map.config").joinpath(name)
    with path.open() as f:
        return yaml.safe_load(f)

def load_style(name: str="datagrams.mplstyle"):
    path = files("cf_map.config").joinpath(name)
    plt.style.use(path)

def read_img(name):
    path = files("cf_map.config").joinpath(name)
    return mpl.image.imread(path)

load_style()

class plotConfig:
    def __init__(self, config_dir: Optional[str|Path]=None):
        self.config = self._load_config()
    def _load_config(self, file=None):
        file = file or 'plot_config.yml'
        config_in = load_yaml(file)
        self.colorbars = self._load_colormaps(config_in.get('colorbars',{}))
        config = {}
        for product, cfg in config_in.get('datagram',{}).items():
            config[product] = self._load_field(cfg)
        return config
    def _load_field(self, cfg):
        # plt_cfgs = next((v for k,v in _plot_cfg.items() if product.lower()==k.lower()),{})
        limits = (cfg.get('limits').get('min'), cfg.get('limits').get('max'))
        scale = cfg.get('scale')
        cm = self.get_colormap(cfg.get('colorbar'))
        if isinstance(scale,dict):
            scale = range(scale.get('min',0.01),scale.get('max',150),scale.get('step',5))
        return SimpleNamespace(limits=limits,scale=scale, colormap=cm,label=cfg.get('label'),
                            cb_fmt=cfg.get('cb_format'),contour_fmt=cfg.get('contour_fmt'))
    def _load_colormaps(self, cb_in):
        cbs = {}
        for name, colors in cb_in.items():
            cbs[name] = mcolors.LinearSegmentedColormap(name, colors)
            if name not in mpl.colormaps.keys():
                mpl.colormaps.register(cbs[name])
        return cbs
    def get_colormap(self,cb_name):
        if cb_name not in self.colorbars:
            self.colorbars[cb_name] = getattr(plt.cm, cb_name)
        return self.colorbars.get(cb_name)
            
    def get_prod_config(self,product):
        return next((v for k,v in self.config.items() if product.lower()==k.lower()),{})