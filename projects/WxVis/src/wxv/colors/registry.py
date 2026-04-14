import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def register(name, colors):
    cmap = LinearSegmentedColormap.from_list(name, normalize(colors))
    if name not in mpl.colormaps:
        mpl.colormaps.register(cmap=cmap, name=name)


def normalize(colors):
    for i, color in enumerate(colors):
        colors[i] = [x / 255.0 for x in color]
    return colors
