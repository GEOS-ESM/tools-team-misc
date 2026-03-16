import sys
import importlib
import wxv.interface as api
from wxv.themes.registry import THEMES

print(THEMES)

ui = api.Interface('Weather Visualizer')
request = ui.get_args()

for theme in request['theme']:

    try:
        module = importlib.import_module(theme)
    except:
        p = THEMES.get(theme, None)
        if p is None:
            print(f'"{theme}" not found')
            sys.exit(2)

    p = THEMES[theme]()

    for name, plt in p.plots.items():
        print(f'plot: {name}')
        print((len(name)+6)*'=')
        print(p.expand(plt, request))

    for name, layer in p.layers.items():
        print(f'layer: {name}')
        print((len(name)+7)*'=')
        print(p.expand(layer, request))
