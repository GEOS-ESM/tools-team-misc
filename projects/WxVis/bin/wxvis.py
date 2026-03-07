import sys
import importlib
import wxv.interface as api
from themes.registry import THEMES

ui = api.Interface('Weather Visualizer')
args = ui.get_args()

for theme in args['theme']:

    try:
        module = importlib.import_module(theme)
    except:
        print(f'"{theme}" not found')
        sys.exit(2)

    p = THEMES[theme]()

    for name, plt in p.plots.items():
        print(name)
        print(len(name)*'=')
        print(plt)
