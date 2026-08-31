from wxvis.colors.registry import register
from pathlib import Path

colors = [
    [255, 255, 255],
    [215, 215, 215],
    [153, 51, 255],
    [33, 15, 45],
]
map_name = Path(__file__).stem
register(map_name, colors)
