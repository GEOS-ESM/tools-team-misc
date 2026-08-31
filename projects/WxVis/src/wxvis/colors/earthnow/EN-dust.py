from wxvis.colors.registry import register
from pathlib import Path

colors = [
    [255, 255, 255],
    [255, 190, 75],
    [204, 0, 0],
    [113, 15, 19],
]
map_name = Path(__file__).stem
register(map_name, colors)
