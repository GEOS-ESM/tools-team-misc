from wxvis.colors.registry import register
from pathlib import Path

colors = [
    [255, 255, 255],
    [75, 200, 168],
    [58, 176, 74],
    [0, 60, 15],
]

map_name = Path(__file__).stem
register(map_name, colors)
