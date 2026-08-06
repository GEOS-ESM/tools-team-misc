from wxvis.colors.registry import register
from pathlib import Path

colors = [
    [255, 255, 255],
    [196, 225, 255],
    [28, 123, 184],
    [2, 57, 89],
]
map_name = Path(__file__).stem
register(map_name, colors)
