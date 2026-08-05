from wxvis.colors.registry import register
from pathlib import Path

colors = [
    [255, 255, 255],
    [185, 185, 185],
    [206, 206, 113],
    [254, 254, 113],
    [252, 16, 22],
    [106, 7, 9],
    [23, 1, 2],
    [102, 0, 102],
    [0, 0, 204],
]

map_name = Path(__file__).stem
register(map_name, colors)
