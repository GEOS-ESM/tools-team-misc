from wxvis.colors.registry import register
from pathlib import Path

colors = [
    [128, 128, 128],
    [200, 231, 255],
    [54, 224, 224],
    [123, 62, 210],
    [59, 54, 135],
    [153, 0, 153],
]

map_name = Path(__file__).stem
register(map_name, colors)
