from wxvis.colors.registry import register
from pathlib import Path

colors = [
    [108, 237, 239],
    [50, 129, 246],
    [0, 33, 245],
    [117, 250, 76],
    [86, 187, 55],
    [55, 125, 34],
    [255, 253, 84],
    [246, 192, 66],
    [239, 134, 51],
    [234, 57, 36],
    [175, 35, 24],
    [117, 20, 12],
    [230, 61, 244],
    [134, 106, 198],
]

map_name = Path(__file__).stem
register(map_name, colors)
