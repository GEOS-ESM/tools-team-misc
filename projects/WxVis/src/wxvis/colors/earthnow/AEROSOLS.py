from wxvis.colors.registry import register

colors_SS = [
    [255, 255, 255],
    [196, 225, 255],
    [28, 123, 184],
    [2, 57, 89],
]

colors_DU = [
    [255, 255, 255],
    [255, 190, 75],
    [204, 0, 0],
    [113, 15, 19],
]

colors_SU = [
    [255, 255, 255],
    [215, 215, 215],
    [153, 51, 255],
    [33, 15, 45],
]

colors_NI = [
    [255, 255, 255],
    [75, 200, 168],
    [58, 176, 74],
    [0, 60, 15],
]

register("AOT-SEASALT", colors_SS)
register("AOT-DUST", colors_DU)
register("AOT-SULFATE", colors_SU)
register("AOT-NITRATE", colors_NI)
