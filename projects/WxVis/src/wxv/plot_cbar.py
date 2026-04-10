#! /usr/bin/env python

import os
import sys
import argparse

from colorbar import Colormaps, Colorbar
from normfuncs import NORMFUNCS

alpha = [(0, 1, 1), (0.5, 1, 1), (1, 1, 1)]


def plot_colorbar(**kwargs):

    cmaps = Colormaps()

    if kwargs.get('list', False):

        print("Colormaps\n")
        for file in cmaps.files:
            bname = os.path.basename(file)
            bname, ext = os.path.splitext(bname)
            print(bname)

        print("\nColor Normalization Functions\n")
        for name in NORMFUNCS:
            print(name)

        sys.exit(0)

    cname = kwargs.get("cmap", "viridis")
    colors = cmaps.get(cname)

    cnorm = kwargs.get("cnorm", "linear")
    kwargs['cnorm'] = NORMFUNCS.get(cnorm, None)

    colors = cmaps.get(cname)
    cb = Colorbar('mycbar', colors, **kwargs)
    cb.draw(cname+".png")

    print(cb.cmap)

if __name__ == "__main__":

    # Get command-line arguments

    parser = argparse.ArgumentParser(description="Plots colorbars")

    parser.add_argument(
        "-o", "--oname", metavar="ONAME", type=str, default=None, help="Filename"
    )
    parser.add_argument(
        "-c", "--cmap", metavar="COLORMAP", type=str, default=None, help="Name of colormap"
    )
    parser.add_argument(
        "--cnorm",
        metavar="CNORM",
        type=str,
        default="linear",
        help="Color normalization method",
    )
    parser.add_argument(
        "--vscale",
        metavar="VSCALE",
        type=str,
        default="linear",
        help="Value scaling method",
    )
    parser.add_argument("--list", action="store_true", help="List colormaps")
    parser.add_argument("--reverse", action="store_true", help="Reverse colormap")
    parser.add_argument(
        "--vmin", metavar="VMIN", type=float, default=None, help="Minimum value"
    )
    parser.add_argument(
        "--vmax", metavar="VMAX", type=float, default=None, help="Maximum value"
    )
    parser.add_argument(
        "--vint", metavar="VINT", type=float, default=None, help="Value increment"
    )
    parser.add_argument(
        "--nsub", metavar="NSUB", type=int, default=1, help="Value Sub-divisions"
    )
    parser.add_argument(
        "--skip", metavar="SKIP", type=int, default=None, help="Label frequency"
    )

    args = vars(parser.parse_args())

    plot_colorbar(**args)
