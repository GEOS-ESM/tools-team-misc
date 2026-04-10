import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from PIL import Image
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap


class Colormaps(object):

    def __init__(self, paths=""):

        self.files = []

        paths += os.getenv("COLORMAPSPATH", "")

        for path in paths.split(":"):
            self.files += Path(path).glob("*.csv")

    def get(self, name):

        # Check for matplotlib color first.

        if plt.colormaps.get(name, None):
            base_cmap = plt.colormaps.get(name, None)
            colors = base_cmap(np.linspace(0, 1, 256))
            return self.segment(colors)

        # Search for CSV file matching the requested name.

        for fname in self.files:

            bname = os.path.basename(fname)
            bname, ext = os.path.splitext(bname)
            nodes = bname.split("-")
            iname = f"{nodes[0]}-{nodes[1]}"
            aname = f"{nodes[0]}-" + "-".join(nodes[2:])
            if name == iname or name == aname or name == bname:
                colors = self.read_csv(fname)
                return self.segment(colors, normalize=True)

        raise FileNotFoundError(f"'{name}' colormap not found.")

    def read_csv(self, fname):

        colors = []

        with open(fname, "r") as f:
            lines = f.read().splitlines()

        for line in lines:
            colors.append(line.split(","))

        return colors

    def segment(self, clist, normalize=False):
        """"""
        segmentdata = {}
        colors = []

        factor = 1.0
        if normalize:
            factor = 255.0

        for color in clist:

            rgba = [float(c) / factor for c in color]
            if len(rgba) < 4:
                rgba.append(1.0)
            colors.append(rgba)

        data = np.linspace(0.0, 1.0, len(colors))

        for i, channel in enumerate(["red", "green", "blue", "alpha"]):

            segmentdata[channel] = []

            for index, rgba in enumerate(colors):
                x = data[index]
                y0 = rgba[i]
                y1 = y0
                values = (x, y0, y1)
                segmentdata[channel].append(values)

        return segmentdata

    __call__ = get


class Colorbar(object):

    def __init__(
        self,
        name,
        colors,
        vmin=None,
        vmax=None,
        vint=None,
        vlevs=None,
        vscale="linear",
        nsub=1,
        skip=None,
        cnorm=None,
        alpha=None,
        discrete=False,
        reverse=False,
        **kwargs,
    ):

        self.name = "_" + name
        self.segmentdata = dict(colors)
        self.vmin = vmin
        self.vmax = vmax
        self.vint = vint
        self.vlevs = vlevs
        self.vscale = vscale
        self.nsub = nsub
        self.skip = skip
        self.cnorm = cnorm
        self.alpha = alpha
        self.discrete = discrete

        if not skip:
            self.skip = nsub

        if not vlevs:
            n = round((vmax - vmin) / vint)
            vlevs = np.linspace(vmin, vmax, n + 1)

        self.vlevs = self.refine_levs(vlevs, nsub, vscale)

        if alpha:
            self.segmentdata["alpha"] = list(alpha)

        if cnorm:
            self.normalize_segment(cnorm)

        if discrete:
            self.discretize_segment()

        if reverse:
            self.reverse_segment()

        # Get a list of colors from the segment data

        N = len(self.vlevs) + 1
        self.cmap = LinearSegmentedColormap(self.name, self.segmentdata, N)
        colors = self.cmap(np.linspace(0, 1, self.cmap.N))

        # Recreate the colormap with the first and last color reserved
        # for the under/over values.

        self.cmap = LinearSegmentedColormap.from_list(
            self.name, colors[1:-1], N=len(colors) - 2
        )

        self.cmap.set_under(colors[0])
        self.cmap.set_over(colors[-1])

    def reverse_segment(self):
        """"""
        segmentdata = self.segmentdata

        for channel in ["red", "green", "blue"]:
            val = segmentdata[channel]
            valnew = [(1.0 - a, b, c) for a, b, c in list(reversed(val))]
            segmentdata[channel] = valnew

    def normalize_segment(self, norm):
        """"""
        segmentdata = self.segmentdata

        for channel in ["red", "green", "blue"]:
            val = segmentdata[channel]
            valnew = [(round(norm(a), 4), b, c) for a, b, c in val]
            segmentdata[channel] = valnew

    def discretize_segment(self):
        """"""
        segmentdata = self.segmentdata

        for channel in ["red", "green", "blue"]:
            vertices = segmentdata[channel]
            for index, val in enumerate(vertices[0:-1]):
                a, b, c = vertices[index]
                c = vertices[index + 1][1]
                vertices[index] = (a, b, c)

    def refine_levs(self, vlevs, nsub, vscale="linear"):
        """
        Refines value levels by sub-dividing each interval into smaller
        intervals.

        This method refines the value levels into a string of values
        where each interval is sub-divided into equally spaced sub-intervals.

        Parameters
        ----------
        vlevs : float[]
            List of value levels
        nsub : int
            Number of sub-divisions for each value interval
            E.g. [0,1] with nsub=10 will yield [0, 0.1, 0.2, ..., 1]
        vscale : string
            linear : default
            log : logarithmic scaling

        Returns
        -------
        levels : float[]
            List value levels

        """
        out_levs = []
        levels = []

        # Set up interpolation parameters

        if vscale.upper() == "LOG":
            interpolate = np.logspace
            options = dict(base=np.e, dtype=float)
            vlevs = [np.log(float(vlev)) for vlev in vlevs]
        else:
            interpolate = np.linspace
            options = dict(dtype=float)
            vlevs = [float(vlev) for vlev in vlevs]

        # Interpolate to sub-divisions

        for index, vlev in enumerate(vlevs[0:-1]):
            levels = interpolate(vlev, vlevs[index + 1], nsub + 1, **options)
            out_levs += list(levels[0:-1])

        out_levs.append(levels[-1])

        return out_levs

    def draw(self, pathname):

        a = np.array([[self.vmin, self.vmax]])
        fig = plt.figure()
        dpi = fig.get_dpi()
        fig.set_size_inches(1800.0 / dpi, 92.0 / dpi)

        img = plt.imshow(a, cmap=self.cmap)

        plt.gca().set_visible(False)
        cax = plt.axes([0.1, 0.2, 0.8, 0.6])
        cax.tick_params(axis="both", colors="white", direction="in")

        cb = plt.colorbar(
            orientation="horizontal",
            extend="both",
            extendrect=True,
            extendfrac="auto",
            drawedges=False,
            cax=cax,
        )
        cb.outline.set_edgecolor("white")
        cb.ax.xaxis.set_tick_params(pad=1)

        levels = []
        labels = []
        for i, lev in enumerate(self.vlevs):

            if i % self.skip != 0:
                continue

            levels.append(lev)
            labels.append(str(round(lev, 3)).strip("0").rstrip("."))
            if not labels[-1]:
                labels[-1] = "0"

        cb.set_ticks(levels)
        cb.ax.patch.set_facecolor("black")
        cb.ax.patch.set_alpha(1.0)

        # Set font properties for x-axis tick labels

        cb.ax.set_xticklabels(
            labels,
            fontsize=20,
            color="white",
            fontfamily="sans-serif",
            fontweight="bold",
            fontstyle="normal",
        )

        name, ext = os.path.splitext(pathname)
        ext = ext.strip(".")
        plt.savefig(
            pathname,
            format=ext,
            facecolor=(0, 0, 0),
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.01,
        )

        plt.close()

        return

def num_convert(val):

    if "." in val:
        return float(val)

    return int(val)
