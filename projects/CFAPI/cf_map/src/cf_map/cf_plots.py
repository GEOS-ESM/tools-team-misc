import os
import sys
import glob
import re
import logging
import json
import importlib
import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from typing import (
    List,
    Generator,
    Tuple,
    Dict,
    Optional,
    Any,
    Mapping,
    Iterable,
    Literal,
)
import datetime as dt

import matplotlib.pyplot as plt

from cf_map.config import plotConfig, load_style
from .datagrams import CFDatagram, set_xticks, custom_date_formatter
from .data_prepper import CFDatagramPrep
from .cfmap_config import IMG_OUTPUT_DIR, file_template

logger = logging.getLogger("cf_map.cf_plots")


class CFSurfacePlot(CFDatagram):
    def __init__(self, data_prepper, lat, lon, **kwargs):
        super().__init__(data_prepper, lat, lon, "sfc", **kwargs)

        # self.make_plot(product,cfg_dir)

    def make_plot(self, product: Optional[str] = None, data: Optional[dict] = None):
        product = (product or self.product).lower()
        data = data or self.data
        self.fig = plt.figure()

        ax1 = plt.subplot2grid(
            (28, 1),
            (1, 0),
            rowspan=2,
        )
        # ax4 = plt.subplot2grid((28, 1), (2, 0), rowspan=15, sharex=ax1)
        ax5 = plt.subplot2grid((28, 1), (3, 0), rowspan=12)
        ax7 = plt.subplot2grid((28, 1), (17, 0), rowspan=12)
        load_style()
        ax1 = self.plot_cdtt(ax1, data.met)

        self.fig.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in self.fig.axes[2:-1]], visible=False)
        ax5 = self.format_xaxis(ax5, data.met.time, "hourly")

        if product == "pm25":
            ax5 = self.plot_sfc_pm(ax5, data.sfc_pm25, product)
        else:
            ax5 = self.plot_sfc(ax5, data.sfc, product)
        ax1 = self.format_xaxis(ax1, data.met.time, "hourly")
        ax7, ax8, ax9 = self.plot_met(ax7, data.met)
        self.add_labels(data.met.time, data.lat, data.lon)
        self.add_logos()
        filename = file_template.format(
            version=self.version,
            type="sfc",
            product=product,
            lat=data.lat,
            lon=data.lon,
            ext="png",
        )
        output_dir = getattr(self, "output_dir") or IMG_OUTPUT_DIR
        output_dir = output_dir / data.met.time[0].strftime("%Y%m%dT%H%M")
        file_out = output_dir / filename
        file_out = self.save_plot(file_out)
        return file_out
        # self.save_plot(filename,output_dir)


class CFReplayPlot(CFDatagram):
    def __init__(self, data_prepper, lat, lon, **kwargs):
        super().__init__(data_prepper, lat, lon, "replay", **kwargs)
        # self.make_plot(product,cfg_dir)

    def make_plot(self, product: Optional[str] = None, data: Optional[dict] = None):
        product = (product or self.product).lower()
        data = data or self.data
        self.start_time = data.met.time[0]
        self.end_time = data.met.time[-1]
        self.fig = plt.figure()

        ax1 = plt.subplot2grid(
            (28, 1),
            (1, 0),
            rowspan=2,
        )
        # ax4 = plt.subplot2grid((28, 1), (2, 0), rowspan=15, sharex=ax1)
        ax5 = plt.subplot2grid((28, 1), (3, 0), rowspan=12)
        ax7 = plt.subplot2grid((28, 1), (17, 0), rowspan=12)
        load_style()
        ax1 = self.plot_cdtt(ax1, data.met)

        self.fig.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in self.fig.axes[2:-1]], visible=False)
        ax5 = self.format_xaxis(ax5, data.met.time, "hourly")

        ax5 = self.plot_sfc(ax5, data.sfc, product)
        ax1 = self.format_xaxis(ax1, data.met.time, "hourly")
        ax7, ax8, ax9 = self.plot_met(ax7, data.met)
        self.add_labels(data.met.time, data.lat, data.lon)
        self.add_logos()
        filename = file_template.format(
            version=self.version,
            type="replay",
            product=product,
            lat=data.lat,
            lon=data.lon,
            ext="png",
        )
        output_dir = getattr(self, "output_dir") or IMG_OUTPUT_DIR
        output_dir = (
            output_dir
            / f"{data.met.time[0].strftime('%Y%m%dT%H%M')}_{data.met.time[-1].strftime('%Y%m%dT%H%M')}"
        )
        file_out = output_dir / filename
        file_out = self.save_plot(file_out)
        return file_out

    def set_xticks(self, ax, times):
        # ax.xaxis.set_ticks([t for t in times if t.hour == 0 or t.hour == 12])
        ax.xaxis.set_ticks(
            [t for t in times if (t - times[0]).days % 5 == 0 and t.hour == 0]
        )
        return ax

    def add_labels(self, times, lat, lon, station: Optional[str] = ""):
        self.fig.suptitle(
            f"GEOS CF Replay  \n {times[0].strftime('%Hz %m/%d/%Y')} - {times[-1].strftime('%Hz %m/%d/%Y')}",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        substr = [f"Lat = {float(lat):.2f}, Lon = {float(lon):.2f}"]
        if station:
            substr += [f"Location = {station}"]
        # substr +=  [f"Fcst_Init = {forecast}"]
        lbl = self.fig.text(
            0.5, -0.01, ", ".join(substr), ha="center", fontsize=11, fontweight="bold"
        )


def make_plotter(lat, lon, plot_type, **kwargs):
    if plot_type.lower() in ["sfc", "surf", "surface", "surf_plot"]:
        plotter = CFSurfacePlot(CFDatagramPrep, lat, lon, **kwargs)
    elif plot_type.lower() in ["pres", "pres_plot", "pressure", "datagram"]:
        plotter = CFDatagram(CFDatagramPrep, lat, lon, **kwargs)
    elif plot_type.lower() in ["replay", "rply", "assim", "ana", "analysis"]:
        plotter = CFReplayPlot(CFDatagramPrep, lat, lon, **kwargs)
    else:
        return ""
    return plotter


def make_plot(lat, lon, product, plot_type, **kwargs):
    plotter = make_plotter(lat, lon, plot_type, **kwargs)
    if not plotter:
        return "", ""
    img = plotter.make_plot(product)
    return img, plotter


# if __name__ == "__main__":
#     from cli_interface import parse_args
#     SCRIPT = Path(__file__).name
#     cfg = parse_args(script=SCRIPT,type='plot')

#     args = {}
#     for k in ['start','end','num_days']:
#         val = getattr(cfg, k)
#         if val:
#             args[k] = val
#     # Example: print normalized config
#     img, plotter = make_plot(cfg.lat,cfg.lon,cfg.product,cfg.plot_type, **args)
#     print(f"Figure saved to {str(img)}")
#     print(cfg)
