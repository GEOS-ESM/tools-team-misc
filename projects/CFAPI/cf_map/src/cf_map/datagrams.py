import os
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import logging
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

import numpy as np
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from cf_map.config import plotConfig, load_style, read_img, width, linewidth
from .cfmap_config import IMG_OUTPUT_DIR, file_template

logger = logging.getLogger("cf_map.datagrams")


def set_xticks(ax, times):
    ax.xaxis.set_ticks([t for t in times if t.hour == 0 or t.hour == 12])
    return ax


def custom_date_formatter(x, p, t0):
    """xaxis formatter for HHz + date + Year"""
    ti = mdates.num2date(x)
    if ti.hour == 0:
        d = mdates.date2num(t0 + dt.timedelta(days=1))
        if x < d:
            return ti.strftime("%Hz\n%a %-d %b\n%Y")
        return ti.strftime("%Hz\n%a %-d %b")
    else:
        return ti.strftime("%Hz")


class CFDatagram:
    def __init__(self, data_prepper, lat, lon, plot_type="pres", **kwargs):
        self.CONFIGS = plotConfig()
        self.version = kwargs.get("version", "v2")
        self.data = data_prepper(lat=lat, lon=lon, plot_type=plot_type, **kwargs)
        self.output_dir = kwargs.get("output_dir")
        load_style()

    def make_plot(self, product: Literal["co", "no2", "pm25", "o3", "so2"] = "no2"):
        self.fig = plt.figure()
        product = product.lower()
        data = self.data
        ax1 = plt.subplot2grid((28, 1), (1, 0))
        ax4 = plt.subplot2grid((28, 1), (2, 0), rowspan=15, sharex=ax1)
        ax5 = plt.subplot2grid((28, 1), (19, 0), rowspan=4)
        ax7 = plt.subplot2grid((28, 1), (24, 0), rowspan=4)
        # plt.style.use('datagrams.mplstyle')
        # self.load_style()
        ax1 = self.plot_cdtt(ax1, data.met)

        self.fig.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in self.fig.axes[2:-1]], visible=False)
        ax4 = self.format_xaxis(ax4, data.met.time, "hourly")
        ax4 = self.plot_pres_contour(ax4, data.contour, product)
        if product == "pm25":
            ax5 = self.plot_sfc_pm(ax5, data.sfc_pm25, product)
        else:
            ax5 = self.plot_sfc(ax5, data.sfc, product)
        ax7, ax8, ax9 = self.plot_met(ax7, data.met)
        self.add_labels(data.met.time, data.lat, data.lon)
        self.add_logos()
        filename = file_template.format(
            version=self.version,
            type="pres",
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

    # def load_style(self, dir_in: Optional[str|Path]=None):
    #     dir_in = Path(dir_in or CONFIG_DIR)
    #     plt.style.use(dir_in / 'datagrams.mplstyle')
    def save_plot(self, file_out: str | Path):
        file_out = Path(file_out)
        file_out.parent.mkdir(mode=0o775, parents=True, exist_ok=True)
        print(f"saving to {str(file_out)}")
        self.fig.savefig(file_out, bbox_inches="tight", dpi=100)
        try:
            os.chmod(file_out, 0o775)
        except Exception:
            pass
        # plt.close()
        return file_out

    def set_xticks(self, ax, times):
        ax.xaxis.set_ticks([t for t in times if t.hour == 0 or t.hour == 12])
        return ax

    def format_xaxis(self, ax, times, type="custom"):
        ax = self.set_xticks(ax, times)
        custom_form = partial(custom_date_formatter, t0=times[0])
        if type.lower() == "custom":
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(custom_form))
        elif type.lower() == "none":
            ax.xaxis.set_major_formatter(mdates.DateFormatter(""))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Hz"))
        return ax

    def get_time_avg(self, time, var, tavg=3):
        times = np.array(time)
        vals = np.array(var)

        # Ensure length is multiple of 3 (truncate remainder)
        n = len(vals) // tavg * tavg
        times = times[:n]
        vals = vals[:n]

        # Compute 3-hour means
        vals_3hr = vals.reshape(-1, tavg).mean(axis=1)

        # Pick representative time (start of each 3-hour block)
        times_3hr = times[::tavg]
        return times_3hr, vals_3hr

    def add_labels(self, times, lat, lon, station: Optional[str] = ""):
        forecast = times[0]
        self.fig.suptitle(
            "GEOS CF Forecast Initialized on %s\n"
            % (forecast.strftime("%Hz %m/%d/%Y")),
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        substr = [f"Lat = {float(lat):.2f}, Lon = {float(lon):.2f}"]
        if station:
            substr += [f"Location = {station}"]
        substr += [f"Fcst_Init = {forecast}"]
        lbl = self.fig.text(
            0.5, -0.01, ", ".join(substr), ha="center", fontsize=11, fontweight="bold"
        )

    def add_logos(self):
        ax = self.fig.add_axes([0, 0, 1, 1])
        gmao_img = read_img("GMAO-logo_small.png")
        nasa_img = read_img("nasa-logo_small.png")

        # Convert them to "OffsetImage" objects:
        gmao_logo_box = OffsetImage(gmao_img, zoom=0.72)  # adjust zoom as needed
        nasa_logo_box = OffsetImage(nasa_img, zoom=0.72)

        # Create AnnotationBbox for each logo:
        gmao_logo_ab = AnnotationBbox(
            gmao_logo_box,
            xy=(1.057, 1.027),  # top-right corner of the Axes
            xycoords="axes fraction",
            box_alignment=(1.0, 1.0),  # sets the alignment of the box center
            frameon=False,
        )
        nasa_logo_ab = AnnotationBbox(
            nasa_logo_box,
            xy=(-0.047, 1.053),  # top-left corner of the Axes
            xycoords="axes fraction",
            box_alignment=(0.0, 1.0),
            frameon=False,
        )

        # Add the logos to the same Axes:
        ax.add_artist(gmao_logo_ab)
        ax.add_artist(nasa_logo_ab)

        ax.axis("off")

    def plot_cdtt(self, ax, met, tavg=3):
        times, vals = self.get_time_avg(met.time, met.cldtt, tavg)
        ax.set_facecolor("b")
        # ax.bar(times, vals, width=width, color='w', align='center', linewidth=linewidth)
        ax.bar(
            met.time[::tavg],
            met.cldtt[::tavg],
            width=width,
            color="w",
            align="center",
            linewidth=linewidth,
        )
        ax.yaxis.set_ticks([])
        ax.set_xlim([met.time[0], met.time[-1]])
        ax.set_ylim(0, 1.5)
        ax.xaxis.set_ticks_position("top")
        ax.tick_params(axis="both", which="both", length=0)
        ax.set_ylabel("Tot Cld (%)", rotation=0)
        ax.yaxis.set_label_coords(-0.07, 0.1)
        return ax

    def plot_met(self, ax, met):
        ax.set_xlim([met.time[0], met.time[-1]])
        ax.bar(
            met.time,
            met.precip,
            color="b",
            width=width * 0.5,
            align="center",
            linewidth=linewidth,
            alpha=0.5,
        )
        ax.yaxis.offsetText.set_color("b")
        ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.yaxis.grid(True, linestyle=":")
        ax.yaxis.set_tick_params(direction="in")
        # ax = self.set_xticks(ax, met.time)
        ax = self.format_xaxis(ax, met.time, "custom")

        ax2 = ax.twinx()
        ax2.plot(met.time, met.temp, color="r")
        ax2.yaxis.offsetText.set_color("r")
        ax2.yaxis.get_major_formatter().set_useOffset(False)
        ax.tick_params(axis="both", which="both", length=0)
        ax.locator_params(axis="y", nbins=4)
        ax2.locator_params(axis="y", nbins=4)
        ax2.yaxis.set_tick_params(direction="in")
        ax.set_ylabel("Tot Precip\n(mm)", color="b", fontsize=11)
        ax2.set_ylabel(
            r"Temp 2m ($^\circ$F)", color="r", rotation=-90, fontsize=11, labelpad=15
        )
        for t1 in ax2.get_yticklabels():
            t1.set_color("r")
        for t1 in ax.get_yticklabels():
            t1.set_color("b")
        # wind speed
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.1))
        ax3.set_frame_on(True)
        ax3.patch.set_visible(False)
        for sp in ax3.spines.values():
            sp.set_visible(False)
        ax3.spines["right"].set_visible(True)

        ax3.plot(met.time, met.wspd, color="k")
        ax3.yaxis.get_major_formatter().set_useOffset(False)
        ax3.locator_params(axis="y", nbins=4)
        ax3.yaxis.set_tick_params(direction="in")
        ax3.set_ylabel(
            "Wind 10m (m/s)", color="k", fontsize=11, rotation=-90, labelpad=15
        )

        # ax3 = self.set_xticks(ax3,met.time)
        ax3 = self.format_xaxis(ax3, met.time, "custom")
        return ax, ax2, ax3

    def plot_sfc(self, ax, chm, product):
        CFG = self.CONFIGS.get_prod_config(product)

        ax.set_xlim([chm.time[0], chm.time[-1]])
        ax.plot(chm.time, getattr(chm, product), "k")
        ax.yaxis.set_tick_params(direction="in")
        ax.xaxis.set_tick_params(top=True, direction="in")
        ax.locator_params(axis="y", nbins=3, min_n_ticks=4)
        ax.yaxis.grid(True, linestyle=":")

        ax = self.format_xaxis(ax, chm.time, "none")
        ax.set_ylabel(f"Surface {CFG.label}", fontsize=11)
        return ax

    def plot_sfc_pm(self, ax, chm, product):
        def get_species_sum(data, bar):
            spec_sum = getattr(data, bar.species[0])
            if len(bar.species) == 1:
                return spec_sum
            for s in bar.species[1:]:
                spec_sum = np.add(spec_sum, getattr(data, s))
            return spec_sum

        CFG = self.CONFIGS.get_prod_config(product)
        ax.set_xlim([chm.time[0], chm.time[-1]])

        bars = [
            SimpleNamespace(species=["ni"], color="g", label="Nitrate"),
            SimpleNamespace(species=["ss"], color="b", label="Sea Salt"),
            SimpleNamespace(species=["du"], color="r", label="Dust"),
            SimpleNamespace(species=["bc", "oc"], color="k", label="OC + BC"),
            SimpleNamespace(species=["su"], color="y", label="Sulfate"),
            SimpleNamespace(species=["soa"], color="darkorange", label="SOA"),
        ]

        # Find the total sun of all PM25 for stacked bar plot
        # get first value so can use np add
        totals = get_species_sum(chm, bars[0])

        for b in bars:
            totals = np.add(totals, get_species_sum(chm, b))
        xlabel = -0.175
        ylabel = 0.5
        nxlab = 0.024
        plot_val = totals
        for b in bars:
            ax.bar(
                chm.time,
                plot_val,
                color=b.color,
                width=width * 0.2,
                align="center",
                linewidth=linewidth,
            )
            # self.fig.text(xlabel, ylabel, b.label, va='center', rotation='vertical', fontsize=11, color=b.color)
            ax.text(
                xlabel,
                ylabel,
                b.label,
                va="center",
                rotation="vertical",
                fontsize=11,
                color=b.color,
                transform=ax.transAxes,  # << use axes coords
                clip_on=False,  # so negatives aren’t clipped away
            )
            plot_val = np.subtract(plot_val, get_species_sum(chm, b))
            xlabel += nxlab

        ax2 = ax.twinx()
        ax2.yaxis.set_ticks([])
        ax2.set_ylabel(CFG.label, color="k", rotation=-90, fontsize=11, labelpad=22)
        return ax

    def plot_pres_contour(self, ax, data, product):
        CFG = self.CONFIGS.get_prod_config(product)
        field = getattr(data, product)
        mcmap = CFG.colormap
        mcmap.set_bad("silver", 1.0)
        mcmap.set_under("w")
        mcmap.set_over("k")

        bounds = np.array(CFG.scale, dtype=float)
        norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=mcmap.N, clip=False)

        ax_img = ax.contourf(
            data.time,
            data.levs,
            field,
            levels=bounds,
            cmap=mcmap,
            norm=norm,
            extend="both",
        )
        ax.invert_yaxis()
        CS = ax.contour(
            data.time,
            data.levs,
            field,
            colors="k",
            levels=CFG.scale,
            linewidths=0.4,
            linestyles="-",
        )
        ax.patch.set_facecolor("silver")
        ax.set_ylim(max(data.levs), min(data.levs))
        self.fig.subplots_adjust(left=0.08, right=0.95)
        cbar_ax = self.fig.add_axes([0.965, 0.437, 0.02, 0.449])
        cbar = self.fig.colorbar(
            ax_img,
            cax=cbar_ax,
            orientation="vertical",
            format=CFG.cb_fmt,
            ticks=CFG.scale,
        )
        cbar.ax.yaxis.set_tick_params(pad=40, direction="in")
        cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), ha="right")
        cbar.set_label(
            CFG.label, fontsize=11, fontweight="bold", rotation=270, labelpad=25
        )
        try:
            tl = ax.clabel(CS, CS.levels, inline=1, fontsize=8, fmt=CFG.contour_fmt)
            for te in tl:
                te.set_bbox(dict(color="w", alpha=0.8, pad=0.1))
        except:
            pass

        ax.yaxis.set_tick_params(direction="out")
        ax.yaxis.tick_left()
        ax.xaxis.set_tick_params(direction="out")
        ax.xaxis.tick_bottom()

        ax.set_ylabel("Pressure (hPa)", fontsize=11)
        return ax
