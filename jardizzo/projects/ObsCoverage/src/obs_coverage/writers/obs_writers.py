import os
import csv
import json
import numpy as np

from .registry import register


@register("fluid_json")
def write_fluid_json(tbin, obs, fname):

    nobs = len(obs)

    oname = tbin.astype(object).strftime(fname) + ".json"
    os.makedirs(os.path.dirname(oname), exist_ok=True)
    buf = [f"{obs.lons[i]:.2f} {obs.lats[i]:.2f}" for i in range(0, nobs)]
    with open(oname, "w") as f:
        json.dump(buf, f)


@register("csv")
def write_csv(tbin, obs, fname):

    nobs = len(obs)

    header = "obs_category,legend_category,isis,idate,obstime,lon,lat"
    header = header.split(",")

    oname = tbin.astype(object).strftime(fname) + ".csv"
    os.makedirs(os.path.dirname(oname), exist_ok=True)
    dattim = tbin.astype(object).strftime("%Y%m%d%H")
    buf = [
        (
            obs.obs_category,
            obs.legend_category,
            obs.isis,
            dattim,
            f"{(obs.times[i] - tbin) / np.timedelta64(1, 'h'):.1f}",
            f"{obs.lons[i]:.1f}",
            f"{obs.lats[i]:.1f}",
        )
        for i in range(0, nobs)
    ]

    with open(oname, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(buf)
