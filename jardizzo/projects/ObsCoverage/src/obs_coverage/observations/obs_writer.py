import csv
import json
import numpy as np

def write_fluid_json(tbin, obs, fname):

    nobs = obs.lons.size

    oname = tbin.astype(object).strftime(fname) + ".json"
    buf = [f"{obs.lons[i]:.2f} {obs.lats[i]:.2f}" for i in range(0, nobs)]
    with open(oname, "w") as f:
        json.dump(buf, f)

def write_csv(tbin, obs, fname):

    nobs = obs.lons.size

    header = "obs_category,legend_category,isis,idate,obstime,lon,lat"
    header = header.split(",")

    oname = tbin.astype(object).strftime(fname) + ".csv"
    dattim = tbin.astype(object).strftime("%Y%m%d%H")
    buf = [
        (
            obs.obs_category,
            obs.legend_category,
            obs.isis,
            dattim,
            f"{(obs.times[i] - tbin) / np.timedelta64(1, "h"):.1f}",
            f"{obs.lons[i]:.1f}",
            f"{obs.lats[i]:.1f}",
        )
        for i in range(0, nobs)
    ]

    with open(oname, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(buf)
