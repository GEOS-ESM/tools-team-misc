import os
import sys
import re
import argparse
import numpy as np
import datetime as dt
from obs_coverage.observations.registry import OBS_TYPES
from obs_coverage.writers.registry import OBS_WRITERS

MAXBUF = 5000000


def execute(
    obtype=None,
    format=None,
    bin_size=None,
    opath=None,
    flist=None,
    files=None,
    start_dt=None,
    end_dt=None,
    **kwargs,
):

    obsobj = OBS_TYPES[obtype]
    writer = OBS_WRITERS[format]
    start_dt = np.datetime64(start_dt)
    end_dt = np.datetime64(end_dt)

    obs = obsobj()
    obstot = obsobj()
    pathname = os.path.join(opath, obtype, f"{obtype}.%Y%m%d%H")
    if not files:
        files = []

    if flist:
        with open(flist, "r") as f:
            files = f.read().splitlines()

    for filename in files:

        obs.read(filename)
        obstot += obs

        #   Write out accumulated observations.

        if len(obstot) > MAXBUF:
            tbins = obstot.bin(bin_size)
            if len(tbins) <= 1:
                raise ValueError(f"Buffer size {MAXBUF} is too small.")

            for tbin in tbins[0:-1]:  # Assume that the last bin is incomplete.

                if tbin >= start_dt and tbin <= end_dt:
                    print(tbin)
                    obs = obstot.get(tbin)
                    obs.thin()
                    writer(tbin, obs, pathname)
                elif tbin > end_dt:
                    sys.exit(0)

            obstot.purge()

    # Write out remaining observations

    for tbin in obstot.bin(bin_size):
        if tbin >= start_dt and tbin <= end_dt:
            print(tbin)
            obs = obstot.get(tbin)
            obs.thin()
            writer(tbin, obs, pathname)


if __name__ == "__main__":
    """
    Observation sampling command-line interface.

    Retrieves command-line arguments and execute the sampler.

    Usage
    -----
    Type "obs_sample.py" -h for a list of command-line options.

    """

    # Retrieve command-line arguments.
    # ================================

    description = "Production driver for observation sampler"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--stime",
        "-s",
        metavar="start_time",
        default="1979-01-01",
        help="Start date/time as an ISO string (e.g. ccyy-mm-ddThh:mm:ss)",
        type=str,
    )

    parser.add_argument(
        "--etime",
        "-e",
        metavar="end_time",
        default="2050-12-31",
        help="End date/time as an ISO string (e.g. ccyy-mm-ddThh:mm:ss)",
        type=str,
    )

    parser.add_argument(
        "--files",
        "-i",
        metavar="files",
        nargs="*",
        help="Input observation files in time sorted order",
        type=str,
    )

    parser.add_argument(
        "--flist",
        "-l",
        metavar="filelist",
        help="Input file containing a time sorted list of files",
        type=str,
    )

    parser.add_argument(
        "--opath",
        "-o",
        metavar="output_dir",
        help="Output directory",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--obtype",
        "-t",
        metavar="obtype",
        choices=OBS_TYPES.keys(),
        help=f"Input observation type",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--format",
        "-f",
        metavar="format",
        choices=OBS_WRITERS.keys(),
        help="Output data format (default: %(default)s)",
        type=str,
        default="csv",
    )

    parser.add_argument(
        "--bin_size",
        "-b",
        metavar="bin_size",
        help="Temporal binning in hours (default: %(default)s)",
        type=int,
        default=6,
    )

    args = parser.parse_args()

    dattim = re.sub("[^0-9]", "", args.stime + "000000")[0:14]
    idate = int(dattim[0:8])
    itime = int(dattim[8:14])
    start_dt = dt.datetime.strptime(dattim, "%Y%m%d%H%M%S")

    dattim = re.sub("[^0-9]", "", args.etime + "000000")[0:14]
    idate = int(dattim[0:8])
    itime = int(dattim[8:14])
    end_dt = dt.datetime.strptime(dattim, "%Y%m%d%H%M%S")

    args_dict = vars(args)
    args_dict.update({"start_dt": start_dt, "end_dt": end_dt})

    execute(**args_dict)

    sys.exit(0)
