import sys
import datetime as dt
from obs_coverage.observations.registry import OBS_TYPES
from obs_coverage.formats.registry import OBS_FORMATS
from obs_coverage.methods.registry import OBS_METHODS

MAXBUF = 5000000

obtype = OBS_TYPES["viirs_snpp"]

obs = obtype()
obstot = obtype()

for filename in sys.argv[1:]:

    obs.read(filename)
    obstot += obs

#   Write out accumulated observations.

    if len(obstot) > MAXBUF:
        tbins = obstot.bin(6)
        if len(tbins) <= 1:
            raise ValueError(
                    f"Buffer size {MAXBUF} is too small."
            )

        for tbin in tbins[0:-1]: # Assume that the last bin is incomplete.
            obs = obstot.get(tbin)
            obs.thin()
            write_csv(tbin, obs, '%Y%m%d%H')

        obstot.purge()

# Write out remaining observations

for tbin in obstot.bin(6):
    obs = obstot.get(tbin)
    obs.thin()
    write_csv(tbin, obs, '%Y%m%d%H')

if __name__ == "__main__":
    """
    Observation sampling command-line interface.

    Retrieves command-line arguments and execute the sampler.

    Usage
    -----
    Type "obs_sample.py" -h for a list of command-line options.

    """

    freeze_support()

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
        "--list",
        "-l",
        metavar="filelist",
        help="Input file containing a time sorted list of files"
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
        help=f"Input observation type: {OBS_TYPES}",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--format",
        "-f",
        metavar="format",
        help=f"Output data format: {OBS_FORMATS}",
        type=str,
        default='csv',
    )

    parser.add_argument(
        "--bin",
        "-b",
        metavar="bin_size",
        help=f"Temporal binning in hours",
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
