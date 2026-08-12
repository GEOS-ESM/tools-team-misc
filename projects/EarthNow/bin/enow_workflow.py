#! /usr/bin/env python

import os
import sys
import re
import argparse
import datetime as dt

from multiprocessing import Pool, freeze_support

from earthnow.workflow.player import Player
from earthnow.workflow.utils import read_yaml
from earthnow.workflow.handlers import make_image, make_movie, purge


def execute(**args):
    """
    Executes Earthnow workflow tasks.

    This method executes EarthNow workflow tasks based on the supplied keyword
    option arguments.

    Parameters
    ----------
    time_dt : datetime object
        Reference date/time.
    config : str
        Configuration filename.
    task : str
        Name of configuration task to execute.
    nproc : int
        Number of processors to use.
    plot : bool
        If true, generate images and movies.
    image : bool
        If true, generate images.
    movie : bool
        If true, geneate movies.
    purge : bool
        If true, delete expired image and movie files.

    Returns
    -------
    None
        No return value

    """

    args_plot = args["plot"]
    time_dt = args["time_dt"]
    task = args["task"]

    # Get configuration.
    # ==================

    config = read_yaml(args["config"])
    ntasks = config.get("nproc", 1)
    if args["nproc"]:
        ntasks = args["nproc"]

    # Generate images
    # ===============

    if args_plot or args["image"]:

        iterator = Player(config, task, time_dt, **args)
        pool = Pool(ntasks)
        pool.map(make_image, iterator)

    # Generate movies
    # ===============

    if args_plot or args["movie"]:

        iterator = Player(config, task, time_dt, tloop=False, seamless=True, **args)
        pool = Pool(ntasks)
        pool.map(make_movie, iterator)

    # Removed expired image and movie files
    # =====================================

    if args["purge"]:

        clean = dict(config["clean"])
        options = {**clean, **args}
        iterator = Player(config, task, time_dt, tloop=False, seamless=True, **options)
        pool = Pool(1)
        pool.map(purge, iterator)


if __name__ == "__main__":
    """
    EarthNow workflow command-line interface.

    Retrieves command-line arguments and execute the workflow.

    Usage
    -----
    Type "enow_workflow" -h for a list of command-line options.

    """

    freeze_support()

    # Retrieve command-line arguments.
    # ================================

    description = "Production driver for EarthNOW visuals"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "datetime",
        metavar="datetime",
        type=str,
        help="Date or datetime ISO string (e.g. ccyy-mm-ddThh:mm:ss)",
    )

    parser.add_argument(
        "config", metavar="config", type=str, help="configuration file (.yml)"
    )
    parser.add_argument(
        "--stime",
        metavar="start_time",
        default=None,
        help="Start time as ISO duration offset",
        type=str,
    )
    parser.add_argument(
        "--etime",
        metavar="end_time",
        default=None,
        help="Ending time as ISO duration offset",
        type=str,
    )
    parser.add_argument(
        "--task",
        metavar="task_name",
        required=True,
        help="Workflow task to execute",
        type=str,
    )
    parser.add_argument(
        "--nproc", metavar="nproc", default=None, help="Number of processors", type=int
    )
    parser.add_argument(
        "--products",
        metavar="products",
        nargs="*",
        help="Products to generate",
        type=str,
    )
    parser.add_argument(
        "--regions", metavar="regions", nargs="*", help="Regions to generate", type=str
    )
    parser.add_argument(
        "--streams", metavar="streams", nargs="*", help="Streams to generate", type=str
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate all images and movies"
    )
    parser.add_argument("--publish", action="store_true", help="Publish visuals")
    parser.add_argument("--movie", action="store_true", help="Generate movie files")
    parser.add_argument("--image", action="store_true", help="Generate images")
    parser.add_argument(
        "--purge", action="store_true", help="Purge expired image files"
    )

    args = parser.parse_args()

    dattim = re.sub("[^0-9]", "", args.datetime + "000000")[0:14]
    idate = int(dattim[0:8])
    itime = int(dattim[8:14])
    time_dt = dt.datetime.strptime(dattim, "%Y%m%d%H%M%S")

    args_dict = vars(args)
    args_dict.update({"date": idate, "time": itime, "time_dt": time_dt})
    specified = [v for v in args_dict.values() if isinstance(v, bool) and v]
    if not specified:
        args_dict["plot"] = True

    execute(**args_dict)

    sys.exit(0)
