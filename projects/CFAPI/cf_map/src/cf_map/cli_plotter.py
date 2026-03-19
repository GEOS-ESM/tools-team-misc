import argparse
import dataclasses as dc
import datetime as dt
from typing import Optional
from pathlib import Path

# import local_config as lc
from . import cfmap_config as lc

SCRIPT = Path(__file__).name


class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
):
    pass


# --------------------
# Parser
# --------------------


def build_parser(script: Optional[str] = None) -> argparse.ArgumentParser:
    script = script or SCRIPT
    examples = [
        "Examples:",
        "# Latest forecast, AQC single-level NO2 at a lat/lon",
        f"python {script} -c fcast -d aqc -l slv -p NO2 -lat 38.9 -lon -77",
        "",
        "# Chemistry pressure levels for O3 over a date range for CF version 1",
        f"python {script} -d chm -l p23 -p O3 -s 20250921 -e 20250923 -v v1",
        "",
        "# Meteorology; product is forced to MET, level will be normalized",
        f"python {script} -d met -l x1 -s latest",
    ]
    examples = "\n".join(examples)

    parser = argparse.ArgumentParser(
        prog=script,
        description="Build query for GEOS-CF model outputs",
        formatter_class=CustomFormatter,
        epilog=examples,
    )
    parser.add_argument(
        "-v",
        "--version",
        type=lc.VERSIONS,
        default="v2",
        metavar="{v1|v2}",
        help=f"GEOS-CF Model Version. {lc.VERSIONS.help_suffix()}",
    )

    parser.add_argument(
        "-p",
        "--product",
        type=lc.PRODUCTS,
        default="no2",
        metavar="{co|pm25|no2|o3|so2}",
        help=f"GEOS-CF plot field. {lc.PRODUCTS.help_suffix()}",
    )

    parser.add_argument(
        "-pt",
        "--plot_type",
        type=lc.PLOTTYPES,
        default="pres",
        metavar="{pres|sfc|replay}",
        help=f"GEOS-CF plot type. {lc.PLOTTYPES.help_suffix()}",
    )

    parser.add_argument(
        "-lat", "--lat", type=lc.lat_type, default=38.9, help="Latitude [-90, 90]."
    )
    parser.add_argument(
        "-lon", "--lon", type=lc.lon_type, default=-77.0, help="Longitude [-180, 180]."
    )

    parser.add_argument(
        "-s",
        "--start",
        type=lc.date_type_allow_latest,
        default=None,
        help="Model start date (YYYYMMDD) or 'latest'.",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=lc.date_type_allow_latest,
        default=None,
        help="Model end date (YYYYMMDD) or 'latest'.",
    )
    parser.add_argument(
        "-days",
        "--num_days",
        type=lc.num_days_type,
        default=None,
        help="Model start date (YYYYMMDD) or 'latest'.",
    )
    # parser.add_argumet(
    #     "-d", "--dir_out",
    #     type=lc.directory_type, default="./plots",
    #     help=f"Output Directory. Defaults to subdirectory of plots from current dir"
    #     )
    # parser.add_argumet(
    #     "-o", "--file_out",
    #     type=lc.plot_file_type, default="./plots/cf{version}_{plot_type}_{product}_{lat}_{lon}.png",
    #     help=f"Output Directory. Defaults to subdirectory of plots from current dir"
    #     )

    return parser


# --------------------
# Main functions
# --------------------

# def parse_args(script: Optional[str]=None, args=None) -> v.CliConfig:
#     script = script or SCRIPT
#     parser = build_parser(script)
#     p = parser.parse_args(args)         # your current CLI path
#     v.enforce_constraints(p)
#     return v.validate_and_build_config(p, parser)


def parse_args(script: Optional[str] = None, args=None):
    script = script or SCRIPT

    parser = build_parser(script)
    p = parser.parse_args()
    return p
    # return v.validate_and_build_config(p, legacy)


def main():
    from .cf_plots import make_plot

    cfg = parse_args()
    args = {}
    for k in ["start", "end", "num_days"]:
        val = getattr(cfg, k)
        if val:
            args[k] = val
    # Example: print normalized config
    img, plotter = make_plot(cfg.lat, cfg.lon, cfg.product, cfg.plot_type, **args)
    print(f"Figure saved to {str(img)}")
    print(cfg)


if __name__ == "__main__":
    main()
