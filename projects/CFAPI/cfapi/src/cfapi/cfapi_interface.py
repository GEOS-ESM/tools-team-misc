import argparse
import dataclasses as dc
import datetime as dt
from typing import Optional
from pathlib import Path
import logging

# import validator as v
from . import validator as v

SCRIPT = Path(__file__).name
logger = logging.getLogger("cfapi.cfapi_interface")


class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
):
    pass


# --------------------
# Parser
# --------------------
def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-v",
        "--version",
        type=v.al.VERSIONS,
        default="v2",
        metavar="{v1|v2}",
        help=f"GEOS-CF Model Version. {v.al.COLLECTIONS.help_suffix()}",
    )
    parser.add_argument(
        "-c",
        "--collection",
        type=v.al.COLLECTIONS,
        default="fcast",
        metavar="{fcast|assim}",
        help=f"Forecast or assimilation stream. {v.al.COLLECTIONS.help_suffix()}",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=v.al.DATASETS,
        default="aqc",
        metavar="{chm|aqc|met}",
        help=f"Target dataset. {v.al.DATASETS.help_suffix()}",
    )
    parser.add_argument(
        "-l",
        "--level",
        type=v.al.LEVELS,
        default="slv",
        metavar="{slv|p23|v72}",
        help=f"Vertical choice. {v.al.LEVELS.help_suffix()}",
    )
    parser.add_argument(
        "-lat", "--lat", type=v.lat_type, default=38.9, help="Latitude [-90, 90]."
    )
    parser.add_argument(
        "-lon", "--lon", type=v.lon_type, default=-77.0, help="Longitude [-180, 180]."
    )
    parser.add_argument(
        "-s",
        "--start",
        type=v.date_type_allow_latest,
        default=None,
        help="Model start date (YYYYMMDD) or 'latest'.",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=v.date_type_allow_latest,
        default=None,
        help="Model end date (YYYYMMDD) or 'latest'.",
    )
    parser.add_argument(
        "-days",
        "--days",
        type=v.days_type,
        default=1,
        metavar="1-31",
        help="Number of days to include in the query (1–31). Only used for assimilation",
    )


def build_legacy_parser(
    script: Optional[str] = None, program: Optional[str] = None
) -> argparse.ArgumentParser:
    script = script or SCRIPT
    program = program or "python"
    examples = [
        "Examples:",
        "# Latest forecast, AQC single-level NO2 at a lat/lon",
        f"{program} {script} -c fcast -d aqc -l slv -p NO2 -lat 38.9 -lon -77",
        "",
        "# Chemistry pressure levels for O3 over a date range for CF version 1",
        f"{program} {script} -d chm -l p23 -p O3 -s 20250921 -e 20250923 -v v1",
        "",
        "# Meteorology; product is forced to MET, level will be normalized",
        f"{program} {script} -d met -l x1 -s latest",
    ]
    examples = "\n".join(examples)

    parser = argparse.ArgumentParser(
        prog=script or SCRIPT,
        description="Build query for GEOS-CF model outputs (LEGACY)",
        formatter_class=CustomFormatter,
        epilog=examples,
    )
    add_common_args(parser)
    # Legacy: only single product, with different validator
    parser.add_argument(
        "-P",
        "--product",
        type=v.al.LEGACY_PRODS,
        default="default",
        metavar="{CO|NO2|O3|PM25|SO2|HCHO|MET}",
        help=f"Single variable/product. {v.al.PRODUCTS.help_suffix()}",
    )
    return parser


def build_parser(
    script: Optional[str] = None, program: Optional[str] = None
) -> argparse.ArgumentParser:
    script = script or SCRIPT
    program = program or "python"
    examples = [
        "Examples:",
        "# Latest forecast, AQC single-level NO2 at a lat/lon",
        f"{program} {script} -c fcast -d aqc -l slv -p NO2 -lat 38.9 -lon -77",
        "",
        "# Chemistry pressure levels for O3 over a date range for CF version 1",
        f"{program} {script} -d chm -l p23 -p O3 -s 20250921 -e 20250923 -v v1",
        "",
        "# Meteorology; product is forced to MET, level will be normalized",
        f"{program} {script} -d met -l x1 -s latest",
    ]
    examples = "\n".join(examples)

    parser = argparse.ArgumentParser(
        prog=script,
        description="Build query for GEOS-CF model outputs",
        formatter_class=CustomFormatter,
        epilog=examples,
    )

    add_common_args(parser)

    # Modern: either multiple --products OR single --product, make them mutually exclusive
    mx = parser.add_mutually_exclusive_group()
    mx.add_argument(
        "-p",
        "--products",
        action=v.SplitProducts,
        nargs="*",
        metavar="VAR",
        help=f"One or more variables (space/comma-separated). {v.al.PRODUCTS.help_suffix()}",
    )
    mx.add_argument(
        "-P",
        "--product",
        type=v.al.PRODUCTS,
        default="default",
        metavar="{CO|NO2|O3|PM25|SO2|HCHO|MET}",
        help=f"Single variable/product.",
    )
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


def parse_args(script: Optional[str] = None, args=None, program=None) -> v.CliConfig:
    script = script or SCRIPT

    # --- Stage 1: detect version early ---
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="v2",
        metavar="{v1|v2}",
        help="GEOS-CF version (detected early)",
    )
    temp_parser.add_argument("--legacy", action="store_true")
    early, remaining = temp_parser.parse_known_args(args)
    version = early.version
    legacy = early.legacy

    # You can now use `version` to choose config defaults, load validators, etc.
    # e.g. dynamically modify v.DATASETS or allowed fields:
    # v.set_version(version)  # if your validator supports dynamic version switching

    # --- Stage 2: full parser using version-aware validation ---
    parser = build_parser(script, program)
    p = parser.parse_args(remaining)
    p.version = version
    # Enforce constraints and validate
    # v.enforce_constraints(p, legacy)
    return v.validate_and_build_config(p, legacy)


def uvmain():
    main("cfapi", "uv run")


def main(script=None, program=None):
    from .core import build_response_cfg

    script = script or ""
    cfg = parse_args(script=script, program=program)
    resp = build_response_cfg(cfg, use_cache=False)
    print(resp)
    print("=" * 25 + "Response Time" + "=" * 25)
    print(resp["schema"]["response time"])


if __name__ == "__main__":
    main()
