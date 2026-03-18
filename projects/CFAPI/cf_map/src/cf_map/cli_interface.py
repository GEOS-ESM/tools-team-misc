import argparse
import dataclasses as dc
import datetime as dt
from typing import Optional, Literal, Dict, Any
from pathlib import Path
import logging
# import local_config as lc
from . import cfmap_config as lc

SCRIPT = Path(__file__).name

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawTextHelpFormatter):
    pass

logger = logging.getLogger('cf_map.cli_interface')
# --------------------
# Parser
# --------------------
def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-v", "--version",
        type=lc.VERSIONS, default="v2", metavar="{v1|v2}",
        help=f"GEOS-CF Model Version. {lc.VERSIONS.help_suffix()}",
    )
    
    parser.add_argument("-lat", "--lat", type=lc.lat_type, default=38.9, help="Latitude [-90, 90].")
    parser.add_argument("-lon", "--lon", type=lc.lon_type, default=-77.0, help="Longitude [-180, 180].")

    parser.add_argument(
        "-s", "--start", type=lc.date_type_allow_latest, default=None,
        help="Model start date (YYYYMMDD) or 'latest'.",
    )
    parser.add_argument(
        "-e", "--end", type=lc.date_type_allow_latest, default=None,
        help="Model end date (YYYYMMDD) or 'latest'.",
    )
    parser.add_argument(
        "-nd", "--num_days", type=lc.num_days_type, default=None,
        help="Model start date (YYYYMMDD) or 'latest'.",
    )    

def build_plot_parser(script: Optional[str]=None, program: Optional[str]=None) -> argparse.ArgumentParser:
    script = script or SCRIPT
    program = program or 'python'
    examples = [
        'Examples:',
        '# Latest forecast, AQC single-level NO2 at a lat/lon',
        f'{program} {script} -c fcast -d aqc -l slv -p NO2 -lat 38.9 -lon -77',
        '',
        '# Chemistry pressure levels for O3 over a date range for CF version 1',
        f'{program} {script} -d chm -l p23 -p O3 -s 20250921 -e 20250923 -v v1',
        '',
        '# Meteorology; product is forced to MET, level will be normalized',
        f'{program} {script} -d met -l x1 -s latest',
    ]
    examples = '\n'.join(examples)

    parser = argparse.ArgumentParser(
        prog=f"{program} {script}",
        description="Build query for GEOS-CF plot outputs",
        formatter_class=CustomFormatter,
        epilog=examples,
    )

    add_common_args(parser)

    parser.add_argument(
        "-p", "--product",
        type=lc.PRODUCTS, default="no2", metavar="{co|pm25|no2|o3|so2}",
        help=f"GEOS-CF plot field. {lc.PRODUCTS.help_suffix()}"
        )
    
    parser.add_argument(
        "-pt", "--plot_type",
        type=lc.PLOTTYPES, default="pres", metavar="{pres|sfc|replay}",
        help=f"GEOS-CF plot type. {lc.PLOTTYPES.help_suffix()}"
        )
    
    parser.add_argument(
        "-o", "--output_dir",
        type=lc.directory_type, default=lc.IMG_OUTPUT_DIR,
        help=f"Output Directory. Defaults to subdirectory of plots from current dir"
        )   
    return parser

def build_file_maker_parser(script: Optional[str]=None, program: Optional[str]=None) -> argparse.ArgumentParser:
    script = script or SCRIPT
    program = program or 'python'
    examples = [
        'Examples:',
        '# Latest forecast, AQC single-level NO2 at a lat/lon',
        f'{program} {script} -c fcast -d aqc -l slv -p NO2 -lat 38.9 -lon -77',
        '',
        '# Chemistry pressure levels for O3 over a date range for CF version 1',
        f'{program} {script} -d chm -l p23 -p O3 -s 20250921 -e 20250923 -v v1',
        '',
        '# Meteorology; product is forced to MET, level will be normalized',
        f'{program} {script} -d met -l x1 -s latest',
    ]
    examples = '\n'.join(examples)

    parser = argparse.ArgumentParser(
        prog=script,
        description="Build query for GEOS-CF data file outputs",
        formatter_class=CustomFormatter,
        epilog=examples,
    )
    
    add_common_args(parser)

    parser.add_argument(
        "-f", "--format",
        type=lc.FORMATS, default="json", metavar="{json|xlsx|txt}",
        help=f"GEOS-CF data file output type. {lc.FORMATS.help_suffix()}"
        )
    
    parser.add_argument(
        "-p", "--product",
        action=lc.SplitProducts, nargs='*', metavar="VAR",
        help=f"One or more variables (space/comma-separated). {lc.FILE_PRODUCTS.help_suffix()}",
    )

    parser.add_argument(
        "-d", "--dataset",
        type=lc.DATASETS, default='chm_slv', metavar="{chm_slv|chm_p23|met_slv}",
        help=f"GEOS-CF dataset type. {lc.DATASETS.help_suffix()}",
    )

    parser.add_argument(
        "-c", "--collection",
        type=lc.COLLECTIONS, default='fcast', metavar="{assim|fcast}",
        help=f"Forecast or assimilation stream. {lc.COLLECTIONS.help_suffix()}",
    )

    parser.add_argument(
        "-o", "--output_dir",
        type=lc.directory_type, default=lc.FILE_OUTPUT_DIR,
        help=f"Output Directory. Defaults to subdirectory of file_downloads from current dir"
        )   

    return parser
# --------------------
# Main functions
# --------------------
def parse_args(script: Optional[str]=None,
            type: Optional[Literal['plot','file']]=None, 
            program: Optional[str]=None ):
    type = type or 'plot'
    script = script or SCRIPT
    if type=='file':
        parser = build_file_maker_parser(script, program)
    else:
        parser = build_plot_parser(script, program)

    p = parser.parse_args()
    args: Dict[str, Any] = {}
    for k in ("start", "end", "num_days"):
        val = getattr(p, k)
        if val:
            args[k] = val
    p.args = args
    return p
    # return v.validate_and_build_config(p, legacy)

def uvmain():
    import sys
    print(sys.argv)
    main('cf-map','uv run')


def main(script=None,program=None):
    from .cf_plots import make_plot
    cfg = parse_args(script=script,program=program)
    args = {}
    for k in ['start','end','num_days']:
        val = getattr(cfg, k)
        if val:
            args[k] = val
    # Example: print normalized config
    img, plotter = make_plot(cfg.lat,cfg.lon,cfg.product,cfg.plot_type, **args)
    print(f"Figure saved to {str(img)}")
    print(cfg)

if __name__ == "__main__":
    main()
