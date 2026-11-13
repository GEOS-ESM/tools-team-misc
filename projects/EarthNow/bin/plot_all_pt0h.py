#!/usr/bin/env python
"""Render all products at a date's 00Z"""

import argparse
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import sys
import os

import yaml


def parse_date(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "date must be a valid YYYY-MM-DD date"
        ) from error


def main():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "date", type=parse_date, help="Forecast date (YYYY-MM-DD), at 00Z"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--product",
        default=None,
        help="Specific product to plot (default: all products)",
    )
    parser.add_argument("--resolution", choices=["hd", "4k", "8k"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()
    date_tag = args.date.strftime("%Y%m%d")
    specific_product = args.product

    with (project_dir / "cylc" / "config.yml").open() as handle:
        config = yaml.safe_load(handle)
    resolution = args.resolution or config["pngsize"].lower()
    lcc_products = config["plot_conus_LCC"]["products"]
    base_path = args.output_dir or config["base_path"]

    output_dir = Path(os.path.expandvars(base_path)) / f"{date_tag}_pt0h"

    for product, options in config["products"].items():
        if specific_product and product != specific_product:
            continue
        for region in ("conus", "global"):
            reader = (
                "conus2kmfc_lcc"
                if region == "conus" and product in lcc_products
                else "conus2kmfc"
            )
            output = output_dir / (
                f"{product}_{region}_{date_tag}_0000_{resolution}.png"
            )
            command = [
                sys.executable,
                str(project_dir / "bin" / "enow_plot.py"),
                "--product",
                options["product"],
                "--style",
                options["style"],
                "--data-reader",
                reader,
                "--map-type",
                region,
                "--fdate",
                f"{date_tag}_00z",
                "--pdate",
                f"{date_tag}_0000z",
                "--resolution",
                resolution,
                "--output",
                str(output),
            ]
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=project_dir, check=True)
                if not output.is_file() or output.stat().st_size == 0:
                    raise RuntimeError(
                        f"Plot command did not create an image: {output}"
                    )


if __name__ == "__main__":
    main()
