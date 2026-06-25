#!/usr/bin/env python3
"""
Standalone runner for EarthNow products via bin/plotall.py.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from earthnow.products import PRODUCTS


def parse_args():
    parser = argparse.ArgumentParser(description="Run plotall.py for EarthNow products")
    parser.add_argument("--fdate", default="20260508_00z", help="Forecast date")
    parser.add_argument("--pdate", default="20260508_0000z", help="Plot date")
    parser.add_argument("--map-type", default="global", help="Map domain")
    parser.add_argument("--style", default="light", help="Style name")
    parser.add_argument("--nproc", type=int, default=1, help="Worker count")
    parser.add_argument(
        "--base-path",
        default=f"/discover/nobackup/{Path.home().name}/EarthNow/plots/tests/",
        help="Output directory root",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only first N products",
    )
    parser.add_argument(
        "--product",
        type=str,
        choices=PRODUCTS.keys(),
        default=None,
        help="Run only one product by name",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    plotall_py = repo_root / "bin" / "plotall.py"

    # With GEOS_WxMaps import commented out in src/earthnow/products/__init__.py,
    # PRODUCTS should already contain only EarthNow products.
    if args.product:
        earthnow_products = [args.product]
    else:
        earthnow_products = list(PRODUCTS.keys())
        # If GEOS_WxMaps import is enabled again later, uncomment this filtered line:
        # earthnow_products = sorted(k for k in PRODUCTS if k.endswith("_EarthNow"))

        if args.limit is not None:
            earthnow_products = earthnow_products[: args.limit]

        if not earthnow_products:
            print("No products found to run.")
            return 2

    print(f"Running {len(earthnow_products)} product(s)")
    print(
        f"fdate={args.fdate} pdate={args.pdate} map={args.map_type} style={args.style}"
    )

    failures = []
    total_start = time.perf_counter()

    for idx, product in enumerate(earthnow_products, start=1):
        cmd = [
            sys.executable,
            str(plotall_py),
            "--product",
            product,
            "--nproc",
            str(args.nproc),
            "--fdate",
            args.fdate,
            "--pdate",
            args.pdate,
            "--map-type",
            args.map_type,
            "--base-path",
            args.base_path,
            "--style",
            args.style,
        ]

        print(f"[{idx}/{len(earthnow_products)}] {product}")
        if args.dry_run:
            print(" ", " ".join(cmd))
            continue

        start = time.perf_counter()
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - start

        if result.returncode == 0:
            print(f"  PASS ({elapsed:.1f}s)")
        else:
            print(f"  FAIL ({elapsed:.1f}s) rc={result.returncode}")
            tail = (result.stderr or result.stdout).strip().splitlines()[-20:]
            failures.append((product, result.returncode, tail))

    total_elapsed = time.perf_counter() - total_start

    if args.dry_run:
        print("Dry run complete")
        return 0

    passed = len(earthnow_products) - len(failures)
    print("\n=== Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {len(failures)}")
    print(f"Total:  {len(earthnow_products)}")
    print(f"Time:   {total_elapsed:.1f}s")

    if failures:
        print("\n=== Failures ===")
        for product, rc, tail in failures:
            print(f"- {product} (rc={rc})")
            for line in tail:
                print(f"    {line}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
