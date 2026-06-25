# validators.py  (or fold into your module)
import argparse
import re
import yaml
import dataclasses as dc
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, List
import logging

# import these from your module
from .aliases import _dedup_keep_order, parse_list_value
from .api_config import aliases as al
from .constants import ArgumentTypeError, CliConfig

logger = logging.getLogger("cfapi")


# --------------------
#  Helpers
# --------------------
def iso2ymd(s):
    if len(s) == 15:
        s = s[:-1]  # remove z
    if len(s) == 14:
        if "T" in s:
            dts = "%Y%m%dT%H:%M"
        else:
            dts = "%Y%m%d %H:%M"
        try:
            return dt.datetime.strptime(s, dts)
        except ValueError:
            return None
    return None


# --------------------
# Types & validators
# --------------------


def lat_type(s: str) -> float:
    try:
        val = float(s)
    except ValueError as e:
        raise ArgumentTypeError(f"Latitude must be a float: {s}") from e
    if not (-90.0 <= val <= 90.0):
        raise ArgumentTypeError(f"Latitude out of range [-90, 90]: {val}")
    return val


def lon_type(s: str) -> float:
    try:
        val = float(s)
    except ValueError as e:
        raise ArgumentTypeError(f"Longitude must be a float: {s}") from e
    if 180.0 < val < 360.0:
        val -= 360
    elif not (-180.0 <= val <= 180.0):
        raise ArgumentTypeError(f"Longitude out of range [-180, 180]: {val}")
    return val


def date_type_allow_latest(s: str) -> Optional[dt.datetime]:
    """Accept 'latest' or YYYYMMDD."""
    if s.lower() == "latest" or s is None:
        return None  # None means "latest"
    try:
        return dt.datetime.strptime(s, "%Y%m%d")
    except ValueError as e:
        dt_out = iso2ymd(s)
        logger.warning(dt_out)
        if dt_out:
            return dt_out
        else:
            raise ArgumentTypeError(
                f"Date must be 'latest' or YYYYMMDD (e.g., 20250131); got: {s}"
            ) from e


def days_type(s: str) -> int:
    """Ensure days is an integer between 1 and 31."""
    try:
        val = int(s)
    except ValueError as e:
        raise ArgumentTypeError(f"Days must be an integer: {s}") from e
    if not (1 <= val <= 31):
        raise ArgumentTypeError(f"Days must be between 1 and 31: {val}")
    return val


# --------------------
# Helpers for product lists
# --------------------
def parse_legacy_products_value(value: Any) -> List[str]:
    return parse_list_value(value, al.LEGACY_PRODS)


def parse_products_value(value: Any) -> List[str]:
    return parse_list_value(value, al.PRODUCTS)


# def parse_products_value(value: Any, aliases=al.PRODUCTS) -> List[str]:
#     return parse_list_value(value, aliases)


# Optional: argparse Action for CLI (splits CSV/space and applies aliases)
class SplitProducts(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        canon_list = parse_products_value(values)
        setattr(namespace, self.dest, canon_list)


class SplitLegacyProducts(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        canon_list = parse_legacy_products_value(values)
        setattr(namespace, self.dest, canon_list)


# --------------------
# Normalization & constraints
# --------------------


def parse_datakey(raw: Mapping[str, Any]):
    datakey = raw.get("datakey")
    level = raw.get("level")
    dataset = raw.get("dataset")
    if "_" in dataset:
        datakey = dataset


def _check_value(
    value: str,
    options: List[str],
    dataset: str,
    key: str = "option",
):
    if options and value not in options:
        raise ArgumentTypeError(
            f"For dataset={dataset}, {key} must be one of {sorted(set(options))}; got {value!r}"
        )


def normalize_level(level: str) -> str:
    """Map aliases x1/v1 → slv; keep others as-is."""
    return "slv" if level in {"x1", "v1", "slv"} else level


def enforce_constraints(p: argparse.Namespace) -> None:
    # Normalize level aliases
    al.set_version(version=p.version)

    # DATASET_OPTS = DATA_OPTS.get(p.dataset,{})
    p.level = normalize_level(p.level)
    p.dataset_key = f"{p.dataset}_{p.level}"
    if p.dataset_key not in al.PRODUCTS.opts:
        prefix = f"{p.dataset}_"
        level_opts = [
            ds.replace(prefix, "")
            for ds in al.PRODUCTS.opts.keys()
            if ds.startswith(prefix)
        ]
        raise ArgumentTypeError(
            f"For dataset={p.dataset}, level must be one of {sorted(set(level_opts))}; got {p.level!r}"
        )
    # _check_value(p.level, DATASET_OPTS.get('levels',{}).get('options',[]),p.dataset, 'level')

    # Ensure p.products exists
    if not hasattr(p, "products") or p.products is None:
        p.products = []

    # Build a working list
    items = list(p.products)
    if p.product and p.product != "default":
        items = [p.product] + items
    if not items:  # nothing provided → dataset-specific default
        items = [al.PRODUCTS.get_default(p.dataset_key)]  # default for CHM/AQC
    items = _dedup_keep_order(items)

    for item in items:
        _check_value(item, al.PRODUCTS.get_opts(p.dataset_key), p.dataset, "product")

    # Persist back
    p.products = items
    # Keep a single representative in p.product for legacy downstreams
    p.product = items[0]
    # Date logic
    if p.start and p.end and p.end < p.start:
        raise ArgumentTypeError("end date cannot be earlier than start date.")


def enforce_constraints_legacy(p: argparse.Namespace) -> None:
    # Normalize level aliases
    al.set_version(version=p.version)
    OPTS = al.legacy_opts.get(p.dataset)
    # DATASET_OPTS = DATA_OPTS.get(p.dataset,{})
    p.level = normalize_level(p.level)
    p.dataset_key = f"{p.dataset}_{p.level}"
    LEVS = OPTS.get("levels", {}).get("options")
    if p.level not in LEVS:
        raise ArgumentTypeError(
            f"For dataset={p.dataset}, level must be one of {sorted(set(LEVS))}; got {p.level!r}"
        )
    # _check_value(p.level, DATASET_OPTS.get('levels',{}).get('options',[]),p.dataset, 'level')

    # Ensure p.products exists
    if not hasattr(p, "products") or p.products is None:
        p.products = []

    # Build a working list
    items = list(p.products)
    if p.product and p.product != "default":
        items = [p.product] + items
    if not items:  # nothing provided → dataset-specific default
        items = [OPTS.get("products", {}).get("default")]  # default for CHM/AQC
    items = _dedup_keep_order(items)

    for item in items:
        _check_value(
            item, OPTS.get("products", {}).get("options", []), p.dataset, "product"
        )

    # Persist back
    p.products = items
    # Keep a single representative in p.product for legacy downstreams
    p.product = items[0]
    # Date logic
    if p.start and p.end and p.end < p.start:
        raise ArgumentTypeError("end date cannot be earlier than start date.")


# Small “spec” for converting raw inputs (strings from CLI/HTTP) to canonical values
CONVERTERS = {
    "version": al.VERSIONS,
    "collection": al.COLLECTIONS,
    "dataset": al.DATASETS,
    "level": al.LEVELS,
    "product": al.PRODUCTS,
    "products": parse_products_value,
    "lat": lat_type,
    "lon": lon_type,
    "start": date_type_allow_latest,
    "end": date_type_allow_latest,
    "days": days_type,
}


def _bad(name: str, value: Any, valid: Optional[list] = None):
    msg = f"Invalid {name!r}: {value!r}"
    if valid:
        msg += f" (valid: {valid})"
    raise ArgumentTypeError(msg)


def _dt_to_str(item):
    if isinstance(item, dt.datetime):
        return item.strftime("%Y%m%d")
    return item  # None or already str


def coerce_inputs(raw: Mapping[str, Any], legacy: bool = False) -> argparse.Namespace:
    """
    Convert raw string-ish inputs (from CLI or HTTP) into canonical values using the same
    converters you'd give to argparse type=...  (case-insensitive aliases included).
    """
    raw = {k: v for k, v in raw.items() if v and v != "None"}
    ns = argparse.Namespace()
    version = al.VERSIONS(raw.get("version"))
    al.set_version(version)

    if legacy:
        CONVERTERS["product"] = al.LEGACY_PRODS
        CONVERTERS["products"] = parse_legacy_products_value

    else:
        CONVERTERS["product"] = al.PRODUCTS
        CONVERTERS["products"] = parse_products_value
    if "datakey" in raw:
        ds, lev = raw.get("datakey").split("_", 1)
        if "dataset" not in raw:
            raw["dataset"] = ds
        if "level" not in raw:
            raw["level"] = lev
    if "start_date" in raw and "start" not in raw:
        raw["start"] = raw["start_date"]
    if "end_date" in raw and "end" not in raw:
        raw["end"] = raw["end_date"]
    if "num_days" in raw and "days" not in raw:
        raw["days"] = raw["num_days"]

    for key, conv in CONVERTERS.items():
        if key in raw and raw[key] is not None:
            setattr(ns, key, conv(raw[key]))
        else:
            # apply same defaults as your CLI (adjust if different)
            default = {
                "version": "v2",
                "collection": "fcast",
                "dataset": "aqc",
                "level": "slv",
                "product": "default",
                "products": [],
                "lat": 38.9,
                "lon": -77.0,
                "start": None,  # latest
                "end": None,
                "days": 1,
            }.get(key)
            setattr(ns, key, default)

    # Normalize aliases like v1/x1 -> slv (same as your CLI path)
    ns.level = normalize_level(ns.level)
    if ns.start and ns.end:
        del_time = ns.end - ns.start
        ns.days = del_time.days
    return ns


def coerce_config(raw: Mapping[str, Any], legacy: bool = False) -> CliConfig:
    ns = coerce_inputs(raw, legacy)
    return validate_and_build_config(ns, legacy)


def validate_and_build_config(
    ns: argparse.Namespace, legacy: bool = False
) -> CliConfig:
    """
    Apply the same cross-field rules as CLI (dataset↔product, date ordering, etc.),
    then return the frozen dataclass for downstream use.
    """
    # Use the same function you already have
    if legacy:
        enforce_constraints_legacy(ns)
    else:
        enforce_constraints(ns)

    # Convert dates to YYYYMMDD strings (or None)
    start = _dt_to_str(ns.start)
    end = _dt_to_str(ns.end)

    return CliConfig(
        version=ns.version,
        collection=ns.collection,
        dataset=ns.dataset,
        level=ns.level,
        product=ns.product,
        products=ns.products,
        lat=ns.lat,
        lon=ns.lon,
        start=start,
        end=end,
        datakey=ns.dataset_key,
        days=ns.days,
    )
