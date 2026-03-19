import copy
import re
import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Tuple, Any
import yaml
from cfapi.config import loader
from .aliases import (
    Aliases,
    make_aliases_from_dict,
    _dedup_keep_order,
    parse_list_value,
)

__all__ = ["CONFIG_DIR", "ROOT", "versionConfig", "ProductAliases"]
here = Path(__file__).parent
CONFIG_DIR = here / "config_files"
ROOT = here.parent
logger = logging.getLogger("cfapi.api_config")


def _dedup_preserve_order(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


class versionConfig:
    def __init__(
        self,
        version: Literal["v1", "v2"] = "v2",
        config_data: Optional[dict] = None,
        config_params: Optional[dict] = None,
        config_dir: Optional[Path | str] = None,
        dataset_key: Optional[str] = None,
    ):
        config_dir = config_dir or CONFIG_DIR
        config_dir = Path(config_dir)
        self.config_dir = config_dir
        if not config_data:
            config_data = loader.load_yaml("config_data.yml")
        if not config_params:
            config_params = loader.load_yaml("config_params.yml")

        data = self._check_version(config_data, version)
        params_in = self._check_version(config_params, version)
        if dataset_key and dataset_key in params_in.keys():
            params_in = params_in.get(dataset_key)
        self.dataset = data.get("dataset", {})
        self.data = data
        self.formats = data.get("format", {})
        self.params_in = params_in
        self.params = self._normalize_params(params_in)
        self.version = version

    def _parse_ds_fields(self, ds_dict):
        fields = {}
        for k, v in ds_dict.items():
            if k == "basic_fields":
                basic_d = {f: {"title": f, "fields": [f]} for f in v}
                fields.update(basic_d)
            elif k == "default":
                fields.update({k: v})
            else:
                fields[k] = {"title": v.get("title", k), "fields": v.get("fields", [k])}
        return fields

    def _parse_defs(self, fields):
        defs = {}
        for k, v in fields.items():
            defs[k] = {
                "units": v.get("units", ""),
                "scaling": v.get("scaling", "1"),
                "var": v.get("var", k),
            }
        return defs

    def _normalize_params(self, params):
        params_norm = {}
        for k, v in params.items():
            if k == "fields":
                params_norm[k] = self._parse_defs(v)
            else:
                params_norm[k] = self._parse_ds_fields(v)
        return params_norm

    def _check_version(self, config: dict | None, version: str) -> dict:
        if version in config:
            return config.get(version)
        else:
            return config

    @property
    def latest_dir(self) -> Path:
        return Path(self.dataset.get("latest")).resolve()

    @property
    def base_dir(self) -> Path:
        return self.latest_dir.parents[1]


class ProductAliases(Aliases):
    """
    Inherit Aliases' callable/type= behavior, but:
      - build groups from products_config
      - expose `opts` (dataset -> list of canonical product names)
      - customize help_suffix() to show per-dataset options
    """

    def __init__(self, products_config: Any, dataset_key: Optional[list | str] = None):

        # keep a reference for later if you want
        self.config = products_config
        self.version = products_config.version
        self.defaults = products_config.defaults
        products_config.set_attrs(dataset_key)
        # opts is what your validators want: {dataset: [canonicals...]}
        self.opts: Dict[str, List[str]] = products_config.opts
        # groups drives the alias mapping (canonical, alias1, alias2, ...)
        groups = list(products_config.tups.values())
        super().__init__(groups=groups)  # builds the LUT

    def get_default(self, ds: str, backup: str = "NO2") -> str:
        return self.defaults.get(ds, backup)

    def get_opts(self, ds: str, backup: list = []) -> str:
        return self.opts.get(ds, backup)

    # You can reuse Aliases._info_help; just customize the suffix:
    def help_suffix(self) -> str:
        lines = ["Available CF products:"]
        for ds, allowed in self.opts.items():
            # only include the groups whose canonical is allowed for this dataset
            ds_groups = [g for g in self.groups if g and g[0] in allowed]
            lines.append(f"{ds} options: {self._info_help(ds_groups)}")
        return "\n".join(lines)

    # Handy helpers for validators (optional niceties):
    def choices(self) -> List[str]:
        """All canonical choices across datasets."""
        return [g[0] for g in self.groups if g]

    def choices_for(self, dataset: str) -> List[str]:
        """Canonical choices limited to a dataset key present in self.opts."""
        return list(self.opts.get(dataset, []))


class ProductConfig(versionConfig):
    """
    Build product alias groups from versioned config and an aliases.yml.

    Surfaces:
      - self.opts: Dict[str, List[str]]       # dataset -> canonical products
      - self.groups: Dict[str, Tuple[str,...]]# canonical -> (canonical, *aliases)
      - self.all_keys: List[str]              # stable order of canonicals across datasets
      - self.all_tups: Dict[str, Tuple[str,...]] # alias tuples keyed by canonical

    Utilities:
      - to_aliases() -> Aliases               # baseline Aliases
      - to_product_aliases() -> ProductAliases# dataset-aware help + opts
    """

    def __init__(
        self,
        version: Optional[Literal["v1", "v2"]] = None,
        config_dir: Optional[Path | str] = None,
        dataset_key: Optional[str] = None,
    ):
        config_dir = Path(config_dir or CONFIG_DIR)
        self.version = version
        version = version or "v2"
        super().__init__(
            version=version, config_dir=config_dir, dataset_key=dataset_key
        )

        # ---- load alias definitions ----
        aliases_doc: Dict[str, Any] = loader.load_yaml("aliases.yml")
        products_aliases = (aliases_doc or {}).get("products", {})
        if not isinstance(products_aliases, dict):
            raise ValueError("aliases.yml -> products must be a dict")

        self._def_aliases: Dict[str, List[str]] = {
            k: (v or []) for k, v in products_aliases.items() if k != "global_subs"
        }
        self._global_subs: Dict[str, List[str]] = (
            products_aliases.get("global_subs", {}) or {}
        )

        # ---- pull defaults for datasets ----
        self.defaults = {k: v.get("default", "NO2") for k, v in self.params.items()}

        # ---- derive dataset -> allowed canonicals (opts) without mutation ----
        self.all_opts: Dict[str, List[str]] = self._get_all_allowed(self.params)
        self.opts = self.all_opts
        # ---- flatten canonical product keys (order preserved across datasets) ----
        self.all_keys: List[str] = _dedup_preserve_order(
            x for vals in self.all_opts.values() for x in vals
        )

        # ---- build alias tuples per canonical ----
        self.all_tups: Dict[str, Tuple[str, ...]] = {
            field: self._make_alias_tuple(field) for field in self.all_keys
        }

        # convenient mirror: canonical -> tuple
        self.all_groups: Dict[str, Tuple[str, ...]] = self.all_tups

        # sanity checks
        self._validate_alias_integrity()
        self.set_attrs(dataset_key)

    # --------------------------
    # Key helpers
    # --------------------------
    def set_attrs(self, dataset_key: Optional[str] = None):
        if not dataset_key:
            self.opts = self.all_opts
            self.keys = self.all_keys
            self.tups = self.all_tups
            self.groups = self.all_groups
        else:
            self.opts = {
                k: v for k, v in self.all_opts.items() if k.startswith(dataset_key)
            }
            self.keys = self._set_keys(self.opts)
            self.tups = self._set_tups(self.keys)
            self.groups = self.tups
        self.dataset_keys = list(self.opts.keys())

    def _set_keys(self, opts):
        return _dedup_preserve_order(x for vals in opts.values() for x in vals)

    def _set_tups(self, keys):
        return {field: self._make_alias_tuple(field) for field in keys}

    # --------------------------
    # Construction helpers
    # --------------------------
    def _get_all_allowed(self, params: Mapping[str, Any]) -> Dict[str, List[str]]:
        opts = self._get_allowed(params)
        if self.version in ["v1", "v2"]:
            return opts
        self.version = "universal"
        v1 = versionConfig("v1", self.config_dir)
        optsv1 = self._get_allowed(v1.params)
        keys = _dedup_preserve_order(list(opts.keys()) + list(optsv1.keys()))

        opts_out = {}
        for key in keys:
            opts_out[key] = _dedup_preserve_order(opts.get(key) + optsv1.get(key))
        return opts_out

    def _get_allowed(self, params: Mapping[str, Any]) -> Dict[str, List[str]]:
        """
        For each dataset block in params, combine basic_fields + keys(fields) (non-destructive).
        """
        out: Dict[str, List[str]] = {}
        for dataset, fields in params.items():
            if dataset == "fields":
                continue
            if not isinstance(fields, Mapping):
                continue
            out[dataset] = self._get_dataset_allowed(fields)
        return out

    def _get_dataset_allowed(self, fields: Mapping[str, Any]) -> List[str]:
        # Copy to avoid accidental mutation
        f = dict(fields)  # shallow copy is fine
        basic = list(f.get("basic_fields", []))
        # non-basic entries excluding helper/meta keys
        # treat keys that map to dicts as “complex field blocks”
        others = [k for k in f.keys() if k not in ("basic_fields",)]
        # preserve order across basic then others, but de-dup in case of repetition
        return _dedup_preserve_order([*basic, *others])

    def _add_subs(self, aliases: List[str], key: str, subs: List[str]) -> List[str]:
        out = list(aliases)
        for a in aliases:
            if key in a:
                out.extend(a.replace(key, s) for s in subs)
        return _dedup_preserve_order(out)

    def get_aliases(self, field: str) -> List[str]:
        """
        Build the alias list for a canonical field:
          1) start from defined aliases in aliases.yml
          2) add generated substitutions via global_subs
          3) always include the lowercase canonical as first alias base
        """
        defined = list(self._def_aliases.get(field, []))
        base = field.lower()
        # fast path: no global_subs apply
        if not any(k in base for k in self._global_subs.keys()):
            return _dedup_preserve_order([*defined])

        # generate variations from global_subs
        generated = [base]
        for key, subs in self._global_subs.items():
            generated = self._add_subs(generated, key, subs or [])
        return _dedup_preserve_order([*defined, *generated])

    def _make_alias_tuple(self, field: str) -> Tuple[str, ...]:
        aliases = self.get_aliases(field)
        return (field, *aliases) if aliases else (field,)

    # --------------------------
    # Public APIs to wire into argparse / validators
    # --------------------------
    def to_aliases(self) -> Aliases:
        """Build a plain Aliases from all tuples."""
        return Aliases(list(self.all_tups.values()))

    def to_product_aliases(self) -> ProductAliases:
        """Build a ProductAliases (inherits Aliases) with dataset-aware help & opts."""
        # from aliases import ProductAliases  # avoid cycles if split across files
        return ProductAliases(self)

    # --------------------------
    # Validation / diagnostics
    # --------------------------
    def _validate_alias_integrity(self) -> None:
        # 1) Each canonical in opts must have a tuple group
        missing = [c for c in self.all_keys if c not in self.all_tups]
        if missing:
            raise ValueError(f"Missing alias tuples for: {missing}")

        # 2) Warn/raise if aliases reference unknown canonicals in def_aliases
        unknown = [k for k in self._def_aliases.keys() if k not in self.all_keys]
        # optional: only warn; here we keep it strict to avoid stale config errors
        if unknown:
            # It’s okay if they exist but simply aren’t allowed in this version.
            # If that’s expected, downgrade this to a log.warning.
            pass

        # 3) Optional: detect collisions across generated aliases (case-insensitive)
        # Build a reverse LUT and detect multi-map collisions
        lut: Dict[str, str] = {}
        collisions: List[Tuple[str, str, str]] = []
        for canon, tup in self.all_tups.items():
            for alias in tup:
                key = alias.casefold()
                prev = lut.get(key)
                if prev is None:
                    lut[key] = canon
                elif prev != canon:
                    collisions.append((alias, prev, canon))
        if collisions:
            details = "; ".join(
                f"{a!r} -> {p!r} vs {c!r}" for a, p, c in collisions[:5]
            )
            raise ValueError(
                f"Alias collisions detected: {details} (and possibly more)"
            )


class apiAliases:
    def __init__(
        self, version: Optional[str] = None, config_dir: Optional[str | Path] = None
    ):
        config_dir = Path(config_dir or CONFIG_DIR)
        data = loader.load_yaml("options.yml")
        aliases_in = data.get("aliases", {})
        self.config_dir = config_dir
        self.legacy_opts = data.get("options", {})
        self.DEFAULTS = data.get("defaults")
        self.COLLECTIONS = make_aliases_from_dict(aliases_in, "collections")
        self.DATASETS = make_aliases_from_dict(aliases_in, "datasets")
        self.LEVELS = make_aliases_from_dict(aliases_in, "levels")
        self.LEGACY_PRODS = make_aliases_from_dict(aliases_in, "legacy_products")
        self.VERSIONS = make_aliases_from_dict(aliases_in, "versions")
        self.set_version(version)

    def set_version(self, version: Optional[str] = None):
        pconfig = ProductConfig(version=version)
        self.PRODUCTS = pconfig.to_product_aliases()


aliases = apiAliases()
