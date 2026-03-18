# cf_file_tools.py

import os
import json
import math
import datetime as dt
from types import SimpleNamespace
from pathlib import Path
import logging
from typing import Dict, List, Mapping, Sequence, Optional, Union, Literal, Any

import pandas as pd

from .cf_plots import make_plotter
from .cfmap_config import dict2df, file_template, FILE_OUTPUT_DIR

ISO_FMT = "%Y-%m-%dT%H:%M:%S"
FileFormat = Literal["csv", "ascii", "txt", "xlsx", "excel", "json"]
logger = logging.getLogger('cf_map.file_maker')

def _parse_time_list(times: Sequence[Union[str, dt.datetime]]) -> List[dt.datetime]:
    """
    Normalize a sequence of ISO strings or datetimes into naive datetimes.

    - Tries datetime.fromisoformat first (tolerant).
    - Falls back to strptime using ISO_FMT.
    """
    out: List[dt.datetime] = []
    for t in times:
        if isinstance(t, dt.datetime):
            out.append(t)
            continue
        s = str(t)
        try:
            out.append(dt.datetime.fromisoformat(s))
        except Exception:
            out.append(dt.datetime.strptime(s, ISO_FMT))
    return out

def is_fill(x,fill):
    return isinstance(x, (float, int)) and math.isclose(x, fill, rel_tol=1e-6)

    

class CFFile:
    """
    Package CF (chem/met) time series into XLSX/TXT/JSON with consistent headers,
    units, and filenames.

    Parameters
    ----------
    data : Mapping[str, Any]
        Must contain keys: 'schema', 'values', 'time'.
        - schema: metadata used by `file_info()`
        - values: mapping variable -> data (nested by level for multilevel datasets)
        - time: list of ISO strings or datetimes
    products : Sequence[str]
        Variables to include. If the first is 'MET', treated as meteorology.
    nan : float | str, optional
        Missing-data fill for outputs (default 1e15).
    output_dir : str | Path, optional
        Base output directory; defaults to FILE_OUTPUT_DIR.
    """

    def __init__(self, data: Mapping[str, Any], products: Sequence[str]|str, **kwargs: Any) -> None:
        if isinstance(products, str):
            products = [products]
        # Required components
        self.schema: Dict[str, Any] = dict(data.get("schema", {}))
        self.values: Dict[str, Any] = dict(data.get("values", {}))
        self.time: List[Union[str, dt.datetime]] = list(data.get("time", []))

        if not self.schema or not self.values or not self.time:
            raise ValueError("data must include non-empty 'schema', 'values', and 'time'.")

        self.dt_list: List[dt.datetime] = _parse_time_list(self.time)
        if 'pm25' in [p.lower() for p in products]:
            products += ['PM25_RH35_GCC', 'PM25_RH35']
        self.products: List[str] = [str(p) for p in products]
        self.nan_val: Union[float, str] = kwargs.get("nan", 1.0e15)

        # Choose keys to write
        if self.products and self.products[0].upper() == "MET":
            self.keys: List[str] = list(self.values.keys()) 
        else:
            prod_lower = {p.lower() for p in self.products}
            self.keys = [k for k in self.values.keys() if str(k).lower() in prod_lower]
        if not self.keys:
            raise ValueError(f"No matching product keys found in values for {self.products!r}")

        # Build DataFrame
        self.df: pd.DataFrame = dict2df(self.values, self.dt_list, self.keys)

        # Derived info and paths
        self.info: SimpleNamespace = self.file_info()
        self.set_filename(**kwargs)

    # ---------------------------
    # Pathing / filenames / info
    # ---------------------------
    def set_filename(self, **kwargs: Any) -> None:
        """
        Configure `self.filename` and `self.output_dir`, creating the directory.

        Directory segment:
          - 'assim' : <t0>_<tN>
          - 'fcst'  : <t0>
        """
        dt_str = "%Y%m%dT%H%M"
        if self.info.collection == "assim":
            time_dir = f"{self.dt_list[0].strftime(dt_str)}_{self.dt_list[-1].strftime(dt_str)}"
        else:
            time_dir = self.dt_list[0].strftime(dt_str)

        type_parts = [self.info.datatype, self.info.levels, self.info.collection]
        self.filename = file_template.format(
            version=self.info.version,
            type="_".join(type_parts),
            product="+".join(self.info.prods),
            lat=self.info.lat,
            lon=self.info.lon,
            ext="{ext}",
        )

        output_dir = Path(kwargs.get("output_dir") or FILE_OUTPUT_DIR)
        self.output_dir = output_dir / time_dir
        self.output_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

    def file_info(self) -> SimpleNamespace:
        """
        Build a compact metadata bundle from `self.schema` and a sample variable.

        Returns
        -------
        SimpleNamespace with attributes:
          version, lat, lon, dataset
          units (dict), longname (dict)
          datatype ('met'/'chm'), prods (list), levels ('slv'/'p23'/'v72')
          collection ('assim' or 'fcst'), time (string), description (string)
        """
        info = SimpleNamespace(
            version=self.schema.get("version", "v2"),
            lat=self.schema.get("near_lat"),
            lon=self.schema.get("near_lon"),
            units={"time": "%Y-%m-%d %H:%M:%S", **self.schema.get("units", {})},
            longname={"time": "isotime", **self.schema.get("longname", {})},
            dataset=self.schema.get("dataset"),
        )

        if self.products and self.products[0].upper() == "MET":
            info.datatype = "met"
            info.prods = ["MET"]
        else:
            info.datatype = "chm"
            info.prods = list(self.keys)

        # Infer level type from first variable
        sample_var = self.values.get(self.keys[0])
        if isinstance(sample_var, dict):
            levs = len(sample_var.keys())
            if levs == 72:
                info.levels = "v72"
                info.units["lev"] = "Model Level"
                info.longname["lev"] = "Model Level"
            else:
                info.levels = "p23"
                info.units["lev"] = "hPa"
                info.longname["lev"] = "Model Pressure Level"
        else:
            info.levels = "slv"

        if "time range" in self.schema:
            info.collection = "assim"
            info.time = f'Time Range: {self.schema.get("time range")}'
            info.description = "Replay of " + ", ".join(self.keys)
        else:
            info.collection = "fcst"
            info.time = f'Forecast init time: {self.schema.get("forecast initialization time")}'
            info.description = "5-day forecast of hourly values of " + ", ".join(self.keys)

        return info

    # ---------------------------
    # Writers
    # ---------------------------
    def write_ascii(self) -> Path:
        """Write a text file with a compact header block plus CSV rows."""
        if isinstance(self.df.index, pd.MultiIndex):
            idx_names = [n or "index" for n in self.df.index.names]
        else:
            idx_names = [self.df.index.name or "time"]

        variables = idx_names + list(self.keys)
        info_lines = [
            ",".join(variables),
            ",".join([self.info.longname.get(v) for v in variables]),
            ",".join([self.info.units.get(v) for v in variables]),
            f"dataset: {self.info.dataset}",
            self.info.time,
            f"FLUID API output for {', '.join(self.info.prods)} at lat: {self.info.lat}, lon: {self.info.lon}",
        ]
        headers = [f"{len(variables)},{len(info_lines) + 1}"] + info_lines
        fill = self.nan_val               
        fmt_str = "%.1e"                     
        df_out = self.df.copy()
        df_out = df_out.map(lambda x: fmt_str % fill if is_fill(x,fill) else x)
        file_out = self.output_dir / self.filename.format(ext="txt")
        with file_out.open("w", encoding="utf-8") as f:
            for line in headers:
                f.write(line + "\n")
            self.df.to_csv(f, header=False, na_rep=str(self.nan_val),float_format=None)
        try:
            os.chmod(file_out, 0o664)
        except Exception:
            pass
        return file_out

    def write_slv_xlsx(self, products: Sequence[str], header: Dict[str, Any], writer: pd.ExcelWriter) -> None:
        """Write single-level (surface) variables into one sheet."""
        header = {**header, "Product": ", ".join(products)}
        title = "Sheet1"
        start_row = len(header) + 2

        self.df[list(products)].fillna(self.nan_val).to_excel(
            writer, sheet_name=title, header=True, startrow=start_row
        )

        ws = writer.sheets[title]
        for i, (k, v) in enumerate(header.items()):
            ws.write(i, 0, k)
            ws.write(i, 1, v)

        ws.write(start_row - 1, 0, "Units")
        for col_idx, prod in enumerate(products, start=1):
            ws.write(start_row - 1, col_idx, str(self.info.units.get(prod)))

    def write_levs_xlsx(self, product: str, header: Dict[str, Any], writer: pd.ExcelWriter) -> None:
        """Write a pressure/levelized variable into its own sheet (lev->columns)."""
        header = {**header, "Product": product, "Units": self.info.units.get(product)}
        title = product
        start_row = len(header) + 2

        df_out = self.df[product].unstack(level="lev")  # rows=time, columns=lev
        df_out.fillna(self.nan_val).to_excel(writer, sheet_name=title, header=True, startrow=start_row)

        ws = writer.sheets[title]
        for i, (k, v) in enumerate(header.items()):
            ws.write(i, 0, k)
            ws.write(i, 1, v)

    def write_xlsx(self) -> Path:
        """Write XLSX (slv: one sheet; p23/v72: one sheet per product)."""
        file_out = self.output_dir / self.filename.format(ext="xlsx")
        header = {
            "Long Name": self.info.time,
            "Initial Time": self.dt_list[0].strftime("%Y-%m%dT%H:%M:%Sz"),
            "Latitude": self.info.lat,
            "Longitude": self.info.lon,
            "CF Version": self.info.version,
            "Missing Value": self.nan_val,
        }

        with pd.ExcelWriter(file_out, engine="xlsxwriter") as writer:
            if self.info.levels == "slv":
                self.write_slv_xlsx(self.keys, header, writer)
            else:
                for p in self.keys:
                    self.write_levs_xlsx(p, header, writer)

        try:
            os.chmod(file_out, 0o664)
        except Exception:
            pass
        return file_out

    def write_json(self) -> Path:
        """Write JSON with a cleaned header and selected variables."""
        file_out = self.output_dir / self.filename.format(ext="json")
        header = dict(self.schema)
        header.update(
            {
                "longname": [self.info.longname.get(k) for k in self.keys],
                "product": list(self.keys),
                "units": {k: self.info.units.get(k) for k in self.keys},
                "description": self.info.description,
            }
        )
        d_out = {"schema": header, "values": {k: self.values.get(k) for k in self.keys}, "time": self.time}

        with file_out.open("w", encoding="utf-8") as fp:
            json.dump(d_out, fp, ensure_ascii=False)
        try:
            os.chmod(file_out, 0o664)
        except Exception:
            pass
        return file_out


# ---------------------------
# Convenience helpers
# ---------------------------

def infer_inputs(
        collection: str, 
        dataset: str,
        products: Sequence[str],
        ):
    """
    Chooses the internal plot_type to build plotter &
    dataset key used by your plotter.data bundle.
    """
    if collection in ['assim','replay','assimilation']:
        plot_type = 'replay'
        ds = 'aqc_slv'
    elif 'p23' in dataset or 'v72' in dataset:
        plot_type = 'pres'
        if 'v72' in dataset:
            ds = 'chm_v72'
        else:
            ds = 'chm_p23'
    else:
        plot_type = 'surf'
        ds = 'chm_slv'
    if products[0].lower() == 'met':
        ds = 'met_slv'
    return plot_type, ds

def raw_get_data_file(
    lat: float,
    lon: float,
    products: Sequence[str],
    dataset: str,
    collection: str,
    fmt: FileFormat,
    **kwargs: Any,
) -> Path:
    """
    One-shot helper to fetch data from the plotter and write a file.
    """

    plot_type, ds_key = infer_inputs(collection, dataset, products)

    plotter = make_plotter(lat, lon, plot_type, **kwargs)
    data = plotter.data.data.get(ds_key)
    if data is None:
        raise ValueError(f"No data found for dataset key '{ds_key}'")

    return get_data_file(data, products, fmt)


def get_data_file(data: Mapping[str, Any], products: Sequence[str], fmt: FileFormat) -> Path:
    """
    Build a CFFile from a data bundle and write to the requested format.
    """
    file_base = CFFile(data, products)
    if fmt in {"csv", "ascii", "txt"}:
        return file_base.write_ascii()
    if fmt in {"xlsx", "excel"}:
        return file_base.write_xlsx()
    return file_base.write_json()


# if __name__ == "__main__":
#     from cli_interface import parse_args

#     SCRIPT = Path(__file__).name
#     cfg = parse_args(script=SCRIPT, type="file")
#     plot_type, ds_key = infer_inputs(cfg.collection, cfg.dataset, cfg.product)

#     plotter = make_plotter(cfg.lat, cfg.lon, plot_type, **cfg.args)
#     data = plotter.data.data.get(ds_key)
#     if data is None:
#         raise SystemExit(f"No data found for dataset key '{ds_key}'")

#     file_out = get_data_file(data, cfg.product, cfg.format)
#     print(f"Datafile saved to {str(file_out)}")

