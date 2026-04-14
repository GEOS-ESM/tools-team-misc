#!/usr/bin/env python
import argparse
import sys
import cartopy.crs as ccrs
import numpy as np

from earthnow.wxmaps_config import WxMapsConfig, StyleConfig
from earthnow.wxmaps_plotting import WxMapPlotter
from earthnow.wxmaps_utils import (
    get_output_filepath,
    parse_date_string,
)

print("success")
