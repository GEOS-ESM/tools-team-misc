#! /usr/bin/env python

import os
import sys

import subprocess
from multiprocessing import Pool

import MSinterface
from player import Player
from myutils import read_yaml
from handlers import make_sarp, make_html, make_sarp_stations

# Retrieve command-line arguments.

ui = MSinterface.Interface("Creates SARP movies")
args = ui.get_args()
arg_dt = args["time_dt"]

# Get configuration.
# ==================

config = read_yaml(args["config"])
PLAYLIST = config.get("PLAYLIST")

for play in PLAYLIST:

    cfg = config[play]
    iterator = Player(cfg, time_dt=arg_dt, tloop=False)
    ntasks = cfg.get("ntasks", 24)
    pool = Pool(ntasks)
    pool.map(make_sarp, iterator)

    request = next(iter(iterator))
    make_sarp_stations(request)
    make_html(request)
