#! /usr/bin/env python

import os
import sys

import subprocess
from multiprocessing import Pool

import EICinterface
from player import Player
from myutils import read_yaml
from handlers import make_image, annotate, make_movie

# Retrieve command-line arguments.

ui = EICinterface.Interface('Creates EIC Images')
args = ui.get_args()
arg_dt = args['time_dt']

# Get configuration.
# ==================

config = read_yaml(args['config'])

cfg_analysis = config['EIC_analysis']
cfg_forecast = config['EIC_forecast']
cfg_annotate = config['EIC_annotate']
cfg_publish = config['EIC_publish']

aname = cfg_analysis['oname']
fname = cfg_forecast['oname']
sname = cfg_annotate['oname']

# Annotate analysis images
# ========================

cfg = dict(cfg_analysis)
cfg.update(cfg_annotate)
iterator = Player(cfg, iname=aname, time_dt=arg_dt)
ntasks = cfg.get('ntasks', 20)
pool = Pool(ntasks)
pool.map(annotate, iterator)

# Annotate forecast images
# ========================

cfg = dict(cfg_forecast)
cfg.update(cfg_annotate)
iterator = Player(cfg, iname=fname, time_dt=arg_dt)
ntasks = cfg.get('ntasks', 20)
pool = Pool(ntasks)
pool.map(annotate, iterator)
