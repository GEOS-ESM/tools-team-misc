#! /usr/bin/env python

import os
import sys

import subprocess
from multiprocessing import Pool

import EICinterface
from player import Player
from myutils import read_yaml
from handlers import make_image, annotate, make_movie, purge

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
mname = cfg_publish['oname']

# Create analysis images
# ======================

cfg = cfg_analysis
iterator = Player(cfg, time_dt=arg_dt)
ntasks = cfg.get('ntasks', 20)
pool = Pool(ntasks) 
pool.map(make_image, iterator)
