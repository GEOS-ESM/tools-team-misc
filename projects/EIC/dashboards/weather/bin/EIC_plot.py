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

# Create forecast images
# ======================

cfg = cfg_forecast
iterator = Player(cfg, time_dt=arg_dt)
ntasks = cfg.get('ntasks', 20)
pool = Pool(ntasks)
pool.map(make_image, iterator)

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

# Create movie files and publish
# ==============================

cfg = dict(cfg_publish)
movies = cfg['movies']

for movie in cfg['movies']:

    cfg.update(config[movie])
    iterator = Player(cfg, tloop=False, iname=sname, time_dt=arg_dt)
    ntasks = cfg.get('ntasks', 4)
    pool = Pool(ntasks)
    pool.map(make_movie, iterator)

# Remove old files
# ================

cfg = dict(config['EIC_clean_movies'])
iterator = Player(cfg, fname=mname, time_dt=arg_dt)
ntasks = cfg.get('ntasks', 2)
pool = Pool(ntasks)
pool.map(purge, iterator)

cfg = dict(config['EIC_clean_images'])
iterator = Player(cfg, fname=aname, time_dt=arg_dt)
ntasks = cfg.get('ntasks', 2)
pool = Pool(ntasks)
pool.map(purge, iterator)

cfg = dict(config['EIC_clean_images'])
iterator = Player(cfg, fname=fname, time_dt=arg_dt)
ntasks = cfg.get('ntasks', 2)
pool = Pool(ntasks)
pool.map(purge, iterator)

cfg = dict(config['EIC_clean_images'])
iterator = Player(cfg, fname=sname, time_dt=arg_dt)
ntasks = cfg.get('ntasks', 2)
pool = Pool(ntasks)
pool.map(purge, iterator)
