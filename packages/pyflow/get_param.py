#! /usr/bin/env python

import os
import re
import sys
import datetime as dt
import argparse

import config
import myutils
import gradstime

# Retrieve command-line arguments
# ===============================

parser = argparse.ArgumentParser(description='Retrieves config parameters')
parser.add_argument('datetime', metavar='datetime', type=str,
                    help='ISO datetime as ccyy-mm-ddThh:mm:ss')
parser.add_argument('config', metavar='config', type=str,
                    help='configuration file (.yml)')
parser.add_argument('param', metavar='parameter', type=str,
                    help='parameter name')
parser.add_argument('--tau', metavar='tau', type=int, required=False,
                    help='hours', default=0) 

args = parser.parse_args()

dattim = re.sub('[^0-9]','', args.datetime+'000000')
idate  = int(dattim[0:8])
itime  = int(dattim[8:14])
tau    = args.tau
param  = args.param

# Get configuration.
# ==================

cfg = config.Config()
cfg.read(args.config)
vars = {k:str(v) for k,v in iter(cfg.items()) if not isinstance(v,dict)}

# Get environment definitions
# ===========================

ut = myutils.Utils()
gt = gradstime.GradsTime(idate,itime)

defs = { k:str(v) for k,v in iter(os.environ.items()) }
defs.update(vars)
defs.update(cfg.get('environment',{}))

# Print resolved parameter
# ========================

v = vars.get(args.param, None)
if v is None: sys.exit(1)

print(gt.strftime(ut.replace(v, **defs), tau))

sys.exit(0)
