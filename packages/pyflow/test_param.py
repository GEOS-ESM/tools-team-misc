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
parser.add_argument('-r', '--root', action='store_true',
                     help='time invariant root path')
parser.add_argument('-t', '--template', action='store_true',
                     help='template only')

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

param = vars.get(args.param, None)
if param is None: sys.exit(1)

# Print unresolved parameter
# ==========================

if args.template:
    print(param)
    sys.exit(0)

# Print resolved parameter
# ========================

if not args.root:
    print(gt.strftime(ut.replace(param, **defs), tau))
    sys.exit(0)

# Print root path of parameter
# ============================

value = gt.strftime(param, tau)
while value != param:
    param = os.path.dirname(param)
    value = gt.strftime(param, tau)

if args.template:
    print(param)
else:
    print(ut.replace(param, **defs))

sys.exit(0)
