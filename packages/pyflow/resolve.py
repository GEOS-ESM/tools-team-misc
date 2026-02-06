#! /usr/bin/env python

import os
import re
import sys
import argparse
import datetime as dt

import config
import myutils
import gradstime

# Retrieve command-line arguments
# ===============================

parser = argparse.ArgumentParser(description='Resolves time tokens and variable names')
parser.add_argument('datetime', metavar='datetime', type=str,
                    help='ISO datetime as ccyy-mm-ddThh:mm:ss')
parser.add_argument('--config', metavar='config', type=str, required=False,
                    help='configuration file (.yml)', default='')
parser.add_argument('--tau', metavar='tau', type=int, required=False,
                    help='hours', default=0)

args = parser.parse_args()

dattim = re.sub('[^0-9]','', args.datetime+'000000')
idate  = int(dattim[0:8])
itime  = int(dattim[8:14])
tau    = args.tau

# Get environment definitions
# ===========================

if args.config:
    cfg = config.Config()
    cfg.read(args.config)
else:
    cfg = {}

defs = { k:str(v) for k,v in iter(os.environ.items()) }
defs.update( {k:str(v) for k,v in iter(cfg.items()) if not isinstance(v,dict)} )

ut = myutils.Utils()
gt = gradstime.GradsTime(idate,itime)

for line in sys.stdin:
    print (gt.strftime(ut.replace(line.rstrip(), **defs),tau))
