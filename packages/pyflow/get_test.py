#! /usr/bin/env python

import os
import re
import sys
import glob
import datetime as dt
import argparse

import config
import myutils
import gradstime
import filehandler as fh

# Retrieve command-line arguments
# ===============================

parser = argparse.ArgumentParser(description='Locates gsi-diag files')
parser.add_argument('datetime', metavar='datetime', type=str,
                    help='ISO datetime as ccyy-mm-ddThh:mm:ss')
parser.add_argument('config', metavar='config', type=str,
                    help='configuration file (.yml)')
parser.add_argument('--tau', metavar='tau', type=int, required=False,
                    help='hours', default=0)

args = parser.parse_args()

dattim = re.sub('[^0-9]','', args.datetime+'000000')
idate  = int(dattim[0:8])
itime  = int(dattim[8:14])
tau    = args.tau

# Get configuration.
# ==================

cfg = config.Config()
cfg.read(args.config)

# Get environment definitions
# ===========================

ut = myutils.Utils()
gt = gradstime.GradsTime(idate,itime)

defs = { k:str(v) for k,v in iter(os.environ.items()) }
defs.update( {k:str(v) for k,v in iter(cfg.items()) if not isinstance(v,dict)} )
defs.update(cfg.get('environment',{}))

# Locate all prerequisite files
# =============================

filelist = []

for k,v in iter(cfg['get_data'].items()):

    if not isinstance(v, dict): continue

    cname      = k
    collection = v
    options    = {k:v for k,v in iter(collection.items())
                         if k not in ['src','dest','files']}

    # Get collection parameters

    paths     = str(collection.get('src', ''))
    paths     = gt.strftime(ut.replace(paths, **defs),tau).split()
    dest      = str(collection.get('dest', ''))
    dest      = gt.strftime(ut.replace(dest, **defs),tau)
    min_count = collection.get('min_count', 1)
    min_time  = collection.get('min_time', None)
    if min_time: min_time = gt.idt + myutils.parse_duration(min_time)
    collection['min_time'] = min_time

    files = {}
    for record in collection['files']:

        record = ut.replace(str(record), **defs)
        record = gt.strftime(record,tau)
        record = record.split() + [None]

        name   = record[0]
        oname  = record[1]

        files[name] = oname

    # Locate Files

    listing = fh.FindFiles(paths, files.keys(), **collection)
    if len(listing) < min_count: sys.exit(1)

    for file in listing:
        print(file.file)

    # Save files to be copied

    if not dest: continue

    for file in listing:

        oname = os.path.join(dest, os.path.basename(file.file))
        if files[file.name]: oname = os.path.join(dest, files[file.name])
        filelist.append( (file, oname, options) )

# Copy designated files

for file, oname, options in filelist:

    print (file)

    try:
        os.makedirs(os.path.dirname(oname), 0o755)
    except:
        pass

    file.copy(oname, **options)

sys.exit(0)
