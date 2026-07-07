#! /usr/bin/env python

import os
import re
import sys
import glob
import shutil
import argparse
import importlib
import datetime as dt

from multiprocessing import Pool

from curtain_cf import plot
from flight_software import Flight, FlightController
from taskmanager import *
from myutils import read_yaml, str_replace, parse_duration, make_dirs

# Get command-line arguments

parser = argparse.ArgumentParser(description='Weather Flights')

parser.add_argument('datetime', metavar='datetime', type=str,
       help='ISO datetime as ccyy-mm-ddThh:mm:ss')
parser.add_argument('model', metavar='model', type=str,
       help='model name')

args = parser.parse_args()

dattim = re.sub('[^0-9]', '', args.datetime+'000000')[0:14]
idate = int(dattim[0:8])
itime = int(dattim[8:14])
time_dt = dt.datetime.strptime(dattim,'%Y%m%d%H%M%S')
ref_date = dt.datetime.strptime(dattim[0:8],'%Y%m%d')

# Set up environment based on field campaign

os.environ['WX_MODEL'] = args.model
campaign_name = os.environ.get('WX_CAMPAIGN', 'default')
bin_path = os.path.abspath(os.path.dirname(sys.argv[0]))
root_path = os.path.dirname(bin_path)
campaign_path = os.path.join(root_path, campaign_name)
resource_file = os.path.join(campaign_path, 'default.yml')

resource = read_yaml(resource_file)
handlers = resource['HANDLERS']

# Create the output flight directories

in_dir = time_dt.strftime(resource['WX_FLIGHT_PATH'])
out_dir = time_dt.strftime(resource['WX_CURTAIN_PATH'])
in_src_dir = os.path.join(in_dir, 'src')
in_data_dir = os.path.join(in_dir, 'data')

make_dirs(out_dir)

fh = Flight(in_dir)
fc = FlightController()

flights = fh.get_flights()
carryon = {'fcst_dt': time_dt, 'ipath': in_dir, 'opath': out_dir}

handler = importlib.import_module(handlers[args.model])

pool = Pool(24)
    
pool.map(handler.plot, fc.iter(flights, **carryon))
pool.close()
pool.join()
