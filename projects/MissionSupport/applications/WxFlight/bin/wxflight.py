#! /usr/bin/env python

import os
import re
import sys
import glob
import shutil
import argparse
import datetime as dt

from flight_software import find_forecast_file
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
collection_file = os.path.join(campaign_path, 'collection.yml')
flight_plan_dir = os.path.join(campaign_path, 'flight_plans')
flight_schedule_file = os.path.join(campaign_path, 'flight_schedule.yml')

resource = read_yaml(resource_file)

# Create the output flight directories

out_dir = time_dt.strftime(resource['WX_FLIGHT_PATH'])
src_out_dir = os.path.join(out_dir, 'src')
data_out_dir = os.path.join(out_dir, 'data')

make_dirs(src_out_dir)
make_dirs(data_out_dir)

# Get the latest flight schedule

schedule = read_yaml(flight_schedule_file)
record = schedule.get('schedule', {})

for period, sched in record.items():

    start_date = period.split('-')[0]
    end_date = period.split('-')[-1]

    apply = True
    if start_date and idate < int(start_date):
        apply = False

    if end_date and idate > int(end_date):
        apply = False

    if apply:
        schedule.update(sched)

print(schedule['aircraft'])
print(schedule['flight_times'])
print(schedule['flight_plans'])

aircraft = schedule.get('aircraft', ['DC8', 'snapshot'])
flight_times = schedule.get('flight_times', ['PT0H'])
flight_plans = schedule.get('flight_plans', ['default'])

# Generate flight times

times = []
for tod in flight_times:
    departure = ref_date+parse_duration(tod)
    times.append(departure.strftime("%Y-%m-%dT%H:%M:%S"))

flight_times = ' '.join(times)

# Generate flight paths

task = TaskManager(ntask=14)
os.chdir(data_out_dir)

for plan in flight_plans:

    pattern = os.path.join(flight_plan_dir, plan, '*.csv')
    filelist = glob.glob(pattern)

    for flight_plan in filelist:

        shutil.copy(flight_plan, src_out_dir)

        for plane in aircraft:

            cmd = ['-v -p', plane, flight_plan, flight_times]
            print('wp2traj.py ' + ' '.join(cmd))
            task.spawn('wp2traj.py ' + ' '.join(cmd))

task.wait()

# Interpolate model data collections in space and time to each point along
# the flight paths.

model_data = read_yaml(collection_file)[args.model]
stride = parse_duration(model_data.get('stride', 'PT12H'))
collections = model_data['collections']

pattern = os.path.join(data_out_dir, '*.csv')
track_files = glob.glob(pattern)

for track_file in track_files:

    with open(track_file, 'r') as f:
        lines = f.readlines()

    dattim = lines[1].split(',')[0]
    dattim = re.sub('[^0-9]', '', dattim+'000000')[0:14]
    start_dt = dt.datetime.strptime(dattim,'%Y%m%d%H%M%S')

    for cname, cdata in collections.items():

        vars = cdata.get('vars', '*')
        cfile = find_forecast_file(start_dt, stride, cdata['file'])

        if not cfile:
            continue
      
        name, ext = os.path.splitext(track_file)
        out_file = '.'.join([name, args.model, cname, 'nc'])
        rc_file = '.'.join([name, args.model, cname, 'rc'])

        with open(rc_file, 'w') as f:
            f.write(','.join(vars) + ': ' + cfile + '\n')

        cmd = ['-d 60 -o', out_file, '-v -t csv -I -r', rc_file, track_file]
        print ('trj_sampler.py ' + ' '.join(cmd))
        task.spawn('trj_sampler.py ' + ' '.join(cmd))

task.wait()

sys.exit(0)
