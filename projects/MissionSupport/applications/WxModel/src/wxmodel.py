#! /usr/bin/env python

import os
import re
import sys
import yaml
import argparse

from myutils import read_yaml

# Get command-line arguments

parser = argparse.ArgumentParser(description='Weather Model Manager')

parser.add_argument('datetime', metavar='datetime', type=str,
       help='ISO datetime as ccyy-mm-ddThh:mm:ss')
parser.add_argument('model', metavar='model', type=str,
       help='model name')
parser.add_argument('-f', '--force', action='store_true',
       help='force processing')

args = parser.parse_args()

dattim = re.sub('[^0-9]', '', args.datetime+'000000')[0:14]
idate = int(dattim[0:8])
itime = int(dattim[8:14])

time_params = { 'idate' : idate, 'itime' : itime }

# Set up environment based on field campaign

campaign_name = os.environ.get('WXMODEL_CAMPAIGN', 'default')
src_path = os.path.abspath(os.path.dirname(sys.argv[0]))
root_path = os.path.dirname(src_path)
campaign_path = os.path.join(root_path, campaign_name)
resource = os.path.join(campaign_path, 'models.yml')
config_file = os.path.join(campaign_path, 'models', args.model+'.yml')
lib_path = os.path.join(campaign_path, 'lib')
bin_path = os.path.join(campaign_path, 'bin')
os.environ['WXMODEL_BIN_DIR'] = bin_path
#sys.path.append(lib_path)
#sys.path.append(bin_path)

import filehandler

environ = read_yaml(resource)
params = {k:v for k,v in environ.iteritems() if k not in os.environ}
os.environ.update(params)

# Read in model configuration

config = read_yaml(config_file)
time_params['shift_dt'] = int(config.get('shift_dt', 0))

status = 0
handlers = config.get('handlers',None)

if handlers:
    handlers = handlers.split(':')
else:
    handlers = [k for k,v in config.iteritems() if isinstance(v,dict)]

# Execute handlers for acquiring and/or processing the model.

for name in handlers:

    # Retrieve the handler method.

    fh  = filehandler.get_handler(name, config, status)
    if not fh: continue

    # Retrieve configured groups for handler to execute.

    cfg    = config[name]
    groups = [v for k,v in cfg.iteritems() if isinstance(v,dict)]
    if not groups: groups = [cfg]

    # Process all groups. Record bad return codes

    for group in groups:
    
        request = dict(cfg)
        request.update(group)
        request.update(time_params)

        iret = fh.handler(request)
        if iret != 0: status = 2

    print name, ': status = ', status

    # Stop processing if a handler fails (unless forced).

    if not args.force and status != 0: break

if args.force:
    sys.exit(0)
else:
    sys.exit(status)
