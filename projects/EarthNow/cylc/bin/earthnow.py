#! /usr/bin/env python

import os
import sys

import enow_interface
from product_generator import ProductGenerator
from myutils import read_yaml, parse_duration

# Retrieve command-line arguments.

ui = enow_interface.Interface('Earth Now Plotter')
args = ui.get_args()
time_dt = args['time_dt']
task = args['task']
nproc = args['nproc']

# Get configuration.
# ==================

config = read_yaml(args['config'])
cfg_readers = config['data_readers']
cfg_products = config['products']

# Plot requested data
# ===================

products = config[task]['products']
data_readers = config[task]['data_readers']
regions = config[task].get('regions', [None])
nproc = config[task].get('nproc', None)

for region in regions:

    options = {'map-type' : region, 'nproc' : nproc}

    for product in products:

        driver = cfg_products[product]['driver']
        options.update(cfg_products[product])

        for data_reader in data_readers:

            options.update(cfg_readers[data_reader])

            for t_opt in ['fdate','pdate','start-pdate','end-pdate']:

                delta_t = options.get(t_opt, None)
                if delta_t:
                    t_dt = time_dt + parse_duration(delta_t)
                    options[t_opt] = t_dt.strftime('%Y%m%d_%Hz')
        
            plot = ProductGenerator(driver, options)
            plot.exe()
