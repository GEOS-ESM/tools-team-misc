#! /usr/bin/env python

import os
import sys
from multiprocessing import Pool

import interface
from handlers import *

request  = interface.parse_args(sys.argv[1:])

fcst_dt  = request['fcst_dt']
start_dt = request['start_dt']
end_dt   = request['end_dt']
t_deltat = request['t_deltat']

args = []
t = start_dt
while t <= end_dt:

    request['time_dt'] = t
    args.append(dict(request))

    t += t_deltat

pool = Pool(8)
pool.map(eta2prs_process, args)

sys.exit(0)
