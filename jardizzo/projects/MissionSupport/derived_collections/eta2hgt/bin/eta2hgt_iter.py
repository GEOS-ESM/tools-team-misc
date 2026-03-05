#! /usr/bin/env python

import os
import sys

import interface
from taskmanager import *

task     = TaskManager()
request  = interface.parse_args(sys.argv[1:])
template = ' '.join(sys.argv[1:])

fcst_dt  = request['fcst_dt']
start_dt = request['start_dt']
end_dt   = request['end_dt']
t_deltat = request['t_deltat']

t = start_dt
while t <= end_dt:

    args = t.strftime(template)
    args = fcst_dt.strftime(args)

    task.spawn('eta2hgt.py ' + args)

    t += t_deltat

task.wait()
