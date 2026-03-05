#! /usr/bin/env python

import re
import os
import sys
import time
import subprocess

# Capture modules state
# =====================

#proc = subprocess.Popen(['csh', '-c', 'module list'], stdout=subprocess.PIPE, shell=False)
#(out, err) = proc.communicate()

#modules = out.split(')')

#for i in range(1,len(modules)):
    #for name in modules[i].split():
        #try:
            #int(name)
        #except:
            #print ("module load " + name)

# Capture local variables from
# exporting shell
# ============================

for line in sys.stdin:

    match = re.match(r'\s*(\w+)\s+(.+)$',line)
    if line.strip() == '_': break

    if match:
        name = match.group(1)
        value = match.group(2)

        try:
            float(value)
            print ('set ' + name + ' = ' + str(value))
        except:
            print ('set ' + name + ' = ' + '"' + value + '"')

# Capture environment variables
# =============================

with open(sys.argv[1],'r') as f:

    for line in f.readlines():

        result = re.findall(r'setenv\s+\w+',line)
        for r in result:

            name = r.split()[-1]
            value = os.environ.get(name, None)
            if not value: continue

            try:
                float(value)
                print ('setenv ' + name + ' ' + str(value))
            except:
                print ('setenv ' + name + ' "' + value + '"')


#for name, value in iter(os.environ.items()):

#    if re.match(r'^SLURM_',name): continue
#    if re.match(r'^PBS_',name): continue
#    if re.match(r'^_ModuleTable',name): continue
#    if re.match(r'^BASH_FUNC',name): continue

#    try:
#        float(value)
#        print ('setenv ' + name + ' ' + str(value))
#    except:
#        print ('setenv ' + name + ' "' + value + '"')

sys.exit(0)
