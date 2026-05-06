#!/usr/bin/env python3
"""
Run flextra in NORMAL mode
"""

import os,sys
from datetime import datetime, timedelta
from   dateutil.parser import parse         as isoparser
from glob import glob
import subprocess

if __name__ == "__main__":
#    lon = 126.99
#    lat = 37.55
#    startname = 'Seoul'

#    lon = 120.98
#    lat = 14.599
#    startname = 'Manila'

    lon = 101.6841
    lat = 3.1319
    startname = "Kuala_Lampur"

#    lon = 100.5018
#    lat = 13.7563
#    startname = "Thailand"


    alts = 100, #500, 1000
    startdate = '2015-02-01T00:00:00'
    enddate   = '2019-03-30T00:00:00'
    tdt       = timedelta(days=-5)

    opendap = '/home/pcastell/opendap/fp/opendap/assim/'
    ctlFile = opendap + 'inst3_3d_asm_Nv'

    # get filename template
    f = open(ctlFile,'r')
    dset = f.readline().split()[1]
    f.close()

    # Edit pathnames
    dsetRoot = '/'.join(dset.split('/')[:-4]) + '/'
    dsetPath = '/'.join(dset.split('/')[-4:]) 
    f = open('pathnames','r')
    data = f.readlines()
    f.close()

    data[2] = dsetRoot +'\n'
    data[4] = dsetRoot +'\n'
    data[6] = dsetRoot +'\n'

    f = open('pathnames','w')
    f.writelines(data)
    f.close()

    # get list of dates
    sdate = isoparser(startdate)
    edate = isoparser(enddate)
    dt    = timedelta(days=1)

    datelist = []
    while sdate <= edate:
        # we're only doing Januray 15 - March 15
        mindate = datetime(sdate.year,1,15,00)
        maxdate = datetime(sdate.year,3,15,00)
        if (sdate >= mindate) and (sdate <= maxdate):
            datelist.append(sdate) 

        sdate += dt

    dt = timedelta(hours=3)
    for date in datelist:
        # Edit COMMAND file
        f = open('options/COMMAND','r')
        data = f.readlines()
        f.close()

        data[19] = date.strftime('   %Y%m%d 000000\n')
#        data[23] = date.strftime('   %Y%m%d 090000\n')
        data[23] = date.strftime('   %Y%m%d 000000\n')

        f = open('options/COMMAND','w')
        f.writelines(data)
        f.close()

        # Edit AVAILABLE file
        sd = min(date,date+tdt) + timedelta(days=-1)
        ed = max(date,date+tdt) + timedelta(days=1)

        f = open('AVAILABLE','w')
        f.write('DATE     TIME         FILENAME     SPECIFICATIONS\n')
        f.write('YYYYMMDD HHMISS\n')
        f.write('________ ______      __________      __________\n')
        while sd <= ed:
            yy = sd.strftime('%Y')
            mm = sd.strftime('%m')
            dd = sd.strftime('%d')
            hh = sd.strftime('%H')
            yyyymmdd_hh = sd.strftime('%Y%m%d %H0000      ')
            filename = dsetPath.replace('%y4',yy).replace('%m2',mm).replace('%d2',dd).replace('%h2',hh).replace('%n2','00')
            line = yyyymmdd_hh + filename + '\n'
            f.write(line)

            sd += dt
        f.close()

        for alt in alts:
            # Edit STARTPOINTS File
            f = open('options/STARTPOINTS','r')
            data = f.readlines()
            f.close()

            data[26] = '{lon:9.4f}\n'.format(lon=lon)
            data[29] = '{lat:9.4f}\n'.format(lat=lat)
            data[38] = '{alt:10.3f}\n'.format(alt=alt)
            data[41] = '{}_{}_{}\n'.format(startname,date.strftime('%Y%m%d_%H'),alt)

            f = open('options/STARTPOINTS','w')
            f.writelines(data)
            f.close()


            # run flexpart
            p = subprocess.Popen(['./FLEXTRA_GEOS'])
            p.wait()
