#!/usr/bin/env python3
"""
Run flextra in NORMAL mode
"""

import os,sys
from datetime import datetime, timedelta
from   dateutil.parser import parse         as isoparser
from glob import glob
import subprocess
import argparse
import csv
from normal import NORMAL
import shutil
import simplekml
import json

if __name__ == "__main__":

    # defaults
    alts = [100]
    direction='both'
    days     = 3
    opendap = '/home/pcastell/opendap/fp/opendap/assim/'
    ctlFile = opendap + 'inst3_3d_asm_Nv'
    inputs = os.getcwd()
    outputs = inputs + '/output'
    # inputs arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("iso_startdate",
           help="start date for trajectory is ISO format")

    parser.add_argument("start_points_file",
           help="input CSV file with starting points name,lon,lat")

    parser.add_argument("--inputs",default=inputs,
           help="inputs directory, default is the current working directory "\
                "(default=%s)"%inputs)

    parser.add_argument("--outputs",default=outputs,
           help="outputs directory, default is the current working directory/output "\
                "(default=%s)"%outputs)

    parser.add_argument("--alts",default=alts,nargs='+',type=float,
           help="list of altitudes (use spaces) to start trajectories from "\
                "(default=%s)"%alts)

    parser.add_argument("--direction",default=direction,
           help="direction for trajectory, can be 'forward','backward', or 'both'" \
                 "(default=%s)"%direction)    

    parser.add_argument("--days",default=days,type=int,
           help="number of days to calcuate for trajectory " \
                 "(default=%i)"%days)

    parser.add_argument("--ctlFile",default=ctlFile,
           help="control file of model wind fields " \
                 "(default=%s)"%ctlFile)

    parser.add_argument("--seamless",action="store_true",
           help="set this flag if you're using a seamless control file " \
                 "(default=False)")

    parser.add_argument("--debug",action="store_true",
           help="debugging, don't delete input and intermdiate files after FLEXTRA run " \
                 "(default=False)")

    args = parser.parse_args()

    # get start date
    sdate = isoparser(args.iso_startdate)

    if args.direction == "forward":
        tdtf       = args.days
        tdtb       = None
    elif args.direction == "backward":
        tdtf       = None
        tdtb       = -1*args.days
    else:
        tdtf      = args.days
        tdtb      = -1*args.days

    
    # read start point CSV
    # ---------------------
    pointsFile = open(args.start_points_file,'r')
    reader = csv.DictReader(pointsFile)
    
    # loop through start points, altitudes, and trajectory directions
    # ---------------------------------------------------------------
    for startp in reader:
        for alt in args.alts:
            for tdt in [tdtf, tdtb]:
                if tdt is not None:
                    if tdt < 0:
                        direction = 'b'
                    else:
                        direction = 'f'

                    ident = '{}_{}_{}_{}'.format(startp['name'],sdate.strftime('%Y%m%d_%H'),alt,direction)
                    if not os.path.exists(ident):
                        os.mkdir(ident)

                    # Edit pathnames input File
                    # -----------------------------
                    # get model filename template (control file)
                    if args.seamless:
                        f = open(args.ctlFile,'r')
                        dsetRoot = f.readline().split()[1][:-4] 
                        ch, assim_s, assim_e, assim_path = f.readline().split()
                        ch, forec_s, forec_e, forec_path = f.readline().split()
                        for l in range(17):
                            f.readline()

                        tdef,t_e,linear,ctl_start, dmin = f.readline().split()
                        ctlsdate = datetime.strptime(ctl_start,'%HZ%d%b%Y')

                        assim_sdate = datetime.strptime(ctl_start,'%HZ%d%b%Y')
                        assim_edate = assim_sdate + timedelta(hours=(3*int(assim_e)))

                        forec_sdate = assim_sdate + timedelta(hours=(3*int(forec_s)))                        
                        
                        f.close()
                    else:
                        f = open(args.ctlFile,'r')
                        dset = f.readline().split()[1]
                        f.close()

                        dsetRoot = '/'.join(dset.split('/')[:-4]) + '/'
                        dsetPath = '/'.join(dset.split('/')[-4:])

                    f = open('pathnames','r')
                    data = f.readlines()
                    f.close()

                    data[0] = args.inputs + '/{}/\n'.format(ident)
                    data[1] = args.outputs + '/\n'
                    data[2] = dsetRoot +'\n'
                    data[3] = args.inputs + '/{}/AVAILABLE\n'.format(ident)
                    data[4] = dsetRoot +'\n'
                    data[5] = args.inputs + '/{}/AVAILABLE\n'.format(ident)
                    data[6] = dsetRoot +'\n'
                    data[7] = args.inputs + '/{}/AVAILABLE\n'.format(ident)

                    pathnames = '{}/pathnames'.format(ident)
                    f = open(pathnames,'w')
                    f.writelines(data)
                    f.close()


                    # Edit STARTPOINTS File
                    # ---------------------
                    f = open('options/STARTPOINTS','r')
                    data = f.readlines()
                    f.close()

                    data[26] = '{lon:9.4f}\n'.format(lon=float(startp['lon']))
                    data[29] = '{lat:9.4f}\n'.format(lat=float(startp['lat']))
                    data[38] = '{alt:10.3f}\n'.format(alt=alt)
                    outFile = '{}_{}_{}_{}'.format(startp['name'],sdate.strftime('%Y%m%d_%H'),alt,direction)
                    data[41] = '{}\n'.format(outFile)

                    f = open('{}/STARTPOINTS'.format(ident),'w')
                    f.writelines(data)
                    f.close()


                    # Edit COMMAND file
                    # ------------------
                    f = open('options/COMMAND','r')
                    data = f.readlines()
                    f.close()

                    if tdt < 0:
                        data[11] = '   -1\n'
                    else:
                        data[11] = '    1\n'

                    tdt_h = str(abs(tdt*24)).zfill(3)
                    data[15] = '   {}0000\n'.format(tdt_h)

                    data[19] = sdate.strftime('   %Y%m%d 000000\n')
                    data[23] = sdate.strftime('   %Y%m%d 000000\n')

                    f = open('{}/COMMAND'.format(ident),'w')
                    f.writelines(data)
                    f.close()

                    # Edit AVAILABLE file
                    # ---------------------
                    dt = timedelta(hours=3) # model time step
                    sd = min(sdate,sdate+timedelta(days=tdt)) + timedelta(days=-1)
                    ed = max(sdate,sdate+timedelta(days=tdt)) + timedelta(days=1)

                    f = open('{}/AVAILABLE'.format(ident),'w')
                    f.write('DATE     TIME         FILENAME     SPECIFICATIONS\n')
                    f.write('YYYYMMDD HHMISS\n')
                    f.write('________ ______      __________      __________\n')
                    while sd <= ed:
                        yy = sd.strftime('%Y')
                        mm = sd.strftime('%m')
                        dd = sd.strftime('%d')
                        hh = sd.strftime('%H')
                        yyyymmdd_hh = sd.strftime('%Y%m%d %H0000      ')

                        if args.seamless:
                            if sd <= assim_edate:
                                filename = assim_path.replace('%y4',yy).replace('%m2',mm).replace('%d2',dd).replace('%h2',hh).replace('%n2','00')
                            else:
                                filename = forec_path.replace('%y4',yy).replace('%m2',mm).replace('%d2',dd).replace('%h2',hh).replace('%n2','00')
                        else:
                            filename = dsetPath.replace('%y4',yy).replace('%m2',mm).replace('%d2',dd).replace('%h2',hh).replace('%n2','00')

                        line = yyyymmdd_hh + filename + '\n'
                        f.write(line)

                        sd += dt
                    f.close()
                    

                    # run flexpart
                    # -------------
                    p = subprocess.Popen(['FLEXTRA_GEOS.x','{}'.format(pathnames)])
                    p.wait()

                    # read formatted outfile, and rewrite output to simple csv & kml & json
                    # ----------------------------------------------------------------
                    tdata = NORMAL('{}/TI_{}'.format(args.outputs,outFile))
                    f = open('{}/TI_{}.csv'.format(args.outputs,outFile),'w')
                    writer = csv.writer(f)
                    fields = ['date','lat','lon','alt']
                    writer.writerow(fields)
                    
                    rows = []
                    urecords = []
                    kml = simplekml.Kml(open=1)
                    linestring = kml.newlinestring(name="{} trajectory".format(direction))
                    coords = []
                    for date,lat,lon,altitude in zip(tdata.sdate[0,:],tdata.lat[0,:],tdata.lon[0,:],tdata.alt[0,:]):
                        rows.append([date,lat,lon,altitude])
                        year,month,dd = date.split('-')
                        day, time = dd.split('T')
                        hour,minute,sec = time.split(':')
                        urecords.append((int(year),int(month),int(day),int(hour),'traj',lat,lon,altitude))
                        coords.append((lon, lat, altitude))
                    linestring.coords = coords
                    linestring.altitudemode = simplekml.AltitudeMode.relativetoground
                    linestring.extrude = 1
                    linestring.style.linestyle.color = simplekml.Color.cyan  
                    linestring.style.linestyle.width = 4 

                    writer.writerows(rows)
                    f.close()
                    kml.save('{}/TI_{}.kml'.format(args.outputs,outFile))

                    f = open('{}/TI_{}.json'.format(args.outputs,outFile),'w')
                    tnew = {}
                    tnew[startp['name']] = urecords
                    tdata = json.dumps(tnew)
                    f.write(tdata)
                    f.close()

                    # if not debug mode, remove input files and FLEXTRA formatted output
                    # ------------------------------------------------------------------
                    if not args.debug:
                        shutil.rmtree(args.inputs + '/{}'.format(ident))
                        os.remove('{}/TI_{}'.format(args.outputs,outFile))
                    

    pointsFile.close()
