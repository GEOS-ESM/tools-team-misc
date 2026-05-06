
"""
 Class to read FLEXTRA trajectoryoutput in NORMAL mode
"""

import os, sys
import numpy as np
from datetime import datetime,timedelta

SDS = 'sec','lon','lat','alt','starttime','starthour'

class NORMAL(object):
    def __init__(self,Path,verb=False):

        self.verb = verb

        for sds in SDS:
            self.__dict__[sds] = []

        # Read each trajectory file
        # appending them to the list
        # ------------------------
        if type(Path) is list:
            if len(Path) == 0:
                print("Warning: Empty NORMAL object created")
                return
        else:
            Path = [Path]
        self._readList(Path)

        # convert to arrays
        for sds in SDS:
            self.__dict__[sds] = np.array(self.__dict__[sds])
        
        self.date = []
        self.sdate = []
        for stime,shour,secs in zip(self.starttime,self.starthour,self.sec):
            date = []
            sdate = []
            dh = timedelta(hours=int(shour))
            for sec in secs:
                ds = timedelta(seconds=int(sec))
                tdate = stime + dh + ds
                date.append(tdate)
                sdate.append(tdate.strftime('%Y-%m-%dT%H:%M:%S'))
            self.date.append(np.array(date))
            self.sdate.append(np.array(sdate))
        self.date = np.array(self.date)
        self.sdate = np.array(self.sdate)

#---
    def _readList(self,List):
        """
        Recursively, look for files in list; list items can
        be files or directories.
        """
        for item in List:
            if os.path.isdir(item):      self._readDir(item)
            elif os.path.isfile(item):   self._readTraj(item)
            else:
                print("%s is not a valid file or directory, ignoring it"%item)
#---
    def _readDir(self,dir):
        """Recursively, look for files in directory."""
        for item in os.listdir(dir):
            path = dir + os.sep + item
            if os.path.isdir(path):      self._readDir(path)
            elif os.path.isfile(path):   self._readTraj(path)
            else:
                print("%s is not a valid file or directorey, ignoring it"%item)                

#---
    def _readTraj(self,filename):
        """Reads a FLEXTRA trajectory file"""

        try:
            if self.verb:
                print("[] Working on "+ filename)
            f = open(filename,'r',errors='ignore')
        except:
            print("- %s: not able to open file"%filename)
            return

        # Read file
        # ------------
        # get number of header lines
        lines = f.readlines()
      # nhead = int(lines[0].split()[0])
      # nhead = int(f.readline().split()[0])-1

        for i, line in enumerate(lines):
            if line[0:5] == 'DATE:':
                nhead = i
                break

        first = True
        # read rest of file
        for line in lines[nhead:]:
            if 'DATE' in line:
                if not first:
                    self.sec.append(np.array(SECS))
                    self.lon.append(np.array(LON))
                    self.lat.append(np.array(LAT))
                    self.alt.append(np.array(ALT))                    
                SECS = []
                LON  = []
                LAT  = []
                ALT  = []               
                first = False
                ll = line.split()
                date = ll[1]
                hour = int(ll[3])//10000
                starttime = datetime.strptime(date,'%Y%m%d')
                starttime += timedelta(hours=hour)
                self.starttime.append(starttime)
                self.starthour.append(hour)
            elif 'SECS' in line:
                continue
            else:
#                secs,longit,latit,eta,press,z,zoro,pv,theta,q = line.split()
                secs = line[:9]
                longit = line[9:18]
                latit  = line[18:27]
                alt    = line[49:57]
                SECS.append(int(secs))
                LON.append(float(longit))
                LAT.append(float(latit))
                ALT.append(float(alt))

        f.close()
        self.sec.append(np.array(SECS))
        self.lon.append(np.array(LON))
        self.lat.append(np.array(LAT))
        self.alt.append(np.array(ALT))


