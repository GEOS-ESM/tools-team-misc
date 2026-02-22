import os
import re
import glob
import fnmatch
import tarfile
import datetime as dt
from shutil import copyfile

def FindFiles(paths, names, **kwargs):

    listing = {}

    for path in paths:

        if os.path.isdir(path):
            fh = DirFileHandler(path, **kwargs)
        elif not os.path.isfile(path):
            continue
        elif tarfile.is_tarfile(path):
            fh = TarFileHandler(path, **kwargs)
        else:
            continue

        for name in names:

            files = fh.find(name)
            for file in files:
                fname = os.path.basename(file)
                if fname in listing: continue

                listing[fname] = FileObj(fh, name, file)

    return listing.values()

class FileObj(object):

    def __init__(self, fh, name, file):

        self.fh   = fh
        self.name = name
        self.file = file

    def copy(self, dest, **kwargs):
        self.fh.copy(self.file, dest, **kwargs)

    def __str__(self):
        return self.file

class FileHandler(object):

    def __init__(self, path, **kwargs):

        self.path      = path
        self.min_age   = kwargs.get('min_age', 0)
        self.min_time  = kwargs.get('min_time', None)
        self.link      = kwargs.get('link',False)

class DirFileHandler(FileHandler):

    def find(self, name):
            
        pathname = name
        if not os.path.isabs(name): pathname = os.path.join(self.path, name)
        filelist = glob.glob(pathname)

        files = []
        for file in filelist:

            if not os.path.isfile(file): continue

            mt   = dt.datetime.fromtimestamp(os.stat(file).st_mtime)
            now  = dt.datetime.now()
            age  = (now - mt).total_seconds()

            if age < self.min_age: continue

            if self.min_time:
                file_time = self.query_time(file)
                if file_time < self.min_time: continue

            files.append(file)

        return files

    def copy(self, src, dest, **kwargs):

        link = kwargs.get('link', False)

        if os.path.isdir(dest): dest = os.path.join(dest,os.path.basename(src))

        if link:
            if os.path.islink(dest): os.remove(dest)
            if not os.path.isfile(dest): os.symlink(src,dest)
        else:
            copyfile(src, dest)

    def query_time(self, filename):

        months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
                  'sep', 'oct', 'nov', 'dec']

        seconds = {'mn': 60, 'hr': 3600, 'dy': 86400}

        with open(filename, 'r') as f:
            lines = f.readlines()

        for line in lines:

            line = line.lower().strip()

            if 'tdef' in line:
                line = line.split()

                dim = int(line[1])
                tstring = line[3]
                tinc = line[4]

                time, date = tstring.split('z')
                day = int(date[0:2])
                month = months.index(date[2:5]) + 1
                year = int(date[5:])

                if ':' in time:
                    hour, minute = [int(n) for n in time.split(':')]
                else:
                    hour, minute = (int(time), 0)

                stride = [c for c in tinc if c.isdigit()]
                stride = int(''.join(stride))
                units = [c for c in tinc if not c.isdigit()]
                units = ''.join(units)
                incsec = dt.timedelta(seconds=seconds[units]*stride)
                totsec = incsec * (dim-1)
                return dt.datetime(year, month, day, hour, minute) + totsec

        raise Exception("File is not a GrADS control file")

class TarFileHandler(FileHandler):

    def __init__(self, path, **kwargs):

        super(TarFileHandler, self).__init__(path, **kwargs)

        self.fh = tarfile.TarFile(path, 'r')
        self.members = self.fh.getmembers()

    def find(self, name):

        rexpr = fnmatch.translate(name)

        files = []
        for member in self.members:

            if not re.search(rexpr, member.name): continue

            mt   = dt.datetime.fromtimestamp(member.mtime)
            now  = dt.datetime.now()
            age  = (now - mt).total_seconds()

            if age < self.min_age: continue

            files.append(member.name)

        return files

    def copy(self, src, dest, **kwargs):

        name = os.path.basename(src)

        if os.path.isdir(dest):
           dir = dest
           dest = os.path.join(dest,name)
        else:
           dir = os.path.dirname(dest)

        self.fh.extract(src, dir)

        oname = os.path.join(dir, name)
        if oname != dest: os.rename(oname, dest)
