import os
import shlex
import ftplib
import subprocess
from shutil import copyfile
from multiprocessing import Pool

def eta2prs_process(request):

    fdt = request['fcst_dt']
    tdt = request['time_dt']

    # Execute command to convert model-level to pressure-level

    levels = request['levels'].replace(',',' ')

    iname = tdt.strftime(request['iname'])
    iname = fdt.strftime(iname)

    oname = tdt.strftime(request['oname'])
    oname = fdt.strftime(oname)

    odir = os.path.dirname(oname)

    try:
        os.makedirs(odir, 0o755)
    except:
        pass

    cmd = 'eta2prs.x -tag ' + oname + ' -levs ' + levels + \
          ' -hdf .TRUE. -noquads -eta ' + iname

    cmd = shlex.split(cmd)

    subprocess.call(cmd)

    # Execute command to create aerosol files

    time  = tdt.strftime('%Y%m%d_%H%M')
    ftime = fdt.strftime('%Y%m%d_%H')
    iname = oname + '.' + time + 'z.nc4'
    oname = oname + '.' + ftime + '+' + time + '.V01.nc4'

    vars = request['vars']
    cmd  = ['aerosol.py', '--iname',iname,'--oname',oname,'--vars',vars]

    subprocess.call(cmd)

    # Remove the intermediate pressure-level file.

    os.remove(iname)
