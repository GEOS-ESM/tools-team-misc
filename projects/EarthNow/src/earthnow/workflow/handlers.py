import os
import glob
import shlex
import shutil
import tempfile 
import subprocess

from earthnow.workflow.utils import str_replace

def make_image(request):

    defs = { k:v for k,v in request.items() if not isinstance(v, dict) }

    time_dt = request['time_dt']
    oname = time_dt.strftime(request['pngname'])
    oname = str_replace(oname, **defs)

    options = request['options']
    options['output'] = oname
    options['resolution'] = request['pngsize']

    cmd = [
        f"--{k} {v}"
        for k, v in options.items()
        if v is not None and not isinstance(v, bool) and k[0] != '_'
    ]
    cmd += [f"--{k}" for k, v in options.items() if isinstance(v, bool)]

    cmd = " ".join(cmd)
    cmd = request['driver'] + " " + cmd

    print(cmd)

    subprocess.call(cmd, shell=True, executable='/bin/bash')

def make_movie(request):
    
    tmpdir = tempfile.TemporaryDirectory()

    defs = { k:v for k,v in request.items() if not isinstance(v, dict) }
    defs['glob'] = os.path.join(tmpdir.name, '*.png')
    defs['resolution'] = request['pngsize']

    iname = request['pngname']
    iname = str_replace(iname, **defs)
    txtname = request['txtname']
    txtname = str_replace(txtname, **defs)
    
    time_dt = request['time_dt']
    fcst_dt = request['fcst_dt']
    start_dt = request['start_dt']
    end_dt = request['end_dt']
    delta_t = request['delta_t']

    os.makedirs(os.path.dirname(txtname), mode=0o755, exist_ok=True)
    with open(txtname, 'w') as f:

        t = start_dt
        while (t <= end_dt):

            src = t.strftime(iname)
            if os.path.isfile(src):

                dst = os.path.join(tmpdir.name, os.path.basename(src))
                os.symlink(src, dst)
                print(src)

                cdattim = t.strftime("%Y-%m-%d, %H%M UTC")
                if t >= fcst_dt:
                    tau  = round((t - fcst_dt).total_seconds() / 3600)
                    cdattim = f"{cdattim} [Forecast Hour: {tau:03d}]"

                f.write(cdattim+'\n')

            t += delta_t

    resolutions = request['resolutions']

    for resolution in request['mp4size']:

        defs['resolution'] = resolution
        defs['frame_size'] = resolutions[resolution]
        oname = request['mp4name']
        oname = str_replace(oname, **defs)

        tname = os.path.join(tmpdir.name, os.path.basename(oname))
        odir = os.path.dirname(oname)
        os.makedirs(odir, mode=0o755, exist_ok=True)

        cmd = 'ffmpeg -loglevel debug -threads 6 -pattern_type glob -r $frame_rate -i "$glob" -y -s $frame_size -c:v libx264 -pix_fmt yuv420p -preset ultrafast -crf $quality $tname'

        defs.update({'oname': oname, 'tname': tname})
        cmd = str_replace(cmd, **defs)
        print(cmd)

        subprocess.call(cmd, shell=True, executable='/bin/bash')
        shutil.move(tname, oname)

    tmpdir.cleanup()

def purge(request):

    defs = { k:v for k,v in request.items() if not isinstance(v, dict) }
        
    iname = request['pngname']
    iname = str_replace(iname, **defs)

    time_dt = request['time_dt']
    fcst_dt = request['fcst_dt']
    start_dt = request['start_dt']
    end_dt = request['end_dt']
    delta_t = request['delta_t']
        
    t = start_dt
    while (t <= end_dt):
    
        src = t.strftime(iname)
        if os.path.isfile(src):
            os.remove(src)

        t += delta_t
