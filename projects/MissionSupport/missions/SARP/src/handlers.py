import os
import glob
import shlex
import shutil
import subprocess

from myutils import str_replace
from html_templates import *

def make_sarp(request):

    make_images(request)
    make_movie(request)
    remove_images(request)

def make_sarp_stations(request):

    defs = get_defs(request)

    # Create station image

    cmd = str_replace('wxmap.py $station_plot', **defs)
    print(cmd)
    subprocess.call(cmd, shell=True, executable='/bin/bash')

def make_images(request):

    defs = get_defs(request)

    options = []
    if request.get('plot_only', 0):
        options.append('--plot_only')
    if request.get('lights_off', 0):
        options.append('--lights_off')

    cmd = 'wxmap.py --theme $themes --stream $stream --fcst_dt $fcst_dt --start_dt $start_dt --end_dt $end_dt --t_deltat $t_deltat --field $field --region $region --level $level --oname $oname --geometry $geometry'

    cmd = str_replace(cmd, **defs)
    cmd = cmd + ' ' + ' '.join(options)
    subprocess.call(cmd, shell=True, executable='/bin/bash')

def make_movie(request):

    defs = get_defs(request)

    # Create movie file

    iname = str_replace(request['oname'], **defs)
    defs['pattern'] = os.path.join(os.path.dirname(iname), "*.png")
    
    mname = str_replace(request['mname'], **defs)
    odir = os.path.dirname(mname)
    os.makedirs(odir, mode=0o755, exist_ok=True)

    cmd = 'ffmpeg -loglevel debug -threads 6 -pattern_type glob -r $frame_rate -i "$pattern" -y -r $frame_rate -s $geometry -c:v libx264 -pix_fmt yuv420p -preset ultrafast -crf $quality $mname'

    cmd = str_replace(cmd, **defs)

    subprocess.call(cmd, shell=True, executable='/bin/bash')


def remove_images(request):

    defs = get_defs(request)

    oname = str_replace(request['oname'], **defs)
    dir = os.path.dirname(oname)

    files = glob.glob(os.path.join(dir, "*.png"))

    for file in files:
        os.remove(file)

def make_html(request):

    level_labels = { 0:'SLV' }
    defs = get_defs(request)
    regions = request['regions']

    stream = request['stream']

    for region in regions:

        defs['region'] = region
        hname = os.path.join('$pub_dir', '$region', 'index.html')
        hname = str_replace(hname, **defs)

        with open(hname, 'w') as f:

            header = str_replace(html_header, **defs)
            f.write(header)

            stations = str_replace(sarp_stations, **defs)
            f.write(stations)

            f.write('<table class="table-spacing">')

            playlist = request['playlist']
            for p, plist in playlist.items():

                play = dict(request)
                play.update(plist)

                fields = play['fields']
                levels = play['levels']

                defs['section'] = play['section']
                section = str_replace(html_section, **defs)
                f.write(section)

                for field in fields:
                    f.write("<tr>\n")
                    defs['field'] = field
                    defs['FIELD'] = field.upper()
                    row_header = str_replace(html_row_header, **defs)
                    f.write(row_header)
                    for level in levels:
                        defs['button'] = level_labels.get(level, str(level))
                        defs['level'] = str(level)
                        row = str_replace(html_row, **defs)
                        f.write(row)
                    f.write("</tr>\n")

            trailer = str_replace(html_trailer, **defs)
            f.write(trailer)

def get_defs(request):

    data_dir = request['data_dir']
    pub_dir = request['pub_dir']
    fcst_dt = request['fcst_dt']
    start_dt = request['start_dt']
    end_dt = request['end_dt']
    themes = request['themes']
    data_dir = request['data_dir']
    pub_dir = request['pub_dir']
    fdate = request['fdate']
    ftitle = request['ftitle']

    defs = dict(request)
    defs['data_dir'] = fcst_dt.strftime(data_dir)
    defs['pub_dir'] = fcst_dt.strftime(pub_dir)
    defs['fcst_dt'] = fcst_dt.strftime('%Y%m%dT%H%M')
    defs['start_dt'] = start_dt.strftime('%Y%m%dT%H%M')
    defs['end_dt'] = end_dt.strftime('%Y%m%dT%H%M')
    defs['themes'] = ' --theme '.join(themes)
    defs['fdate'] = fcst_dt.strftime(fdate)
    defs['ftitle'] = fcst_dt.strftime(ftitle)

    return defs
