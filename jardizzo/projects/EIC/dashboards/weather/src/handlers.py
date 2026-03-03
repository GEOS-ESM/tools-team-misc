import os
import shlex
import shutil
import tempfile
import subprocess
from dateutil import tz
from PIL import Image

from imutils import im_paste_file, HersheyDraw
from myutils import str_replace, parse_duration

def make_image(request):

    options = []
    if request.get('plot_only', 0):
        options.append('--plot_only')
    if request.get('lights_off', 0):
        options.append('--lights_off')

    defs = dict(request)
    defs['fcst_dt'] = request.get('fcst_dt').strftime('%Y%m%dT%H%M')
    defs['start_dt'] = request.get('t_start').strftime('%Y%m%dT%H%M')
    defs['end_dt'] = request.get('t_end').strftime('%Y%m%dT%H%M')
    defs['t_deltat'] = int(request.get('t_deltat') / parse_duration('PT1H'))
    defs['geometry'] = request.get('geometry', '1024x768')

    cmd = '$pre_script; wxmap.py --theme $theme --stream $stream --fcst_dt $fcst_dt --start_dt $start_dt --end_dt $end_dt --t_deltat $t_deltat --field $field --region $region --level $level --oname $oname --geometry $geometry'

    cmd = str_replace(cmd, **defs)

    print (cmd + ' ' + ' '.join(options))

  # subprocess.call(cmd.split() + options)
    cmd = cmd + ' ' + ' '.join(options)
    subprocess.call(cmd, shell=True, executable='/bin/bash')

# -----------------------------------------------------------------------------

def annotate(request):

    defs = dict(request)

    time_dt = request['t_start']
    fname = time_dt.strftime(request['iname'])
    fname = str_replace(fname, **defs)
    oname = time_dt.strftime(request['oname'])
    oname = str_replace(oname, **defs)

    os.makedirs(os.path.dirname(oname), mode=0o755, exist_ok=True)

    # Switch timezone to EST/EDST

    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('America/New_York')
    time_dt = time_dt.replace(tzinfo=from_zone).astimezone(to_zone)

    # Set the format for the time string label

    cdattim = time_dt.strftime("%d %b %Y %H:%M:%S %Z")
    print(cdattim)

    # Paste image onto a black canvas
    # Use RGB mode (i.e. no alpha channel for the canvas)

    im_main = Image.open(fname).convert("RGBA")
    im_main = im_main.resize((2048, 1024), Image.LANCZOS)
    
    im_final = Image.new('RGB', (im_main.width, im_main.height), color='black')
    im_final.paste(im_main, (0, 0), im_main)
    im_main.close()

    # Set the font and font color

    font_name = request['font_name']
    font_color = request['font_color'].split()
    font_color = (255, 255, 255)

    # Paste the date shadow box onto the final image

    box_name = request['date_box_name']
    im_paste_file(im_final, box_name, 0, 925)

    # Add the date/time label

    x = 30
    y = 954
    d = HersheyDraw(im_final, font_name, 37, font_color)

    s = cdattim[0:6]
    w, h = d.text_size(s)
    d.draw_text(x, y, s)
    x += 123

    s = cdattim[6:]
    w, h = d.text_size(s)
    d.draw_text(x, y, s)

    # Paste the logo shadow box onto the final image

    box_name = request['logo_box_name']
    im_paste_file(im_final, box_name, 0, 0)

    # Add logos

    x = 51
    y = 18
    logo_name = request['nasa_logo_name']
    xs, ys = im_paste_file(im_final, logo_name, x, y, ysize=50)

    x += xs + 10
    y = 25
    logo_name = request['gmao_logo_name']
    xs, ys = im_paste_file(im_final, logo_name, x, y, ysize=35)

    # Save the final annotated image.

    im_final.save(oname, format='png')

# -----------------------------------------------------------------------------

def make_movie(request):

    tmpdir = tempfile.TemporaryDirectory()

    defs = dict(request) 
    defs['glob'] = os.path.join(tmpdir.name, '*.png')

    time_dt = request['t_end']
    start_dt = request['start_dt']
    end_dt = request['end_dt']
    t_deltat = request['t_deltat']

    iname = str_replace(request['iname'], **defs)
    oname = str_replace(request['oname'], **defs)
    oname = time_dt.strftime(oname)
    tname = os.path.join(tmpdir.name, os.path.basename(oname))
    odir = os.path.dirname(oname)
    os.makedirs(odir, mode=0o755, exist_ok=True)

    t = start_dt
    while (t <= end_dt):

        src = t.strftime(iname)
        dst = os.path.join(tmpdir.name, os.path.basename(src)) 
        os.symlink(src, dst)

        t += t_deltat

    cmd = 'ffmpeg -loglevel debug -threads 6 -pattern_type glob -r $frame_rate -i "$glob" -y -r $frame_rate -s $frame_size -c:v libx264 -pix_fmt yuv420p -preset ultrafast -crf $quality $tname'

    defs.update({'oname': oname, 'tname': tname})
    cmd = str_replace(cmd, **defs)
    
  # subprocess.call(shlex.split(cmd))
    subprocess.call(cmd, shell=True, executable='/bin/bash')

    shutil.move(tname, oname)

    tmpdir.cleanup()

def purge(request):

    defs = dict(request)

    time_dt = request['t_start']

    src = str_replace(request['fname'], **defs)
    src = time_dt.strftime(src)
    if os.path.exists(src):
        print("Removing: ", src)
        os.remove(src)
