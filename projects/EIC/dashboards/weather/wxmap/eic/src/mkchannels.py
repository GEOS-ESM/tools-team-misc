#! /usr/bin/env python

import sys
import os
import datetime as dt
from dateutil import tz
from PIL import Image, ImageOps

from imutils import im_paste_file, HersheyDraw, create_rounded_rectangle_mask

request = {
    'font_color': '255 255 255',
    'font_name': 'helr45w.ttf',
    'date_box_name': 'date-box-curved-corner-500x100px.png',
    'logo_box_name': 'logo-box-curved-corner-350x75px.png',
    'nasa_logo_name': 'nasa-logo.png',
    'gmao_logo_name': 'gmao-logo-white.png'
    }

start_lon = -180
loninc = int(round(360.0 / 4))
pxtodeg = 2048/360.0

for fname in sys.argv[1:]:

    bname = os.path.basename(fname)
    dirname = os.path.dirname(fname)
    nodes = bname.split('.')

    nodes[1] = 'channels'
    oname = '.'.join(nodes)

    im_final = Image.new('RGB', (2048, 1024), color='black')

    lon = start_lon
    for field in ['pm25sfc','cosfc','o3sfc','nox']:

        nodes[1] = field
        iname = '.'.join(nodes)
        iname = os.path.join(dirname, iname)

        # Set crop/region to extract from image

        if lon >= 180:
            lon -= 360

        lon1 = lon
        lon2 = lon1 + loninc

        x1 = int(round((lon1 + 180.0) * pxtodeg))
        y1 = 0

        x2 = int(round((lon2 + 180.0) * pxtodeg))
        y2 = 1024

        if x2 >= 2048:
            x2 = 2047

        # Extract region and paste onto final image.

        im_main = Image.open(iname).convert("RGBA")
        im_main = im_main.resize((2048, 1024), Image.ANTIALIAS)
        im_channel = im_main.crop((x1, y1, x2, y2))

        print lon1, lon2, x1, y1, x2, y2, iname
        im_final.paste(im_channel, (x1,y1), im_channel)

        lon += loninc

    dattim = oname.split('.')[-2]
    time_dt = dt.datetime.strptime(dattim, "%Y%m%d%H")

    # Switch timezone to EST/EDST

    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('America/New_York')
    time_dt = time_dt.replace(tzinfo=from_zone).astimezone(to_zone)

    # Set the format for the time string label

    cdattim = time_dt.strftime("%d %b %Y %H:%M:%S %Z")
    print(cdattim)

    # Set the font and font color

    font_name = request['font_name']
    font_color = request['font_color'].split()
    font_color = tuple([int(v) for v in font_color])

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

    start_lon += 1
    if start_lon >= 180:
        start_lon = -180
