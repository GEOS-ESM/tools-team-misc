import os
import datetime as dt

def find_forecast_file(time_dt, stride, file_template, window=None):

    if not window: window = dt.timedelta(days=1)
    t_dt = dt.datetime(time_dt.year, time_dt.month, time_dt.day) - window

    if t_dt+stride <= t_dt:
        stride = dt.timedelta(hours=12)

    fcst_file = None
    while t_dt <= time_dt:

        file = t_dt.strftime(file_template)
        if os.path.isfile(file):

            start_dt, end_dt = get_file_times(file)
            if start_dt > time_dt: break

            fcst_file = file

        t_dt += stride

    return fcst_file

def get_file_times(filename):

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
            time_dt = dt.datetime(year, month, day, hour, minute)
            return (time_dt, time_dt+totsec)

    raise Exception("File is not a GrADS control file")
