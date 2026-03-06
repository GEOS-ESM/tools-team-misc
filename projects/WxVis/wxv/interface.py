import re
import argparse
import datetime as dt

class Interface(object):

    def __init__(self, description):

        self.parser = argparse.ArgumentParser(description=description)

        self.parser.add_argument('--theme', metavar='THEME',
                    default=[],action='append', required=True,
                    help='Name of configuration file or directory ' +
                    'referencing a theme')
        self.parser.add_argument('--fcst_dt', metavar='FCST_DT',
                    default=None, help='Forecast date/time in ISO format')   
        self.parser.add_argument('--time_dt', metavar='TIME_DT',
                    default=None, help='date/time in ISO format')   

    def get_args(self):

        args = self.parser.parse_args()

        fcst_dt = args.fcst_dt
        time_dt = args.time_dt

        if args.time_dt is not None:
            dattim = re.sub('[^0-9]', '', args.time_dt+'000000')[0:14]
            idate = int(dattim[0:8])
            itime = int(dattim[8:14])
            time_dt = dt.datetime.strptime(dattim,'%Y%m%d%H%M%S')

        if args.fcst_dt is not None:
            dattim = re.sub('[^0-9]', '', args.fcst_dt+'000000')[0:14]
            idate = int(dattim[0:8])
            itime = int(dattim[8:14])
            fcst_dt = dt.datetime.strptime(dattim,'%Y%m%d%H%M%S')

        return {'fcst_dt': fcst_dt, 'time_dt': time_dt, 'theme': args.theme}
