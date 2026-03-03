
import os
import sys
import argparse
import datetime as dt

def parse_args(args=None):

    levels  = [ 1000,975,950,925,900,850,800,750,700,650,
                 600,550,500,450,400,350,300,250,200,150,
                 100,70,50,30,20,10
              ]
    myvars   = "PM25,PM10,PM,DUST,BCOC"

    mylevels = ','.join([str(v) for v in levels])

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--iname', metavar='INPUT', default='',
        help='Input filename (default: %(default)s)',required=True
    )
    parser.add_argument(
        '-o', '--oname', metavar='ONAME', default='',
        help='Output filename (default: %(default)s)',required=True
    )
    parser.add_argument(
        '--aname', metavar='ANAME', default='',
        help='Name of file containing meteorology (default: %(default)s)'
    )
    parser.add_argument(
        '-v', '--vars', metavar='VARS', default=myvars,
        help='Comma-separated list of variables (default: %(default)s)'
    )
    parser.add_argument(
        '-l', '--levels', metavar='LEVELS', default=mylevels,
        help='Comma-separated list of levels (default: %(default)s)'
    )
    parser.add_argument(
        '--time_dt', metavar='YYYYMMDDTHHMMSS', default=None,
        help='Time in ISO format'
    )
    parser.add_argument(
        '--fcst_dt', metavar='YYYYMMDDTHHMMSS', default=None,
        help='Forecast start time in ISO format'
    )
    parser.add_argument(
        '--start_dt', metavar='YYYYMMDDTHHMMSS', default=None,
        help='Start time in ISO format'
    )
    parser.add_argument(
        '--end_dt', metavar='YYYYMMDDTHHMMSS', default=None,
        help='Ending time in ISO format'
    )
    parser.add_argument(
        '--t_deltat', metavar='HOURS', type=int, default='3',
        help='Time increment in hours (default: %(default)s)'
    )

    if not len(args):
        parser.print_help()
        sys.exit(1)

    args = vars(parser.parse_args(args))

    if args['time_dt'] is None:
        now    = dt.datetime.utcnow()
        hour   = int(now.hour / 12) * 12
        hhmmss = "%06d"%(hour*10000,)
        args['time_dt'] = now.strftime('%Y%m%dT' + hhmmss)

    if args['start_dt'] is None:
        args['start_dt'] = args['time_dt']
    else:
        args['time_dt']  = args['start_dt']

    if args['end_dt'] is None:
        args['end_dt'] = args['start_dt']

    if args['fcst_dt']:
        args['fcst_dt'] = dt.datetime.strptime(args['fcst_dt'],'%Y%m%dT%H%M%S')

    args['time_dt']  = dt.datetime.strptime(args['time_dt'],'%Y%m%dT%H%M%S')
    args['start_dt'] = dt.datetime.strptime(args['start_dt'],'%Y%m%dT%H%M%S')
    args['end_dt']   = dt.datetime.strptime(args['end_dt'],'%Y%m%dT%H%M%S')
    args['t_deltat'] = dt.timedelta(hours=args['t_deltat'])

    return args
