
import os
import sys
import argparse
import datetime as dt

def parse_args(args=None):

    parser = argparse.ArgumentParser()

#   # required
#   required = parser.add_argument_group('required arguments')

#   required.add_argument(
#       '--rc', metavar='RESOURCE FILE', required=True,
#       help='Name of resource file'
#   )

    # optional
    parser.add_argument(
        '-i', '--iname', metavar='INPUT', default='',
        help='Input filename (default: %(default)s)',required=True
    )
    parser.add_argument(
        '-o', '--oname', metavar='ONAME', default='',
        help='Output filename (default: %(default)s)',required=True
    )
    parser.add_argument(
        '--hname', metavar='HNAME', default='',required=True,
        help='Name of file containing heights (ETA) (default: %(default)s)'
    )
    parser.add_argument(
        '-v', '--vars', metavar='VARS', default='',required=True,
        help='Comma-separated list of variables (default: %(default)s)'
    )
    parser.add_argument(
        '-l', '--levels', metavar='LEVELS', default='',required=True,
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
    parser.add_argument(
        '--strict', action='store_true', help='Do not extrapolate'
    )
    parser.add_argument(
        '--ground', action='store_true', help='Use ground as reference'
    )
    parser.add_argument(
        '--feet', action='store_true', help='Levels in units of feet'
    )
    parser.add_argument(
        '--alt', action='store_true', help='Use altimetric height'
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
