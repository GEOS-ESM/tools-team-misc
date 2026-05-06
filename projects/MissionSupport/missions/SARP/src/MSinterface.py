import re
import argparse
import datetime as dt


class Interface(object):

    def __init__(self, description):

        self.parser = argparse.ArgumentParser(description=description)

        self.parser.add_argument(
            "datetime",
            metavar="datetime",
            type=str,
            help="ISO datetime as ccyy-mm-ddThh:mm:ss",
        )
        self.parser.add_argument(
            "config", metavar="config", type=str, help="configuration file (.yml)"
        )
        self.parser.add_argument(
            "--tau", metavar="tau", type=int, required=False, help="hours", default=0
        )

    def get_args(self):

        args = self.parser.parse_args()

        dattim = re.sub("[^0-9]", "", args.datetime + "000000")[0:14]
        idate = int(dattim[0:8])
        itime = int(dattim[8:14])
        tau = args.tau
        time_dt = dt.datetime.strptime(dattim, "%Y%m%d%H%M%S")

        return {
            "datetime": args.datetime,
            "config": args.config,
            "date": idate,
            "time": itime,
            "tau": args.tau,
            "time_dt": time_dt,
        }
