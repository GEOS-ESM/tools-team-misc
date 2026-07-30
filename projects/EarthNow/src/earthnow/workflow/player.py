import copy
from earthnow.workflow.utils import parse_duration


class Player(object):

    def __init__(self, configuration, task_name, ref_dt, tloop=True, seamless=False, **kwargs):

        self.config = configuration
        self.tloop = tloop
        self.seamless = seamless
        self.time_dt = ref_dt
        self.options = dict(kwargs)

        self.products = configuration["products"]
        self.streams = configuration["streams"]

        self.plays = self.get_plays(configuration, task_name)

    def get_plays(self, config, play):

        plays = []
        playlist = config.get(play, {})
        if isinstance(playlist, list):
            for play in playlist:
                plays += self.get_plays(config, play)
        else:
            return [playlist]

        return plays

    def add_user_options(self, request):

        user_options = { k:v for k,v in self.options.items() if v }
        request.update(user_options)

    def __iter__(self):

        for play in self.plays:

            request = copy.deepcopy(self.config)
            request.update(play)

            self.add_user_options(request)

            products = request["products"]
            streams = request["streams"]
            regions = request["regions"]

            for region in regions:

                request["region"] = region

                for product in products:

                    request["product"] = product

                    for name in streams:

                        stream = self.streams[name]
                        readers = stream.get('streams', [name])
                        if self.seamless:
                            readers = [name]

                        for reader in readers:

                            options = {}
                            request["options"] = options
                            options["map-type"] = region
                            options["data-reader"] = reader
                            options.update(self.products[product])
                            request.update(self.streams[reader])

                            self.add_user_options(request)

                            ftime = request.get("ftime", "PT0H")
                            stime = request.get("stime", "PT0H")
                            etime = request.get("etime", "PT0H")
                            tinc = request.get("tinc", "PT1H")

                            time_dt = self.time_dt
                            fcst_dt = time_dt + parse_duration(ftime)
                            start_dt = time_dt + parse_duration(stime)
                            end_dt = time_dt + parse_duration(etime)
                            delta_t = parse_duration(tinc)

                            request["delta_t"] = delta_t
                            request["fcst_dt"] = fcst_dt
                            request["start_dt"] = start_dt
                            request["end_dt"] = end_dt
                            request["ref_dt"] = time_dt

                          # if self.streams[reader].get("ftime", None):
                            options["fdate"] = fcst_dt.strftime("%Y%m%d_%Hz")

                            t_start = start_dt
                            if not self.tloop:
                                t_start = end_dt

                            t = t_start
                            while t <= end_dt:

                                options["pdate"] = t.strftime("%Y%m%d_%H%Mz")
                                request["time_dt"] = t

                                yield copy.deepcopy(request)

                                t += delta_t
