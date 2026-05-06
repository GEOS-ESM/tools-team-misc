from myutils import parse_duration

class Player(object):

    def __init__(self, config, tloop=True, **kwargs):

        self.config = dict(config)
        self.config.update(kwargs)
        self.tloop = tloop

    def __iter__(self):

        playlist = self.config['playlist']

        for p, plist in playlist.items():

            fields = plist.get('fields', [])

            for field in fields:

                play = dict(self.config)
                play.update(plist)

                levels = play.get('levels', [0])
                regions = play.get('regions', [])

                fcst_dt = play.get('fcst_dt', 'PT0H')
                start_dt = play.get('start_dt', 'PT0H')
                end_dt = play.get('end_dt', 'PT0H')
                t_deltat = play.get('t_deltat', 'PT1H')
                time_dt = play['time_dt']

                fcst_dt = time_dt + parse_duration(fcst_dt)
                start_dt = time_dt + parse_duration(start_dt)
                end_dt = time_dt + parse_duration(end_dt)
                delta_t = parse_duration(t_deltat)

                t_start = start_dt
                if not self.tloop: t_start = end_dt

                for level in levels:

                    for region in regions:

                        t = t_start
                        while t <= end_dt:

                            play.update({'field': field,
                                         'region': region,
                                         'level': level,
                                         't_start': t,
                                         't_end': t,
                                         't_deltat': t_deltat,
                                         'fcst_dt': fcst_dt,
                                         'start_dt': start_dt,
                                         'end_dt': end_dt})

                            yield dict(play)

                            t += delta_t
