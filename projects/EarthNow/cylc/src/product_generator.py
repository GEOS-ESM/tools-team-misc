"""
EarthNow product configuration and generator.

Sets configuration parameters and executes product generation.
"""

import os

class ProductGenerator(object):
    """
    Default products generator.
    """
    
    def __init__(self,
                 cmd,
                 driver=None,
                 nproc=1,
                 product=None,
                 data_reader=None,
                 region=None,
                 style=None,
                 fdate=None,
                 pdate=None,
                 start_pdate=None,
                 end_pdate=None,
                 plot_path=None):

        self.driver = cmd
        self.options = {}

        self.options['driver'] = None # Remove from options.
        self.options['product'] = product
        self.options['data-reader'] = data_reader
        self.options['nproc'] = nproc
        self.options['style'] = style
        self.options['map-type'] = region
        self.options['fdate'] = fdate
        self.options['pdate'] = pdate
        self.options['start-pdate'] = start_pdate
        self.options['end-pdate'] = end_pdate
        self.options['base-path'] = plot_path

    def exe(self):

        cmd = [f'--{k} {v}' for k,v in self.options.items() if v is not None and not isinstance(v, bool)]
        cmd += [f'--{k}' for k,v in self.options.items() if isinstance(v, bool)]

        cmd = ' '.join(cmd)
        print(self.driver + ' ' + cmd)
        os.system(self.driver + ' ' + cmd)
