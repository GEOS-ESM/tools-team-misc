"""
EarthNow product configuration and generator.

Sets configuration parameters and executes product generation.
"""

import os

class ProductGenerator(object):
    """
    Default products generator.
    """
    
    def __init__(self, cmd, options, **kwargs):

        self.driver = cmd
        self.options = dict(options)
        self.options.update(kwargs)

    def exe(self):

        cmd = [f'--{k} {v}' for k,v in self.options.items() if v is not None and not isinstance(v, bool)]
        cmd += [f'--{k}' for k,v in self.options.items() if isinstance(v, bool)]

        cmd = ' '.join(cmd)
        print(self.driver + ' ' + cmd)
 #      os.system(self.driver + ' ' + cmd)
