"""
EarthNow product configuration and generator.

Sets configuration parameters and executes product generation.
"""

from registry import register

class Product(object):
    """
    Default products generator.
    """
    
    def __init__(self,
                 nproc=1,
                 data_reader=None,
                 region=None,
                 fdate=None,
                 pdate=None,
                 start_pdate=None,
                 end_pdate=None):

        self.driver = 'plotall.py'
        self.options = {}

        self.options['data_reader'] = data_reader
        self.options['nproc'] = nproc
        self.options['map-style'] = region
        self.options['fdate'] = fdate
        self.options['pdate'] = pdate
        self.options['start-pdate'] = start_pdate
        self.options['end-pdate'] = end_pdate

    def exe(self):

        cmd = [f'--{k} {v}' for k,v in self.options.items() if v is not None and not isinstance(v, bool)]
        cmd += [f'--{k}' for k,v in self.options.items() if isinstance(v, bool)]

        cmd = ' '.join(cmd)
    #   os.system(self.driver + ' ' + cmd)
        print(self.driver + ' ' + cmd)

@register("max_reflectivity")
class max_reflectivity(Product):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.options['product'] = 'max_reflectivity'
        self.options['style'] = 'light'

@register("sea_level_pressure")
class sea_level_pressure(Product):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.options['product'] = 'sea_level_pressure'
        self.options['style'] = 'light'

@register("vorticity_heights_500mb")
class vorticity_heights_500mb(Product):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.options['product'] = 'vorticity_heights_500mb'
        self.options['style'] = 'light'
