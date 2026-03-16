from wxv.themes.wxmapsclassicpub import *
from wxv.themes.registry import *
from wxv.conditional import Conditional

@register("user_inherit_theme")
class user_inherit_theme(wxmapsclassicpub):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.define_plots()

    def define_plots(self):

        super().define_plots()

        attr = self.attributes

        self.add_plot('vort-2', long_name='Vorticity', levels=attr.levels,
                 layers=['vorticity', 'vort_contour', 'heights'],
                 title='$level hPa Relative Vorticity [10`a-5`n/sec]'+
                       ' and Heights [dam]')

        self.add_plot('tmpu-2', long_name='Temperature', levels=attr.levels,
                 map=attr.map1,
                 layers=['temperature', 'heights'],
                 title='$level hPa Temperature [C] and Heights [dam]')
