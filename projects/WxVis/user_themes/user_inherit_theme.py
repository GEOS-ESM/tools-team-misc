from themes.wxmapsclassicpub import *
from themes.registry import *
from wxv.conditional import Conditional

@register("usertheme")
class usertheme(wxmapsclassicpub):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.add_plot('myvort', long_name='Vorticity', levels=self.levels,
                 layers=['vorticity', 'vort_contour', 'heights'],
                 title='$level hPa Relative Vorticity [10`a-5`n/sec]'+
                       ' and Heights [dam]')

        self.add_plot('mytmpu', long_name='Temperature', level=self.levels,
                 map=self.map1,
                 layers=['temperature', 'heights'],
                 title='$level hPa Temperature [C] and Heights [dam]')

        self.add_layer('myvorticity', gxout='shaded',
                       expr='smth9(regrid($field,$res,$res,bl)*100000)',
                       field='_vort', cbar='Vorticity', nsub=4, skip=4,
                       clevs=self.vort_clevs, res=0.25, mask=self.vort_mask)
