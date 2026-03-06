from themes.wxmapsclassicpub import *
from themes.registry import *
from wxv.conditional import Conditional

@register("usertheme")
class usertheme(wxmapsclassicpub):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        levels = [10, 30, 50, 100, 200, 300, 500, 700, 850]

        map1 = dict(line_color=(150,150,150),
                    land_color=(255,255,255),
                    line_width=5)

        self.add_plot('myvort', long_name='Vorticity', levels=levels,
                 layers=['vorticity', 'vort_contour', 'heights'],
                 title='$level hPa Relative Vorticity [10`a-5`n/sec]'+
                       ' and Heights [dam]')

        self.add_plot('mytmpu', long_name='Temperature', level=levels, map=map1,
                 layers=['temperature', 'heights'],
                 title='$level hPa Temperature [C] and Heights [dam]')

        self.add_layer('myvorticity', gxout='shaded',
                       expr='smth9(regrid($field,$res,$res,bl)*100000)',
                       field='_vort', cbar='Vorticity', nsub=4, skip=4,
                       clevs=vort_clevs, res=0.25, mask=vort_mask)
