from .theme import theme
from .registry import register
from wxv.myutils import dict_merge
from wxv.conditional import Conditional


@register("wxmapsclassicpub")
class wxmapsclassicpub(theme):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        vort_clevs = Conditional('level')
        self.vort_clevs = vort_clevs
        vort_clevs('default', (4,8,12,16,20))
        vort_clevs(300, (8,12,16,20))
        vort_clevs(200, (8,12,16,20))
        vort_clevs(100, (2,4,6,8,10))
        vort_clevs(50, (2,4,6,8,10))
        vort_clevs(30, (2,4,6,8,10))
        vort_clevs(10, (2,4,6,8,10))

        vort_mask = Conditional('level')
        self.vort_mask = vort_mask
        vort_mask('default', -4)
        vort_mask(300, -8)
        vort_mask(200, -8)

        self.levels = [10, 30, 50, 100, 200, 300, 500, 700, 850]

        self.map1 = dict(line_color=(150,150,150),
                         land_color=(255,255,255),
                         line_width=5)

        self.add_plot('vort', long_name='Vorticity', levels=self.levels,
                 layers=['vorticity', 'vort_contour', 'heights'],
                 title='$level hPa Relative Vorticity [10`a-5`n/sec]'+
                       ' and Heights [dam]')

        self.add_plot('tmpu', long_name='Temperature',
                 level=self.levels, map=self.map1,
                 layers=['temperature', 'heights'],
                 title='$level hPa Temperature [C] and Heights [dam]')

        self.add_layer('vorticity', gxout='shaded',
                       expr='smth9(regrid($field,$res,$res,bl)*100000)',
                       field='_vort', cbar='Vorticity', nsub=4, skip=4,
                       clevs=self.vort_clevs, res=0.25, mask=self.vort_mask)
