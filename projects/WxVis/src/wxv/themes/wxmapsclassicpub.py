from wxv.themes.theme import theme
from wxv.themes.registry import register
from wxv.myutils import dict_merge
from wxv.conditional import Conditional


@register("wxmapsclassicpub")
class wxmapsclassicpub(theme):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        attr = self.attributes

        vort_clevs = Conditional('level')
        attr.vort_clevs = vort_clevs
        vort_clevs('default', (4,8,12,16,20))
        vort_clevs(300, (8,12,16,20))
        vort_clevs(200, (8,12,16,20))
        vort_clevs(100, (2,4,6,8,10))
        vort_clevs(50, (2,4,6,8,10))
        vort_clevs(30, (2,4,6,8,10))
        vort_clevs(10, (2,4,6,8,10))

        vort_mask = Conditional('level')
        attr.vort_mask = vort_mask
        vort_mask('default', -4)
        vort_mask(300, -8)
        vort_mask(200, -8)

        regrid = Conditional('region')
        attr.regrid = regrid
        regrid('midatl', 0.1)
        regrid('pacnw', 0.25)

        attr.levels = [10, 30, 50, 100, 200, 300, 500, 700, 850]

        attr.map1 = dict(line_color=(150,150,150),
                         land_color=(255,255,255),
                         line_width=5)

        self.define_plots()

    def define_plots(self):

        super().define_plots()

        attr = self.attributes

        self.add_plot('vort', long_name='Vorticity', levels=attr.levels,
                 layers=['vorticity', 'vort_contour', 'heights'],
                 title='$level hPa Relative Vorticity [10`a-5`n/sec]'+
                       ' and Heights [dam]')

        self.add_plot('tmpu', long_name='Temperature',
                 level=attr.levels, map=attr.map1,
                 layers=['temperature', 'heights'],
                 title='$level hPa Temperature [C] and Heights [dam]')

        self.add_layer('vorticity', gxout='shaded',
                       expr='smth9(regrid($field,$res,$res,bl)*100000)',
                       field='_vort', cbar='Vorticity', nsub=4, skip=4,
                       clevs=attr.vort_clevs, res=attr.regrid,
                       mask=attr.vort_mask)
