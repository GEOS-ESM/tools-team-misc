from .registry import register
from myutils import dict_merge

class wxtheme(object):

    def __init__(self, *args, **kwargs):

        self.plots = {}

    def add_plot(self, name, **kwargs):

        p = self.plots.get(name, {})
        dict_merge(p, kwargs)
        self.plots[name] = p

    def __str__(self):

        return str(self.plots)


@register("wxmapsclassicpub")
class wxmapsclassicpub(wxtheme):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        levels = [10, 30, 50, 100, 200, 300, 500, 700, 850]

        map1 = dict(line_color=(150,150,150),
                    land_color=(255,255,255),
                    line_width=5)

        self.add_plot('vort', long_name='Vorticity', levels=levels,
                 layers=['vorticity', 'vort_contour', 'heights'],
                 title='$level hPa Relative Vorticity [10`a-5`n/sec]'+
                       ' and Heights [dam]')

        self.add_plot('tmpu', long_name='Temperature', level=levels, map=map1,
                 layers=['temperature', 'heights'],
                 title='$level hPa Temperature [C] and Heights [dam]')
