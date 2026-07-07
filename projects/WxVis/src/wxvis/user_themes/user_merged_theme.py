from wxvis.themes.wxmapsclassicpub import *
from wxvis.themes.registry import *
from wxvis.conditional import Conditional


@register("user_merged_theme")
class user_merged_theme(theme):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        p = wxmapsclassicpub()
        self.merge(p)

        self.define_plots()

    def define_plots(self):

        super().define_plots()

        self.add_plot("vort-2", title="New Vorticity Plot")

        self.add_plot("tmpu-2", title="New Temperature Plot")
