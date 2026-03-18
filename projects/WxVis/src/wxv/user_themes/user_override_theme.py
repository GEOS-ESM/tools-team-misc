from wxv.themes.wxmapsclassicpub import *
from wxv.themes.registry import *
from wxv.conditional import Conditional

@register("user_override_theme")
class user_override_theme(wxmapsclassicpub):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

      # Change the regrid resolution for the midatl region
      # by overriding the attribute

        regrid = self.attributes.regrid
        regrid('midatl', 0.2)
        regrid('pacnw', 0.25)

        self.define_plots()

    def define_plots(self):

        super().define_plots()

      # Override the titles for the inherited plots

        self.add_plot('vort', title='New Vorticity Title')

        self.add_plot('tmpu', title='New Temperature Title')
