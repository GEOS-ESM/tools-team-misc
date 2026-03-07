from themes.wxmapsclassicpub import *
from themes.registry import *
from wxv.conditional import Conditional

@register("user_override_theme")
class user_override_theme(wxmapsclassicpub):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.add_plot('vort', title='New Vorticity Title')

        self.add_plot('tmpu', title='New Temperature Title')
