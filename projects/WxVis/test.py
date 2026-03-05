from themes.registry import THEMES
#import themes

print(THEMES)

p = THEMES['wxmapsclassicpub']()

request = dict(level=800)
print (p.expand(p.plots['vort'], request))
print (p.expand(p.layers['vorticity'], request))
