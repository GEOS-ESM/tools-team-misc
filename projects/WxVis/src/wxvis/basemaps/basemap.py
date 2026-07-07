import os
import numpy as np
import matplotlib.pyplot as plt

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

xsize=10800
ysize=5400
name = 'shadedrelief_grayscale.10800x5400.png'
extent=[-180, 180, -90, 90]

fig = plt.figure(figsize=(xsize*2/300.0, ysize*2/300.0), dpi=100,facecolor=(1,1,1,0))

# 1. Setup projection and extent
proj = ccrs.PlateCarree()
ax = plt.axes(projection=proj)
ax.set_extent(extent, crs=ccrs.PlateCarree())

# 2. Load and plot custom image
img = plt.imread(name)
ax.imshow(img, origin='upper', extent=extent, transform=ccrs.PlateCarree())

#ax.coastlines(resolution='50m', color='white')

mapProj=ccrs.PlateCarree(central_longitude=np.mean(extent[:2]))
map=fig.add_subplot(1,1,1,projection=mapProj)
map.set_extent(extent, crs=ccrs.PlateCarree())
map.axis('off')

#map.add_feature(cfeature.LAND, zorder=100, facecolor='red')
map.add_feature(cfeature.OCEAN, zorder=100, facecolor=(0.85,0.85,0.85))
map.add_feature(cfeature.LAKES,facecolor=(0.85,0.85,0.85))

plt.savefig('map.png', format='png', bbox_inches='tight', pad_inches=0,
                        dpi=300, facecolor='black')
