import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.io.img_tiles import GoogleTiles

# Custom class for ESRI Shaded Relief
class ShadedReliefESRI(GoogleTiles):
    def _image_url(self, tile):
        x, y, z = tile
        url = ('https://server.arcgisonline.com/ArcGIS/rest/services/'
               'World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg').format(
            z=z, y=y, x=x)
        return url

# Setup Plot
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add imagery with grayscale conversion
ax.add_image(ShadedReliefESRI(), 8, cmap='gray')

ax.set_extent([-10, 20, 30, 60])
plt.show()
