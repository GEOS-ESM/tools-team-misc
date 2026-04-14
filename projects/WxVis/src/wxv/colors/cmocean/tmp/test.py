import sys
import matplotlib.pyplot as plt
import numpy as np

import wxv.colors

name = sys.argv[1]

base_cmap = plt.colormaps.get(name, None)
print(base_cmap._segmentdata)
colors = base_cmap(np.linspace(0, 1, 256))

# 1. Create data
data = np.random.rand(10, 10)

# 2. Create the plot (this returns a mappable object 'im')
fig, ax = plt.subplots()
im = ax.imshow(data, cmap=name)

# 3. Add the colorbar
fig.colorbar(im, ax=ax)

plt.show()
