import h3
import numpy as np
import pandas as pd

# 1. Sample dense data (some points are very close together)
data = {
    "latitude": [37.7749, 37.7748, 37.7750, 34.0522, 34.0523],
    "longitude": [-122.4194, -122.4195, -122.4193, -118.2437, -118.2436],
    "value": [0,1,2,3,4]
}
df = pd.DataFrame(data)

# 2. Define your target resolution (e.g., Res 9 ~ 174m edge length)
RESOLUTION = 9

# 3. Map lat/lng to H3 cell IDs
df["h3_cell"] = df.apply(
    lambda row: h3.geo_to_h3(row["latitude"], row["longitude"], RESOLUTION),
    axis=1,
)

# 4. Thin the data by taking the first observation in each hexagon
df_thinned = df.drop_duplicates(subset=["h3_cell"])["h3_cell"]

# 5. Convert cells to (lat, lng) tuples
coords = np.array([h3.h3_to_geo(cell) for cell in df_thinned])

# 6. Split into two separate NumPy arrays
latitudes = coords[:, 0]
longitudes = coords[:, 1]
values = coords[:, 2]

print("Original data size:", len(df))
print("Thinned data size:", len(df_thinned))
print(latitudes)
print(longitudes)
print(values)
