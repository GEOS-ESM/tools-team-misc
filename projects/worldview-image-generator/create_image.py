#!/usr/bin/env python

from hybig import create_browse

# geotiff_path = "micasa-cog/monthly/2001/MiCASA_v1_NEE_x3600_y1800_monthly_200101.tif"
# geotiff_folder = "test-images/test-image-one/"
geotiff_folder = "test-images/test-image-tiles/"
geotiff_path = (
    geotiff_folder + "MiCASA_v1_NEE_x3600_y1800_monthly_200101.tif"
)  # The TIF has to be in the folder where the browse imagery will be created

results = create_browse(
    geotiff_path,
    {
        # "mime": "image/png",
        "crs": {"epsg": "EPSG:4326"},
        "scale_extent": {
            "x": {"min": -180, "max": 180},
            "y": {"min": -90, "max": 90},
        },
        # "scale_size": {"x": 10, "y": 10},
        "scale_size": {"x": 0.005, "y": 0.005},
    },
)
