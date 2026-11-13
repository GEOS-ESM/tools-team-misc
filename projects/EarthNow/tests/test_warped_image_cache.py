from types import SimpleNamespace

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from earthnow.wxmaps_base_images import BaseImagePlotter
from earthnow.wxmaps_config import WxMapsConfig


def test_warped_image_is_cached_and_reused(tmp_path, monkeypatch):
    monkeypatch.setitem(
        WxMapsConfig.RESOLUTIONS, "test", SimpleNamespace(width=8, height=4)
    )
    image_path = tmp_path / "base.png"
    Image.new("RGB", (16, 8), (40, 80, 120)).save(image_path)
    cache_directory = tmp_path / "cache"

    figure = plt.figure()
    try:
        ax = figure.add_subplot(projection=ccrs.PlateCarree())
        ax.set_global()

        first = BaseImagePlotter(
            str(image_path), cache_directory, target_resolution="test"
        )
        expected = np.asarray(first.plot_on_map(ax).get_array()).copy()
        assert len(list(cache_directory.glob("*.npz"))) == 1

        def unexpected_load(self):
            raise AssertionError("cache hit should not decode the source image")

        monkeypatch.setattr(BaseImagePlotter, "_load_source_image", unexpected_load)
        second = BaseImagePlotter(
            str(image_path), cache_directory, target_resolution="test"
        )
        np.testing.assert_array_equal(second.plot_on_map(ax).get_array(), expected)
    finally:
        plt.close(figure)
