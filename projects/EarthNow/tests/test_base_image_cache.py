import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from earthnow import paths
from earthnow.wxmaps_base_images import BaseImageCache
from earthnow.wxmaps_config import WxMapsConfig
from earthnow.wxmaps_base_images import BaseImagePlotter


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    BaseImageCache.clear_cache()
    directory = tmp_path / "cache"
    monkeypatch.setattr(paths, "BASE_IMAGE_CACHE_DIR", directory)
    monkeypatch.setenv("EARTHNOW_BASE_IMAGE_CACHE_DIR", str(directory))
    monkeypatch.setitem(
        WxMapsConfig.RESOLUTIONS, "test", SimpleNamespace(width=4, height=3)
    )
    yield directory
    BaseImageCache.clear_cache()


def write_image(path, color):
    Image.new("RGB", (8, 4), color).save(path)


def test_disk_cache_is_reused_by_a_fresh_process(cache_dir, tmp_path):
    image_path = tmp_path / "base.png"
    write_image(image_path, (20, 40, 60))

    expected = BaseImageCache.get_image(str(image_path), target_resolution="test")
    assert expected.shape == (2, 4, 4)
    assert list(cache_dir.glob("*.npy"))

    script = """
import sys
import earthnow.wxmaps_base_images as base_images
from types import SimpleNamespace
base_images.WxMapsConfig.RESOLUTIONS["test"] = SimpleNamespace(width=4, height=3)

def unexpected_decode(*args, **kwargs):
    raise AssertionError("source image was decoded instead of using disk cache")

base_images.Image.open = unexpected_decode
image = base_images.BaseImageCache.get_image(sys.argv[1], sys.argv[2])
print(image.shape, int(image.sum()))
"""
    environment = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", script, str(image_path), "test"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert str(expected.shape) in result.stdout
    assert str(int(expected.sum())) in result.stdout


def test_changed_source_invalidates_memory_and_disk_entries(cache_dir, tmp_path):
    image_path = tmp_path / "base.png"
    write_image(image_path, (255, 0, 0))
    original = BaseImageCache.get_image(str(image_path), target_resolution="test")

    previous_stat = image_path.stat()
    write_image(image_path, (0, 0, 255))
    os.utime(
        image_path,
        ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns + 1_000_000_000),
    )
    updated = BaseImageCache.get_image(str(image_path), target_resolution="test")

    assert original[0, 0].tolist() == [255, 0, 0, 255]
    assert updated[0, 0].tolist() == [0, 0, 255, 255]
    assert len(list(cache_dir.glob("*.npy"))) == 2


def test_corrupt_disk_entry_is_rebuilt(cache_dir, tmp_path):
    image_path = tmp_path / "base.png"
    write_image(image_path, (10, 20, 30))
    expected = BaseImageCache.get_image(str(image_path), target_resolution="test")
    cache_path = next(cache_dir.glob("*.npy"))
    cache_path.write_bytes(b"not a numpy image")
    BaseImageCache.clear_cache()

    actual = BaseImageCache.get_image(str(image_path), target_resolution="test")

    np.testing.assert_array_equal(actual, expected)
    with cache_path.open("rb") as cache_file:
        np.testing.assert_array_equal(np.load(cache_file, allow_pickle=False), expected)


def test_concurrent_cold_cache_writes_are_atomic(cache_dir, tmp_path, monkeypatch):
    image_path = tmp_path / "base.png"
    write_image(image_path, (70, 80, 90))
    barrier = Barrier(4)
    load_disk_cache = BaseImageCache._load_disk_cache

    def synchronized_load(cache_path):
        result = load_disk_cache(cache_path)
        barrier.wait(timeout=10)
        return result

    monkeypatch.setattr(
        BaseImageCache, "_load_disk_cache", staticmethod(synchronized_load)
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        images = list(
            executor.map(
                lambda _: BaseImageCache.get_image(
                    str(image_path), target_resolution="test"
                ),
                range(4),
            )
        )

    for image in images[1:]:
        np.testing.assert_array_equal(image, images[0])
    assert len(list(cache_dir.glob("*.npy"))) == 1
    assert not list(cache_dir.glob("*.tmp"))


@pytest.mark.parametrize("region", ["conus", "global"])
def test_render_uses_requested_shape_and_loads_lazily(
    cache_dir, tmp_path, monkeypatch, region
):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from earthnow.wxmaps_transform_cache import TransformCache, reset_transform_cache

    monkeypatch.setitem(
        WxMapsConfig.RESOLUTIONS, "test", SimpleNamespace(width=80, height=45)
    )
    monkeypatch.setattr(paths, "TRANSFORM_CACHE_DIR", tmp_path / "transforms")
    monkeypatch.setattr(TransformCache, "CACHE_DIR", tmp_path / "transforms")
    reset_transform_cache()
    image_path = tmp_path / "base.png"
    pixels = np.zeros((90, 180, 3), dtype=np.uint8)
    pixels[:45] = [255, 0, 0]
    pixels[45:] = [0, 0, 255]
    Image.fromarray(pixels).save(image_path)
    config = WxMapsConfig.get_all_standard_maps()[region]
    figure = plt.figure(figsize=(4, 2.25))
    try:
        axes = figure.add_subplot(projection=config.projection)
        axes.set_extent(config.extent, crs=ccrs.PlateCarree())
        plotter = BaseImagePlotter(str(image_path), target_resolution="test")

        artist = plotter.plot_on_map(axes)
        np.testing.assert_allclose(artist.get_extent(), axes.get_extent(crs=axes.projection))
        rendered = np.asarray(artist.get_array())
        assert rendered.shape == (45, 80, 4)
        assert rendered[..., 3].max() == 255
        if region == "conus":
            assert rendered[22, 40, 0] > rendered[22, 40, 2]
        else:
            assert rendered[-10, 40, 0] > rendered[-10, 40, 2]
            assert rendered[10, 40, 2] > rendered[10, 40, 0]
        figure.savefig(tmp_path / f"{region}.png")
    finally:
        plt.close(figure)
        reset_transform_cache()


def test_missing_base_image_falls_back_with_requested_feature_resolution(monkeypatch):
    import matplotlib.pyplot as plt
    from cartopy.mpl.geoaxes import GeoAxes
    from earthnow.wxmaps_config import StyleConfig
    from earthnow.wxmaps_plotting import WxMapPlotter

    features = []

    def missing_image(*args, **kwargs):
        raise FileNotFoundError("synthetic missing base image")

    monkeypatch.setattr(BaseImagePlotter, "load_image", missing_image)
    monkeypatch.setattr(
        GeoAxes, "add_feature", lambda self, feature, **kwargs: features.append(feature)
    )
    style = StyleConfig.greyblue_plain()
    style.boundaries = []
    plotter = WxMapPlotter(WxMapsConfig.get_conus_map(), resolution="hd", style=style)
    try:
        plotter.create_basemap(feature_resolution="110m")
        assert plotter.feature_resolution == "110m"
        assert any(feature.name == "lakes" and feature.scale == "110m" for feature in features)
    finally:
        if plotter.fig is not None:
            plt.close(plotter.fig)
