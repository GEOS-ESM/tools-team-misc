import os
import shlex
import subprocess
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
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
    monkeypatch.setattr(paths, "WARPED_IMAGE_CACHE_DIR", directory / "warped")
    monkeypatch.setenv("EARTHNOW_BASE_IMAGE_CACHE_DIR", str(directory))
    monkeypatch.setitem(
        WxMapsConfig.RESOLUTIONS, "test", SimpleNamespace(width=4, height=3)
    )
    yield directory
    BaseImageCache.clear_cache()


def write_image(path, color):
    Image.new("RGB", (8, 4), color).save(path)


def test_cache_directory_configures_warped_subdirectory(tmp_path, monkeypatch):
    root = tmp_path / "custom-cache"
    monkeypatch.setattr(paths, "BASE_IMAGE_CACHE_DIR", tmp_path / "old-base")
    monkeypatch.setattr(paths, "WARPED_IMAGE_CACHE_DIR", tmp_path / "old-warped")

    paths.configure_base_image_cache_directory(root)

    assert paths.BASE_IMAGE_CACHE_DIR == root
    assert paths.WARPED_IMAGE_CACHE_DIR == root / "warped"


def test_workflow_forwards_cache_directory(tmp_path, monkeypatch):
    from earthnow.workflow import handlers

    commands = []
    monkeypatch.setattr(
        handlers.subprocess,
        "call",
        lambda command, **kwargs: commands.append(command),
    )
    now = datetime(2026, 10, 23)
    cache_directory = tmp_path / "custom cache"
    request = {
        "time_dt": now,
        "fcst_dt": now,
        "isa_forecast": False,
        "fcname": str(tmp_path / "forecast.png"),
        "rpname": str(tmp_path / "replay.png"),
        "options": {"product": "carbon", "map-type": "conus"},
        "pngsize": "8k",
        "driver": "enow_plot.py",
        "cache_directory": str(cache_directory),
    }

    handlers.make_image(request)

    assert len(commands) == 1
    assert f"--cache-directory {shlex.quote(str(cache_directory))}" in commands[0]


def test_source_image_cache_is_reused_in_process(cache_dir, tmp_path, monkeypatch):
    image_path = tmp_path / "base.png"
    write_image(image_path, (20, 40, 60))

    expected = BaseImageCache.get_image(str(image_path), target_resolution="test")
    assert expected.shape == (2, 4, 4)
    monkeypatch.setattr(
        Image, "open", lambda *args, **kwargs: pytest.fail("image decoded twice")
    )
    actual = BaseImageCache.get_image(str(image_path), target_resolution="test")

    np.testing.assert_array_equal(actual, expected)
    assert not list(cache_dir.glob("*.npy"))


def test_changed_source_invalidates_memory_entry(cache_dir, tmp_path):
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
    assert not list(cache_dir.glob("*.npy"))


def test_concurrent_warped_cache_writes_are_atomic(cache_dir, tmp_path):
    image_path = tmp_path / "base.png"
    write_image(image_path, (70, 80, 90))
    warped_image = np.full((3, 4, 4), 123, dtype=np.uint8)
    extent = (-120.0, -70.0, 25.0, 50.0)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(
            executor.map(
                lambda _: BaseImageCache.save_warped_image(
                    str(image_path), "test", "same-transform", warped_image, extent
                ),
                range(4),
            )
        )

    BaseImageCache.clear_cache()
    cached = BaseImageCache.load_warped_image(
        str(image_path), "test", "same-transform", (3, 4)
    )
    assert cached is not None
    np.testing.assert_array_equal(cached[0], warped_image)
    assert cached[1] == extent
    assert len(list((cache_dir / "warped").glob("*.npz"))) == 1
    assert not list((cache_dir / "warped").glob("*.tmp"))


@pytest.mark.parametrize("region", ["conus", "global"])
def test_render_uses_requested_shape_and_loads_lazily(
    cache_dir, tmp_path, monkeypatch, region
):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from earthnow.wxmaps_transform_cache import reset_transform_cache

    monkeypatch.setitem(
        WxMapsConfig.RESOLUTIONS, "test", SimpleNamespace(width=80, height=45)
    )
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
        np.testing.assert_allclose(
            artist.get_extent(), axes.get_extent(crs=axes.projection)
        )
        rendered = np.asarray(artist.get_array())
        assert rendered.shape == (45, 80, 4)
        assert rendered[..., 3].max() == 255
        if region == "conus":
            assert rendered[22, 40, 0] > rendered[22, 40, 2]
        else:
            assert rendered[-10, 40, 0] > rendered[-10, 40, 2]
            assert rendered[10, 40, 2] > rendered[10, 40, 0]
        assert list((cache_dir / "warped").glob("*.npz"))
        assert not list(cache_dir.glob("*.npy"))
        assert not list((tmp_path / "transforms").glob("*.pkl"))
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
        assert any(
            feature.name == "lakes" and feature.scale == "110m" for feature in features
        )
    finally:
        if plotter.fig is not None:
            plt.close(plotter.fig)


def test_warped_rgba_is_reused_without_source_or_transform(
    cache_dir, tmp_path, monkeypatch
):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from earthnow.wxmaps_transform_cache import TransformCache

    monkeypatch.setattr(paths, "WARPED_IMAGE_CACHE_DIR", tmp_path / "warped")
    monkeypatch.setenv("EARTHNOW_WARPED_IMAGE_CACHE_DIR", str(tmp_path / "warped"))

    image_path = tmp_path / "base.png"
    pixels = np.zeros((4, 8, 3), dtype=np.uint8)
    pixels[:2] = [255, 0, 0]
    pixels[2:] = [0, 0, 255]
    Image.fromarray(pixels).save(image_path)

    config = WxMapsConfig.get_all_standard_maps()["global"]
    figure = plt.figure(figsize=(4, 2.25))
    try:
        axes = figure.add_subplot(projection=config.projection)
        axes.set_extent(config.extent, crs=ccrs.PlateCarree())
        first_artist = BaseImagePlotter(
            str(image_path), target_resolution="test"
        ).plot_on_map(axes)
        expected = np.asarray(first_artist.get_array()).copy()
        expected_extent = first_artist.get_extent()
        assert list((tmp_path / "warped").glob("*.npz"))

        BaseImageCache.clear_cache()

        def unexpected_call(*args, **kwargs):
            raise AssertionError(
                "warped cache hit should skip source/transform loading"
            )

        with monkeypatch.context() as cache_hit_patch:
            cache_hit_patch.setattr(BaseImageCache, "get_image", unexpected_call)
            cache_hit_patch.setattr(
                TransformCache, "compute_transform_data", unexpected_call
            )
            second_artist = BaseImagePlotter(
                str(image_path), target_resolution="test"
            ).plot_on_map(axes)

        np.testing.assert_array_equal(np.asarray(second_artist.get_array()), expected)
        assert second_artist.get_extent() == expected_extent

        transform_key = TransformCache.make_cache_key(
            ccrs.PlateCarree(),
            axes.projection,
            (-180.0, 180.0, -90.0, 90.0),
            axes.get_extent(crs=axes.projection),
            (3, 4),
        )
        script = """
import sys
from types import SimpleNamespace
from earthnow.wxmaps_base_images import BaseImageCache
from earthnow.wxmaps_config import WxMapsConfig
from earthnow.wxmaps_transform_cache import TransformCache

WxMapsConfig.RESOLUTIONS["test"] = SimpleNamespace(width=4, height=3)
def unexpected(*args, **kwargs):
    raise AssertionError("fresh process tried to recompute the warped image")
BaseImageCache.get_image = unexpected
TransformCache.compute_transform_data = unexpected
cached = BaseImageCache.load_warped_image(
    sys.argv[1], "test", sys.argv[2], (3, 4)
)
assert cached is not None
print(int(cached[0].sum()), cached[1])
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(image_path), transform_key],
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        assert str(int(expected.sum())) in result.stdout

        cache_path = next((tmp_path / "warped").glob("*.npz"))
        cache_path.write_bytes(b"corrupt warped image")
        BaseImageCache.clear_cache()
        rebuilt_artist = BaseImagePlotter(
            str(image_path), target_resolution="test"
        ).plot_on_map(axes)
        np.testing.assert_array_equal(np.asarray(rebuilt_artist.get_array()), expected)
        with np.load(cache_path, allow_pickle=False) as cached:
            np.testing.assert_array_equal(cached["rgba"], expected)
            np.testing.assert_allclose(cached["extent"], expected_extent)
    finally:
        plt.close(figure)
