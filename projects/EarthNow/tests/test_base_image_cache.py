import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest
from PIL import Image

from earthnow import paths
from earthnow.wxmaps_base_images import BaseImageCache


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    BaseImageCache.clear_cache()
    directory = tmp_path / "cache"
    monkeypatch.setattr(paths, "BASE_IMAGE_CACHE_DIR", directory)
    monkeypatch.setenv("EARTHNOW_BASE_IMAGE_CACHE_DIR", str(directory))
    yield directory
    BaseImageCache.clear_cache()


def write_image(path, color):
    Image.new("RGB", (8, 4), color).save(path)


def test_disk_cache_is_reused_by_a_fresh_process(cache_dir, tmp_path):
    image_path = tmp_path / "base.png"
    write_image(image_path, (20, 40, 60))

    expected = BaseImageCache.get_image(str(image_path), target_width=4)
    assert list(cache_dir.glob("*.npy"))

    script = """
import sys
import earthnow.wxmaps_base_images as base_images

def unexpected_decode(*args, **kwargs):
    raise AssertionError("source image was decoded instead of using disk cache")

base_images.Image.open = unexpected_decode
image = base_images.BaseImageCache.get_image(sys.argv[1], int(sys.argv[2]))
print(image.shape, int(image.sum()))
"""
    environment = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", script, str(image_path), "4"],
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
    original = BaseImageCache.get_image(str(image_path), target_width=4)

    previous_stat = image_path.stat()
    write_image(image_path, (0, 0, 255))
    os.utime(
        image_path,
        ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns + 1_000_000_000),
    )
    updated = BaseImageCache.get_image(str(image_path), target_width=4)

    assert original[0, 0].tolist() == [255, 0, 0, 255]
    assert updated[0, 0].tolist() == [0, 0, 255, 255]
    assert len(list(cache_dir.glob("*.npy"))) == 2


def test_corrupt_disk_entry_is_rebuilt(cache_dir, tmp_path):
    image_path = tmp_path / "base.png"
    write_image(image_path, (10, 20, 30))
    expected = BaseImageCache.get_image(str(image_path), target_width=4)
    cache_path = next(cache_dir.glob("*.npy"))
    cache_path.write_bytes(b"not a numpy image")
    BaseImageCache.clear_cache()

    actual = BaseImageCache.get_image(str(image_path), target_width=4)

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
                lambda _: BaseImageCache.get_image(str(image_path), target_width=4),
                range(4),
            )
        )

    for image in images[1:]:
        np.testing.assert_array_equal(image, images[0])
    assert len(list(cache_dir.glob("*.npy"))) == 1
    assert not list(cache_dir.glob("*.tmp"))
