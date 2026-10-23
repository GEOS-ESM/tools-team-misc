import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import cartopy.crs as ccrs

from earthnow.wxmaps_transform_cache import (
    TransformCache,
    get_transform_cache,
    reset_transform_cache,
)


def request_transform(cache, shape=(4, 8)):
    return cache.get_or_compute_transform(
        ccrs.PlateCarree(),
        ccrs.PlateCarree(),
        (-180, 180, -90, 90),
        (-120, -70, 25, 50),
        shape,
    )


def test_cache_survives_singleton_reset(tmp_path, monkeypatch):
    reset_transform_cache()
    try:
        expected, key = request_transform(get_transform_cache(tmp_path))
        reset_transform_cache()

        def unexpected_compute(*args, **kwargs):
            raise AssertionError("persistent transform was recomputed")

        monkeypatch.setattr(TransformCache, "compute_transform", unexpected_compute)
        actual, actual_key = request_transform(get_transform_cache(tmp_path))
        assert actual_key == key
        np.testing.assert_array_equal(
            actual["x_source_norm"], expected["x_source_norm"]
        )
    finally:
        reset_transform_cache()


def test_resolution_is_part_of_key(tmp_path):
    cache = TransformCache(tmp_path)
    _, first_key = request_transform(cache, (4, 8))
    result, second_key = request_transform(cache, (8, 16))
    assert first_key != second_key
    assert result["mask"].shape == (8, 16)


def test_corrupt_transform_is_rebuilt(tmp_path):
    cache = TransformCache(tmp_path)
    expected, key = request_transform(cache)
    cache.get_cache_path(key).write_bytes(b"incomplete")
    actual, _ = request_transform(TransformCache(tmp_path))
    np.testing.assert_array_equal(actual["mask"], expected["mask"])


def test_disk_reuse_in_separate_process(tmp_path):
    _, key = request_transform(TransformCache(tmp_path))
    script = """
import sys
import cartopy.crs as ccrs
from earthnow.wxmaps_transform_cache import TransformCache

def unexpected_compute(*args, **kwargs):
    raise AssertionError("warm transform was recomputed")

TransformCache.compute_transform = unexpected_compute
cache = TransformCache(sys.argv[1])
result, key = cache.get_or_compute_transform(
    ccrs.PlateCarree(), ccrs.PlateCarree(),
    (-180, 180, -90, 90), (-120, -70, 25, 50), (4, 8),
)
assert result["mask"].shape == (4, 8)
print(key)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert key in result.stdout


def test_concurrent_transform_writers(tmp_path, monkeypatch):
    barrier = Barrier(4)
    compute = TransformCache.compute_transform

    def synchronized_compute(self, *args, **kwargs):
        barrier.wait(timeout=10)
        return compute(self, *args, **kwargs)

    monkeypatch.setattr(TransformCache, "compute_transform", synchronized_compute)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda _: request_transform(TransformCache(tmp_path)), range(4)
            )
        )
    key = results[0][1]
    persisted = TransformCache(tmp_path).load(key)
    for data, result_key in results:
        assert result_key == key
        np.testing.assert_array_equal(data["x_source_norm"], persisted["x_source_norm"])
    assert not list(tmp_path.glob("*.tmp"))


def test_default_root_ignores_job_scratch(tmp_path, monkeypatch):
    from earthnow import paths

    monkeypatch.setenv("TMPDIR", str(tmp_path / "job-scratch"))
    monkeypatch.setattr(paths, "TRANSFORM_CACHE_DIR", tmp_path / "persistent")
    assert TransformCache._get_default_cache_dir() == tmp_path / "persistent"
