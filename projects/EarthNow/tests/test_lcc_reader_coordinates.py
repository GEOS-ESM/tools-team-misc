from types import SimpleNamespace

import cartopy.crs as ccrs
import numpy as np
import pytest
from netCDF4 import Dataset

from earthnow.data_readers.dataservice import SimpleNetCDFDataService


@pytest.mark.parametrize(
    "values, expected",
    [
        ([[225, 260], [280, 299]], [[-135, -100], [-80, -61]]),
        ([[-135, -100], [-80, -61]], [[-135, -100], [-80, -61]]),
        ([[0, 180], [360, -180]], [[0, -180], [0, -180]]),
    ],
)
def test_lcc_coordinates_normalized_without_reordering(tmp_path, values, expected):
    path = tmp_path / "coordinates.nc4"
    latitudes = np.array([[25, 30], [40, 50]], dtype=float)
    with Dataset(path, "w") as nc:
        nc.createDimension("y", 2)
        nc.createDimension("x", 2)
        nc.createVariable("lats", "f8", ("y", "x"))[:] = latitudes
        nc.createVariable("lons", "f8", ("y", "x"))[:] = values
    reader = SimpleNetCDFDataService(SimpleNamespace(grid="lcc", coords=path))
    lats, lons = reader.get_coords()
    np.testing.assert_array_equal(lats, latitudes)
    np.testing.assert_array_equal(lons, expected)

    projection = ccrs.RotatedPole(pole_longitude=83, pole_latitude=60)
    before = projection.transform_points(ccrs.PlateCarree(), np.array(values), lats)
    after = projection.transform_points(ccrs.PlateCarree(), lons, lats)
    np.testing.assert_allclose(before, after, atol=1e-10, rtol=0)


def test_lcc_longitude_mask_preserved(tmp_path):
    path = tmp_path / "masked.nc4"
    values = np.ma.array([[230., 260.], [280., 290.]], mask=[[0, 1], [0, 0]])
    with Dataset(path, "w") as nc:
        nc.createDimension("y", 2)
        nc.createDimension("x", 2)
        nc.createVariable("lats", "f8", ("y", "x"))[:] = 35
        nc.createVariable("lons", "f8", ("y", "x"), fill_value=-9999)[:] = values
    reader = SimpleNetCDFDataService(SimpleNamespace(grid="lcc", coords=path))
    _, lons = reader.get_coords()
    np.testing.assert_array_equal(np.ma.getmaskarray(lons), values.mask)
    np.testing.assert_array_equal(lons.compressed(), [-130, -80, -70])


def test_regular_grid_longitudes_unchanged(tmp_path):
    path = tmp_path / "regular.nc4"
    with Dataset(path, "w") as nc:
        nc.createDimension("lat", 2)
        nc.createDimension("lon", 3)
        nc.createVariable("lat", "f8", ("lat",))[:] = [25, 50]
        nc.createVariable("lon", "f8", ("lon",))[:] = [0, 180, 300]
    reader = SimpleNetCDFDataService(SimpleNamespace(grid="latlon"))
    with Dataset(path) as nc:
        reader.nc = nc
        lats, lons = reader.get_coords()
    np.testing.assert_array_equal(lats, [25, 50])
    np.testing.assert_array_equal(lons, [0, 180, 300])
