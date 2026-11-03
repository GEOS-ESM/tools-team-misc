import numpy as np
import pytest

from earthnow.wxmaps_config import StyleConfig, WxMapsConfig
from earthnow.wxmaps_plotting import WxMapPlotter


@pytest.fixture
def plotter():
    instance = WxMapPlotter(
        WxMapsConfig.get_conus_map(),
        resolution="hd",
        style=StyleConfig(use_base_shapefiles=False),
    )
    instance.create_basemap()
    try:
        yield instance
    finally:
        instance.close()


@pytest.mark.parametrize("positive_longitudes", [False, True])
@pytest.mark.parametrize("descending", [False, True])
def test_crop_reduces_grid_and_preserves_values_masks_and_coverage(
    plotter, positive_longitudes, descending
):
    lats = np.linspace(-90, 90, 181)
    lons = np.arange(360.) if positive_longitudes else np.arange(-180., 180.)
    if descending:
        lats = lats[::-1]
        lons = lons[::-1]
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    field = np.ma.masked_where(lat_grid > 40, lon_grid + lat_grid)
    rgba = np.stack([lon_grid, lat_grid, lon_grid, lat_grid], axis=-1)

    cropped_lats, cropped_lons, cropped, cropped_rgba = plotter.crop_data_to_view(
        lats, lons, field, rgba
    )
    assert cropped.size < field.size * 0.1
    assert cropped_rgba.shape == (*cropped.shape, 4)
    expected_lons, expected_lats = np.meshgrid(cropped_lons, cropped_lats)
    np.testing.assert_array_equal(cropped.data, expected_lons + expected_lats)
    np.testing.assert_array_equal(np.ma.getmaskarray(cropped), expected_lats > 40)
    np.testing.assert_array_equal(cropped_rgba[:, :, 0], expected_lons)
    np.testing.assert_array_equal(cropped_rgba[:, :, 1], expected_lats)

    geographic_lons = (cropped_lons + 180) % 360 - 180
    # IDL map-frame envelope, including corners outside the input limit box.
    assert geographic_lons.min() < -134.91380
    assert geographic_lons.max() > -59.08620
    assert cropped_lats.min() < 19.82857
    assert cropped_lats.max() > 52.43527


def test_native_lcc_grid_is_not_cropped(plotter):
    lons, lats = np.meshgrid(np.linspace(-130, -60, 30), np.linspace(20, 55, 20))
    data = lats + lons
    result = plotter.crop_data_to_view(lats, lons, data)
    assert result[0] is lats
    assert result[1] is lons
    assert result[2] is data
    image, extent = plotter.crop_image_to_view(data, lats, lons)
    assert image is data
    assert extent == (-130, -60, 20, 55)


def test_other_views_are_not_cropped():
    instance = WxMapPlotter(
        WxMapsConfig.get_global_map(),
        resolution="hd",
        style=StyleConfig(use_base_shapefiles=False),
    )
    lats = np.linspace(-90, 90, 19)
    lons = np.linspace(-180, 180, 37)
    data = np.zeros((19, 37))
    # No axes are required for an unchanged non-CONUS path.
    assert instance.crop_data_to_view(lats, lons, data)[2] is data
    image, extent = instance.crop_image_to_view(data, lats, lons)
    assert image is data
    assert extent == (-180, 180, -90, 90)


@pytest.mark.parametrize("channels", [None, 4])
@pytest.mark.parametrize("positive_longitudes", [False, True])
def test_image_crop_preserves_original_pixel_registration(
    plotter, channels, positive_longitudes
):
    lats = np.linspace(-90, 90, 181)
    lons = np.linspace(0, 360, 361) if positive_longitudes else np.linspace(-180, 180, 361)
    data = np.arange(len(lats) * len(lons)).reshape(len(lats), len(lons))
    if channels:
        data = np.repeat(data[:, :, None], channels, axis=2)
    cropped, extent = plotter.crop_image_to_view(data, lats, lons)
    assert cropped.shape[0] * cropped.shape[1] < len(lats) * len(lons) * 0.1
    dx = (lons.max() - lons.min()) / len(lons)
    dy = (lats.max() - lats.min()) / len(lats)
    source_west = extent[0] + (360 if positive_longitudes else 0)
    column = round((source_west - lons.min()) / dx)
    row = round((extent[2] - lats.min()) / dy)
    np.testing.assert_array_equal(
        cropped,
        data[row:row + cropped.shape[0], column:column + cropped.shape[1], ...],
    )
    assert extent[1] == pytest.approx(extent[0] + cropped.shape[1] * dx)
    assert extent[3] == pytest.approx(extent[2] + cropped.shape[0] * dy)
    assert -180 < extent[0] < extent[1] < 0


def test_crop_reports_mismatched_field_shape(plotter):
    with pytest.raises(ValueError, match="does not match grid"):
        plotter.crop_data_to_view(
            np.linspace(-90, 90, 181),
            np.linspace(-180, 180, 361),
            np.zeros((10, 10)),
        )


def test_crop_reports_nonmonotonic_coordinates(plotter):
    with pytest.raises(ValueError, match="monotonic"):
        plotter.crop_data_to_view(
            np.array([20, 40, 30]),
            np.linspace(-180, 180, 361),
            np.zeros((3, 361)),
        )


def test_crop_reports_nonoverlapping_grid(plotter):
    with pytest.raises(ValueError, match="does not overlap"):
        plotter.crop_data_to_view(
            np.linspace(-50, -20, 31),
            np.linspace(50, 80, 31),
            np.zeros((31, 31)),
        )
