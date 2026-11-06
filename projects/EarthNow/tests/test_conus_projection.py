import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from earthnow.wxmaps_config import StyleConfig, WxMapsConfig
from earthnow.wxmaps_plotting import WxMapPlotter


@pytest.mark.parametrize("resolution", ["hd", "4k", "8k"])
def test_conus_matches_idl_viewport(resolution):
    config = WxMapsConfig.get_conus_map()
    plotter = WxMapPlotter(
        config,
        resolution=resolution,
        style=StyleConfig(use_base_shapefiles=False),
    )
    try:
        _, ax = plotter.create_basemap()
        assert isinstance(ax.projection, ccrs.RotatedPole)
        assert config.extent == [-125.75, -68.25, 23, 50]
        assert config.standard_parallels is None
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        actual = ccrs.PlateCarree().transform_points(
            ax.projection,
            np.array([xmin, xmin, xmax, xmax, 0, 0]),
            np.array([ymin, ymax, ymin, ymax, ymin, ymax]),
        )[:, :2]
        # IDL 8.9 MAP_SET region 104 map-window corners and edge midpoints.
        expected = [
            [-124.9138031, 19.82857323],
            [-134.9138031, 48.13189316],
            [-69.0861969, 19.82857323],
            [-59.0861969, 48.13189316],
            [-97, 23],
            [-97, 52.43526840],
        ]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)
        np.testing.assert_allclose(
            ax.get_position().bounds,
            [0, 0.00332517107, 1, 0.99334965766],
            atol=1e-8,
            rtol=0,
        )
    finally:
        plotter.close()


@pytest.mark.parametrize("map_name", ["global", "conus_east"])
def test_other_maps_keep_geographic_extent(map_name):
    config = WxMapsConfig.get_all_standard_maps()[map_name]
    assert config.native_extent is None
    plotter = WxMapPlotter(
        config,
        resolution="hd",
        style=StyleConfig(use_base_shapefiles=False),
    )
    reference = plt.figure()
    try:
        _, ax = plotter.create_basemap()
        reference_ax = reference.add_subplot(projection=config.projection)
        reference_ax.set_extent(config.extent, crs=ccrs.PlateCarree())
        np.testing.assert_allclose(ax.get_xlim(), reference_ax.get_xlim())
        np.testing.assert_allclose(ax.get_ylim(), reference_ax.get_ylim())
    finally:
        plotter.close()
        plt.close(reference)


@pytest.mark.parametrize("source_grid", ["latlon", "lcc"])
def test_conus_renders_geographic_coordinates_from_both_grids(tmp_path, source_grid):
    geographic = ccrs.PlateCarree()
    if source_grid == "latlon":
        lons, lats = np.meshgrid(np.linspace(-140, -55, 80), np.linspace(15, 60, 45))
    else:
        source = ccrs.LambertConformal(
            central_longitude=-97,
            central_latitude=35,
            standard_parallels=(33, 45),
        )
        x, y = np.meshgrid(
            np.linspace(-2500000, 2500000, 80),
            np.linspace(-1500000, 1500000, 45),
        )
        coordinates = geographic.transform_points(source, x, y)
        lons, lats = coordinates[:, :, 0], coordinates[:, :, 1]

    plotter = WxMapPlotter(
        WxMapsConfig.get_conus_map(),
        resolution="hd",
        style=StyleConfig(use_base_shapefiles=False),
    )
    try:
        _, ax = plotter.create_basemap()
        limits = (ax.get_xlim(), ax.get_ylim())
        ax.pcolormesh(
            lons,
            lats,
            lats,
            transform=geographic,
            shading="nearest",
            vmin=15,
            vmax=60,
        )
        output = tmp_path / f"conus_{source_grid}.png"
        plotter.save(str(output))
        assert (ax.get_xlim(), ax.get_ylim()) == limits
        with Image.open(output) as image:
            pixels = np.asarray(image.convert("RGB"))
            assert pixels.shape[0] > 1000
            assert pixels.shape[1] > 1800
            assert pixels.std() > 10
    finally:
        plotter.close()
