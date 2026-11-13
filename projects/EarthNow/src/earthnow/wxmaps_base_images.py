"""
WxMaps Base Image Module
Support for custom base earth imagery backgrounds
"""

import hashlib
import os
import tempfile
import zipfile

from PIL import Image
import numpy as np
import cartopy.crs as ccrs
from pathlib import Path
from typing import Optional, Tuple
from enum import Enum
from earthnow import paths
from earthnow.wxmaps_config import WxMapsConfig
from earthnow.wxmaps_transform_cache import TransformCache

import logging

logger = logging.getLogger(__name__)

Extent = Tuple[float, float, float, float]

# Bump when warping/limb-darkening output changes so stale cache files are ignored.
WARPED_CACHE_VERSION = 1


class BaseImageType(Enum):
    """Predefined base image types"""

    NATURAL_EARTH_GREY = "natural_earth_grey"
    NATURAL_EARTH_GREYBLUE = "natural_earth_greyblue"
    NATURAL_EARTH_LIGHT = "natural_earth_light"
    GEOCOLOR_LIGHT = "geocolor_light"
    BMNG_NO_SNOW = "bmng_no_snow"
    BMNG_MONTHLY = "bmng_monthly"
    DNB_NIGHTLIGHTS = "dnb_nightlights"
    CUSTOM = "custom"


class BaseImageConfig:
    """Configuration for base images"""

    BASE_IMAGE_DIR = paths.BASE_IMAGE_DIR

    # Predefined image paths
    IMAGES = {
        "natural_earth_grey": {
            "path": "natural_earth_grey_noice_16200x8100.jpg",
            "extent": [-180, 180, -90, 90],
            "description": "Natural Earth Greyscale Blue (no arctic ice)",
            "projection": "platecarree",
        },
        "natural_earth_greyblue": {
            "path": "natural_earth_greyblue_noice_16200x8100.jpg",
            "extent": [-180, 180, -90, 90],
            "description": "Natural Earth Grey-Blue (no arctic ice)",
            "projection": "platecarree",
        },
        "natural_earth_light": {
            "path": "natural_earth_noice_16200x8100.jpeg",
            "extent": [-180, 180, -90, 90],
            "description": "Natural Earth Light Colors (no arctic ice)",
            "projection": "platecarree",
        },
        "geocolor_light": {
            "path": "New/eo_base_2020_clean_geo.3x21600x10800.jpg",
            "extent": [-180, 180, -90, 90],
            "description": "GeoColor Light Map (2020)",
            "projection": "platecarree",
        },
        "bmng_no_snow": {
            "path": "bmng.3x21600x10800.jpg",
            "extent": [-180, 180, -90, 90],
            "description": "Blue Marble (no snow/ice, consistent vegetation)",
            "projection": "platecarree",
        },
        "dnb_nightlights": {
            "path": "dnb_land_ocean_ice.2012.54000x27000.jpg",
            "extent": [-180, 180, -90, 90],
            "description": "VIIRS Day/Night Band - Earth at Night (2012)",
            "projection": "platecarree",
        },
    }

    @staticmethod
    def get_monthly_bmng_path(month: int) -> str:
        """Get Blue Marble monthly image path"""
        return f"world.2004{month:02d}.3x21600x10800.jpg"

    @staticmethod
    def get_image_path(image_type: str, month: Optional[int] = None) -> str:
        """
        Get full path to base image

        Parameters:
        -----------
        image_type : str
            Type of base image
        month : int, optional
            Month (1-12) for monthly Blue Marble images

        Returns:
        --------
        str : Full path to image file
        """
        if image_type == "bmng_monthly":
            if month is None:
                raise ValueError("Month parameter required for bmng_monthly")
            filename = BaseImageConfig.get_monthly_bmng_path(month)
        elif image_type in BaseImageConfig.IMAGES:
            filename = BaseImageConfig.IMAGES[image_type]["path"]
        else:
            # Assume it's a custom path
            return image_type

        return str(Path(BaseImageConfig.BASE_IMAGE_DIR) / filename)

    @staticmethod
    def list_available_images():
        """Print available base images"""
        print("\nAvailable Base Images:")
        print("=" * 70)
        for key, info in BaseImageConfig.IMAGES.items():
            print(f"\n{key}:")
            print(f"  Description: {info['description']}")
            print(f"  Path: {BaseImageConfig.BASE_IMAGE_DIR}/{info['path']}")


class BaseImagePlotter:
    """Handler for plotting base earth images"""

    def __init__(
        self,
        image_path: str,
        cache_directory: str | Path,
        extent: Extent = (-180.0, 180.0, -90.0, 90.0),
        target_resolution: str = "4k",
    ):
        """
        Initialize base image plotter

        Parameters:
        -----------
        image_path : str
            Path to base image file
        cache_directory : str | Path
            Directory for warped-image cache files
        extent : tuple, optional
            Image extent [lon_min, lon_max, lat_min, lat_max]
            Default: (-180, 180, -90, 90)
        target_resolution : str
            Output resolution key ('hd', 'fhd', '2k', '4k', '8k')

        """
        self.image_path = image_path
        self.extent = extent
        self.cache_directory = cache_directory
        self.target_resolution = target_resolution
        resolution_config = WxMapsConfig.RESOLUTIONS[target_resolution]
        self.target_shape = (resolution_config.height, resolution_config.width)

    @staticmethod
    def apply_limb_darkening(img, strength=0.3, gamma=3.0):
        """
        Apply limb darkening to a geostationary full-disk image.

        Parameters
        ----------
        img : ndarray
            Warped GEO image, shape (H, W, C)
        strength : float
            0 = none, 1 = strong darkening
        gamma : float
            Controls falloff shape (1–2 typical)

        Returns
        -------
        img_out : ndarray
        """

        h, w = img.shape[:2]

        # Normalized disk coordinates
        y, x = np.ogrid[-1 : 1 : h * 1j, -1 : 1 : w * 1j]
        r = np.sqrt(x * x + y * y)

        # μ = cos(theta) approximation
        mu = np.clip(1.0 - r**2, 0, 1)
        mu = mu**gamma

        # Limb mask
        limb = (1.0 - strength) + strength * mu

        img_out = img.astype(np.float32)
        img_out[..., :3] *= limb[..., None]

        return np.clip(img_out, 0, 255).astype(img.dtype)

    def _load_source_image(self) -> np.ndarray:
        """Decode the source image as RGBA, downsampled to the output width."""
        target_width = self.target_shape[1]
        # Source basemaps are legitimately larger than PIL's decompression-bomb limit.
        Image.MAX_IMAGE_PIXELS = None

        print(f"   Loading source base image and resampling: {self.image_path}")
        with Image.open(self.image_path) as img:
            orig_width, orig_height = img.size
            img = img.convert("RGBA")
            if orig_width > target_width:
                target_height = max(1, round(target_width * orig_height / orig_width))
                img = img.resize(
                    (target_width, target_height), Image.Resampling.LANCZOS
                )
            return np.array(img)

    def _create_warped_image(
        self, projection, target_extent: Extent
    ) -> Tuple[np.ndarray, Extent]:
        """Warp the source image onto the target projection and extent."""
        source_image = self._load_source_image()
        transform_data = TransformCache.compute_transform_data(
            source_proj=ccrs.PlateCarree(),
            target_proj=projection,
            source_extent=self.extent,
            target_extent=target_extent,
            target_shape=self.target_shape,
        )
        # Image rows run north-to-south; the transform expects south-to-north.
        warped_image, warped_extent = TransformCache.apply_transform(
            source_data=source_image,
            transform_data=transform_data,
            fill_value=0,
            flip_source_y=True,
        )
        if isinstance(projection, ccrs.Geostationary):
            warped_image = BaseImagePlotter.apply_limb_darkening(warped_image)
        print("   Created warped image")
        return warped_image, warped_extent

    @staticmethod
    def _save_warped_image(
        cache_path: Path, warped_image: np.ndarray, warped_extent: Extent
    ) -> None:
        """Atomically write the warped image so concurrent jobs never read a partial file."""
        temporary_path = None
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=cache_path.parent,
                prefix=f".{cache_path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                np.savez(
                    temporary_file,
                    rgba=warped_image,
                    extent=np.asarray(warped_extent, dtype=np.float64),
                )
            os.replace(temporary_path, cache_path)
            print(f"   Saved warped base image to cache: {cache_path}")
        except (OSError, ValueError) as error:
            print(
                f"   Warning: Unable to write warped image cache {cache_path}: {error}"
            )
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _warped_cache_path(
        self,
        target_extent: Extent,
        projection: ccrs.Projection,
        source_projection: ccrs.Projection = ccrs.PlateCarree(),
    ) -> Path:
        """Cache file for one source image warped onto one map geometry."""
        transform_key = TransformCache.make_cache_key(
            projection,
            source_projection,
            self.extent,
            target_extent,
            self.target_shape,
        )

        source_path = Path(self.image_path).expanduser().resolve(strict=True)
        source_stat = source_path.stat()
        fingerprint = "\0".join(
            map(
                str,
                (
                    WARPED_CACHE_VERSION,
                    source_path,
                    source_stat.st_size,
                    source_stat.st_mtime_ns,
                    transform_key,
                ),
            )
        )
        digest = hashlib.sha256(fingerprint.encode()).hexdigest()
        return Path(self.cache_directory).expanduser() / f"{digest}.npz"

    def _load_warped_image(
        self, cache_path: Path
    ) -> Optional[Tuple[np.ndarray, Extent]]:
        """Return the cached warped RGBA image and extent, or None if unusable."""
        try:
            with np.load(cache_path, allow_pickle=False) as cache_file:
                warped_image = cache_file["rgba"]
                warped_extent = cache_file["extent"]
            if (
                warped_image.dtype != np.uint8
                or warped_image.shape != (*self.target_shape, 4)
                or warped_extent.shape != (4,)
                or not np.isfinite(warped_extent).all()
            ):
                raise ValueError("invalid warped image cache data")
        except FileNotFoundError:
            return None
        except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile) as error:
            print(f"   Ignoring invalid warped image cache {cache_path}: {error}")
            return None

        x0, x1, y0, y1 = (float(value) for value in warped_extent)
        print(f"   Loaded warped image from cache: {cache_path}")
        return warped_image, (x0, x1, y0, y1)

    def load_or_create_warped_image(
        self, target_extent: Extent, projection: ccrs.Projection
    ) -> Tuple[np.ndarray, Extent]:
        """Load the warped image from cache, or create it and cache it."""
        cache_path = self._warped_cache_path(target_extent, projection)
        cached = self._load_warped_image(cache_path)
        if cached is not None:
            return cached

        warped_image, warped_extent = self._create_warped_image(
            projection, target_extent
        )
        self._save_warped_image(cache_path, warped_image, warped_extent)
        return warped_image, warped_extent

    def plot_on_map(
        self,
        ax,
        alpha: float = 1.0,
        interpolation: str = "bilinear",
        zorder: int = 0,
    ):
        """Plot the warped base image, loading it from cache when available."""
        target_extent = ax.get_extent(crs=ax.projection)

        warped_image, warped_extent = self.load_or_create_warped_image(
            target_extent, ax.projection
        )

        return ax.imshow(
            warped_image,
            origin="lower",
            extent=warped_extent,
            transform=ax.projection,
            interpolation=interpolation,
            resample=False,
            alpha=alpha,
            zorder=zorder,
            rasterized=True,
        )
