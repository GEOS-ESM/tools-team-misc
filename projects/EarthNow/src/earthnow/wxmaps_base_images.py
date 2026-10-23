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

import logging

logger = logging.getLogger(__name__)


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


class BaseImageCache:
    """In-memory source images and persistent warped RGBA outputs."""

    _cache = {}
    _warped_cache = {}
    _warped_cache_version = 1

    @classmethod
    def _get_cache_identity(cls, image_path: str, target_resolution: str):
        source_path = Path(image_path).expanduser()
        try:
            source_path = source_path.resolve(strict=True)
            source_stat = source_path.stat()
        except FileNotFoundError:
            raise FileNotFoundError(f"Base image not found: {image_path}") from None

        cache_key = (
            str(source_path),
            target_resolution,
            source_stat.st_size,
            source_stat.st_mtime_ns,
        )
        return cache_key

    @classmethod
    def _get_warped_cache_identity(
        cls, image_path: str, target_resolution: str, transform_key: str
    ):
        source_key = cls._get_cache_identity(image_path, target_resolution)
        fingerprint = "\0".join(
            map(
                str,
                (cls._warped_cache_version, *source_key, transform_key),
            )
        )
        cache_key = hashlib.sha256(fingerprint.encode()).hexdigest()
        cache_path = paths.WARPED_IMAGE_CACHE_DIR / f"{cache_key}.npz"
        return cache_key, cache_path

    @classmethod
    def load_warped_image(
        cls,
        image_path: str,
        target_resolution: str,
        transform_key: str,
        target_shape: Tuple[int, int],
    ) -> Optional[Tuple[np.ndarray, Tuple[float, float, float, float]]]:
        cache_key, cache_path = cls._get_warped_cache_identity(
            image_path, target_resolution, transform_key
        )
        if cache_key in cls._warped_cache:
            return cls._warped_cache[cache_key]

        try:
            with np.load(cache_path, allow_pickle=False) as cache_file:
                warped_image = cache_file["rgba"]
                warped_extent = cache_file["extent"]
            if (
                warped_image.dtype != np.uint8
                or warped_image.shape != (*target_shape, 4)
                or warped_extent.shape != (4,)
                or not np.isfinite(warped_extent).all()
            ):
                raise ValueError("invalid warped image cache data")
            extent_values = tuple(float(value) for value in warped_extent)
            result = (
                warped_image,
                (
                    extent_values[0],
                    extent_values[1],
                    extent_values[2],
                    extent_values[3],
                ),
            )
            cls._warped_cache[cache_key] = result
            return result
        except FileNotFoundError:
            return None
        except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile) as error:
            logger.warning(
                "Ignoring invalid warped image cache entry %s: %s", cache_path, error
            )
            return None

    @classmethod
    def save_warped_image(
        cls,
        image_path: str,
        target_resolution: str,
        transform_key: str,
        warped_image: np.ndarray,
        warped_extent: Tuple[float, float, float, float],
    ) -> None:
        cache_key, cache_path = cls._get_warped_cache_identity(
            image_path, target_resolution, transform_key
        )
        result = (warped_image, tuple(float(value) for value in warped_extent))
        cls._warped_cache[cache_key] = result
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
                temporary_file.flush()
            os.replace(temporary_path, cache_path)
        except (OSError, ValueError) as error:
            logger.warning(
                "Unable to write warped image cache entry %s: %s", cache_path, error
            )
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    @classmethod
    def get_image(cls, image_path: str, target_resolution: str) -> np.ndarray:
        """
        Load and downsample base image with caching.

        Parameters:
        -----------
        image_path : str
            Path to base image
        target_resolution : str
            Target resolution for downsampling ('hd', 'fhd', '2k', '4k', '8k')

        Returns:
        --------
        np.ndarray : Downsampled image array
        """
        cache_key = cls._get_cache_identity(image_path, target_resolution)

        # Return cached version if available
        if cache_key in cls._cache:
            print("   Basemap image found in memory cache")
            return cls._cache[cache_key]

        print(f"  No image cache found. Loading base image: {image_path}")

        resolution_config = WxMapsConfig.RESOLUTIONS[target_resolution]
        target_width = resolution_config.width
        # Disable decompression bomb check for legitimate large images
        Image.MAX_IMAGE_PIXELS = None

        with Image.open(image_path) as img:
            orig_width, orig_height = img.size
            orig_pixels = (orig_width * orig_height) / 1e6

            print(
                f"  Original size: {orig_width}x{orig_height} ({orig_pixels:.1f}M pixels)"
            )
            # Convert to RBGA if in greyscale luminance alpha
            img = img.convert("RGBA")

            # Only downsample if image is larger than target
            if orig_width > target_width:
                target_height = max(1, round(target_width * orig_height / orig_width))

                # Downsample with high-quality Lanczos filter
                print(
                    f"  Downsampling to: {target_width}x{target_height} ({(target_width * target_height) / 1e6:.1f}M pixels)"
                )
                img_resized = img.resize(
                    (target_width, target_height), Image.Resampling.LANCZOS
                )
                img_array = np.array(img_resized)
            else:
                # Use original size if already smaller than target
                print("  Using original size (already smaller than target)")
                img_array = np.array(img)

        cls._cache[cache_key] = img_array
        print(
            f"  Cached base image in memory. Memory: {img_array.nbytes / (1024**2):.1f} MB   Image converted: {img_array.shape}"
        )
        return img_array

    @classmethod
    def clear_cache(cls):
        """Clear the image cache to free memory"""
        cls._cache = {}
        cls._warped_cache = {}
        print("Base image cache cleared.")


class BaseImagePlotter:
    """Handler for plotting base earth images"""

    def __init__(
        self,
        image_path: str,
        extent: Optional[Tuple[float, float, float, float]] = None,
        target_resolution: str = "4k",
    ):
        """
        Initialize base image plotter

        Parameters:
        -----------
        image_path : str
            Path to base image file
        extent : tuple, optional
            Image extent [lon_min, lon_max, lat_min, lat_max]
            Default: [-180, 180, -90, 90] for global images
        target_resolution : int
            Target width for downsampling (default: 4000)
            Recommended: 4000 for fast, 8000 for high quality
        """
        self.image_path = image_path
        self.extent = extent or (-180.0, 180.0, -90.0, 90.0)
        self.target_resolution = target_resolution
        self.image_array = None

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

    def load_image(self):
        """
        Load the base image using cache
        """
        if self.image_array is None:
            self.image_array = BaseImageCache.get_image(
                self.image_path, target_resolution=self.target_resolution
            )

        return self.image_array

    def plot_on_map(
        self,
        ax,
        alpha: float = 1.0,
        interpolation: str = "bilinear",
        zorder: int = 0,
    ):
        """
        Plot base image on a Cartopy map using cached transform
        """
        print("   Looking for cached warped base image...")

        target_extent = ax.get_extent(crs=ax.projection)
        resolution_config = WxMapsConfig.RESOLUTIONS[self.target_resolution]
        target_shape = (resolution_config.height, resolution_config.width)
        print(f"      Target extent: {target_extent}, target shape: {target_shape}")

        from earthnow.wxmaps_transform_cache import TransformCache

        source_projection = ccrs.PlateCarree()
        transform_key = TransformCache.make_cache_key(
            source_projection,
            ax.projection,
            self.extent,
            target_extent,
            target_shape,
        )
        cached_image = BaseImageCache.load_warped_image(
            self.image_path,
            self.target_resolution,
            transform_key,
            target_shape,
        )

        if cached_image is not None:
            warped_img, warped_extent = cached_image
            print("      Loaded warped RGBA image and extent from disk cache.")
        else:
            source_image = self.load_image()
            transform_data = TransformCache.compute_transform_data(
                source_proj=source_projection,
                target_proj=ax.projection,
                source_extent=self.extent,
                target_extent=target_extent,
                target_shape=target_shape,
            )
            warped_img, warped_extent = TransformCache.apply_transform(
                source_data=source_image,
                transform_data=transform_data,
                fill_value=0,
                flip_source_y=True,
            )

            if isinstance(ax.projection, ccrs.Geostationary):
                warped_img = BaseImagePlotter.apply_limb_darkening(warped_img)

            BaseImageCache.save_warped_image(
                self.image_path,
                self.target_resolution,
                transform_key,
                warped_img,
                warped_extent,
            )

        # Plot the transformed image
        # The warped_img array is constructed with row 0 = bottom (target_extent[2])
        # This is origin='lower' convention, matching how compute_transform() builds the coordinate mapping
        im = ax.imshow(
            warped_img,
            origin="lower",  # Output array has row 0 at bottom of map
            extent=warped_extent,
            transform=ax.projection,
            interpolation=interpolation,
            resample=False,
            alpha=alpha,
            zorder=zorder,
            rasterized=True,
        )

        print(
            f"   Basemap plotting complete (interpolation={interpolation}, alpha={alpha})"
        )

        return im
