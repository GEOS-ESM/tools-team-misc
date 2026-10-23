"""
WxMaps Base Image Module
Support for custom base earth imagery backgrounds
"""

import hashlib
import os
import tempfile

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.img_transform import warp_array
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
    """Memory and disk cache for decoded/downsampled base images."""

    _cache = {}
    _preloaded = {}  # Separate dict for images preloaded before forking (I dont think we need this for our workflow)
    _disk_cache_version = 2

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
        fingerprint = "\0".join(map(str, (cls._disk_cache_version, *cache_key)))
        cache_filename = hashlib.sha256(fingerprint.encode()).hexdigest() + ".npy"
        return cache_key, paths.BASE_IMAGE_CACHE_DIR / cache_filename

    @staticmethod
    def _load_disk_cache(cache_path: Path) -> Optional[np.ndarray]:
        try:
            image_array = np.load(cache_path, allow_pickle=False)
            if not isinstance(image_array, np.ndarray):
                image_array.close()
                raise ValueError("cache entry is not an ndarray")
            if (
                image_array.dtype != np.uint8
                or image_array.ndim != 3
                or image_array.shape[2] != 4
            ):
                raise ValueError("cache entry is not an RGBA uint8 image")
            return image_array
        except FileNotFoundError:
            return None
        except (OSError, ValueError, EOFError) as error:
            logger.warning(
                "Ignoring invalid base-image cache entry %s: %s", cache_path, error
            )
            return None

    @staticmethod
    def _save_disk_cache(cache_path: Path, image_array: np.ndarray) -> None:
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
                np.save(temporary_file, image_array, allow_pickle=False)
                temporary_file.flush()
            os.replace(temporary_path, cache_path)
        except (OSError, ValueError) as error:
            logger.warning(
                "Unable to write base-image cache entry %s: %s", cache_path, error
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
        cache_key, disk_cache_path = cls._get_cache_identity(
            image_path, target_resolution
        )

        # Check preloaded cache first
        if cache_key in cls._preloaded:
            print("   Cache preloaded variable found")
            return cls._preloaded[cache_key]

        # Return cached version if available
        if cache_key in cls._cache:
            print("   Basemap image found in memory cache")
            return cls._cache[cache_key]

        image_array = cls._load_disk_cache(disk_cache_path)
        if image_array is not None:
            cls._cache[cache_key] = image_array
            print(f"   Loaded base image from disk cache: {disk_cache_path}")
            return image_array

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
        cls._save_disk_cache(disk_cache_path, img_array)
        print(
            f"  Cached base image for other workers. Memory: {img_array.nbytes / (1024**2):.1f} MB   Image converted: {img_array.shape}\n  Cache: {disk_cache_path}"
        )
        return img_array

    @classmethod
    def mark_as_preloaded(cls, image_path: str, target_resolution: str):
        """
        Mark an image as preloaded so workers know to use the forked copy.
        Call this BEFORE forking workers.
        This doesn't work for our workflow, SLURM distributes each date to a worker, there is no central process that is performed first.
        """

        cache_key, _ = cls._get_cache_identity(image_path, target_resolution)
        if cache_key in cls._cache:
            cls._preloaded[cache_key] = cls._cache[cache_key]

    @classmethod
    def clear_cache(cls):
        """Clear the image cache to free memory"""
        cls._cache = {}
        cls._preloaded = {}
        print("Base image cache cleared.")


class BaseImagePlotter:
    """Handler for plotting base earth images"""

    def __init__(
        self,
        image_path: str,
        extent: Tuple[float, float, float, float] = None,
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
        self.extent = extent if extent else [-180, 180, -90, 90]
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
        if self.image_array is None:
            self.load_image()

        print("   Adding base image to map...")

        # Use cached coordinate transformation
        from earthnow.wxmaps_transform_cache import get_transform_cache

        transform_cache = get_transform_cache()

        target_extent = ax.get_extent(crs=ax.projection)
        resolution_config = WxMapsConfig.RESOLUTIONS[self.target_resolution]
        target_shape = (resolution_config.height, resolution_config.width)
        print(f"      Target extent: {target_extent}, target shape: {target_shape}")

        # Why can't we just cache the transformed image? That would be faster right?
        # Get or compute transform (should hit cache now!)
        transform_data, cache_key = transform_cache.get_or_compute_transform(
            source_proj=ccrs.PlateCarree(),
            target_proj=ax.projection,
            source_extent=self.extent,
            target_extent=target_extent,
            target_shape=target_shape,
        )

        # Apply the cached transform
        # CRITICAL: Base images (JPEG/PNG) have origin='upper' (row 0 = North)
        # But geographic data convention is origin='lower' (row 0 = South)
        # The transformation expects source data in origin='lower' format, so we must flip
        warped_img, warped_extent = transform_cache.apply_transform(
            source_data=self.image_array,
            transform_data=transform_data,
            fill_value=0,
            flip_source_y=True,  # Flip source from origin='upper' to origin='lower' before interpolation
        )
        print(f"      Warped image dims: {warped_img.shape}")

        # Automatic limb darkening for full disk GEO
        if isinstance(ax.projection, ccrs.Geostationary):
            warped_img = BaseImagePlotter.apply_limb_darkening(warped_img)

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


def preload_base_images_from_style(style, target_resolution: str = "4k"):
    """
    Preload base images into cache based on StyleConfig.

    Parameters:
    -----------
    style : StyleConfig
        Style configuration that may request base images
    target_resolution : str
        Target resolution key for WxMapsConfig.RESOLUTIONS (default: "4k")
    """
    if not style.use_base_image:
        return  # No base image requested - why not do this prior to the function invocation?

    # Determine which image to load
    if style.base_image_path:
        image_path = style.base_image_path
    else:
        image_path = BaseImageConfig.get_image_path(
            style.base_image_type, month=style.base_image_month
        )

    print(
        f"Checking for cached base image: ({style.base_image_type}) at {target_resolution} resolution..."
    )

    # Load into cache
    BaseImageCache.get_image(image_path, target_resolution=target_resolution)

    # Mark as preloaded: I don't think this does anything for our workflow?
    BaseImageCache.mark_as_preloaded(image_path, target_resolution=target_resolution)

    # print("Base images preloaded successfully.")
    # print("Workers will use the preloaded copy from forked memory.\n")
