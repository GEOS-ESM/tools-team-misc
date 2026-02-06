"""
WxMaps Base Image Module
Support for custom base earth imagery backgrounds
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.img_transform import warp_array
from pathlib import Path
from typing import Optional, Tuple
from enum import Enum


class BaseImageType(Enum):
    """Predefined base image types"""
    NATURAL_EARTH_GREYBLUE = 'natural_earth_greyblue'
    NATURAL_EARTH_LIGHT = 'natural_earth_light'
    GEOCOLOR_LIGHT = 'geocolor_light'
    BMNG_NO_SNOW = 'bmng_no_snow'
    BMNG_MONTHLY = 'bmng_monthly'
    DNB_NIGHTLIGHTS = 'dnb_nightlights'
    CUSTOM = 'custom'


class BaseImageConfig:
    """Configuration for base images"""
    
    BASE_IMAGE_DIR = '/discover/nobackup/projects/gmao/g6dev/pub/BMNG'
    
    # Predefined image paths
    IMAGES = {
        'natural_earth_greyblue': {
            'path': 'natural_earth_greyblue_noice_16200x8100.jpg',
            'extent': [-180, 180, -90, 90],
            'description': 'Natural Earth Grey-Blue (no arctic ice)',
            'projection': 'platecarree'
        },
        'natural_earth_light': {
            'path': 'natural_earth_noice_16200x8100.jpeg',
            'extent': [-180, 180, -90, 90],
            'description': 'Natural Earth Light Colors (no arctic ice)',
            'projection': 'platecarree'
        },
        'geocolor_light': {
            'path': 'New/eo_base_2020_clean_geo.3x21600x10800.jpg',
            'extent': [-180, 180, -90, 90],
            'description': 'GeoColor Light Map (2020)',
            'projection': 'platecarree'
        },
        'bmng_no_snow': {
            'path': 'bmng.3x21600x10800.jpg',
            'extent': [-180, 180, -90, 90],
            'description': 'Blue Marble (no snow/ice, consistent vegetation)',
            'projection': 'platecarree'
        },
        'dnb_nightlights': {
            'path': 'dnb_land_ocean_ice.2012.54000x27000.jpg',
            'extent': [-180, 180, -90, 90],
            'description': 'VIIRS Day/Night Band - Earth at Night (2012)',
            'projection': 'platecarree'
        },
    }
    
    @staticmethod
    def get_monthly_bmng_path(month: int) -> str:
        """Get Blue Marble monthly image path"""
        return f'world.2004{month:02d}.3x21600x10800.jpg'
    
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
        if image_type == 'bmng_monthly':
            if month is None:
                raise ValueError("Month parameter required for bmng_monthly")
            filename = BaseImageConfig.get_monthly_bmng_path(month)
        elif image_type in BaseImageConfig.IMAGES:
            filename = BaseImageConfig.IMAGES[image_type]['path']
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
    """Singleton cache for base images to avoid reloading in parallel workers"""
    
    _cache = {}
    _preloaded = {}  # Separate dict for images preloaded before forking
    
    @classmethod
    def get_image(cls, image_path: str, target_width: int = 4000) -> np.ndarray:
        """
        Load and downsample base image with caching.
        
        Parameters:
        -----------
        image_path : str
            Path to base image
        target_width : int
            Target width for downsampling (default: 4000)
            Higher = better quality but slower (4000-8000 recommended)
        
        Returns:
        --------
        np.ndarray : Downsampled image array
        """
        cache_key = (image_path, target_width)
        
        # Check preloaded cache first (set before forking)
        if cache_key in cls._preloaded:
            return cls._preloaded[cache_key]
        
        # Return cached version if available
        if cache_key in cls._cache:
            return cls._cache[cache_key]
        
        # Load image
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Base image not found: {image_path}")
        
        print(f"Loading base image: {image_path}")
        
        # Disable decompression bomb check for legitimate large images
        Image.MAX_IMAGE_PIXELS = None
        
        with Image.open(image_path) as img:
            orig_width, orig_height = img.size
            orig_pixels = (orig_width * orig_height) / 1e6
            
            print(f"  Original size: {orig_width}x{orig_height} ({orig_pixels:.1f}M pixels)")
            
            # Only downsample if image is larger than target
            if orig_width > target_width:
                # Calculate target height maintaining aspect ratio
                aspect_ratio = orig_height / orig_width
                target_height = int(target_width * aspect_ratio)
                
                # Downsample with high-quality Lanczos filter
                print(f"  Downsampling to: {target_width}x{target_height} ({(target_width*target_height)/1e6:.1f}M pixels)")
                img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                img_array = np.array(img_resized)
            else:
                # Use original size if already smaller than target
                print(f"  Using original size (already smaller than target)")
                img_array = np.array(img)
            
            print(f"  Cached. Memory: {img_array.nbytes / (1024**2):.1f} MB")
        
        # Cache it
        cls._cache[cache_key] = img_array
        
        return img_array
    
    @classmethod
    def mark_as_preloaded(cls, image_path: str, target_width: int):
        """
        Mark an image as preloaded so workers know to use the forked copy.
        Call this BEFORE forking workers.
        """
        cache_key = (image_path, target_width)
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
    
    def __init__(self, image_path: str, extent: Tuple[float, float, float, float] = None,
                 target_resolution: int = 4000):
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
    def apply_limb_darkening(
        img,
        strength=0.3,
        gamma=3.0
    ):
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
        y, x = np.ogrid[-1:1:h*1j, -1:1:w*1j]
        r = np.sqrt(x*x + y*y)

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
                self.image_path,
                target_width=self.target_resolution
            )
        return self.image_array
    
    def plot_on_map(self, ax, alpha: float = 1.0, 
                   interpolation: str = 'bilinear',
                   zorder: int = 0,
                   style=None):
        """
        Plot base image on a Cartopy map using cached transform
        """
        # Load from cache (fast after first load)
        if self.image_array is None:
            self.load_image()
       
        print(f"  Plotting base image: {Path(self.image_path).name}")
        
        # Use cached coordinate transformation
        from wxmaps_transform_cache import get_transform_cache
        
        transform_cache = get_transform_cache()
        
        # Use pre-computed extent/shape if available (ensures cache hit)
        if style and hasattr(style, 'cached_target_extent'):
            target_extent = style.cached_target_extent
            target_shape = style.cached_target_shape
            print(f"    Using pre-computed extent from style")
        else:
            target_extent = ax.get_extent()
            target_shape = (2160, 4320)
        
        # Get or compute transform (should hit cache now!)
        transform_data, cache_key = transform_cache.get_or_compute_transform(
            source_proj=ccrs.PlateCarree(),
            target_proj=ax.projection,
            source_extent=self.extent,
            target_extent=target_extent,
            target_shape=target_shape
        )
        
        # Apply the cached transform - FLIP for base images (they have origin='upper')
        print(f"    Applying cached transform to base image...")
        warped_img, warped_extent = transform_cache.apply_transform(
            source_data=self.image_array,  # Don't pre-flip, let apply_transform do it
            transform_data=transform_data,
            fill_value=0,
            flip_source_y=False
        )

        # Automatic limb darkening for full disk GEO
        if isinstance(ax.projection, ccrs.Geostationary):
            warped_img = BaseImagePlotter.apply_limb_darkening(warped_img)

        # Plot the transformed image
        im = ax.imshow(
            warped_img, 
            origin='lower',
            extent=warped_extent,
            transform=ax.projection,
            interpolation=interpolation,
            resample=False,
            alpha=alpha,
            zorder=zorder,
            rasterized=True
        )
        
        print(f"    Done (interpolation={interpolation}, alpha={alpha})")
        
        return im

def preload_base_images_from_style(style, target_resolution: int = 4000):
    """
    Preload base images into cache based on StyleConfig.
    Call this once in main process before forking workers.
    
    Parameters:
    -----------
    style : StyleConfig
        Style configuration that may request base images
    target_resolution : int
        Target width for downsampling (default: 4000)
    """
    if not style.use_base_image:
        return  # No base image requested
    
    print("\n" + "="*70)
    print("Preloading base images into cache...")
    print("="*70)
    
    # Determine which image to load
    if style.base_image_path:
        image_path = style.base_image_path
    else:
        image_path = BaseImageConfig.get_image_path(
            style.base_image_type,
            month=style.base_image_month
        )
    
    # Load into cache
    BaseImageCache.get_image(image_path, target_width=target_resolution)
    
    # Mark as preloaded so workers use the forked copy
    BaseImageCache.mark_as_preloaded(image_path, target_resolution)
    
    print("Base images preloaded successfully.")
    print("Workers will use the preloaded copy from forked memory.\n")

