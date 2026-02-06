"""
WxMaps Transform Cache Module
Pre-compute and cache coordinate transformations to avoid repeated expensive calculations
"""
import numpy as np
import pickle
import hashlib
import json
import os
from pathlib import Path
from typing import Tuple, Optional
import cartopy.crs as ccrs
from scipy.interpolate import RegularGridInterpolator


class TransformCache:
    """
    Cache coordinate transformations between projections.
    Stores the mapping coordinates so workers can quickly apply transforms to their data.
    """
    
    # Default cache directory - use TMPDIR if available, otherwise user's home
    @staticmethod
    def _get_default_cache_dir():
        """Get default cache directory based on environment"""
        # Try TMPDIR first (SLURM sets this)
        tmpdir = os.environ.get('TMPDIR')
        if tmpdir:
            cache_dir = Path(tmpdir) / 'wxmaps_transform_cache'
        else:
            # Try /discover/nobackup/username
            username = os.environ.get('USER', 'unknown')
            discover_tmp = Path(f'/discover/nobackup/{username}/tmp/wxmaps_transform_cache')
            
            # Check if parent exists
            if discover_tmp.parent.parent.exists():
                cache_dir = discover_tmp
            else:
                # Fall back to home directory
                cache_dir = Path.home() / '.wxmaps_cache' / 'transforms'
        
        return cache_dir
    
    CACHE_DIR = _get_default_cache_dir()
    
    def __init__(self, cache_dir: Optional[Path] = None, clear_on_init: bool = False):
        """
        Initialize transform cache
        
        Parameters:
        -----------
        cache_dir : Path, optional
            Directory to store cached transforms (default: auto-detect from TMPDIR or user home)
        clear_on_init : bool
            If True, clear all cached transforms on initialization (default: True for robustness)
        """
        if cache_dir is None:
            cache_dir = self.CACHE_DIR
        
        self.cache_dir = Path(cache_dir)
        
        # Create cache directory
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(
                f"Cannot create cache directory: {self.cache_dir}\n"
                f"Error: {e}\n"
                f"Please specify a writable cache directory:\n"
                f"  cache = TransformCache(cache_dir='/path/to/writable/dir')"
            ) from e
        
        # In-memory cache for current process
        self._memory_cache = {}
        
        print(f"Transform cache directory: {self.cache_dir}")
        
        # Clear old cache on initialization for robustness
        if clear_on_init:
            old_files = list(self.cache_dir.glob("*.pkl"))
            if old_files:
                print(f"  Clearing {len(old_files)} old cached transform(s)...")
                for cache_file in old_files:
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        print(f"    Warning: Could not delete {cache_file.name}: {e}")
                print("  ✓ Old cache cleared")
    
    @staticmethod
    def _make_cache_key(
        source_proj,
        target_proj,
        source_extent: Tuple[float, float, float, float],
        target_extent: Tuple[float, float, float, float],
        target_shape: Tuple[int, int]
    ) -> str:
        """
        Create a unique cache key for a transformation
        
        Parameters:
        -----------
        source_proj : cartopy.crs.Projection
            Source projection
        target_proj : cartopy.crs.Projection
            Target projection
        source_extent : tuple
            Source extent [lon_min, lon_max, lat_min, lat_max]
        target_extent : tuple
            Target extent in target projection coordinates
        target_shape : tuple
            Target shape (height, width)
        
        Returns:
        --------
        str : Cache key (hash)
        """

        def to_py(vals, ndigits):
            return tuple(float(round(float(v), ndigits)) for v in vals)

        key_data = {
            "src": source_proj.proj4_init,
            "dst": target_proj.proj4_init,
            "src_extent": to_py(source_extent, 6),
            "dst_extent": to_py(target_extent, 2),
        }

        s = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(s.encode()).hexdigest()
    
    def get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def exists(self, cache_key: str) -> bool:
        """Check if cached transform exists"""
        return self.get_cache_path(cache_key).exists()
    
    def compute_transform(
        self,
        source_proj,
        target_proj,
        source_extent: Tuple[float, float, float, float],
        target_extent: Tuple[float, float, float, float],
        target_shape: Tuple[int, int]
    ) -> dict:
        """
        Compute the coordinate transformation mapping
        
        This computes where each pixel in the target image should sample from
        in the source image.
        
        Parameters:
        -----------
        source_proj : cartopy.crs.Projection
            Source projection
        target_proj : cartopy.crs.Projection
            Target projection
        source_extent : tuple
            Source extent
        target_extent : tuple
            Target extent
        target_shape : tuple
            Target shape (height, width)
        
        Returns:
        --------
        dict : Transform data containing:
            - 'x_source_norm': normalized X coordinates for interpolation
            - 'y_source_norm': normalized Y coordinates for interpolation
            - 'target_shape': target shape
            - 'target_extent': target extent
            - 'mask': valid pixel mask
            - 'source_extent': source extent
        """
        height, width = target_shape
        
        # Create target coordinate grid
        # Note: Y is from bottom to top (lat_min to lat_max)
        x_target = np.linspace(target_extent[0], target_extent[1], width)
        y_target = np.linspace(target_extent[2], target_extent[3], height)
        xx_target, yy_target = np.meshgrid(x_target, y_target)
        
        # Transform target coordinates to source coordinates
        source_coords = source_proj.transform_points(
            target_proj,
            xx_target.flatten(),
            yy_target.flatten()
        )
        
        # Reshape back to grid
        x_source = source_coords[:, 0].reshape(height, width)
        y_source = source_coords[:, 1].reshape(height, width)
        
        # Create mask for valid pixels (within source extent)
        mask = (
            (x_source >= source_extent[0]) &
            (x_source <= source_extent[1]) &
            (y_source >= source_extent[2]) &
            (y_source <= source_extent[3])
        )
        
        # Normalize source coordinates to [0, 1] range for interpolation
        x_source_norm = (x_source - source_extent[0]) / (source_extent[1] - source_extent[0])
        # NO FLIP - we'll handle image origin in the apply step
        y_source_norm = (y_source - source_extent[2]) / (source_extent[3] - source_extent[2])
        
        return {
            'x_source_norm': x_source_norm,
            'y_source_norm': y_source_norm,
            'target_shape': target_shape,
            'target_extent': target_extent,
            'mask': mask,
            'source_extent': source_extent
        }
    
    def save(self, cache_key: str, transform_data: dict):
        """
        Save transform data to cache
        
        Parameters:
        -----------
        cache_key : str
            Cache key
        transform_data : dict
            Transform data from compute_transform()
        """
        cache_path = self.get_cache_path(cache_key)
        
        # Save to disk
        with open(cache_path, 'wb') as f:
            pickle.dump(transform_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Also cache in memory for this process
        self._memory_cache[cache_key] = transform_data
        
        size_mb = cache_path.stat().st_size / (1024**2)
        print(f"    Cached transform: {cache_path.name} ({size_mb:.2f} MB)")
    
    def load(self, cache_key: str) -> dict:
        """
        Load transform data from cache
        
        Parameters:
        -----------
        cache_key : str
            Cache key
        
        Returns:
        --------
        dict : Transform data
        """
        # Check memory cache first
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        # Load from disk
        cache_path = self.get_cache_path(cache_key)
        
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")
        
        with open(cache_path, 'rb') as f:
            transform_data = pickle.load(f)
        
        # Cache in memory
        self._memory_cache[cache_key] = transform_data
        
        return transform_data
    
    def get_or_compute_transform(
        self,
        source_proj,
        target_proj,
        source_extent: Tuple[float, float, float, float],
        target_extent: Tuple[float, float, float, float],
        target_shape: Tuple[int, int],
        force_recompute: bool = False
    ) -> Tuple[dict, str]:
        """
        Get cached transform or compute and cache it
        
        Parameters:
        -----------
        source_proj : cartopy.crs.Projection
        target_proj : cartopy.crs.Projection
        source_extent : tuple
        target_extent : tuple
        target_shape : tuple
        force_recompute : bool
        
        Returns:
        --------
        tuple : (transform_data, cache_key)
        """
        # Generate cache key
        cache_key = self._make_cache_key(
            source_proj,
            target_proj,
            source_extent,
            target_extent,
            target_shape
        )
        
        # Check if cached (only matters within same run, since cache cleared on init)
        if not force_recompute and cache_key in self._memory_cache:
            print(f"    Using in-memory cached transform: {cache_key[:16]}...")
            return self._memory_cache[cache_key], cache_key
        
        if not force_recompute and self.exists(cache_key):
            print(f"    Using cached transform: {cache_key[:16]}...")
            transform_data = self.load(cache_key)
            return transform_data, cache_key
        
        # Compute new transform
        print(f"    Computing coordinate transformation: {force_recompute}")
        print(f"DEBUG: source_extent: [{source_extent[0]:.2f}, {source_extent[1]:.2f}], [{source_extent[2]:.2f}, {source_extent[3]:.2f}]")
        print(f"DEBUG: target_extent: [{target_extent[0]:.2f}, {target_extent[1]:.2f}], [{target_extent[2]:.2f}, {target_extent[3]:.2f}]")
        transform_data = self.compute_transform(
            source_proj,
            target_proj,
            source_extent,
            target_extent,
            target_shape
        )
        
        # Save to cache
        self.save(cache_key, transform_data)
        
        return transform_data, cache_key
    
    def apply_transform(
        self,
        source_data: np.ndarray,
        transform_data: dict,
        fill_value: float = 0.0,
        flip_source_y: bool = False  # ADD THIS PARAMETER
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """
        Apply a cached transform to data
        
        Parameters:
        -----------
        source_data : np.ndarray
            Source data array (height, width) or (height, width, channels)
        transform_data : dict
            Transform data from get_or_compute_transform()
        fill_value : float
            Value for pixels outside source extent
        flip_source_y : bool
            If True, flip source data vertically before interpolation
            (use for image files with origin='upper')
        
        Returns:
        --------
        tuple : (transformed_data, target_extent)
        """
        # Flip source data if needed (for images with origin='upper')
        if flip_source_y:
            source_data = np.flipud(source_data)
        
        # Get transform coordinates
        x_source_norm = transform_data['x_source_norm']
        y_source_norm = transform_data['y_source_norm']
        target_shape = transform_data['target_shape']
        mask = transform_data['mask']
        
        # Handle multi-channel data
        if source_data.ndim == 3:
            # RGB or RGBA
            height, width, channels = source_data.shape
            
            # Determine output dtype - use float for intermediate calculations
            if source_data.dtype == np.uint8:
                output_dtype = np.uint8
                use_float = True
            else:
                output_dtype = source_data.dtype
                use_float = False
            
            # Create output in float for interpolation, convert back later if needed
            output = np.full((*target_shape, channels), fill_value, dtype=np.float32)
            
            for c in range(channels):
                # Create interpolator for this channel
                y_coords = np.linspace(0, 1, height)
                x_coords = np.linspace(0, 1, width)
                
                # Convert source data to float for interpolation
                source_channel = source_data[:, :, c].astype(np.float32)
                
                interpolator = RegularGridInterpolator(
                    (y_coords, x_coords),
                    source_channel,
                    method='linear',
                    bounds_error=False,
                    fill_value=fill_value
                )
                
                # Apply transform
                points = np.stack([
                    y_source_norm.flatten(),
                    x_source_norm.flatten()
                ], axis=1)
                
                result = interpolator(points).reshape(target_shape)
                
                # Handle NaN/Inf values
                result = np.nan_to_num(result, nan=fill_value, posinf=fill_value, neginf=fill_value)
                
                output[:, :, c] = result
            
            # Apply mask
            output[~mask] = fill_value
            
            # Convert back to original dtype if needed
            if use_float:
                output = np.clip(output, 0, 255).astype(output_dtype)
            else:
                output = output.astype(output_dtype)
            
        else:
            # Single channel
            height, width = source_data.shape
            
            # Use float for interpolation
            source_float = source_data.astype(np.float32)
            output = np.full(target_shape, fill_value, dtype=np.float32)
            
            # Create interpolator
            y_coords = np.linspace(0, 1, height)
            x_coords = np.linspace(0, 1, width)
            
            interpolator = RegularGridInterpolator(
                (y_coords, x_coords),
                source_float,
                method='linear',
                bounds_error=False,
                fill_value=fill_value
            )
            
            # Apply transform
            points = np.stack([
                y_source_norm.flatten(),
                x_source_norm.flatten()
            ], axis=1)
            
            result = interpolator(points).reshape(target_shape)
            
            # Handle NaN/Inf values
            result = np.nan_to_num(result, nan=fill_value, posinf=fill_value, neginf=fill_value)
            
            output = result
            
            # Apply mask
            output[~mask] = fill_value
            
            # Convert back to original dtype
            output = output.astype(source_data.dtype)
        
        return output, transform_data['target_extent']
    
    def clear_cache(self, max_age_days: Optional[int] = None):
        """Clear cache files"""
        import time
        
        deleted = 0
        total_size = 0
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            if max_age_days is not None:
                file_age_days = (time.time() - cache_file.stat().st_mtime) / 86400
                if file_age_days < max_age_days:
                    continue
            
            file_size = cache_file.stat().st_size
            cache_file.unlink()
            deleted += 1
            total_size += file_size
        
        if deleted > 0:
            print(f"Deleted {deleted} cache files ({total_size / (1024**2):.1f} MB)")
        
        self._memory_cache = {}


# Global cache instance
_global_cache = None

def get_transform_cache(cache_dir: Optional[Path] = None) -> TransformCache:
    """
    Get the global transform cache instance
    
    Parameters:
    -----------
    cache_dir : Path, optional
        Custom cache directory (only used on first call)
    
    Note:
    -----
    Cache is cleared on first initialization for robustness.
    Within the same run, transforms are reused from memory.
    """
    global _global_cache
    if _global_cache is None:
        print("DEBUG: Creating NEW transform cache instance")  # ADD THIS
        _global_cache = TransformCache(cache_dir=cache_dir, clear_on_init=True)
    else:
        print("DEBUG: Reusing existing transform cache instance")  # ADD THIS
    return _global_cache

def reset_transform_cache():
    """
    Reset the global cache instance.
    Useful for testing or if you need to reinitialize with different settings.
    """
    global _global_cache
    _global_cache = None
