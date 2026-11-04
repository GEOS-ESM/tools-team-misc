"""
WxMaps Transform Cache Module
Pre-compute and cache coordinate transformations to avoid repeated expensive calculations
"""

import numpy as np
import pickle
import hashlib
import json
import os
import logging
import tempfile
from pathlib import Path
from typing import Tuple, Optional
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


class TransformCache:
    """
    Cache coordinate transformations between projections.
    Stores the mapping coordinates so workers can quickly apply transforms to their data.
    """

    def __init__(
        self, cache_dir: Optional[Path | str] = None, clear_on_init: bool = False
    ):
        """
        Initialize transform cache

        Parameters:
        -----------
        cache_dir : Path | str, optional
            Directory to store transforms; None keeps transforms in memory only
        clear_on_init : bool
            If True, explicitly clear cached transforms (default: False).
        """

        # In-memory cache for current process
        self._memory_cache = {}
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is None:
            return

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(
                f"Cannot create cache directory: {self.cache_dir}\n"
                f"Error: {e}\n"
                f"Please specify a writable cache directory:\n"
                f"  cache = TransformCache(cache_dir='/path/to/writable/dir')"
            ) from e

        print(f"      Transform cache directory: {self.cache_dir}")

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
    def make_cache_key(
        source_proj,
        target_proj,
        source_extent: Tuple[float, float, float, float],
        target_extent: Tuple[float, float, float, float],
        target_shape: Tuple[int, int],
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

        key_data = {
            "version": 2,
            "src": source_proj.proj4_init,
            "dst": target_proj.proj4_init,
            "src_extent": tuple(float(value) for value in source_extent),
            "dst_extent": tuple(float(value) for value in target_extent),
            "target_shape": tuple(int(value) for value in target_shape),
        }

        s = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(s.encode()).hexdigest()

    def get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key"""
        return self.cache_dir / f"{cache_key}.pkl"

    def exists(self, cache_key: str) -> bool:
        """Check if cached transform exists"""
        return self.cache_dir is not None and self.get_cache_path(cache_key).exists()

    @staticmethod
    def compute_transform_data(
        source_proj,
        target_proj,
        source_extent: Tuple[float, float, float, float],
        target_extent: Tuple[float, float, float, float],
        target_shape: Tuple[int, int],
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
        # IMPORTANT: This creates output with origin='lower' convention:
        # - Row 0 corresponds to target_extent[2] (bottom/minimum Y)
        # - Row height-1 corresponds to target_extent[3] (top/maximum Y)
        # For geographic data: extent[2]=-90 (South), extent[3]=90 (North)
        # For projection data: extent[2]=bottom, extent[3]=top in projection coordinates
        x_target = np.linspace(target_extent[0], target_extent[1], width)
        y_target = np.linspace(target_extent[2], target_extent[3], height)
        xx_target, yy_target = np.meshgrid(x_target, y_target)

        # Transform target coordinates to source coordinates
        source_coords = source_proj.transform_points(
            target_proj, xx_target.flatten(), yy_target.flatten()
        )

        # Reshape back to grid
        x_source = source_coords[:, 0].reshape(height, width)
        y_source = source_coords[:, 1].reshape(height, width)

        # Create mask for valid pixels (within source extent)
        mask = (
            (x_source >= source_extent[0])
            & (x_source <= source_extent[1])
            & (y_source >= source_extent[2])
            & (y_source <= source_extent[3])
        )

        # Normalize source coordinates to [0, 1] range for interpolation
        # This assumes source data will be in origin='lower' format:
        # - Source row 0 should contain data at source_extent[2] (bottom/minimum Y)
        # - Source row height-1 should contain data at source_extent[3] (top/maximum Y)
        # For geographic: extent[2]=-90 (South), extent[3]=90 (North)
        # If source data is an image with origin='upper', it MUST be flipped before interpolation!
        x_source_norm = (x_source - source_extent[0]) / (
            source_extent[1] - source_extent[0]
        )
        y_source_norm = (y_source - source_extent[2]) / (
            source_extent[3] - source_extent[2]
        )

        return {
            "x_source_norm": x_source_norm,
            "y_source_norm": y_source_norm,
            "target_shape": target_shape,
            "target_extent": target_extent,
            "mask": mask,
            "source_extent": source_extent,
        }

    def compute_transform(
        self,
        source_proj,
        target_proj,
        source_extent: Tuple[float, float, float, float],
        target_extent: Tuple[float, float, float, float],
        target_shape: Tuple[int, int],
    ) -> dict:
        return self.compute_transform_data(
            source_proj, target_proj, source_extent, target_extent, target_shape
        )

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
        self._memory_cache[cache_key] = transform_data
        if self.cache_dir is None:
            return
        cache_path = self.get_cache_path(cache_key)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.cache_dir, prefix=f".{cache_key}.", suffix=".tmp", delete=False
            ) as cache_file:
                temporary_path = Path(cache_file.name)
                pickle.dump(
                    transform_data, cache_file, protocol=pickle.HIGHEST_PROTOCOL
                )
                cache_file.flush()
            os.replace(temporary_path, cache_path)
        except OSError as error:
            logger.warning("Unable to persist transform %s: %s", cache_path, error)
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as error:
                    logger.warning(
                        "Unable to clean transform temporary file %s: %s",
                        temporary_path,
                        error,
                    )

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

        with open(cache_path, "rb") as f:
            transform_data = pickle.load(f)

        if not isinstance(transform_data, dict):
            raise ValueError("Transform cache entry must be a mapping")
        shape = tuple(transform_data["target_shape"])
        for name in ("x_source_norm", "y_source_norm", "mask"):
            if (
                not isinstance(transform_data[name], np.ndarray)
                or transform_data[name].shape != shape
            ):
                raise ValueError(f"Invalid transform array: {name}")
        if transform_data["mask"].dtype != np.bool_:
            raise ValueError("Invalid transform mask dtype")

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
        force_recompute: bool = False,
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
        cache_key = self.make_cache_key(
            source_proj, target_proj, source_extent, target_extent, target_shape
        )

        # Check memory, then persistent storage.
        if not force_recompute and cache_key in self._memory_cache:
            print(f"      Using in-memory cached transform: {cache_key[:16]}...")
            return self._memory_cache[cache_key], cache_key

        if not force_recompute and self.exists(cache_key):
            try:
                transform_data = self.load(cache_key)
                if tuple(transform_data["target_shape"]) != tuple(target_shape):
                    raise ValueError("Cached transform has the wrong target shape")
                print(f"      Using cached transform: {cache_key}")
                return transform_data, cache_key
            except (
                OSError,
                EOFError,
                pickle.UnpicklingError,
                ValueError,
                KeyError,
                TypeError,
            ) as error:
                self._memory_cache.pop(cache_key, None)
                logger.warning("Rebuilding invalid transform %s: %s", cache_key, error)

        # Compute new transform
        print(f"      Computing coordinate transformation: {force_recompute}")
        print(
            f"      DEBUG: source_extent: [{source_extent[0]:.2f}, {source_extent[1]:.2f}], [{source_extent[2]:.2f}, {source_extent[3]:.2f}]"
        )
        print(
            f"      DEBUG: target_extent: [{target_extent[0]:.2f}, {target_extent[1]:.2f}], [{target_extent[2]:.2f}, {target_extent[3]:.2f}]"
        )
        transform_data = self.compute_transform(
            source_proj, target_proj, source_extent, target_extent, target_shape
        )

        # Save to cache
        self.save(cache_key, transform_data)

        return transform_data, cache_key

    @staticmethod
    def apply_transform(
        source_data: np.ndarray,
        transform_data: dict,
        fill_value: float = 0.0,
        flip_source_y: bool = False,
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
            If True, flip source data vertically before interpolation.
            REQUIRED for image files (JPEG/PNG) which have origin='upper':
            - origin='upper': row 0 = top (North for geographic images)
            - origin='lower': row 0 = bottom (South for geographic images)
            The interpolation expects origin='lower', so images must be flipped.

        Returns:
        --------
        tuple : (transformed_data, target_extent)
            The output array has origin='lower' convention (row 0 = bottom)
        """
        # Flip source data if needed (for images with origin='upper')
        if flip_source_y:
            source_data = np.flipud(source_data)

        # Get transform coordinates
        x_source_norm = transform_data["x_source_norm"]
        y_source_norm = transform_data["y_source_norm"]
        target_shape = transform_data["target_shape"]
        mask = transform_data["mask"]

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
                    method="linear",
                    bounds_error=False,
                    fill_value=fill_value,
                )

                # Apply transform
                points = np.stack(
                    [y_source_norm.flatten(), x_source_norm.flatten()], axis=1
                )

                result = interpolator(points).reshape(target_shape)

                # Handle NaN/Inf values
                result = np.nan_to_num(
                    result, nan=fill_value, posinf=fill_value, neginf=fill_value
                )

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
                method="linear",
                bounds_error=False,
                fill_value=fill_value,
            )

            # Apply transform
            points = np.stack(
                [y_source_norm.flatten(), x_source_norm.flatten()], axis=1
            )

            result = interpolator(points).reshape(target_shape)

            # Handle NaN/Inf values
            result = np.nan_to_num(
                result, nan=fill_value, posinf=fill_value, neginf=fill_value
            )

            output = result

            # Apply mask
            output[~mask] = fill_value

            # Convert back to original dtype
            output = output.astype(source_data.dtype)

        return output, transform_data["target_extent"]

    def clear_cache(self, max_age_days: Optional[int] = None):
        """Clear cache files"""
        import time

        deleted = 0
        total_size = 0
        self._memory_cache = {}
        if self.cache_dir is None:
            return

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
    Existing disk entries are retained so a warm-up job can serve later jobs.
    """
    global _global_cache
    if _global_cache is None:
        print("      New session-local transform cache object; ")
        _global_cache = TransformCache(cache_dir=cache_dir)
    else:
        print("      Reusing session-local transform cache object.")
    return _global_cache


def reset_transform_cache():
    """
    Reset the global cache instance.
    Useful for testing or if you need to reinitialize with different settings.
    """
    global _global_cache
    _global_cache = None
