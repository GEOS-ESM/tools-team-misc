"""
Radar Reflectivity Product
Maximum Composite Reflectivity (DBZ_MAX / REFC)
"""

import time
import numpy as np
import cartopy.crs as ccrs
from matplotlib import cm
from matplotlib.colors import ListedColormap, BoundaryNorm
from .registry import register
from wxmaps_utils import load_color_table
from typing import Tuple, Optional
import datetime

def solar_alpha(isr, min_frac=0.02, max_frac=0.15):
    """
    Create smooth alpha mask from incoming solar radiation.
    """
    isr_max = np.nanmax(isr)
    lo = min_frac * isr_max
    hi = max_frac * isr_max

    alpha = (isr - lo) / (hi - lo)
    alpha = np.nan_to_num(alpha, nan=0.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha

def auto_enhance_rgb_luminance(rgb, strength=0.35, debug=False):
    """IDL-style auto_enhance_rgb (luminance preserving)"""
    strength = np.clip(strength, 0.0, 1.0)

    # RGB → luminance (IDL weights)
    lum = (
        0.2126 * rgb[..., 0] +
        0.7152 * rgb[..., 1] +
        0.0722 * rgb[..., 2]
    )

    valid = np.isfinite(lum)
    if np.count_nonzero(valid) < 10:
        return rgb

    data = lum[valid]
    data = np.clip(data, 0, 1)

    # IDL histogram percentiles
    low = np.percentile(data, 2)
    high = np.percentile(data, 98)

    if debug:
        print(f"Luminance stretch: {low:.4f} → {high:.4f}")

    lum_stretch = (lum - low) / (high - low)
    lum_stretch = np.clip(lum_stretch, 0, 1)

    # Blend
    lum_new = (1 - strength) * lum + strength * lum_stretch

    # Avoid divide-by-zero
    scale = np.ones_like(lum)
    mask = lum > 1e-6
    scale[mask] = lum_new[mask] / lum[mask]

    rgb_out = rgb * scale[..., None]
    rgb_out = np.clip(rgb_out, 0, 1)

    return rgb_out

def auto_enhance_rgb_histogram(red, green, blue, strength=0.5, debug=False):
    """IDL-style auto_enhance_rgb (HISTOGRAM method only)"""
    
    strength = np.clip(strength, 0.0, 1.0)

    out_r = red.astype(np.float32, copy=True)
    out_g = green.astype(np.float32, copy=True)
    out_b = blue.astype(np.float32, copy=True)

    channels = [
        ("Red",   red,   out_r),
        ("Green", green, out_g),
        ("Blue",  blue,  out_b),
    ]

    low_pct  = (2.0 - strength) / 100.0
    high_pct = 1.0 - low_pct

    if debug:
        print(f"Histogram enhancement:")
        print(f"  strength = {strength}")
        print(f"  low_pct  = {low_pct*100:.2f}%")
        print(f"  high_pct = {high_pct*100:.2f}%")

    for name, src, dst in channels:
        valid = np.isfinite(src)
        n_valid = np.count_nonzero(valid)

        if n_valid <= 1:
            if debug:
                print(f"{name}: skipped (n_valid={n_valid})")
            continue

        data = src[valid]
        dmin = data.min()
        dmax = data.max()

        if dmin == dmax:
            if debug:
                print(f"{name}: skipped (constant field)")
            continue

        sorted_data = np.sort(data)
        n = sorted_data.size

        low_idx  = max(int(n * low_pct), 0)
        high_idx = min(int(n * high_pct), n - 1)

        min_val = sorted_data[low_idx]
        max_val = sorted_data[high_idx]

        if debug:
            print(f"{name}: n={n} min_val={min_val:.4f} max_val={max_val:.4f}")

        if min_val == max_val:
            continue

        stretched = (data - min_val) / (max_val - min_val)
        result = (1.0 - strength) * data + strength * stretched
        result = np.clip(result, 0.0, 1.0)
        dst[valid] = result

    return out_r, out_g, out_b

class DaytimeRGB:
    """Create daytime RGB composites from solar radiation bands."""
    
    def __init__(self, reader, fdate, pdate):
        self.reader = reader
        self.fdate = fdate
        self.pdate = pdate
        
        self.swtdn = 'SWTDN'
        self.isr_blue = 'ISRB11RG'
        self.isr_red = 'ISRB10RG'
        self.isr_nir = 'ISRB09RG'
        
        self.osr_blue = 'OSRB11RG'
        self.osr_red = 'OSRB10RG'
        self.osr_nir = 'OSRB09RG'
        
        self.lats = None
        self.lons = None
        self.meta = None
        self._cache = {}
    
    def _load_variable(self, variable_name: str) -> np.ndarray:
        if variable_name not in self._cache:
            data, lats, lons, meta = self.reader.read_variable(
                self.fdate,
                self.pdate,
                variables=[variable_name]
            )
            
            if self.lats is None:
                self.lats = lats
                self.lons = lons
                self.meta = meta
            
            self._cache[variable_name] = data.astype(np.float32)
        
        return self._cache[variable_name]

    def _solar_distance_au(self):
        """
        Compute Earth–Sun distance in AU.
        LAZY IMPORT astropy here - only when actually called
        """
        # MOVED IMPORTS HERE
        from astropy.time import Time
        from astropy.coordinates import get_sun
        import astropy.units as u
        
        pdate = self.pdate

        if isinstance(pdate, datetime.datetime):
            t = Time(pdate)
        elif isinstance(pdate, np.datetime64):
            t = Time(pdate.astype('datetime64[s]').astype(datetime.datetime))
        elif isinstance(pdate, str):
            p = pdate.strip()
            if p.lower().endswith("z"):
                p = p[:-1]
            for fmt in ("%Y%m%d_%H%M", "%Y%m%d%H", "%Y-%m-%d %H:%M:%S"):
                try:
                    t = Time.strptime(p, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Unrecognized pdate format: {pdate}")
        else:
            raise TypeError(f"Unsupported pdate type: {type(pdate)}")

        return get_sun(t).distance.to(u.au).value

    def _compute_reflectance(self, osr: np.ndarray, isr: np.ndarray,
                            sun_distance_au: float,
                            min_solar_threshold: float = 0.001) -> np.ndarray:
        isr_masked = np.where(isr < min_solar_threshold, np.nan, isr)
        reflectance = (np.pi * osr) / (isr_masked * sun_distance_au**2)
        return reflectance
    
    def _normalize_channel(self, data: np.ndarray, 
                          vmin: Optional[float] = None,
                          vmax: Optional[float] = None,
                          gamma: float = 1.0) -> np.ndarray:
        data_masked = np.ma.masked_invalid(data)
        
        if vmin is None:
            valid_data = data_masked.compressed()
            if len(valid_data) > 0:
                vmin = np.percentile(valid_data, 0.5)
            else:
                vmin = 0
        if vmax is None:
            valid_data = data_masked.compressed()
            if len(valid_data) > 0:
                vmax = np.percentile(valid_data, 99.5)
            else:
                vmax = 1
        
        if vmax == vmin:
            vmax = vmin + 1e-6
        
        normalized = (data_masked - vmin) / (vmax - vmin)
        normalized = np.clip(normalized, 0, 1)
        
        if gamma != 1.0:
            normalized = np.power(normalized, gamma)
        
        return normalized.filled(0)
  
    def _normalize_reflectance(self, data):
        """IDL-equivalent GeoColor normalization"""
        data = np.clip(data, 0.025, 1.2)
        data = np.log10(data)

        y1 = -1.6
        y2 = 0.176

        out = np.zeros_like(data)
        mask = (data >= y1) & (data <= y2)
        out[mask] = (data[mask] - y1) / (y2 - y1)
        out[data > y2] = 1.0

        return out

    def get_reflectance_channels(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        print("Loading solar radiation bands...")
        
        isr_blue = self._load_variable(self.isr_blue)
        isr_red = self._load_variable(self.isr_red)
        isr_nir = self._load_variable(self.isr_nir)
        
        print(f"ISR Blue: min={np.nanmin(isr_blue):.6f}, max={np.nanmax(isr_blue):.6f}, mean={np.nanmean(isr_blue):.6f}")
        print(f"ISR Red:  min={np.nanmin(isr_red):.6f}, max={np.nanmax(isr_red):.6f}, mean={np.nanmean(isr_red):.6f}")
        print(f"ISR NIR:  min={np.nanmin(isr_nir):.6f}, max={np.nanmax(isr_nir):.6f}, mean={np.nanmean(isr_nir):.6f}")

        isr = self._load_variable(self.swtdn)
        print(f"ISR : min={np.nanmin(isr):.6f}, max={np.nanmax(isr):.6f}, mean={np.nanmean(isr):.6f}")
        isr_blue = isr
        isr_red  = isr
        isr_nir  = isr
 
        osr_blue = self._load_variable(self.osr_blue)
        osr_red = self._load_variable(self.osr_red)
        osr_nir = self._load_variable(self.osr_nir)
        
        print(f"OSR Red:  min={np.nanmin(osr_red):.6f}, max={np.nanmax(osr_red):.6f}, mean={np.nanmean(osr_red):.6f}")
        print(f"OSR NIR:  min={np.nanmin(osr_nir):.6f}, max={np.nanmax(osr_nir):.6f}, mean={np.nanmean(osr_nir):.6f}")
        print(f"OSR Blue: min={np.nanmin(osr_blue):.6f}, max={np.nanmax(osr_blue):.6f}, mean={np.nanmean(osr_blue):.6f}")
 
        print("\nComputing reflectance...")

        sun_dist = self._solar_distance_au()
        blu_refl = self._compute_reflectance(osr_blue, isr_blue, sun_dist)
        red_refl = self._compute_reflectance(osr_red, isr_red, sun_dist)
        nir_refl = self._compute_reflectance(osr_nir, isr_nir, sun_dist)
       
        print(f"\nReflectance Red:  min={np.nanmin(red_refl):.6f}, max={np.nanmax(red_refl):.6f}, mean={np.nanmean(red_refl):.6f}")
        print(f"Reflectance NIR:    min={np.nanmin(nir_refl):.6f}, max={np.nanmax(nir_refl):.6f}, mean={np.nanmean(nir_refl):.6f}")
        print(f"Reflectance Blue: min={np.nanmin(blu_refl):.6f}, max={np.nanmax(blu_refl):.6f}, mean={np.nanmean(blu_refl):.6f}")

        isr = self._load_variable(self.swtdn)
        alpha = solar_alpha(isr, min_frac=0.02, max_frac=0.66)
        print(f"Alpha Layer: min={np.nanmin(alpha):.6f}, max={np.nanmax(alpha):.6f}, mean={np.nanmean(alpha):.6f}")

        return blu_refl, red_refl, nir_refl, alpha
    
    def create_true_color(self, 
                         gamma: float = 1.0,
                         enhance: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        blu_refl, red_refl, nir_refl, alpha = self.get_reflectance_channels()

        print(f"\nCreating true-color composite (enhance={enhance}, gamma={gamma})...")

        grn_refl = 0.45 * red_refl + 0.45 * blu_refl + 0.1 * nir_refl

        print(f"\nReflectance Red:  min={np.nanmin(red_refl):.6f}, max={np.nanmax(red_refl):.6f}, mean={np.nanmean(red_refl):.6f}")
        print(f"Reflectance Green:  min={np.nanmin(grn_refl):.6f}, max={np.nanmax(grn_refl):.6f}, mean={np.nanmean(grn_refl):.6f}")
        print(f"Reflectance Blue: min={np.nanmin(blu_refl):.6f}, max={np.nanmax(blu_refl):.6f}, mean={np.nanmean(blu_refl):.6f}")

        blu_refl = self._normalize_reflectance(blu_refl)
        red_refl = self._normalize_reflectance(red_refl)
        grn_refl = self._normalize_reflectance(grn_refl)

        blu_refl = blu_refl.filled(0.0)
        red_refl = red_refl.filled(0.0)
        grn_refl = grn_refl.filled(0.0)

        print(f"\nReflectance-Norm Red:   min={np.nanmin(red_refl):.6f}, max={np.nanmax(red_refl):.6f}, mean={np.nanmean(red_refl):.6f}")
        print(f"Reflectance-Norm Green: min={np.nanmin(grn_refl):.6f}, max={np.nanmax(grn_refl):.6f}, mean={np.nanmean(grn_refl):.6f}")
        print(f"Reflectance-Norm Blue:  min={np.nanmin(blu_refl):.6f}, max={np.nanmax(blu_refl):.6f}, mean={np.nanmean(blu_refl):.6f}")

        red_refl, grn_refl, blu_refl = auto_enhance_rgb_histogram(
            red_refl, grn_refl, blu_refl,
            strength=0.9,
            debug=False
        )

        rgb = np.dstack([red_refl, grn_refl, blu_refl])

        if gamma != 1.0:
            rgb = np.power(rgb, 1.0 / gamma)

        print(f"Enhanced RGB:")
        print(f"  Red:   min={np.nanmin(rgb[:,:,0]):.6f}, max={np.nanmax(rgb[:,:,0]):.6f}, mean={np.nanmean(rgb[:,:,0]):.6f}")
        print(f"  Green: min={np.nanmin(rgb[:,:,1]):.6f}, max={np.nanmax(rgb[:,:,1]):.6f}, mean={np.nanmean(rgb[:,:,1]):.6f}")
        print(f"  Blue:  min={np.nanmin(rgb[:,:,2]):.6f}, max={np.nanmax(rgb[:,:,2]):.6f}, mean={np.nanmean(rgb[:,:,2]):.6f}")
        
        print(f"Final RGB: shape={rgb.shape}, min={np.nanmin(rgb):.6f}, max={np.nanmax(rgb):.6f}, mean={np.nanmean(rgb):.6f}")
        print("True-color composite created.")
        
        return rgb, alpha, self.lats, self.lons
    
    # ... rest of methods unchanged ...
    
    def clear_cache(self):
        self._cache = {}
        print("Cache cleared.")

    def create_natural_color(self,
                            gamma: float = 2.2,
                            veggie_factor: float = 0.4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a natural-color (vegetation-corrected) RGB composite.
        Uses NIR channel to enhance vegetation appearance.
        
        Parameters:
        -----------
        gamma : float
            Gamma correction for display
        veggie_factor : float
            Weight for NIR contribution to green channel (0-1)
            Higher values = greener vegetation
        
        Returns:
        --------
        tuple : (rgb_array, lats, lons)
        """
        # Get reflectance channels
        blue_refl, red_refl, grn_refl = self.get_reflectance_channels(veggie_factor=veggie_factor)
        
        print("Creating natural-color composite...")
        
        # Normalize channels
        blue_norm = self._normalize_channel(blue_refl, gamma=gamma)
        red_norm = self._normalize_channel(red_refl, gamma=gamma)
        green_norm = self._normalize_channel(grn_refl, gamma=gamma)
        
        # Stack into RGB
        rgb = np.dstack([red_norm, green_norm, blue_norm])
        
        print("Natural-color composite created.")
        
        return rgb, self.lats, self.lons
    
    def create_day_snow_fog(self, gamma: float = 1.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a Day Snow-Fog RGB composite.
        Uses NIR in red channel to distinguish snow/ice from clouds.
        
        Red:   NIR reflectance - Snow appears bright, clouds darker
        Green: Red reflectance - Distinguishes vegetation
        Blue:  Blue reflectance - Standard blue channel
        
        Snow/ice has high reflectance in all channels → appears white
        Clouds have lower NIR reflectance → appear cyan
        Vegetation has low visible, high NIR → appears green
        
        Parameters:
        -----------
        gamma : float
            Gamma correction (default 1.7 for this product)
        
        Returns:
        --------
        tuple : (rgb_array, lats, lons)
        """
        # Get reflectance channels
        blue_refl, red_refl, nir_refl = self.get_reflectance_channels()
        
        print("Creating day snow-fog composite...")
        
        # Snow-fog specific normalization
        red_channel = self._normalize_channel(nir_refl, gamma=gamma)
        green_channel = self._normalize_channel(red_refl, gamma=gamma)
        blue_channel = self._normalize_channel(blue_refl, gamma=gamma)
        
        rgb = np.dstack([red_channel, green_channel, blue_channel])
        
        print("Day snow-fog composite created.")
        
        return rgb, self.lats, self.lons
    
    def create_day_cloud_phase(self, gamma: float = 1.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a Day Cloud Phase RGB composite.
        Helps distinguish ice clouds from water clouds.
        
        Red:   NIR reflectance - Ice vs water distinction
        Green: NIR reflectance - Particle size sensitivity  
        Blue:  Blue reflectance - Cloud thickness
        
        Ice clouds → yellow/orange
        Water clouds → cyan/blue
        
        Parameters:
        -----------
        gamma : float
            Gamma correction
        
        Returns:
        --------
        tuple : (rgb_array, lats, lons)
        """
        # Get reflectance channels
        blue_refl, red_refl, nir_refl = self.get_reflectance_channels()
        
        print("Creating day cloud-phase composite...")
        
        # Cloud phase specific normalization
        red_channel = self._normalize_channel(nir_refl, vmin=0, vmax=0.6, gamma=gamma)
        green_channel = self._normalize_channel(nir_refl, vmin=0, vmax=0.7, gamma=gamma)
        blue_channel = self._normalize_channel(blue_refl, vmin=0, vmax=1.0, gamma=gamma)
        
        rgb = np.dstack([red_channel, green_channel, blue_channel])
        
        print("Day cloud-phase composite created.")
        
        return rgb, self.lats, self.lons
    
    def get_solar_zenith_mask(self, threshold_fraction: float = 0.01) -> np.ndarray:
        """
        Create a mask for areas with sufficient solar illumination.
        
        Parameters:
        -----------
        threshold_fraction : float
            Fraction of maximum ISR to use as threshold (default=0.01 = 1%)
            Areas with ISR < threshold are masked (too dark)
        
        Returns:
        --------
        np.ndarray : Boolean mask (True = daytime, False = night/twilight)
        """
        # Use blue band ISR as proxy for solar illumination
        isr_blue = self._load_variable(self.isr_blue)
        
        # Threshold based on fraction of maximum ISR
        threshold = np.nanmax(isr_blue) * threshold_fraction
        mask = isr_blue > threshold
        
        return mask
    
    def clear_cache(self):
        """Clear the variable cache to free memory."""
        self._cache = {}
        print("Cache cleared.")

# ------------------------------------------------------------------
# Reflectivity colormap + levels (wxmaps-style)
# ------------------------------------------------------------------

COLORS = load_color_table(
        "/discover/nobackup/projects/gmao/g6dev/pub/ColorTables/NESDIS_IR_10p3micron.txt"
    )

clevs = [-110., -59, -20, 6, 31, 57] # Celcius
LEVELS = np.interp(5 * np.arange(256) / 255.0, np.arange(len(clevs)), clevs)

# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------

@register("geocolor_rgb")
def plot_geocolor_rgb(fig, ax, plotter, reader, args):
    """
    Plot GeoColor RGB Image
    """

    # ------------------------------------------------------------
    # Longwave IR night image
    # ------------------------------------------------------------
    data, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["TBRB06RG"]
    )
    data = data.astype(np.float32) - 273.15 # Celcius
    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    COLORS = load_color_table(
            "/discover/nobackup/projects/gmao/g6dev/pub/ColorTables/NESDIS_IR_10p3micron.txt"
        )
    clevs = [-110., -59, -20, 6, 31, 57] # Celcius
    LEVELS = np.interp(5 * np.arange(256) / 255.0, np.arange(len(clevs)), clevs)
    cmap = cm.get_cmap('gray_r', 256)  # discrete 256 colors
    norm = BoundaryNorm(LEVELS, ncolors=cmap.N, clip=True)
    normalized = norm(data)
    print(f"IR normalized: min={np.nanmin(normalized):.6f}, max={np.nanmax(normalized):.6f}, mean={np.nanmean(normalized):.6f}")
    ir_rgba = cmap(normalized)
    # create IR alpha layer
    ir_rgba[..., 3] = np.clip(255-normalized, 0.0, 125.0)/125.0
    # ------------------------------------------------------------
    # True Color Daytime field
    # ------------------------------------------------------------
    # Create RGB compositor
    rgb_maker = DaytimeRGB(reader, args.fdate, args.pdate)
    # Get RGB array and coordinates
    rgb, alpha, lats, lons = rgb_maker.create_true_color()
    rgba = np.dstack([rgb, alpha])   # shape (H, W, 4)
    # ------------------------------------------------------------
    # Blend IR + RGB  (night replacement)
    # ------------------------------------------------------------
    # Get day/night mask from RGB alpha
    day_mask = rgba[..., 3]  # 1.0 = full daylight, 0.0 = full night
    # Blend RGB colors: daylight shows RGB, nighttime shows IR
    rgba[..., :3] = (
        day_mask[..., None] * rgba[..., :3] +           # Daylight RGB
        (1 - day_mask[..., None]) * ir_rgba[..., :3]    # Nighttime IR colors
    )
    # Blend alpha channels: daylight is opaque, nighttime uses IR alpha
    # This makes cold clouds transparent at night, warm surfaces opaque
    rgba[..., 3] = (
        day_mask * 1.0 +                    # Daylight: fully opaque
        (1 - day_mask) * ir_rgba[..., 3]    # Nighttime: IR alpha (temp-based)
    )
    # Limb Darkening
    rgba = plotter.apply_limb_darkening(
        rgba,
        lats,
        lons
    )
    # The RGB array is shape (height, width, 3) with values in range [0, 1]
    # Plot image on map - let Cartopy handle the reprojection
    start = time.time()
    from wxmaps_transform_cache import get_transform_cache
    transform_cache = get_transform_cache()
    # Get or load cached transform
    target_extent = ax.get_extent()
    target_shape = (2160, 4320)  # height, width for target
    transform_data, cache_key = transform_cache.get_or_compute_transform(
        source_proj=ccrs.PlateCarree(),
        target_proj=ax.projection,
        source_extent=[-180, 180, -90, 90],  # Adjust to your data extent
        target_extent=target_extent,
        target_shape=target_shape
    )
    # Apply the cached transform to this worker's data
    warped_rgba, warped_extent = transform_cache.apply_transform(
        source_data=rgba,
        transform_data=transform_data,
        fill_value=0.0,
        flip_source_y=True  # Don't flip for NetCDF data
    )
    # Plot the transformed image (FAST!)
    im = ax.imshow(
        warped_rgba,
        origin='upper',
        extent=warped_extent,
        transform=ax.projection,
        interpolation='antialiased',
        resample=False,
        zorder=4,
        rasterized=True
    )
    print(f"GeoColor composite finished: {time.time() - start:.2f}s")

def generate_colorbar():
    """Generate colorbar for longwave window IR"""
    from wxmaps_utils import save_colorbar_single
    
    # Use representative tick levels instead of all 256
    tick_levels = np.array([-110, -80, -50, -20, 0, 20, 40, 57])
    
    output = "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/longwave_window_ir.png"
    save_colorbar_single(
        COLORS, 
        LEVELS, 
        output, 
        label="11.2 μm Longwave Window Brightness Temperature (°C)",
        extend='both'
    )


