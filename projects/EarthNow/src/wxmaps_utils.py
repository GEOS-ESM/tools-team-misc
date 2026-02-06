"""
WxMaps Utility Module
Helper functions for file paths, date parsing, and directory management
"""
import os
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional


def parse_date_string(date_str: str) -> datetime:
    """
    Parse date string in format YYYYMMDD_HHz or YYYYMMDD_HHMMz
    
    Parameters:
    -----------
    date_str : str
        Date string (e.g., '20260116_00z' or '20260118_1200z')
    
    Returns:
    --------
    datetime object
    """
    # Remove 'z' suffix if present
    date_str = date_str.rstrip('z')
    
    # Try different formats
    formats = [
        '%Y%m%d_%H',      # 20260116_00
        '%Y%m%d_%H%M',    # 20260116_0000
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Unable to parse date string: {date_str}")


def format_fdate_for_filename(dt: datetime) -> str:
    """Format datetime object for filename (YYYYMMDD_HHz)"""
    return dt.strftime('%Y%m%d_%Hz')

def format_pdate_for_filename(dt: datetime) -> str:
    """Format datetime object for filename (YYYYMMDD_HHMMz)"""
    return dt.strftime('%Y%m%d_%H%Mz')


def format_date_for_display(dt: datetime) -> str:
    """Format datetime object for display (YYYY-MM-DD HH:MM UTC)"""
    return dt.strftime('%Y-%m-%d %H:%M UTC')


def calculate_forecast_hour(fdate: datetime, pdate: datetime) -> int:
    """Calculate forecast hour from forecast and valid dates"""
    delta = pdate - fdate
    return int(delta.total_seconds() / 3600)


def create_output_path(base_path: str, exp_res: str, exp_id: str, 
                      product: str, map_type: str,
                      fdate: str, pdate: str) -> Tuple[str, str]:
    """
    Create output directory path and filename with new naming convention
    
    Parameters:
    -----------
    base_path : str
        Base directory path
    exp_res : str
        Experiment resolution (e.g., 'CONUS02KM')
    exp_id : str
        Experiment ID (e.g., 'Feature-c2160_L137')
    product : str
        Product name (e.g., 'composite-reflectivity', 'basemaps')
    map_type : str
        Map type (e.g., 'conus', 'europe')
    fdate : str
        Forecast date (e.g., '20260116_00z')
    pdate : str
        Valid date (e.g., '20260116_1200z')
    
    Returns:
    --------
    output_dir : str
        Full output directory path
    filename : str
        Output filename
    """
    # Parse dates
    pdate_dt = parse_date_string(pdate)
    fdate_dt = parse_date_string(fdate)
    
    # Format dates for filename
    fdate_str = format_fdate_for_filename(fdate_dt)
    pdate_str = format_pdate_for_filename(pdate_dt)
    
    # Create directory structure
    year_dir = f"Y{pdate_dt.year}"
    month_dir = f"M{pdate_dt.month:02d}"
    day_dir = f"D{pdate_dt.day:02d}"
    
    # Product directory name (convert hyphens to underscores, uppercase)
    product_dir = f"PLOTALL_{product.upper().replace('-', '_')}"
    
    output_dir = os.path.join(
        base_path,
        f"Ops{exp_res}",
        product_dir,
        year_dir,
        month_dir,
        day_dir
    )
    
    # Create filename: plotall_{product}_{map}_{exp_res}_{exp_id}.fdate.{fdate}.pdate.{pdate}.png
    filename = f"plotall_{product}_{map_type}_{exp_res}_{exp_id}.fdate.{fdate_str}.pdate.{pdate_str}.png"
    
    return output_dir, filename


def get_output_filepath(base_path: str, exp_res: str, exp_id: str,
                       product: str, map_type: str,
                       fdate: str, pdate: str) -> str:
    """
    Get full output filepath, creating directories as needed
    
    Returns:
    --------
    filepath : str
        Full path to output file
    """
    output_dir, filename = create_output_path(base_path, exp_res, exp_id, 
                                              product, map_type, fdate, pdate)
    ensure_directory_exists(output_dir)
    return os.path.join(output_dir, filename)

def ensure_directory_exists(directory: str):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def normalize(data, vmin=None, vmax=None, gamma=1.0):

    data = np.asarray(data, dtype=np.float32)
    mask = ~np.isfinite(data)

    if vmin is None:
        vmin = np.nanpercentile(data, 0.5)
    if vmax is None:
        vmax = np.nanpercentile(data, 99.5)

    if vmax == vmin:
        vmax += 1e-6

    norm = (data - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)

    if gamma != 1:
        norm = np.power(norm, 1/gamma)

    norm[mask] = 0

    return norm.astype(np.float32)

def auto_enhance_rgb_luminance(rgb, strength=0.35, debug=False):
    """
    IDL-style auto_enhance_rgb (luminance preserving)
    """

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

def auto_enhance_rgb_histogram(
    red, green, blue, strength=0.5, debug=False
):
    """
    IDL-style auto_enhance_rgb (HISTOGRAM method only)

    Parameters
    ----------
    red, green, blue : np.ndarray
        RGB channels, float, expected in [0,1], NaNs allowed
    strength : float
        Enhancement strength [0,1]
    debug : bool

    Returns
    -------
    enhanced_red, enhanced_green, enhanced_blue : np.ndarray
    """

    # Clamp strength
    strength = np.clip(strength, 0.0, 1.0)

    # Copy inputs (preserve NaNs)
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

        # IDL SORT + percentile index behavior
        sorted_data = np.sort(data)
        n = sorted_data.size

        low_idx  = max(int(n * low_pct), 0)
        high_idx = min(int(n * high_pct), n - 1)

        min_val = sorted_data[low_idx]
        max_val = sorted_data[high_idx]

        if debug:
            print(
                f"{name}: n={n} "
                f"min_val={min_val:.4f} max_val={max_val:.4f}"
            )

        if min_val == max_val:
            continue

        # Stretch
        stretched = (data - min_val) / (max_val - min_val)

        # Blend original + stretched
        result = (1.0 - strength) * data + strength * stretched

        # Clamp
        result = np.clip(result, 0.0, 1.0)

        # Write back only valid pixels
        dst[valid] = result

    return out_r, out_g, out_b

def load_color_table(filepath):
    """
    Load a color table from a text file.
    
    Parameters
    ----------
    filepath : str
        Path to the color table file (space-separated RGB values, 0-255)
    
    Returns
    -------
    colors : np.ndarray
        Color table normalized to [0, 1], shape (N, 3)
    """
    import numpy as np
    colors = np.loadtxt(filepath)
    return colors / 255.0

def save_colorbar_grid(colorbar_specs, output_path, title="", 
                       width=6600, height=600, grid_shape=(2, 2)):
    """
    Generate a colorbar PNG with multiple colorbars arranged in a grid.
    
    Parameters
    ----------
    colorbar_specs : list of dict
        List of colorbar specifications, each containing:
        - 'colors': array (N, 3) normalized to 0-1
        - 'levels': array of contour levels
        - 'label': label string
        - 'extend': 'neither', 'both', 'min', 'max' (default 'neither')
    output_path : str
        Full path to save PNG
    title : str
        Overall title for the figure
    width, height : int
        Total image dimensions in pixels
    grid_shape : tuple
        (nrows, ncols) for grid layout
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    
    nrows, ncols = grid_shape
    dpi = 100
    figsize = (width / dpi, height / dpi)
    
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='white')
    
    # Add title if provided with bigger font
    if title:
        fig.suptitle(title, fontsize=42, fontweight='bold', y=0.98)
    
    # Create grid of subplots
    for idx, spec in enumerate(colorbar_specs):
        if idx >= nrows * ncols:
            break
            
        # Position: [left, bottom, width, height]
        row = idx // ncols
        col = idx % ncols
        
        # Calculate position with larger margins
        left_margin = 0.15      # 15% left margin
        right_margin = 0.15     # 15% right margin
        top_margin = 0.20       # 20% top margin
        bottom_margin = 0.20    # 20% bottom margin
        h_spacing = 0.08        # 8% horizontal spacing between panels
        v_spacing = 0.12        # 12% vertical spacing between panels
        
        plot_width = (1.0 - left_margin - right_margin - h_spacing * (ncols - 1)) / ncols
        plot_height = (1.0 - top_margin - bottom_margin - v_spacing * (nrows - 1)) / nrows
        
        left = left_margin + col * (plot_width + h_spacing)
        bottom = 1.0 - top_margin - (row + 1) * plot_height - row * v_spacing
        
        # Make colorbar thinner (reduce height)
        cbar_height = plot_height * 0.4  # 30% of available height
        bottom_adjusted = bottom + (plot_height - cbar_height) / 2
        
        ax = fig.add_axes([left, bottom_adjusted, plot_width, cbar_height])
        
        # Create colormap
        cmap = ListedColormap(spec['colors'])
        norm = BoundaryNorm(spec['levels'], ncolors=cmap.N, clip=True)
        
        # Create colorbar
        extend = spec.get('extend', 'neither')
        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation='horizontal',
            extend=extend
        )
        # Set colorbar outline width
        cb.outline.set_linewidth(3)
        
        # Set label with even bigger font
        cb.set_label(spec['label'], fontsize=36, fontweight='bold', labelpad=12)
        cb.ax.tick_params(labelsize=24, width=3, length=10)
        
        # Set tick positions (every other level for clarity)
        tick_positions = spec['levels'][::2]
        cb.set_ticks(tick_positions)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', pad_inches=0.1)
    plt.close(fig)
    
    print(f"Saved colorbar to: {output_path}")

def save_colorbar_single(colors, levels, output_path, label="", 
                         width=6600, height=600, extend='neither'):
    """
    Generate a single horizontal colorbar PNG.
    
    Parameters
    ----------
    colors : array-like
        Color array (N, 3) normalized to 0-1
    levels : array-like
        Contour levels
    output_path : str
        Full path to save PNG
    label : str
        Label for the colorbar
    width, height : int
        Image dimensions in pixels
    extend : str
        'neither', 'both', 'min', 'max'
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import os
    
    dpi = 100
    figsize = (width / dpi, height / dpi)
    
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='white')
    
    # Single axes with margins
    ax = fig.add_axes([0.15, 0.35, 0.70, 0.25])  # [left, bottom, width, height]
    
    # Create colormap
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    # Create colorbar
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='horizontal',
        extend=extend
    )
    
    # Set label with large font
    cb.set_label(label, fontsize=48, fontweight='bold', labelpad=15)
    cb.ax.tick_params(labelsize=42, width=4, length=12)
    # Set colorbar outline width
    cb.outline.set_linewidth(3)
    
    # Set tick positions
    if len(levels) > 20:
        tick_positions = levels[::len(levels)//10]  # ~10 ticks
    else:
        tick_positions = levels[::2]
    cb.set_ticks(tick_positions)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', pad_inches=0.2)
    plt.close(fig)
    
    print(f"Saved colorbar to: {output_path}")
