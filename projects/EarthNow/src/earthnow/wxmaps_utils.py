"""
WxMaps Utility Module
Helper functions for file paths, date parsing, and directory management
"""

import os
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Optional
import matplotlib.pyplot as plt


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
    date_str = date_str.rstrip("z")

    # Try different formats
    formats = [
        "%Y%m%d_%H",  # 20260116_00
        "%Y%m%d_%H%M",  # 20260116_0000
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unable to parse date string: {date_str}")


def format_fdate_for_filename(dt: datetime) -> str:
    """Format datetime object for filename (YYYYMMDD_HHz)"""
    return dt.strftime("%Y%m%d_%Hz")


def format_pdate_for_filename(dt: datetime) -> str:
    """Format datetime object for filename (YYYYMMDD_HHMMz)"""
    return dt.strftime("%Y%m%d_%H%Mz")


def format_date_for_display(dt: datetime) -> str:
    """Format datetime object for display (YYYY-MM-DD HH:MM UTC)"""
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def calculate_forecast_hour(fdate: datetime, pdate: datetime) -> int:
    """Calculate forecast hour from forecast and valid dates"""
    delta = pdate - fdate
    return int(delta.total_seconds() / 3600)


def create_output_path(
    base_path: str,
    exp_res: str,
    exp_id: str,
    product: str,
    map_type: str,
    fdate: str,
    pdate: str,
) -> Tuple[str, str]:
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
        base_path, f"Ops{exp_res}", product_dir, year_dir, month_dir, day_dir
    )

    # Create filename: plotall_{product}_{map}_{exp_res}_{exp_id}.fdate.{fdate}.pdate.{pdate}.png
    filename = f"plotall_{product}_{map_type}_{exp_res}_{exp_id}.fdate.{fdate_str}.pdate.{pdate_str}.png"

    return output_dir, filename


def get_output_filepath(
    base_path: str,
    exp_res: str,
    exp_id: str,
    product: str,
    map_type: str,
    fdate: str,
    pdate: str,
) -> str:
    """
    Get full output filepath, creating directories as needed

    Returns:
    --------
    filepath : str
        Full path to output file
    """
    output_dir, filename = create_output_path(
        base_path, exp_res, exp_id, product, map_type, fdate, pdate
    )
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
        norm = np.power(norm, 1 / gamma)

    norm[mask] = 0

    return norm.astype(np.float32)


def auto_enhance_rgb_luminance(rgb, strength=0.35, debug=False):
    """
    IDL-style auto_enhance_rgb (luminance preserving)
    """

    strength = np.clip(strength, 0.0, 1.0)

    # RGB → luminance (IDL weights)
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

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
        ("Red", red, out_r),
        ("Green", green, out_g),
        ("Blue", blue, out_b),
    ]

    low_pct = (2.0 - strength) / 100.0
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

        low_idx = max(int(n * low_pct), 0)
        high_idx = min(int(n * high_pct), n - 1)

        min_val = sorted_data[low_idx]
        max_val = sorted_data[high_idx]

        if debug:
            print(f"{name}: n={n} " f"min_val={min_val:.4f} max_val={max_val:.4f}")

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


def colorbar_alpha_fade(
    cmap,
    pct_float: float,
    ncolors=256,
):
    """
    Apply an alpha fade to an existing colormap
    Parameters
    -------------
    cmap: matplotlib.colors.Colormap
    """
    if not (0 <= pct_float <= 1):
        raise ValueError("Percent must be between 0 and 1")
    color_matrix = cmap(np.linspace(0, 1, ncolors))
    alphas = np.ones(ncolors)
    n_fade = int(256 * pct_float)
    alphas[:n_fade] = np.linspace(0, 1, n_fade)
    color_matrix[:, -1] = alphas

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(color_matrix)
    return cmap


def build_colorbar(
    fig,
    ax,
    mappable,
    ticks,
    label="",
    format=None,
    **kwargs,
):
    """
    Build a single horizontal colorbar PNG given an existing fig, ax

    Parameters
    ----------
    fig: matplotlib.pyplot.figure
        Colorbar figure
    axis: matplotlib.pyplot.axis
        Colorbar axis
    mappable: matplotlib.pyplot.plot or matplotlib.colorizer.ColorizingArtist
        Mappable object
    ticks : np.array
        Colorbar tickmarks
    label : str
        Label for the colorbar
    format: str
        Tick value formatting
    """

    cbar_kwargs = {}
    cb = fig.colorbar(mappable, cax=ax, orientation="horizontal")
    if format:
        cbar_kwargs["format"] = format
    cbar_kwargs.update(kwargs)
    cb = fig.colorbar(mappable, cax=ax, orientation="horizontal", **cbar_kwargs)

    # Set label with large font
    cb.set_label(label, fontsize=96, loc="left", labelpad=30, color="white")
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=48, width=4, length=12, colors="white")
    cb.ax.tick_params(length=0)
    cb.ax.minorticks_off()

    tick_positions = ticks
    cb.set_ticks(tick_positions)

    return fig


def save_colorbar_single(
    plot,
    output_path,
    ticks,
    label="",
    width=6600,
    height=600,
):
    """
    Generate a single horizontal colorbar PNG.

    Parameters
    ----------
    plot: matplotlib.pyplot obj
    output_path : str
        Full path to save PNG
    ticks : np.array
        Colorbar tickmarks
    label : str
        Label for the colorbar
    width, height : int
        Image dimensions in pixels
    """

    dpi = 100
    figsize = (width / dpi, height / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("none")

    # Single axes with margins
    ax = fig.add_axes([0.15, 0.20, 0.70, 0.25])  # [left, bottom, width, height]

    build_colorbar(fig, ax, plot, ticks, label)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(
        output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2, transparent=True
    )
    plt.close(fig)

    print(f"Saved colorbar to: {output_path}")


def build_and_save_colorbars(
    mappables,
    ticks,
    output_path,
    labels,
    formats=None,
    label_fontsize=96,
    tick_labelsize=48,
    text_color="white",
    bg_color=None,
    tick_width=4,
    tick_length=0,
    width=6600,
    height=600,
    dpi=100,
    cbar_left=0.15,  # Left margin (figure fraction)
    cbar_bottom=0.2,  # Distance from bottom (figure fraction)
    cbar_width=0.7,  # Colorbar width (figure fraction)
    cbar_height=0.25,  # Colorbar height (figure fraction)
    **kwargs,
):
    """
    Dynamically generates a colorbar figure with single or multiple rows of colorbars and saves it. This function completely replaces the build_colorbar + save_colorbar_single.


    Parameters
    ----------
    mappables : object or list
        Single mappable object or list of mappable objects.
    ticks : array-like or list of array-like
        Single tickmark array or list of arrays.
    labels : str or list of str, optional
        Label or list of labels. Defaults to empty strings.
    formats : str or list of str, optional
        Format string or list of format strings.
    save_path : str
        Filepath to save the final image (default: "colorbars.png").
    figsize : tuple, optional
        Figure size. If None, it dynamically scales based on the number of rows.
    label_fontsize : int, optional
        Font size for the label (default: 96)
    """

    # 1. Standardize inputs to lists if single objects are provided
    if not isinstance(mappables, (list, tuple)):
        mappables = [mappables]
        ticks = [ticks]
        labels = [labels]
        if formats is not None and isinstance(formats, str):
            formats = [formats]

    n_bars = len(mappables)

    # 2. Handle optional lists
    if formats is None:
        formats = [None] * n_bars

    #  Validate lenghts
    if not (n_bars == len(mappables) == len(ticks) == len(labels) == len(formats)):
        raise ValueError(
            "axes, mappables, ticks, labels, and formats must have the same length."
        )

    # 3. Generate Figure and Axes dynamically
    height = 600 * n_bars
    figsize = (width / dpi, height / dpi)

    fig, axes = plt.subplots(nrows=n_bars, ncols=1, figsize=figsize)
    fig.patch.set_facecolor(bg_color)

    # Ensure axes is iterable
    if n_bars == 1:
        axes = [axes]
    elif hasattr(axes, "flatten"):
        axes = axes.flatten()

    colorbars = []

    for i, (ax, mappable, tick_arr, label, fmt) in enumerate(
        zip(axes, mappables, ticks, labels, formats)
    ):
        # Set strict axis positioning
        row_idx_from_bottom = n_bars - 1 - i
        bottom = (row_idx_from_bottom + cbar_bottom) / n_bars
        ax.set_position([cbar_left, bottom, cbar_width, cbar_height / n_bars])
        # ax.set_position([0.15, 0.2 * (i + 1) * n_bars, 0.70, 0.25 / n_bars])
        cbar_kwargs = kwargs.copy()
        if fmt is not None:
            cbar_kwargs["format"] = fmt

        # Generate the colorbar on  the specific axis
        cb = fig.colorbar(mappable, cax=ax, orientation="horizontal", **cbar_kwargs)

        # Set label with configurable properties
        cb.set_label(
            label, fontsize=label_fontsize, loc="left", labelpad=30, color=text_color
        )
        cb.ax.xaxis.set_label_position("top")

        # Format and set ticks
        cb.ax.tick_params(
            labelsize=tick_labelsize,
            width=tick_width,
            length=tick_length,
            colors=text_color,
        )
        cb.ax.minorticks_off()
        cb.set_ticks(tick_arr)

        colorbars.append(cb)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(
        output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2, transparent=True
    )
    plt.close(fig)

    print(f"Saved colorbar to: {output_path}")
