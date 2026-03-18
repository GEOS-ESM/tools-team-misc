"""
WxMaps Colorbars Utility Module
Helper functions and class to generate colormaps and colorbars
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List, Literal, Sequence

import numpy as np

from .paths import ensure_directory_exists

__all__ = (
    "load_color_table",
    "ColorbarSpec",
    "save_colorbar_grid",
    "save_colorbar_single",
)


_log = setup_logger("earthnow.utils.colorbars")

ExtendType = Literal["neither", "both", "min", "max"]


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


@dataclass
class ColorbarSpec:
    product: str
    colors: np.ndarray  # shape (N, 3), normalized 0–1
    levels: np.ndarray
    label: str = ""
    extend: ExtendType = "neither"
    maptype: Literal["continuous","discrete"] = "discrete"
    clip: bool = True

    def __post_init__(self):
        self.colors  = np.asarray(self.colors, dtype=float)
        self.levels  = np.asarray(self.levels, dtype=float)
        self.extend  = self.extend.lower()
        self.maptype = self.maptype.lower() 

        if self.colors.ndim != 2 or self.colors.shape[1] != 3:
            raise ValueError("colors must be shape (N, 3)")

        if not np.all((0.0 <= self.colors) & (self.colors <= 1.0)):
            raise ValueError("colors must be normalized to [0, 1]")

        if not np.all(np.diff(self.levels) >= 0):
            raise ValueError("levels must be monotonically increasing")

        if self.extend not in ("neither", "both", "min", "max"):
            raise ValueError("invalid extend value")

    def create_cmap(self):
        if self.maptype == "discrete":
            return create_discrete_cmap(self.product,self.colors)
        else:
            return create_continuous_cmap(self.product,self.colors,self.levels)

    def create_norm(self):
        if self.maptype == "discrete":
            return create_discrete_norm()
        else:
            return create_continuous_norm()
            
def create_discrete_cmap(
    name: str, 
    colors: Sequence, 
    *args
    ) -> ListedColormap:
    from matplotlib.colors import ListedColormap

    return ListedColormap(colors, name=name)

def create_discrete_norm(
    levels: Sequence,
    ncolors: int,
    clip: bool=True
    ):
    from matplotlib.colors import BoundaryNorm

    return BoundaryNorm(levels, ncolors=ncolors, clip=clip)
    
def create_continuous_cmap(self):
    name: str, 
    colors: Sequence, 
    levels: Sequence
    ) -> LinearSegmentedColormap:
    """Build the continuous colormap from anchor colors."""
    from matplotlib.colors import LinearSegmentedColormap
    vmin = min(levels)
    vmax = max(levels)

    positions = (np.array(levels) - vmin) / (vmax - vmin)    
    return LinearSegmentedColormap.from_list(
        name, 
        list(zip(positions, colors[:-1])), 
        N=256
    )

def create_continuous_norm(
    levels: Sequence,
    clip: bool=True,
    *args
    ):
    from matplotlib.colors import Normalize

    return Normalize(vmin=min(levels), vmax=max(levels), clip=clip)

def save_colorbar_grid(
    colorbar_specs, output_path, title="", width=6600, height=600, grid_shape=(2, 2)
):
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

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")

    # Add title if provided with bigger font
    if title:
        fig.suptitle(title, fontsize=42, fontweight="bold", y=0.98)

    # Create grid of subplots
    for idx, spec in enumerate(colorbar_specs):
        if idx >= nrows * ncols:
            break

        # Position: [left, bottom, width, height]
        row = idx // ncols
        col = idx % ncols

        # Calculate position with larger margins
        left_margin = 0.15  # 15% left margin
        right_margin = 0.15  # 15% right margin
        top_margin = 0.20  # 20% top margin
        bottom_margin = 0.20  # 20% bottom margin
        h_spacing = 0.08  # 8% horizontal spacing between panels
        v_spacing = 0.12  # 12% vertical spacing between panels

        plot_width = (
            1.0 - left_margin - right_margin - h_spacing * (ncols - 1)
        ) / ncols
        plot_height = (
            1.0 - top_margin - bottom_margin - v_spacing * (nrows - 1)
        ) / nrows

        left = left_margin + col * (plot_width + h_spacing)
        bottom = 1.0 - top_margin - (row + 1) * plot_height - row * v_spacing

        # Make colorbar thinner (reduce height)
        cbar_height = plot_height * 0.4  # 30% of available height
        bottom_adjusted = bottom + (plot_height - cbar_height) / 2

        ax = fig.add_axes([left, bottom_adjusted, plot_width, cbar_height])

        # Create colormap
        cmap = ListedColormap(spec["colors"])
        norm = BoundaryNorm(spec["levels"], ncolors=cmap.N, clip=True)

        # Create colorbar
        extend = spec.get("extend", "neither")
        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation="horizontal",
            extend=extend,
        )
        # Set colorbar outline width
        cb.outline.set_linewidth(3)

        # Set label with even bigger font
        cb.set_label(spec["label"], fontsize=36, fontweight="bold", labelpad=12)
        cb.ax.tick_params(labelsize=24, width=3, length=10)

        # Set tick positions (every other level for clarity)
        tick_positions = spec["levels"][::2]
        cb.set_ticks(tick_positions)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(
        output_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.1
    )
    plt.close(fig)

    print(f"Saved colorbar to: {output_path}")


def save_colorbar_single(
    colors, levels, output_path, label="", width=6600, height=600, extend="neither"
):
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

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")

    # Single axes with margins
    ax = fig.add_axes([0.15, 0.35, 0.70, 0.25])  # [left, bottom, width, height]

    # Create colormap
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    # Create colorbar
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="horizontal",
        extend=extend,
    )

    # Set label with large font
    cb.set_label(label, fontsize=48, fontweight="bold", labelpad=15)
    cb.ax.tick_params(labelsize=42, width=4, length=12)
    # Set colorbar outline width
    cb.outline.set_linewidth(3)

    # Set tick positions
    if len(levels) > 20:
        tick_positions = levels[:: len(levels) // 10]  # ~10 ticks
    else:
        tick_positions = levels[::2]
    cb.set_ticks(tick_positions)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(
        output_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.2
    )
    plt.close(fig)

    print(f"Saved colorbar to: {output_path}")