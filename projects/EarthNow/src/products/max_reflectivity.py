"""
Radar Reflectivity Product
Maximum Composite Reflectivity (DBZ_MAX / REFC)
"""

import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from .registry import register

# ------------------------------------------------------------------
# Reflectivity colormap + levels (wxmaps-style)
# ------------------------------------------------------------------

REFL_COLORS = np.array([
    [  0, 224, 227],  # Light cyan
    [  0, 141, 243],  # Blue
    [  0,  12, 243],  # Dark blue
    [  0, 239,   8],  # Bright green
    [  0, 183,   0],  # Green
    [  0, 123,   0],  # Dark green
    [255, 246,   0],  # Yellow
    [228, 173,   0],  # Gold
    [255, 129,   0],  # Orange
    [255,   0,   0],  # Red
    [209,   0,   0],  # Dark red
    [180,   0,   0],  # Darker red
    [249,   7, 253],  # Magenta
    [133,  67, 186],  # Purple
    [245, 245, 245],  # White
]) / 255.0

REFL_LEVELS = np.arange(5.0, 80.0, 5.0)  # 5–75 dBZ


# ------------------------------------------------------------------
# Main product function
# ------------------------------------------------------------------

@register("max_reflectivity")
def plot_max_reflectivity(fig, ax, plotter, reader, args):
    """
    Plot maximum composite radar reflectivity (dBZ)
    """
    # Read from reader (reader decides the collection)
    data, lats, lons, meta = reader.read_variable(
        args.fdate,
        args.pdate,
        variables=["REFC", "DBZ_MAX"]
    )

    data = data.astype(np.float32)

    # Mask invalid reflectivity
    data = np.ma.masked_where(data < 0.0, data)
    data = np.ma.masked_where(data > 80.0, data)

    # ------------------------------------------------------------
    # Report data resolution
    # ------------------------------------------------------------
    data_shape = data.shape
    print(f"Data resolution: {data_shape[0]} x {data_shape[1]} (height x width)")
    print(f"  Total data points: {data_shape[0] * data_shape[1]:,}")
    
    # Calculate approximate grid spacing
    if lons.ndim == 1 and lats.ndim == 1:
        lon_spacing = (lons.max() - lons.min()) / (len(lons) - 1)
        lat_spacing = (lats.max() - lats.min()) / (len(lats) - 1)
        print(f"  Grid spacing: {lon_spacing:.4f}° lon x {lat_spacing:.4f}° lat")
    elif lons.ndim == 2 and lats.ndim == 2:
        lon_spacing = np.median(np.diff(lons[0, :]))
        lat_spacing = np.median(np.diff(lats[:, 0]))
        print(f"  Grid spacing (median): {lon_spacing:.4f}° lon x {lat_spacing:.4f}° lat")
    
    # ------------------------------------------------------------
    # Colormap + normalization
    # ------------------------------------------------------------
    cmap = ListedColormap(REFL_COLORS)
    norm = BoundaryNorm(REFL_LEVELS, ncolors=cmap.N, clip=True)

    # ------------------------------------------------------------
    # Plot field
    # ------------------------------------------------------------
    ax.pcolormesh(
        lons,
        lats,
        data,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="nearest",
        zorder=4,
        rasterized=True
    )
    
    # ------------------------------------------------------------
    # Report image resolution
    # ------------------------------------------------------------
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_inches, height_inches = bbox.width, bbox.height
    dpi = fig.dpi
    
    img_width_px = int(width_inches * dpi)
    img_height_px = int(height_inches * dpi)
    
    print(f"Image resolution: {img_height_px} x {img_width_px} pixels (height x width)")
    print(f"  Figure size: {width_inches:.2f} x {height_inches:.2f} inches")
    print(f"  DPI: {dpi}")
    print(f"  Total image pixels: {img_height_px * img_width_px:,}")
    print(f"  Pixel ratio (image/data): {(img_height_px * img_width_px) / (data_shape[0] * data_shape[1]):.2f}x")

def generate_colorbar():
    """Generate colorbar for max reflectivity"""
    from wxmaps_utils import save_colorbar_single
    
    output = "/discover/nobackup/projects/gmao/g6dev/pub/WxMaps/ColorBars/max_reflectivity.png"
    save_colorbar_single(
        REFL_COLORS, 
        REFL_LEVELS, 
        output, 
        label="Composite Reflectivity (dBZ)",
        extend='max'
    )
