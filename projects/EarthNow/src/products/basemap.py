"""
basemap_product.py

Basemap-only product for WxMaps.
Used to validate map configuration, styling, projections, and boundaries.
"""

from typing import Optional
from datetime import datetime
from .registry import register

from wxmaps_plotting import WxMapPlotter

@register("basemap")
def plot_basemap(fig, ax, plotter: WxMapPlotter, reader, args):
    """
    Plot a basemap using WxMapPlotter.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure created by plotter
    ax : matplotlib.axes.Axes
        Axes created by plotter
    plotter : WxMapPlotter
        Initialized plotter instance
    reader : object
        Data reader (unused for basemap, included for API consistency)
    args : argparse.Namespace
        Parsed CLI arguments
    """

    # ------------------------------------------------------------
    # Create basemap background + projection
    # ------------------------------------------------------------
    fig, ax = plotter.create_basemap(
        boundaries=None,  # boundaries added explicitly below
        feature_resolution=args.feature_resolution
        if hasattr(args, "feature_resolution") else "10m",
    )

    # ------------------------------------------------------------
    # Draw boundaries (order handled internally)
    # ------------------------------------------------------------
    if args.boundaries:
        plotter.add_boundaries(args.boundaries)

    # ------------------------------------------------------------
    # Optional overlays (safe no-ops if disabled)
    # ------------------------------------------------------------
    if getattr(args, "cities", False):
        plotter.add_cities()

    if getattr(args, "roads", False):
        plotter.add_roads(
            road_scale="10m" if plotter.resolution in ("4k", "8k") else "50m",
            major_only=getattr(args, "major_roads_only", False),
        )

    # ------------------------------------------------------------
    # Timestamp (wxmaps-style)
    # ------------------------------------------------------------
    if args.fdate and args.pdate:
        plotter.add_forecast_timestamp(
            fdate=args.fdate,
            pdate=args.pdate,
            exp=args.experiment if hasattr(args, "experiment") else "",
        )

