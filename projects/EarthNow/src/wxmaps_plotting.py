"""
WxMaps Plotting Module
Core functionality for creating base maps with various boundary features
Supports HD, 4K, and 8K resolutions with strict 16:9 aspect ratio
wxmaps design with full style control
"""
import gc
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for high-res rendering
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from typing import Optional, List, Tuple
import numpy as np
import warnings
from datetime import datetime
import pytz

from wxmaps_config import MapConfig, WxMapsConfig, ResolutionConfig, StyleConfig

# =============================================================================
# Boundary draw order (bottom → top)
# =============================================================================

PHYSICAL_BOUNDARIES = (
    "coastlines",
    "rivers",
)

POLITICAL_BOUNDARIES = (
    "countries",
    "states",
    "counties",
)

BOUNDARY_ORDER = PHYSICAL_BOUNDARIES + POLITICAL_BOUNDARIES

class WxMapPlotter:
    """Main plotting class for weather maps with HD/4K/8K support"""
    
    def __init__(self, map_config: MapConfig, resolution: str = '4k', 
                 style: Optional[StyleConfig] = None):
        """
        Initialize map plotter
        
        Parameters:
        -----------
        map_config : MapConfig
            Map configuration object
        resolution : str
            Resolution preset: 'hd', 'fhd', '2k', '4k', '8k' (default: '4k')
        style : StyleConfig, optional
            Style configuration. If None, uses wxmaps style.
        """
        self.config = map_config
        
        if resolution not in WxMapsConfig.RESOLUTIONS:
            raise ValueError(f"Unknown resolution '{resolution}'. "
                           f"Available: {list(WxMapsConfig.RESOLUTIONS.keys())}")
        
        self.resolution_config = WxMapsConfig.RESOLUTIONS[resolution]
        self.resolution = resolution
        self.figsize = self.resolution_config.figsize
        self.dpi = self.resolution_config.dpi
        
        # Use wxmaps style by default
        if style is None:
            style = StyleConfig.wxmaps()
        
        # Scale style for resolution
        self.style = WxMapsConfig.scale_style_for_resolution(style, resolution)
        
        self.fig = None
        self.ax = None
        
        print(f"Initialized {self.resolution_config.name} plotter: "
              f"{self.resolution_config.width}x{self.resolution_config.height} "
              f"({self.figsize[0]:.1f}x{self.figsize[1]:.1f} in @ {self.dpi} DPI)")
        
    def create_basemap(self, boundaries: Optional[List[str]] = None,
                      feature_resolution: str = '50m') -> Tuple[plt.Figure, plt.Axes]:
        """Create base map with specified boundaries"""
        
        if boundaries is None:
            boundaries = ['coastlines', 'countries', 'states']
        
        # Create figure with high DPI
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.fig.patch.set_facecolor('white')
        
        # Create axes with projection - no margins for wxmaps look
        self.ax = self.fig.add_axes([0, 0, 1, 1], projection=self.config.projection)
        
        # Set extent - handle geostationary projections differently
        if self.config.extent is None:
            print("Skipping set_extent")
        else:
            self.ax.set_extent(self.config.extent, crs=ccrs.PlateCarree())
        
        # Set frame properties
        if self.style.show_frame:
            for spine in self.ax.spines.values():
                spine.set_linewidth(self.style.frame_width)
                spine.set_edgecolor(self.style.frame_color)
        else:
            # Hide frame completely for wxmaps look
            self.ax.spines['geo'].set_visible(False)
        
        # ===================================================================
        # Add background - Priority: Base Image > GSHHS > Cartopy features
        # ===================================================================
        if self.style.use_base_image:
            # Use base earth image
            print("Using base earth image...")
            self._add_base_image()
        elif self.style.use_gshhs:
            # Use GSHHS for high-resolution land/water
            print("Using GSHHS for land/water boundaries...")
            self._add_gshhs_background()
        else:
            # Use default Cartopy features
            self.ax.add_feature(cfeature.OCEAN, facecolor=self.style.ocean_color, zorder=0)
            self.ax.add_feature(cfeature.LAND, facecolor=self.style.land_color, zorder=0)
            self.ax.add_feature(
                cfeature.LAKES.with_scale(feature_resolution),
                edgecolor='none',
                facecolor=self.style.ocean_color,
                zorder=0
            )
        
        # ===================================================================
        # Add boundary features (countries, states, etc.)
        # Skip coastlines if using GSHHS (already included)
        # ===================================================================
        if 'coastlines' in boundaries:
            print("  Adding Cartopy coastline+lake borders")
            self.ax.coastlines(
                resolution=feature_resolution,
                linewidth=self.style.coastline_width,
                color=self.style.coastline_color,
                alpha=self.style.coastline_alpha,
                zorder=6
            )
            # lakes included with coastlines
            self.ax.add_feature(
                cfeature.LAKES.with_scale(feature_resolution),
                linewidth=self.style.coastline_width,
                edgecolor=self.style.coastline_color,
                facecolor='none',
                alpha=self.style.coastline_alpha,
                zorder=6
            )
        
        if 'countries' in boundaries:
            print("  Adding Cartopy country borders") 
            self.ax.add_feature(
                cfeature.BORDERS.with_scale(feature_resolution),
                linewidth=self.style.country_width,
                edgecolor=self.style.country_color,
                alpha=self.style.country_alpha,
                facecolor='none',
                zorder=5
            )
        
        if 'states' in boundaries:
            print("  Adding Cartopy state borders")
            self.ax.add_feature(
                cfeature.STATES.with_scale(feature_resolution),
                linewidth=self.style.state_width,
                edgecolor=self.style.state_color,
                alpha=self.style.state_alpha,
                facecolor='none',
                zorder=5
            )
        
        if 'counties' in boundaries:
            # Counties require additional Natural Earth data
            try:
                from cartopy.io.shapereader import Reader
                from cartopy.feature import ShapelyFeature
                # This would need the counties shapefile
                print("Warning: County boundaries require additional Natural Earth data")
            except:
                print("Warning: Could not load county boundaries")
        
        if 'rivers' in boundaries:
            print("  Adding Cartopy rivers")
            self.ax.add_feature(
                cfeature.RIVERS.with_scale(feature_resolution),
                linewidth=self.style.river_width,
                edgecolor=self.style.river_color,
                alpha=self.style.river_alpha,
                zorder=4
            )
        
        # Add gridlines if requested
        if self.style.show_gridlines:
            print("  Adding grid lines")
            self._add_gridlines()
        
        return self.fig, self.ax
   
    def _add_base_image(self):
        """Add base earth image background"""
        try:
            from wxmaps_base_images import BaseImageConfig, BaseImagePlotter
            
            # Get image path
            if self.style.base_image_path:
                # Custom image path provided
                image_path = self.style.base_image_path
                print(f"Using custom base image: {image_path}")
            else:
                # Use predefined image type
                image_path = BaseImageConfig.get_image_path(
                    self.style.base_image_type,
                    month=self.style.base_image_month
                )
                print(f"Using base image: {self.style.base_image_type}")

            # Determine target resolution - must match what was preloaded!
            target_resolution = getattr(self.style, 'base_image_target_resolution', 4000)

            # Create plotter with matching resolution
            img_plotter = BaseImagePlotter(
                image_path,
                target_resolution=target_resolution
            )
            
            # Load image (will use preloaded cache if available)
            img_plotter.load_image()

            # Plot on map
            img_plotter.plot_on_map(
                self.ax,
                alpha=self.style.base_image_alpha,
                interpolation=self.style.base_image_interpolation,
                zorder=0
            )
            
        except Exception as e:
            print(f"Warning: Could not load base image: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Falling back to solid color background")
            # Fallback to solid colors
            self.ax.add_feature(cfeature.OCEAN, facecolor=self.style.ocean_color, zorder=0)
            self.ax.add_feature(cfeature.LAND, facecolor=self.style.land_color, zorder=0)
            self.ax.add_feature(
                cfeature.LAKES.with_scale(self.feature_resolution),  # Fixed: was undefined variable
                edgecolor='none',
                facecolor=self.style.ocean_color,
                zorder=0
            )
 
    def apply_limb_darkening(
        self,
        rgba,
        lats,
        lons,
        gamma=1.25,
        mode="rgb",
    ):
        """
        Apply limb darkening using the actual geostationary projection geometry.

        Parameters
        ----------
        rgba : (H,W,4)
        lats, lons : (H,W) degrees
        projection : cartopy.crs.Geostationary
            Use map_config.projection directly
        gamma : float
            Darkening strength (1=physical, >1 stronger)
        mode : 'rgb' | 'alpha' | 'both'
        """

        # ---------------------------------------------------------
        # Only apply for geostationary projections
        # ---------------------------------------------------------
        if not isinstance(self.config.projection, ccrs.Geostationary):
            return rgba

        # -------------------------------------------------
        # Handle 1D or 2D lat/lon automatically
        # -------------------------------------------------
        if lats.ndim == 1 and lons.ndim == 1:
            lons, lats = np.meshgrid(lons, lats)

        # pull parameters directly from Cartopy
        sat_lon = self.config.center_lon
        H = 35786023.0 #self.config.projection.satellite_height

        Re = 6378137.0
        Rs = Re + H

        lat = np.deg2rad(lats)
        lon = np.deg2rad(lons)
        lon0 = np.deg2rad(sat_lon)
        cos_c = np.cos(lat) * np.cos(lon - lon0)
        mu = (Rs*cos_c - Re) / np.sqrt(Rs**2 + Re**2 - 2*Rs*Re*cos_c)
        mu = np.clip(mu, 0, 1)
        start = 0.55          # where darkening begins
        t = np.clip(mu / start, 0, 1)
        dark = t**gamma
        out = rgba.copy()
        if mode in ("rgb", "both"):
            out[..., :3] *= dark[..., None]
        if mode in ("alpha", "both"):
            out[..., 3] *= dark
        return out

    def _use_gshhs(self):
        return bool(self.style.use_gshhs)

# NOTE:
# GSHHS provides land/water polygons only (including lakes/ponds).
# Coastlines and lake fills are implicit; no separate GSHHS boundary layers
# should be drawn elsewhere.
    def _add_gshhs_background(self):
        """Add GSHHS-based land/water background"""
        try:
            from wxmaps_gshhs import GSHHSReader, get_gshhs_resolution_file
            import matplotlib.patches as mpatches
            
            # Get GSHHS file path
            gshhs_file = get_gshhs_resolution_file(
                self.style.gshhs_path,
                self.style.gshhs_resolution
            )
            
            print(f"  Loading GSHHS from: {gshhs_file}")
            print(f"  Resolution: {self.style.gshhs_resolution} (min area: {self.style.gshhs_min_area} km²)")
            print(f"  Ocean color: {self.style.ocean_color}")
            print(f"  Land color: {self.style.land_color}")
            print(f"  Map extent: {self.config.extent}")
            
            # Create reader
            reader = GSHHSReader(gshhs_file)
            
            # First, try reading WITHOUT extent filter to see if file works
            print(f"  Testing: reading first 10 polygons from file (no filters)...")
            test_polygons = reader.read_polygons(
                min_area=0.0,
                max_level=4,
                extent=None
            )
            
            if len(test_polygons) == 0:
                print("  ERROR: Could not read ANY polygons from GSHHS file!")
                print(f"  Check if file exists and is readable: {gshhs_file}")
                raise FileNotFoundError(f"GSHHS file appears empty or unreadable: {gshhs_file}")
            
            print(f"  File is readable: found {len(test_polygons)} total polygons")
            
            # Now read with filters and expanded extent
            extent = self.config.extent
            expanded_extent = (
                extent[0] - 10,  # Expand more
                extent[1] + 10,
                extent[2] - 10,
                extent[3] + 10
            )
            
            print(f"  Reading polygons for map extent (expanded by 10°)...")
            polygons = reader.read_polygons(
                min_area=self.style.gshhs_min_area,
                max_level=self.style.gshhs_max_level,
                extent=expanded_extent
            )
            
            print(f"  Loaded {len(polygons)} GSHHS polygons for map")
            
            if len(polygons) == 0:
                print("  WARNING: No GSHHS polygons found in map extent! Falling back to Cartopy.")
                self.ax.add_feature(cfeature.OCEAN, facecolor=self.style.ocean_color, zorder=0)
                self.ax.add_feature(cfeature.LAND, facecolor=self.style.land_color, zorder=0)
                self.ax.add_feature(
                    cfeature.LAKES.with_scale(feature_resolution),
                    edgecolor='none',
                    facecolor=self.style.ocean_color,
                    zorder=0
                )   
                return
            
            # Show polygon level distribution
            level_counts = {}
            for p in polygons:
                level = p['level']
                level_counts[level] = level_counts.get(level, 0) + 1
            print(f"  Polygon levels: {level_counts}")
            print(f"    Level 1 = land, Level 2 = lakes, Level 3 = islands in lakes, Level 4 = ponds")
            
            # Fill entire background with ocean color
            print(f"  Setting background to ocean color: {self.style.ocean_color}")
            
            # Use Cartopy OCEAN feature as base
            self.ax.add_feature(cfeature.OCEAN, facecolor=self.style.ocean_color, zorder=0)
            
            # Background rectangle as backup
            background = mpatches.Rectangle((0, 0), 1, 1, 
                                          transform=self.ax.transAxes,
                                          facecolor=self.style.ocean_color,
                                          edgecolor='none',
                                          zorder=-1)
            self.ax.add_patch(background)
            
            print(f"  Plotting GSHHS polygons:")
            print(f"    Land (levels 1,3) = {self.style.land_color}")
            print(f"    Water (levels 2,4) = {self.style.ocean_color}")
            
            # Plot GSHHS polygons
            reader.plot_polygons_on_map(
                self.ax,
                polygons,
                land_color=self.style.land_color,
                water_color=self.style.ocean_color,
                fill=True,
                outline=False,
                transform=ccrs.PlateCarree()
            )
            
            print(f"  GSHHS background complete")
            
        except Exception as e:
            print(f"ERROR: Could not load GSHHS data: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Falling back to standard Cartopy features")
            # Fallback to cartopy features
            self.ax.add_feature(cfeature.OCEAN, facecolor=self.style.ocean_color, zorder=0)
            self.ax.add_feature(cfeature.LAND, facecolor=self.style.land_color, zorder=0)
 
    def _add_gridlines(self):
        """Add gridlines to the map with resolution-appropriate styling"""
        fontsize_scale = {
            'hd': 10,
            'fhd': 12,
            '2k': 14,
            '4k': 16,
            '8k': 24
        }
        
        fs = fontsize_scale.get(self.resolution, 10) if self.style.show_grid_labels else 0
        
        gl = self.ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=self.style.show_grid_labels,
            linewidth=self.style.gridline_width,
            color=self.style.gridline_color,
            alpha=self.style.gridline_alpha,
            linestyle=self.style.gridline_style,
            zorder=7
        )
        
        if self.style.show_grid_labels:
            # Configure gridline labels
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': fs, 'color': 'black', 'weight': 'bold'}
            gl.ylabel_style = {'size': fs, 'color': 'black', 'weight': 'bold'}
            
            # Set gridline intervals based on map extent
            extent = self.config.extent
            lon_range = extent[1] - extent[0]
            lat_range = extent[3] - extent[2]
            
            if lon_range > 180:
                lon_interval = 30
            elif lon_range > 60:
                lon_interval = 10
            elif lon_range > 20:
                lon_interval = 5
            else:
                lon_interval = 2
            
            if lat_range > 90:
                lat_interval = 20
            elif lat_range > 30:
                lat_interval = 10
            elif lat_range > 10:
                lat_interval = 5
            else:
                lat_interval = 2
            
            gl.xlocator = mticker.MultipleLocator(lon_interval)
            gl.ylocator = mticker.MultipleLocator(lat_interval)
    
    # ===================================================================
    # Boundary drawing primitives
    # ===================================================================

    def draw_gshhs_coastlines(self):
        """Draw coastline outlines from GSHHS (no fill)"""
        if not self.style.use_gshhs:
            return

        try:
            from wxmaps_gshhs import GSHHSReader, get_gshhs_resolution_file

            gshhs_file = get_gshhs_resolution_file(
                self.style.gshhs_path,
                self.style.gshhs_resolution
            )

            reader = GSHHSReader(gshhs_file)

            extent = self.config.extent
            expanded_extent = (
                extent[0] - 5,
                extent[1] + 5,
                extent[2] - 5,
                extent[3] + 5
            )

            polygons = reader.read_polygons(
                min_area=self.style.gshhs_min_area,
                max_level=self.style.gshhs_max_level,
                extent=expanded_extent
            )

            reader.plot_polygons_on_map(
                self.ax,
                polygons,
                land_color=None,
                water_color=None,
                fill=False,
                outline=True,
                outline_color=self.style.coastline_color,
                outline_width=self.style.coastline_width,
                transform=ccrs.PlateCarree()
            )

        except Exception as e:
            print(f"Warning: GSHHS coastline outlines failed: {e}")

    def draw_coastlines(self, feature_resolution: str = '50m'):
        self.ax.coastlines(
            resolution=feature_resolution,
            color=self.style.coastline_color,
            linewidth=self.style.coastline_width,
            alpha=self.style.coastline_alpha,
            zorder=6,
        )

    def draw_countries(self, feature_resolution: str = '50m'):
        """Draw country boundaries"""
        self.ax.add_feature(
            cfeature.BORDERS.with_scale(feature_resolution),
            linewidth=self.style.country_width,
            edgecolor=self.style.country_color,
            alpha=self.style.country_alpha,
            facecolor='none',
            zorder=5
        )

    def draw_states(self, feature_resolution: str = '50m'):
        """Draw state boundaries"""
        self.ax.add_feature(
            cfeature.STATES.with_scale(feature_resolution),
            linewidth=self.style.state_width,
            edgecolor=self.style.state_color,
            alpha=self.style.state_alpha,
            facecolor='none',
            zorder=5
        )

    def draw_counties(self, feature_resolution: str = '50m'):
        """Draw county boundaries (requires external Natural Earth data)"""
        print("Warning: County boundaries are not yet implemented")

    def draw_rivers(self, feature_resolution: str = '50m'):
        """Draw rivers"""
        self.ax.add_feature(
            cfeature.RIVERS.with_scale(feature_resolution),
            linewidth=self.style.river_width,
            edgecolor=self.style.river_color,
            alpha=self.style.river_alpha,
            zorder=4
        )

    def add_boundaries(self, boundaries):
        """
        Draw requested map boundaries in a deterministic order.
        """
        if not boundaries:
            return

        requested = set(boundaries)

        for b in BOUNDARY_ORDER:
            if b not in requested:
                continue

            if b == "coastlines":
                self.draw_coastlines()
            elif b == "rivers":
                self.draw_rivers()
            elif b == "countries":
                self.draw_countries()
            elif b == "states":
                self.draw_states()
            elif b == "counties":
                self.draw_counties()

    def add_cities(self, city_list: Optional[List[dict]] = None, 
                   min_population: int = 1000000):
        """
        Add city markers and labels
        
        Parameters:
        -----------
        city_list : list of dict, optional
            List of cities with 'name', 'lon', 'lat', 'population'
        min_population : int
            Minimum population threshold
        """
        marker_scale = {
            'hd': 5,
            'fhd': 6,
            '2k': 7,
            '4k': 10,
            '8k': 15
        }
        
        fontsize_scale = {
            'hd': 8,
            'fhd': 9,
            '2k': 10,
            '4k': 12,
            '8k': 18
        }
        
        ms = marker_scale.get(self.resolution, 5)
        fs = fontsize_scale.get(self.resolution, 8)
        
        if city_list is None:
            # Default major US cities
            city_list = [
                {'name': 'New York', 'lon': -74.006, 'lat': 40.7128, 'population': 8336817},
                {'name': 'Los Angeles', 'lon': -118.2437, 'lat': 34.0522, 'population': 3979576},
                {'name': 'Chicago', 'lon': -87.6298, 'lat': 41.8781, 'population': 2693976},
                {'name': 'Houston', 'lon': -95.3698, 'lat': 29.7604, 'population': 2320268},
                {'name': 'Phoenix', 'lon': -112.0740, 'lat': 33.4484, 'population': 1680992},
                {'name': 'Miami', 'lon': -80.1918, 'lat': 25.7617, 'population': 467963},
                {'name': 'Atlanta', 'lon': -84.3880, 'lat': 33.7490, 'population': 498715},
                {'name': 'Boston', 'lon': -71.0589, 'lat': 42.3601, 'population': 692600},
                {'name': 'Seattle', 'lon': -122.3321, 'lat': 47.6062, 'population': 753675},
                {'name': 'Denver', 'lon': -104.9903, 'lat': 39.7392, 'population': 727211},
                {'name': 'Dallas', 'lon': -96.7970, 'lat': 32.7767, 'population': 1343573},
                {'name': 'Philadelphia', 'lon': -75.1652, 'lat': 39.9526, 'population': 1584064},
                {'name': 'San Francisco', 'lon': -122.4194, 'lat': 37.7749, 'population': 873965},
                {'name': 'San Diego', 'lon': -117.1611, 'lat': 32.7157, 'population': 1423851},
            ]
        
        for city in city_list:
            if city['population'] >= min_population:
                extent = self.config.extent
                if (extent[0] <= city['lon'] <= extent[1] and 
                    extent[2] <= city['lat'] <= extent[3]):
                    
                    self.ax.plot(city['lon'], city['lat'], 'o', 
                               markersize=ms, color='red', 
                               transform=ccrs.PlateCarree(),
                               zorder=10)
                    
                    self.ax.text(city['lon'], city['lat'], f" {city['name']}", 
                               fontsize=fs, fontweight='bold',
                               ha='left', va='center',
                               transform=ccrs.PlateCarree(),
                               zorder=10,
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='white', alpha=0.7,
                                       edgecolor='black', linewidth=0.5))
    
    def add_roads(self, road_scale: str = '50m', major_only: bool = False):
        """
        Add road features
        
        Parameters:
        -----------
        road_scale : str
            '110m', '50m', '10m'
        major_only : bool
            If True, only show major highways/interstates
            If False, show all roads
        """
        try:
            if major_only:
                # Load roads shapefile and filter for major roads
                from cartopy.io.shapereader import Reader, natural_earth
                import shapely.geometry as sgeom
                
                # Get the roads shapefile
                roads_path = natural_earth(resolution=road_scale, 
                                          category='cultural', 
                                          name='roads')
                
                # Read and filter roads
                roads_reader = Reader(roads_path)
                major_roads = []
                
                for road in roads_reader.records():
                    # Filter by road type - keep only major highways
                    # Natural Earth road types: 'Major Highway', 'Secondary Highway', 'Road'
                    road_type = road.attributes.get('type', '')
                    if 'Major' in road_type or 'Highway' in road_type:
                        major_roads.append(road.geometry)
                
                # Add filtered roads to map
                for geom in major_roads:
                    self.ax.add_geometries([geom], ccrs.PlateCarree(),
                                         edgecolor=self.style.road_color,
                                         facecolor='none',
                                         linewidth=self.style.road_width,
                                         alpha=self.style.road_alpha,
                                         zorder=4)
                
                print(f"Added {len(major_roads)} major roads")
            else:
                # Add all roads using feature
                roads = cfeature.NaturalEarthFeature(
                    category='cultural',
                    name='roads',
                    scale=road_scale,
                    edgecolor=self.style.road_color,
                    facecolor='none'
                )
                self.ax.add_feature(roads, 
                                  linewidth=self.style.road_width, 
                                  alpha=self.style.road_alpha, 
                                  zorder=4)
        except Exception as e:
            print(f"Warning: Could not load roads feature: {e}")
            print(f"  This may require downloading Natural Earth road data")

    def add_nws_warnings(self, valid_time: datetime):
        """
        Add NWS watches/warnings overlay
        
        Parameters:
        -----------
        valid_time : datetime
            Valid time for warnings
        """
        if not self.style.show_nws_warnings:
            return
        
        try:
            from wxmaps_nws_warnings import NWSWarnings, get_nws_shapefile_path
            
            # Get shapefile for the year
            year = valid_time.year
            shapefile = get_nws_shapefile_path(self.style.nws_shapefile_base, year)
            
            print(f"Loading NWS warnings from: {shapefile}")
            
            # Create warnings handler
            nws = NWSWarnings(shapefile)
            nws.load_shapefile()
            
            # Determine outline width from style config (default 1.0)
            # This gives warnings=1.0, watches/advisories=0.5 automatically
            outline_width = getattr(self.style, 'nws_warning_outline_width', 1.0)
            
            # Determine fill setting
            fill = getattr(self.style, 'nws_warning_fill', False)
            
            # Determine alpha
            alpha = getattr(self.style, 'nws_warning_alpha', 0.5)
            
            # Plot warnings based on configuration
            if hasattr(self.style, 'nws_custom_type') and hasattr(self.style, 'nws_custom_status') \
               and self.style.nws_custom_type and self.style.nws_custom_status:
                # Plot specific custom warning type/status
                print(f"Plotting custom warning: {self.style.nws_custom_type}.{self.style.nws_custom_status}")
                nws.plot_warnings(
                    self.ax,
                    warning_type=self.style.nws_custom_type,
                    warning_status=self.style.nws_custom_status,
                    valid_time=valid_time,
                    outline_width=outline_width,
                    fill=fill,
                    alpha=alpha,
                    transform=ccrs.PlateCarree()
                )
            elif hasattr(self.style, 'nws_warning_types') and self.style.nws_warning_types:
                # Plot multiple specific warning types (list of tuples like [('TO', 'W'), ('SV', 'W')])
                for warning_type, warning_status in self.style.nws_warning_types:
                    type_name = nws.VTEC_PHENOMENA.get(warning_type, warning_type)
                    status_name = nws.VTEC_SIGNIFICANCE.get(warning_status, warning_status)
                    print(f"Plotting {type_name} {status_name}")
                    nws.plot_warnings(
                        self.ax,
                        warning_type=warning_type,
                        warning_status=warning_status,
                        valid_time=valid_time,
                        outline_width=outline_width,
                        fill=fill,
                        alpha=alpha,
                        transform=ccrs.PlateCarree()
                    )
            elif hasattr(self.style, 'nws_severe_only') and self.style.nws_severe_only:
                # Show only severe weather warnings (TO.W and SV.W)
                print(f"Plotting SEVERE warnings only (Tornado + Severe Thunderstorm) at {valid_time}")
                nws.plot_all_warnings(
                    self.ax,
                    valid_time=valid_time,
                    outline_width=outline_width,
                    fill=fill,
                    alpha=alpha,
                    severe_only=True,
                    transform=ccrs.PlateCarree()
                )
            else:
                # DEFAULT: Show ALL warnings and watches at this time
                print(f"Plotting ALL active warnings and watches at {valid_time}")
                nws.plot_all_warnings(
                    self.ax,
                    valid_time=valid_time,
                    outline_width=outline_width,
                    fill=fill,
                    alpha=alpha,
                    severe_only=False,
                    transform=ccrs.PlateCarree()
                )
            
        except Exception as e:
            print(f"Warning: Could not load NWS warnings: {e}")
            import traceback
            traceback.print_exc()

    def add_city_data_values(self, data, lons, lats, text_color='white'):
        """     
        Add data labels at city locations
        
        Parameters
        ----------
        data : np.ndarray
            Temperature data array (2D)
        lons : np.ndarray
            Longitude array matching data shape
        lats : np.ndarray
            Latitude array matching data shape
        """     
        import os
        from scipy.interpolate import RegularGridInterpolator
        from scipy.spatial import cKDTree

        # Determine city file based on map domain
        map_name = self.config.name.lower()

        if 'maryland' in map_name or 'midatlantic' in map_name:
            if 'maryland' in map_name:
                cityfile = '/home/wputman/IDL_BASE/CITIES/all_cities_md.txt'
                size_multiplier = 1.5
                spacing_multiplier = 0.75
            else:
                cityfile = '/home/wputman/IDL_BASE/CITIES/all_cities.txt'
                size_multiplier = 1.0
                spacing_multiplier = 1.0
        else:
            cityfile = '/home/wputman/IDL_BASE/CITIES/world_cities.csv'
            size_multiplier = 1.0
            spacing_multiplier = 1.0

        if not os.path.exists(cityfile):
            print(f"Warning: City file not found: {cityfile}")
            return

        # Calculate size parameters based on image resolution
        img_width = self.fig.get_size_inches()[0] * self.fig.dpi

        # Scale factor relative to reference 4K image (3840 x 2160)
        scale_factor = img_width / 3840.0

        # Base font size scales with image, with reasonable limits
        base_fontsize = np.clip(10 * scale_factor, 8, 24) * size_multiplier

        # City spacing to avoid overlap (in degrees)
        city_spacing_degrees = np.clip(0.5 / scale_factor, 0.2, 2.0) * spacing_multiplier

        # Create interpolator for data
        if lons.ndim == 1 and lats.ndim == 1:
            # 1D coordinate arrays
            interp = RegularGridInterpolator(
                (lats, lons),
                data,
                method='nearest',
                bounds_error=False,
                fill_value=np.nan
            )
            use_1d = True
        else:
            # 2D coordinate arrays - use KDTree for fast lookup
            lons_flat = lons.ravel()
            lats_flat = lats.ravel()
            
            # Fill masked values with nan
            if np.ma.is_masked(data):
                data_flat = data.filled(np.nan).ravel()
            else:
                data_flat = data.ravel()
            
            # Build KDTree for fast nearest neighbor lookup
            points = np.column_stack([lons_flat, lats_flat])
            tree = cKDTree(points)
            use_1d = False

        # Determine if we need to convert longitude convention
        lon_min = np.nanmin(lons)
        use_360_convention = lon_min > 180

        # Get map bounds once
        extent = self.ax.get_extent(crs=ccrs.PlateCarree())
        
        # Read all cities first, filter by bounds
        cities_to_plot = []
        
        print('------------------CITIES------------------')
        
        with open(cityfile, 'r') as f:
            for line in f:
                parts = line.strip().split(',')

                # Handle different file formats
                if len(parts) >= 6:
                    try:
                        lat = float(parts[2])
                        lon = float(parts[3])
                        name = parts[0].strip()
                    except (ValueError, IndexError):
                        try:
                            name = parts[1].strip() if parts[1].strip() else parts[0].strip()
                            lat = float(parts[-2].strip().lstrip('+'))
                            lon = float(parts[-1].strip().lstrip('+'))
                        except (ValueError, IndexError):
                            continue
                else:
                    continue

                if not name:
                    continue

                # Quick bounds check
                if not (extent[0] <= lon <= extent[1] and extent[2] <= lat <= extent[3]):
                    continue

                # Convert for data lookup if needed
                lon_data = lon + 360.0 if (use_360_convention and lon < 0) else lon

                # Get data value
                if use_1d:
                    temp_val = interp([lat, lon_data])[0]
                else:
                    # Fast KDTree lookup
                    dist, idx = tree.query([lon_data, lat])
                    temp_val = data_flat[idx]

                if np.isnan(temp_val) or temp_val < 1.0:
                    continue

                # Check if value is masked (missing/invalid)
                if np.ma.is_masked(temp_val):
                    # Skip this city if data is masked
                    continue
        
                temp_val = round(temp_val)
                cities_to_plot.append((lon, lat, name, temp_val))

        # Sort cities by value descending (highest values plotted first)
        # This ensures important/extreme values are plotted when there's overlap
        cities_to_plot.sort(key=lambda x: x[3], reverse=True)
        
        # Use spatial grid for fast overlap checking
        plotted_locations = []
        
        for lon, lat, name, temp_val in cities_to_plot:
            # Fast overlap check using list comprehension
            too_close = any(
                abs(lon - prev_lon) < city_spacing_degrees and 
                abs(lat - prev_lat) < city_spacing_degrees
                for prev_lon, prev_lat in plotted_locations
            )
            
            if too_close:
                continue

            # Plot temperature label
            self.ax.text(
                lon, lat,
                str(int(temp_val)),
                transform=ccrs.PlateCarree(),
                fontsize=base_fontsize,
                color=text_color,
                weight='bold',
                ha='center',
                va='center',
                zorder=15
            )

            plotted_locations.append((lon, lat))
            print(f"{name}: {temp_val} ({text_color})")

        print(f"Plotted {len(plotted_locations)} cities")
        print('------------------------------------------')

    def add_city_temperatures(self, data, lons, lats, temperature_unit='F'):
        """
        Add temperature values at city locations with color coding based on temperature.
        
        Parameters
        ----------
        data : np.ndarray
            Temperature data array (2D)
        lons : np.ndarray
            Longitude array matching data shape
        lats : np.ndarray
            Latitude array matching data shape
        temperature_unit : str
            'F' for Fahrenheit or 'C' for Celsius (affects color thresholds)
        """
        import os
        from scipy.interpolate import RegularGridInterpolator
        from scipy.spatial import cKDTree
        
        # Temperature-based color scheme (Fahrenheit thresholds)
        if temperature_unit == 'F':
            temp_colors = [
                (-np.inf, -20, '#FF00FF', 1.00),  # magenta for < -20°F
                (-20, 40, '#FFFFFF', 0.95),        # white for -20 to 40°F
                (40, 70, '#000000', 0.90),         # black for 40 to 70°F
                (70, 80, '#FFFF00', 0.90),         # yellow (c70) for 70-80°F
                (80, 90, '#FFFFAF', 0.95),         # light yellow (c80) for 80-90°F
                (90, 100, '#FFFFFF', 1.00),        # white (c90) for 90-100°F
                (100, 105, '#FF4BD2', 1.05),       # pink (c100) for 100-105°F
                (105, 110, '#FF00FF', 1.10),       # magenta (c105) for 105-110°F
                (110, np.inf, '#E100E1', 1.15),    # dark magenta (c110) for > 110°F
            ]
        else:  # Celsius
            temp_colors = [
                (-np.inf, -29, '#FF00FF', 1.00),   # < -29°C
                (-29, 4, '#FFFFFF', 0.95),          # -29 to 4°C
                (4, 21, '#000000', 0.90),           # 4 to 21°C
                (21, 27, '#FFFF00', 0.90),          # 21-27°C
                (27, 32, '#FFFFAF', 0.95),          # 27-32°C
                (32, 38, '#FFFFFF', 1.00),          # 32-38°C
                (38, 41, '#FF4BD2', 1.05),          # 38-41°C
                (41, 43, '#FF00FF', 1.10),          # 41-43°C
                (43, np.inf, '#E100E1', 1.15),      # > 43°C
            ]
        
        # Determine city file based on map domain
        map_name = self.config.name.lower()
        
        if 'maryland' in map_name or 'midatlantic' in map_name:
            if 'maryland' in map_name:
                cityfile = '/home/wputman/IDL_BASE/CITIES/all_cities_md.txt'
                size_multiplier = 1.5
                spacing_multiplier = 0.75
            else:
                cityfile = '/home/wputman/IDL_BASE/CITIES/all_cities.txt'
                size_multiplier = 1.0
                spacing_multiplier = 1.0
        else:
            cityfile = '/home/wputman/IDL_BASE/CITIES/world_cities.csv'
            size_multiplier = 1.0
            spacing_multiplier = 1.0
        
        if not os.path.exists(cityfile):
            print(f"Warning: City file not found: {cityfile}")
            return
        
        # Calculate size parameters based on image resolution
        img_width = self.fig.get_size_inches()[0] * self.fig.dpi
        
        # Scale factor relative to reference 4K image (3840 x 2160)
        scale_factor = img_width / 3840.0
        
        # Base font size scales with image, with reasonable limits
        base_fontsize = np.clip(10 * scale_factor, 8, 24) * size_multiplier

        # City spacing to avoid overlap (in degrees)
        city_spacing_degrees = np.clip(0.5 / scale_factor, 0.2, 2.0) * spacing_multiplier

        # Create interpolator for data
        if lons.ndim == 1 and lats.ndim == 1:
            # 1D coordinate arrays - use argmin for nearest neighbor
            use_1d = True
            
            # Pre-check data shape
            if data.shape[0] != len(lats) or data.shape[1] != len(lons):
                raise ValueError(f"Data shape {data.shape} doesn't match lat/lon arrays ({len(lats)}, {len(lons)})")
            
            print(f"Using 1D grid lookup: data shape {data.shape}, lats {len(lats)}, lons {len(lons)}")
        else:
            # 2D coordinate arrays - use KDTree for fast lookup
            lons_flat = lons.ravel()
            lats_flat = lats.ravel()

            # Fill masked values with nan
            if np.ma.is_masked(data):
                data_flat = data.filled(np.nan).ravel()
            else:
                data_flat = data.ravel()

            # Build KDTree for fast nearest neighbor lookup
            points = np.column_stack([lons_flat, lats_flat])
            tree = cKDTree(points)
            use_1d = False
            
            print(f"Using 2D KDTree lookup: {len(lons_flat)} points")

        # Determine if we need to convert longitude convention
        lon_min = np.nanmin(lons)
        lon_max = np.nanmax(lons)
        
        # Grid uses 0-360 if maximum is significantly > 180
        use_360_convention = lon_max > 180
        
        print(f"DEBUG: Grid lons range: [{lon_min:.2f}, {lon_max:.2f}]")
        print(f"DEBUG: Grid lats range: [{np.min(lats):.2f}, {np.max(lats):.2f}]")
        print(f"DEBUG: Using 0-360 convention: {use_360_convention}")
        print(f"DEBUG: Data shape: {data.shape}")
        print(f"DEBUG: Lats shape: {lats.shape}, Lons shape: {lons.shape}")
        print(f"DEBUG: First 5 lats: {lats[:5]}")
        print(f"DEBUG: Last 5 lats: {lats[-5:]}")
        print(f"DEBUG: First 5 lons: {lons[:5]}")
        print(f"DEBUG: Last 5 lons: {lons[-5:]}")
 
        # Get map bounds once
        extent = self.ax.get_extent(crs=ccrs.PlateCarree())
        
        # Read all cities first, filter by bounds
        cities_to_plot = []
        
        print('------------------CITIES------------------')
        
        with open(cityfile, 'r') as f:
            for line in f:
                parts = line.strip().split(',')

                # Handle different file formats
                if len(parts) >= 6:
                    try:
                        lat = float(parts[2])
                        lon = float(parts[3])
                        name = parts[0].strip()
                    except (ValueError, IndexError):
                        try:
                            name = parts[1].strip() if parts[1].strip() else parts[0].strip()
                            lat = float(parts[-2].strip().lstrip('+'))
                            lon = float(parts[-1].strip().lstrip('+'))
                        except (ValueError, IndexError):
                            continue
                else:
                    continue
                
                if not name:
                    continue
                
                print(f"DEBUG: Map extent: {extent}")
                print(f"DEBUG: Extent in -180/180: [{extent[0]:.2f}, {extent[1]:.2f}], [{extent[2]:.2f}, {extent[3]:.2f}]")

                # Quick bounds check
                if not (extent[0] <= lon <= extent[1] and extent[2] <= lat <= extent[3]):
                    continue
                
                # Convert for data lookup if needed
                lon_data = lon + 360.0 if (use_360_convention and lon < 0) else lon

                # Get temperature value at this location
                if use_1d:
                    # Find nearest grid indices
                    lat_idx = np.argmin(np.abs(lats - lat))
                    lon_idx = np.argmin(np.abs(lons - lon_data))
                    temp_val = data[lat_idx, lon_idx]
                else:
                    # Fast KDTree lookup
                    dist, idx = tree.query([lon_data, lat])
                    temp_val = data_flat[idx]
                
                if np.isnan(temp_val):
                    continue
                
                temp_val = round(temp_val)
                cities_to_plot.append((lon, lat, name, temp_val))

        # Sort cities by temperature descending (hottest plotted first)
        # cities_to_plot.sort(key=lambda x: x[3], reverse=True)
        
        print(f"Found {len(cities_to_plot)} cities in bounds...")
        
        # Plot cities with overlap checking
        plotted_locations = []
        
        for lon, lat, name, temp_val in cities_to_plot:
            # Fast overlap check
            too_close = any(
                abs(lon - prev_lon) < city_spacing_degrees and 
                abs(lat - prev_lat) < city_spacing_degrees
                for prev_lon, prev_lat in plotted_locations
            )
            
            if too_close:
                continue
            
            # Determine color and size based on temperature
            color = '#FFFFFF'  # default
            size_factor = 1.0
            
            for t_min, t_max, t_color, t_size in temp_colors:
                if t_min <= temp_val < t_max:
                    color = t_color
                    size_factor = t_size
                    break
            
            fontsize = base_fontsize * size_factor
            
            # Plot temperature label
            self.ax.text(
                lon, lat, 
                str(int(temp_val)),
                transform=ccrs.PlateCarree(),
                fontsize=fontsize,
                color=color,
                weight='bold',
                ha='center',
                va='center',
                zorder=15
            )
            
            plotted_locations.append((lon, lat))
            print(f"{name}: {temp_val}°{temperature_unit} ({color})")
        
        print(f"Plotted {len(plotted_locations)} cities")
        print('------------------------------------------')

    def _is_within_bounds(self, lon, lat):
        """Check if lon/lat is within current map bounds"""
        extent = self.ax.get_extent(crs=ccrs.PlateCarree())
        result = (extent[0] <= lon <= extent[1] and 
                  extent[2] <= lat <= extent[3])
        #print(f"    Bounds check: extent={extent}, lon={lon:.2f}, lat={lat:.2f}, result={result}")
        return result
    
    def add_title(self, title: str, loc: str = 'center'):
        """Add title to the map with resolution-appropriate sizing"""
        if not self.style.show_title:
            return
        
        fontsize_scale = {
            'hd': 16,
            'fhd': 18,
            '2k': 20,
            '4k': 24,
            '8k': 36
        }
        
        fs = self.style.title_fontsize if self.style.title_fontsize else fontsize_scale.get(self.resolution, 16)
        
        self.ax.set_title(title, fontsize=fs, loc=loc, fontweight='bold', pad=20)
    
    def add_forecast_timestamp(self, fdate: str, pdate: str, exp: str = 'CONUS02KM',
                              timezone: str = 'US/Eastern'):
        """
        Add wxmaps forecast timestamp
        
        Format:
        012 Forecast Hour
        2026-01-16 12:00Z (Fri 07:00am EST)
        
        Parameters:
        -----------
        fdate : str
            Forecast date (e.g., '20260116_00z')
        pdate : str
            Valid/plot date (e.g., '20260118_12z')
        exp : str
            Experiment name (not displayed in wxmaps mode)
        timezone : str
            Timezone for local time display (default: US/Eastern)
        """
        if not self.style.show_timestamp:
            return
        
        from wxmaps_utils import parse_date_string, calculate_forecast_hour
        
        fontsize_scale = {
            'hd': 8,
            'fhd': 8,
            '2k': 8,
            '4k': 14,
            '8k': 28
        }
        
        fs = self.style.timestamp_fontsize if self.style.timestamp_fontsize else fontsize_scale.get(self.resolution, 12)
        
        # Parse dates
        fdate_dt = parse_date_string(fdate)
        pdate_dt = parse_date_string(pdate)
        fhour = calculate_forecast_hour(fdate_dt, pdate_dt)
        
        # Convert to specified timezone
        utc = pytz.UTC
        local_tz = pytz.timezone(timezone)
        pdate_utc = utc.localize(pdate_dt)
        pdate_local = pdate_utc.astimezone(local_tz)
        
        # Get timezone abbreviation
        tz_abbr = pdate_local.strftime('%Z')
        
        # Format timestamp
        if self.style.timestamp_format == 'detailed':
            line1 = f"{fhour:03d} Forecast Hour"
            line2 = f"{pdate_dt.strftime('%Y-%m-%d %H:%M')}Z ({pdate_local.strftime('%a %I:%M%p')} {tz_abbr})"
            timestamp_text = f"{line1}\n{line2}"
        else:  # simple
            timestamp_text = f"F{fhour:03d} | {pdate_dt.strftime('%Y-%m-%d %H:%M')}Z"
        
        # Position
        loc = self.style.timestamp_location
        if loc == 'lower left':
            x, y = 0.01, 0.01
            ha, va = 'left', 'bottom'
        elif loc == 'lower right':
            x, y = 0.99, 0.01
            ha, va = 'right', 'bottom'
        elif loc == 'upper left':
            x, y = 0.01, 0.99
            ha, va = 'left', 'top'
        elif loc == 'upper right':
            x, y = 0.99, 0.99
            ha, va = 'right', 'top'
        else:
            x, y = 0.01, 0.01
            ha, va = 'left', 'bottom'
        
        # Add text - wxmaps style with no box
        self.fig.text(x, y, timestamp_text, 
                     fontsize=fs,
                     ha=ha, va=va, 
                     transform=self.fig.transFigure,
                     color=self.style.text_color,
                     fontweight='bold',
                     family='monospace')

    def save(self, filename: str, optimize: bool = True):
        """
        Save the figure at native resolution
        
        Parameters:
        -----------
        filename : str
            Output filename
        optimize : bool
            Apply optimization for file size
        """
        import os
        import gc
        
        # Save at native DPI
        save_kwargs = {
            'dpi': self.dpi,
            'facecolor': self.style.background_color,
            'edgecolor': 'none',
            'bbox_inches': None,
            'pad_inches': 0
        }
        
        if optimize and filename.endswith('.png'):
            save_kwargs['pil_kwargs'] = {'compress_level': 6, 'optimize': True}
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Suppress warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self.fig.savefig(filename, **save_kwargs)
        
        # Calculate file size
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        
        print(f"Saved: {filename}")
        print(f"  Resolution: {self.resolution_config.width}x{self.resolution_config.height}")
        print(f"  File size: {file_size_mb:.2f} MB")
        
        # CRITICAL: Close figure to free memory
        plt.close(self.fig)
        self.fig = None
        self.ax = None
        gc.collect()

    def close(self):
        """Close the figure and free memory"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
