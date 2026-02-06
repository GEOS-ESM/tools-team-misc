"""
WxMaps Configuration Module
Defines standard projections, extents, and map parameters for weather visualization
with support for HD, 4K, and 8K resolutions at 16:9 aspect ratio
"""
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
import cartopy.crs as ccrs
import numpy as np

@dataclass
class MapConfig:
    """Configuration for a map type"""
    name: str
    projection: ccrs.Projection
    extent: Tuple[float, float, float, float]  # [lon_min, lon_max, lat_min, lat_max]
    center_lon: Optional[float] = None
    center_lat: Optional[float] = None
    standard_parallels: Optional[Tuple[float, float]] = None

@dataclass
class StyleConfig:
    """Style configuration for map appearance"""
    # Basic Options
    background_color: str = 'white'
    text_color: str = 'black'

    # Surface colors
    ocean_color: str = '#D0E8F2'  # Light blue
    land_color: str = '#F5F5DC'   # Beige

    # Base image options (ADD THESE)
    # Base image options
    use_base_image: bool = False
    base_image_type: str = 'natural_earth_greyblue'
    base_image_path: Optional[str] = None
    base_image_month: Optional[int] = None
    base_image_alpha: float = 1.0
    base_image_interpolation: str = 'nearest'
    base_image_target_resolution: int = 4000 

    # Transform cache fields (set by plotall.py before forking workers)
    cached_target_extent: Optional[Tuple[float, float, float, float]] = None
    cached_target_shape: Optional[Tuple[int, int]] = None
    cached_transform_key: Optional[str] = None
    
    # GSHHS options
    use_gshhs: bool = False
    gshhs_path: str = '/discover/nobackup/projects/gmao/osse2/GSHHG/v2.3.7'
    gshhs_resolution: str = 'h'  # f, h, i, l, c
    gshhs_min_area: float = 1.0  # km^2
    gshhs_max_level: int = 2     # 1=land, 2=land+lakes, 3=+islands, 4=+ponds
    gshhs_outline: bool = False
    gshhs_outline_color: str = '#FFFFFF'
    gshhs_outline_width: float = 0.3
    
    # NWS Warnings options
    show_nws_warnings: bool = False
    nws_shapefile_base: str = '/discover/nobackup/projects/gmao/osse2/TSE_staging/SHAPE_FILES/ALL'
    nws_warning_types: Optional[List[Tuple[str, str]]] = None  # List of (type, status) tuples: [('TO', 'W'), ('SV', 'W')]
    nws_warning_alpha: float = 0.5
    nws_warning_fill: bool = False  # Default to outline only
    nws_warning_outline_width: float = 1.0  # Warnings get this value, watches/advisories get 0.5x
    nws_custom_type: Optional[str] = None  # Single warning type: 'TO', 'SV', etc.
    nws_custom_status: Optional[str] = None  # Single status: 'W', 'A', 'Y', 'S'
    nws_severe_only: bool = False  # Show only tornado and severe thunderstorm warnings

    # Boundary styles
    coastline_color: str = '#333333'
    coastline_width: float = 1.0
    coastline_alpha: float = 1.0
    
    country_color: str = '#666666'
    country_width: float = 0.6
    country_alpha: float = 0.8
    
    state_color: str = '#999999'
    state_width: float = 0.4
    state_alpha: float = 0.6
    
    county_color: str = '#CCCCCC'
    county_width: float = 0.2
    county_alpha: float = 0.4
    
    river_color: str = '#5AA0D0'
    river_width: float = 0.2
    river_alpha: float = 0.6
    
    road_color: str = '#FF8C00'
    road_width: float = 0.3
    road_alpha: float = 0.7
    
    # Frame
    show_frame: bool = False
    frame_width: float = 2.0
    frame_color: str = 'black'
    
    # Grid
    show_gridlines: bool = False
    show_grid_labels: bool = False
    gridline_color: str = 'gray'
    gridline_width: float = 0.5
    gridline_alpha: float = 0.5
    gridline_style: str = '--'
    
    # Title
    show_title: bool = False
    title_fontsize: Optional[int] = None  # Auto-scaled if None
    
    # Timestamp
    show_timestamp: bool = True
    timestamp_location: str = 'lower left'
    timestamp_fontsize: Optional[int] = None  # Auto-scaled if None
    timestamp_format: str = 'detailed'  # 'detailed' or 'simple'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy parameter passing"""
        return {k: v for k, v in self.__dict__.items()}
    

    @staticmethod
    def wxmaps() -> 'StyleConfig':
        """WxMaps theme style"""
        return StyleConfig(
            background_color='white',
            text_color='black',
            use_gshhs=True,
            ocean_color="#EEEEEE",
            land_color="#FFFFFF",
            coastline_color="black",
            coastline_width=0.5,
            coastline_alpha=0.6,
            country_color="black",
            country_width=0.5,
            country_alpha=0.6,
            state_color="black",
            state_width=0.5,
            state_alpha=0.6,
            show_nws_warnings=False,
            show_timestamp=True
        )

    @staticmethod
    def light() -> 'StyleConfig':
        """Light theme style"""
        return StyleConfig(
            background_color='white',
            text_color='black',
            use_gshhs=False,
            ocean_color="#EEEEEE",
            land_color="#FFFFFF",
            coastline_color="black",
            coastline_width=0.5,
            coastline_alpha=0.8,
            country_color="black",
            country_width=0.5,
            country_alpha=0.8,
            state_color="black",
            state_width=0.5,
            state_alpha=0.8,
            show_nws_warnings=False,
            show_timestamp=True,
            show_frame=False,
            show_gridlines=False,
            show_title=False
        )
 
    @staticmethod
    def dark() -> 'StyleConfig':
        """Dark theme style"""
        return StyleConfig(
            background_color='#454545',
            text_color='white',
            use_gshhs=False, 
            ocean_color="#454545",
            land_color="#6e6e6e",
            coastline_color="#FFFFFF",
            coastline_width=0.5,
            coastline_alpha=0.8,
            country_color="#FFFFFF",
            country_width=0.5,
            country_alpha=0.8,
            state_color="#FFFFFF",
            state_width=0.5,
            state_alpha=0.8,
            show_nws_warnings=False,
            show_timestamp=True,
            show_frame=False,
            show_gridlines=False,
            show_title=False
        )

    @staticmethod
    def nightlights() -> 'StyleConfig':
        """nightlights theme style"""
        return StyleConfig(
            background_color='black',
            text_color='white',
            use_gshhs=False,
            use_base_image=True,
            base_image_type='dnb_nightlights',
            coastline_color="#FFFFFF",
            coastline_width=0.5,
            coastline_alpha=0.8,
            country_color="#FFFFFF",
            country_width=0.5,
            country_alpha=0.8,
            state_color="#FFFFFF",
            state_width=0.5,
            state_alpha=0.8,
            show_nws_warnings=False,
            show_timestamp=True,
            show_frame=False,
            show_gridlines=False,
            show_title=False
        )
 
    @staticmethod
    def satellite() -> 'StyleConfig':
        """Satellite-like style"""
        return StyleConfig(
            background_color='black',
            text_color='white',
            use_base_image=False,
            coastline_color='#FFFFFF',
            coastline_width=0.4,
            coastline_alpha=0.8,
            country_color='#FFFFFF',
            country_width=0.4,
            country_alpha=0.6,
            state_color='#FFFFFF',
            state_width=0.3,
            state_alpha=0.4,
            show_frame=False,
            show_gridlines=False,
            show_title=False
        )
    
    @staticmethod
    def print_quality() -> 'StyleConfig':
        """High-contrast style for printing"""
        return StyleConfig(
            background_color='white',
            text_color='black',
            ocean_color='white',
            land_color='white',
            coastline_color='black',
            coastline_width=1.2,
            coastline_alpha=1.0,
            country_color='black',
            country_width=0.8,
            country_alpha=1.0,
            state_color='black',
            state_width=0.5,
            state_alpha=0.8,
            show_frame=True,
            frame_width=2.0,
            frame_color='black',
            show_gridlines=True,
            show_grid_labels=True,
            gridline_color='black',
            gridline_alpha=0.3,
            show_title=True
        )
    
@dataclass
class ResolutionConfig:
    """Resolution configuration with 16:9 aspect ratio"""
    name: str
    width: int
    height: int
    dpi: int
    figsize: Tuple[float, float]
    
    def __post_init__(self):
        """Validate 16:9 aspect ratio"""
        aspect = self.width / self.height
        if not np.isclose(aspect, 16/9, rtol=0.01):
            raise ValueError(f"Resolution must have 16:9 aspect ratio. Got {aspect:.3f}")

@dataclass
class WxMapsConfig:
    """Standard map configurations for weather visualization"""
    
    # Standard resolutions with strict 16:9 aspect ratio
    RESOLUTIONS = {
        'hd': ResolutionConfig(
            name='HD (1080p)',
            width=1920,
            height=1080,
            dpi=100,
            figsize=(19.2, 10.8)
        ),
        '4k': ResolutionConfig(
            name='4K (2160p)',
            width=3840,
            height=2160,
            dpi=200,
            figsize=(19.2, 10.8)
        ),
        '8k': ResolutionConfig(
            name='8K (4320p)',
            width=7680,
            height=4320,
            dpi=400,
            figsize=(19.2, 10.8)
        ),
        'fhd': ResolutionConfig(
            name='Full HD',
            width=1920,
            height=1080,
            dpi=120,
            figsize=(16.0, 9.0)
        ),
        '2k': ResolutionConfig(
            name='2K',
            width=2560,
            height=1440,
            dpi=150,
            figsize=(17.067, 9.6)
        ),
    }
    
    @staticmethod
    def scale_style_for_resolution(style: StyleConfig, resolution: str) -> StyleConfig:
        """
        Scale style properties based on resolution
        
        Parameters:
        -----------
        style : StyleConfig
            Base style configuration
        resolution : str
            Resolution key ('hd', '4k', '8k')
        
        Returns:
        --------
        StyleConfig : Scaled style configuration
        """
        scale_factors = {
            'hd': 1.0,
            'fhd': 1.0,
            '2k': 1.5,
            '4k': 2.0,
            '8k': 4.0
        }
        
        scale = scale_factors.get(resolution, 1.0)
        
        # Create a copy and scale linewidths
        scaled_style = StyleConfig(**style.to_dict())
        scaled_style.coastline_width *= scale
        scaled_style.country_width *= scale
        scaled_style.state_width *= scale
        scaled_style.county_width *= scale
        scaled_style.river_width *= scale
        scaled_style.road_width *= scale
        scaled_style.frame_width *= scale
        scaled_style.gridline_width *= scale
        
        return scaled_style
    
    @staticmethod
    def get_global_map() -> MapConfig:
        """Global Plate Carrée projection optimized for 16:9"""
        extent = [-180, 180, -90, 90]
        return MapConfig(
            name='global',
            projection=ccrs.PlateCarree(),
            extent=extent,
        )
   
    @staticmethod
    def get_northamerica_map() -> MapConfig:
        """North America - Lambert Conformal optimized for 16:9"""
        extent = [-160, -40, 15, 70]
        return MapConfig(
            name='northamerica',
            projection=ccrs.LambertConformal(
                central_longitude=-100,
                central_latitude=35,
                standard_parallels=(33, 45)
            ),
            extent=extent,
            center_lon=-100,
            center_lat=35,
            standard_parallels=(33, 45),
        )
 
    @staticmethod
    def get_conus_map() -> MapConfig:
        """Continental United States - Lambert Conformal optimized for 16:9"""
        extent = [-125.75, -68.25, 23, 50]
        return MapConfig(
            name='conus',
            projection=ccrs.LambertConformal(
                central_longitude=-97,
                central_latitude=30,
                standard_parallels=(33, 45)
            ),
            extent=extent,
            center_lon=-96,
            center_lat=37,
            standard_parallels=(33, 45),
        )

    @staticmethod
    def get_conus_midatlantic_map() -> MapConfig:
        """Eastern CONUS optimized for 16:9"""
        extent = [-90, -60, 35, 45]
        return MapConfig(
            name='conus_midatlantic',
            projection=ccrs.LambertConformal(
                central_longitude=-75,
                central_latitude=35.5,
                standard_parallels=(33, 45)
            ),
            extent=extent,
            center_lon=-75,
            center_lat=35.5,
        )

    @staticmethod
    def get_conus_east_map() -> MapConfig:
        """Eastern CONUS optimized for 16:9"""
        extent = [-97, -63, 23, 48]
        return MapConfig(
            name='conus_east',
            projection=ccrs.LambertConformal(
                central_longitude=-80,
                central_latitude=35.5,
                standard_parallels=(33, 45)
            ),
            extent=extent,
            center_lon=-80,
            center_lat=35.5,
        )
    
    @staticmethod
    def get_conus_west_map() -> MapConfig:
        """Western CONUS optimized for 16:9"""
        extent = [-132, -98, 29, 51]
        return MapConfig(
            name='conus_west',
            projection=ccrs.LambertConformal(
                central_longitude=-115,
                central_latitude=40,
                standard_parallels=(33, 45)
            ),
            extent=extent,
            center_lon=-115,
            center_lat=40,
        )
    
    @staticmethod
    def get_conus_central_map() -> MapConfig:
        """Central CONUS optimized for 16:9"""
        extent = [-110, -82, 27, 50]
        return MapConfig(
            name='conus_central',
            projection=ccrs.LambertConformal(
                central_longitude=-96,
                central_latitude=38.5,
                standard_parallels=(33, 45)
            ),
            extent=extent,
            center_lon=-96,
            center_lat=38.5,
        )
    
    @staticmethod
    def get_conus_southeast_map() -> MapConfig:
        """Southeastern CONUS - Hurricane alley"""
        extent = [-95, -70, 24, 40]
        return MapConfig(
            name='conus_southeast',
            projection=ccrs.LambertConformal(
                central_longitude=-82.5,
                central_latitude=32,
                standard_parallels=(30, 40)
            ),
            extent=extent,
            center_lon=-82.5,
            center_lat=32,
        )
    
    @staticmethod
    def get_conus_northeast_map() -> MapConfig:
        """Northeastern CONUS"""
        extent = [-84, -64, 37, 49]  # Expanded
        return MapConfig(
            name='conus_northeast',
            projection=ccrs.LambertConformal(
                central_longitude=-73.5,
                central_latitude=43,
                standard_parallels=(40, 48)
            ),
            extent=extent,
            center_lon=-73.5,
            center_lat=43,
        )
    
    @staticmethod
    def get_conus_southwest_map() -> MapConfig:
        """Southwestern CONUS"""
        extent = [-125, -102, 28, 42]
        return MapConfig(
            name='conus_southwest',
            projection=ccrs.LambertConformal(
                central_longitude=-113.5,
                central_latitude=35,
                standard_parallels=(30, 40)
            ),
            extent=extent,
            center_lon=-113.5,
            center_lat=35,
        )
    
    @staticmethod
    def get_conus_northwest_map() -> MapConfig:
        """Northwestern CONUS"""
        extent = [-127, -105, 40, 50]
        return MapConfig(
            name='conus_northwest',
            projection=ccrs.LambertConformal(
                central_longitude=-116,
                central_latitude=45,
                standard_parallels=(40, 50)
            ),
            extent=extent,
            center_lon=-116,
            center_lat=45,
        )
    
    @staticmethod
    def get_great_plains_map() -> MapConfig:
        """Great Plains - Tornado Alley"""
        extent = [-108, -88, 32, 48]
        return MapConfig(
            name='great_plains',
            projection=ccrs.LambertConformal(
                central_longitude=-98,
                central_latitude=40,
                standard_parallels=(35, 45)
            ),
            extent=extent,
            center_lon=-98,
            center_lat=40,
        )
    
    @staticmethod
    def get_europe_map() -> MapConfig:
        """Europe - Lambert Conformal optimized for 16:9"""
        extent = [-15, 40, 34, 72]
        return MapConfig(
            name='europe',
            projection=ccrs.LambertConformal(
                central_longitude=12.5,
                central_latitude=53,
                standard_parallels=(40, 60)
            ),
            extent=extent,
            center_lon=12.5,
            center_lat=53,
        )
    
    @staticmethod
    def get_europe_west_map() -> MapConfig:
        """Western Europe"""
        extent = [-12, 15, 40, 60]
        return MapConfig(
            name='europe_west',
            projection=ccrs.LambertConformal(
                central_longitude=1.5,
                central_latitude=50,
                standard_parallels=(40, 55)
            ),
            extent=extent,
            center_lon=1.5,
            center_lat=50,
        )
    
    @staticmethod
    def get_europe_east_map() -> MapConfig:
        """Eastern Europe"""
        extent = [10, 45, 42, 62]
        return MapConfig(
            name='europe_east',
            projection=ccrs.LambertConformal(
                central_longitude=27.5,
                central_latitude=52,
                standard_parallels=(45, 58)
            ),
            extent=extent,
            center_lon=27.5,
            center_lat=52,
        )
    
    @staticmethod
    def get_atlantic_hurricane_map() -> MapConfig:
        """Atlantic Hurricane Basin optimized for 16:9"""
        extent = [-105, -10, 8, 52]
        return MapConfig(
            name='atlantic_hurricane',
            projection=ccrs.PlateCarree(central_longitude=-57.5),
            extent=extent,
            center_lon=-57.5,
        )
    
    @staticmethod
    def get_pacific_hurricane_map() -> MapConfig:
        """Eastern Pacific Hurricane Basin optimized for 16:9"""
        extent = [-165, -75, 5, 42]
        return MapConfig(
            name='pacific_hurricane',
            projection=ccrs.PlateCarree(central_longitude=-120),
            extent=extent,
            center_lon=-120,
        )
    
    @staticmethod
    def get_gulf_of_mexico_map() -> MapConfig:
        """Gulf of Mexico"""
        extent = [-100, -78, 18, 31]
        return MapConfig(
            name='gulf_of_mexico',
            projection=ccrs.LambertConformal(
                central_longitude=-89,
                central_latitude=24.5,
                standard_parallels=(20, 30)
            ),
            extent=extent,
            center_lon=-89,
            center_lat=24.5,
        )
    
    @staticmethod
    def get_caribbean_map() -> MapConfig:
        """Caribbean Sea"""
        extent = [-90, -58, 10, 24]
        return MapConfig(
            name='caribbean',
            projection=ccrs.PlateCarree(central_longitude=-74),
            extent=extent,
            center_lon=-74,
        )
    
    @staticmethod
    def get_north_atlantic_map() -> MapConfig:
        """North Atlantic optimized for 16:9"""
        extent = [-85, 20, 18, 72]
        return MapConfig(
            name='north_atlantic',
            projection=ccrs.Stereographic(
                central_longitude=-32.5,
                central_latitude=45
            ),
            extent=extent,
            center_lon=-32.5,
            center_lat=45,
        )
    
    @staticmethod
    def get_north_pacific_map() -> MapConfig:
        """North Pacific optimized for 16:9"""
        extent = [-180, -100, 8, 62]
        return MapConfig(
            name='north_pacific',
            projection=ccrs.Stereographic(
                central_longitude=-140,
                central_latitude=35
            ),
            extent=extent,
            center_lon=-140,
            center_lat=35,
        )
    
    @staticmethod
    def get_arctic_map() -> MapConfig:
        """Arctic region"""
        extent = None
        return MapConfig(
            name='arctic',
            projection=ccrs.NorthPolarStereo(central_longitude=0),
            extent=extent,
            center_lon=0,
            center_lat=90,
        )
    
    @staticmethod
    def get_goes_east_full_disk() -> MapConfig:
        """GOES-East Full Disk (Geostationary) optimized for 16:9"""
        extent = None
        return MapConfig(
            name='goes_east_full_disk',
            projection=ccrs.Geostationary(
                central_longitude=-75.0,
                satellite_height=35786023.0,
                sweep_axis='x'
            ),
            extent=extent,
            center_lon=-75.0,
        )
    
    @staticmethod
    def get_goes_west_full_disk() -> MapConfig:
        """GOES-West Full Disk (Geostationary) optimized for 16:9"""
        extent = None
        return MapConfig(
            name='goes_west_full_disk',
            projection=ccrs.Geostationary(
                central_longitude=-137.0,
                satellite_height=35786023.0,
                sweep_axis='x'
            ),
            extent=extent,
            center_lon=-137.0,
        )
    
    @staticmethod
    def get_goes_east_conus() -> MapConfig:
        """GOES-East CONUS sector"""
        extent = [-135, -60, 15, 60]
        return MapConfig(
            name='goes_east_conus',
            projection=ccrs.Geostationary(
                central_longitude=-75.0,
                satellite_height=35786023.0,
                sweep_axis='x'
            ),
            extent=extent,
            center_lon=-75.0,
        )
    
    @staticmethod
    def get_goes_west_conus() -> MapConfig:
        """GOES-West CONUS sector"""
        extent = [-165, -95, 15, 60]
        return MapConfig(
            name='goes_west_conus',
            projection=ccrs.Geostationary(
                central_longitude=-137.0,
                satellite_height=35786023.0,
                sweep_axis='x'
            ),
            extent=extent,
            center_lon=-137.0,
        )
    
    @staticmethod
    def create_custom_map(name: str, center_lon: float, center_lat: float,
                         extent: Tuple[float, float, float, float],
                         proj_type: str = 'lambert') -> MapConfig:
        """
        Create a custom map configuration optimized for 16:9
        """
        if proj_type == 'lambert':
            projection = ccrs.LambertConformal(
                central_longitude=center_lon,
                central_latitude=center_lat,
                standard_parallels=(center_lat - 10, center_lat + 10)
            )
        elif proj_type == 'mercator':
            projection = ccrs.Mercator(central_longitude=center_lon)
        elif proj_type == 'stereographic':
            projection = ccrs.Stereographic(
                central_longitude=center_lon,
                central_latitude=center_lat
            )
        elif proj_type == 'platecarree':
            projection = ccrs.PlateCarree(central_longitude=center_lon)
        elif proj_type == 'geostationary':
            projection = ccrs.Geostationary(
                central_longitude=center_lon,
                satellite_height=35786023.0,
                sweep_axis='x'
            )
        else:
            raise ValueError(f"Unknown projection type: {proj_type}")
        
        return MapConfig(
            name=name,
            projection=projection,
            extent=extent,
            center_lon=center_lon,
            center_lat=center_lat,
        )
    
    @staticmethod
    def get_all_standard_maps() -> dict:
        """Return dictionary of all standard map configurations"""
        return {
            'global': WxMapsConfig.get_global_map(),
            'northamerica': WxMapsConfig.get_northamerica_map(),
            'conus': WxMapsConfig.get_conus_map(),
            'conus_east': WxMapsConfig.get_conus_east_map(),
            'conus_west': WxMapsConfig.get_conus_west_map(),
            'conus_central': WxMapsConfig.get_conus_central_map(),
            'conus_southeast': WxMapsConfig.get_conus_southeast_map(),
            'conus_northeast': WxMapsConfig.get_conus_northeast_map(),
            'conus_southwest': WxMapsConfig.get_conus_southwest_map(),
            'conus_northwest': WxMapsConfig.get_conus_northwest_map(),
            'conus_midatlantic': WxMapsConfig.get_conus_midatlantic_map(),
            'great_plains': WxMapsConfig.get_great_plains_map(),
            'europe': WxMapsConfig.get_europe_map(),
            'europe_west': WxMapsConfig.get_europe_west_map(),
            'europe_east': WxMapsConfig.get_europe_east_map(),
            'atlantic_hurricane': WxMapsConfig.get_atlantic_hurricane_map(),
            'pacific_hurricane': WxMapsConfig.get_pacific_hurricane_map(),
            'gulf_of_mexico': WxMapsConfig.get_gulf_of_mexico_map(),
            'caribbean': WxMapsConfig.get_caribbean_map(),
            'north_atlantic': WxMapsConfig.get_north_atlantic_map(),
            'north_pacific': WxMapsConfig.get_north_pacific_map(),
            'arctic': WxMapsConfig.get_arctic_map(),
            'goes_east_full_disk': WxMapsConfig.get_goes_east_full_disk(),
            'goes_west_full_disk': WxMapsConfig.get_goes_west_full_disk(),
            'goes_east_conus': WxMapsConfig.get_goes_east_conus(),
            'goes_west_conus': WxMapsConfig.get_goes_west_conus(),
        }
