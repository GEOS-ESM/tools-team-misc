"""
GSHHS (Global Self-consistent, Hierarchical, High-resolution Geography Database) Reader
Provides high-resolution land/water boundaries from GSHHS binary files
"""
import numpy as np
import struct
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


class GSHHSReader:
    """Reader for GSHHS binary format files"""
    
    # GSHHS polygon levels
    LEVEL_LAND = 1
    LEVEL_LAKE = 2
    LEVEL_ISLAND_IN_LAKE = 3
    LEVEL_POND_IN_ISLAND = 4
    
    def __init__(self, filename: str):
        """
        Initialize GSHHS reader
        
        Parameters:
        -----------
        filename : str
            Path to GSHHS binary file (e.g., gshhs_f.b, gshhs_h.b, gshhs_i.b, gshhs_l.b, gshhs_c.b)
            f = full resolution
            h = high resolution
            i = intermediate resolution
            l = low resolution
            c = crude resolution
        """
        self.filename = filename
        if not Path(filename).exists():
            raise FileNotFoundError(f"GSHHS file not found: {filename}")
        
        # 44-byte header structure for GSHHS v2.2+
        self.header_struct = struct.Struct('>IIIiiiiIIii')  # Big-endian
        self.header_size = 44
        
    def read_polygons(self, min_area: float = 0.0, max_level: int = 4,
                     extent: Optional[Tuple[float, float, float, float]] = None) -> List[dict]:
        """
        Read polygons from GSHHS file
        
        Parameters:
        -----------
        min_area : float
            Minimum polygon area in km^2 (default: 0.0 = all polygons)
        max_level : int
            Maximum polygon level to include (1-4)
            1 = land
            2 = lake
            3 = island in lake
            4 = pond in island
        extent : tuple, optional
            Map extent [lon_min, lon_max, lat_min, lat_max] in -180/180 format
            Will be converted to 0/360 for GSHHS comparison
            
        Returns:
        --------
        list of dict : List of polygon dictionaries with keys:
            - 'level': polygon level (1-4)
            - 'area': area in km^2
            - 'lon': array of longitudes (converted to -180/180)
            - 'lat': array of latitudes
            - 'west', 'east', 'south', 'north': bounding box
        """
        polygons = []
        total_read = 0
        filtered_by_level = 0
        filtered_by_area = 0
        filtered_by_extent = 0
        
        # Convert extent from -180/180 to 0/360 for GSHHS comparison
        extent_360 = None
        if extent:
            lon_min, lon_max, lat_min, lat_max = extent
            # Convert to 0-360
            lon_min_360 = lon_min if lon_min >= 0 else lon_min + 360
            lon_max_360 = lon_max if lon_max >= 0 else lon_max + 360
            extent_360 = (lon_min_360, lon_max_360, lat_min, lat_max)
        
        with open(self.filename, 'rb') as f:
            while True:
                # Read header
                header_bytes = f.read(self.header_size)
                if len(header_bytes) < self.header_size:
                    break  # EOF
                
                header = self._parse_header(header_bytes)
                total_read += 1
                
                # Get polygon bounding box from header (in 0-360 format from GSHHS)
                poly_west = header['west']
                poly_east = header['east']
                poly_south = header['south']
                poly_north = header['north']
                
                # Filter by level
                if header['level'] > max_level:
                    # Skip this polygon's data
                    f.seek(header['npoints'] * 8, 1)  # 8 bytes per point (2 longs)
                    filtered_by_level += 1
                    continue
                
                # Filter by area
                if header['area'] < min_area:
                    f.seek(header['npoints'] * 8, 1)
                    filtered_by_area += 1
                    continue
                
                # Read polygon coordinates
                npoints = header['npoints']
                coord_bytes = f.read(npoints * 8)  # 2 longs (4 bytes each) per point
                
                if len(coord_bytes) < npoints * 8:
                    break  # EOF
                
                # Parse coordinates (GSHHS uses 0-360 longitude)
                coords = struct.unpack(f'>{npoints * 2}i', coord_bytes)
                lon = np.array(coords[0::2]) * 1.0e-6  # Convert from micro-degrees
                lat = np.array(coords[1::2]) * 1.0e-6
                
                # Convert longitudes from 0-360 to -180-180
                lon = np.where(lon > 180, lon - 360, lon)
                
                # Also convert bounding box to -180/180 for storage
                poly_west_180 = poly_west if poly_west <= 180 else poly_west - 360
                poly_east_180 = poly_east if poly_east <= 180 else poly_east - 360
                
                # Filter by extent if provided (compare in 0-360 space)
                if extent_360 is not None:
                    lon_min_360, lon_max_360, lat_min, lat_max = extent_360
                    
                    # Check latitude first (simple)
                    if poly_south > lat_max or poly_north < lat_min:
                        filtered_by_extent += 1
                        continue
                    
                    # Check longitude (handle wrapping)
                    # Polygon crosses dateline if west > east
                    if poly_west > poly_east:
                        # Polygon crosses dateline
                        # Check if extent intersects either side
                        intersects_lon = (poly_west <= lon_max_360 or poly_east >= lon_min_360)
                    else:
                        # Normal case - no dateline crossing
                        intersects_lon = not (poly_east < lon_min_360 or poly_west > lon_max_360)
                    
                    # Also check if extent crosses dateline
                    if lon_min_360 > lon_max_360:
                        # Extent crosses dateline
                        intersects_lon = (poly_west >= lon_min_360 or poly_east <= lon_max_360)
                    
                    if not intersects_lon:
                        filtered_by_extent += 1
                        continue
                
                # Store polygon (with coordinates in -180/180 format)
                polygons.append({
                    'level': header['level'],
                    'area': header['area'],
                    'lon': lon,
                    'lat': lat,
                    'west': poly_west_180,
                    'east': poly_east_180,
                    'south': poly_south,
                    'north': poly_north,
                    'npoints': npoints
                })
        
        return polygons
    
    def _parse_header(self, header_bytes: bytes) -> dict:
        """Parse GSHHS header structure"""
        values = self.header_struct.unpack(header_bytes)
        
        # Unpack header fields
        polygon_id, npoints, flag, west, east, south, north, area, area_full, container, ancestor = values
        
        # Parse flag byte
        level = flag & 255
        version = (flag >> 8) & 255
        greenwich = (flag >> 16) & 3
        source = (flag >> 24) & 1
        river = (flag >> 25) & 1
        
        # For version 9+ (GSHHS 2.2+), calculate magnitude
        if version >= 9:
            magnitude = (flag >> 26) & 255
            area_km2 = float(area) / (10.0 ** magnitude)
        else:
            area_km2 = float(area) * 0.1
        
        return {
            'id': polygon_id,
            'npoints': npoints,
            'level': level,
            'version': version,
            'greenwich': greenwich,
            'source': source,
            'river': river,
            'west': west * 1.0e-6,
            'east': east * 1.0e-6,
            'south': south * 1.0e-6,
            'north': north * 1.0e-6,
            'area': area_km2,
            'area_full': area_full,
            'container': container,
            'ancestor': ancestor
        }
    
    def plot_polygons_on_map(self, ax, polygons: List[dict],
                            land_color: str = '#F5F5DC',
                            water_color: str = '#D0E8F2',
                            fill: bool = True,
                            outline: bool = False,
                            outline_color: str = 'black',
                            outline_width: float = 0.5,
                            transform: Optional[ccrs.Projection] = None):
        """
        Plot GSHHS polygons on a Cartopy map
        
        Parameters:
        -----------
        ax : GeoAxes
            Cartopy GeoAxes to plot on
        polygons : list
            List of polygon dictionaries from read_polygons()
        land_color : str
            Fill color for land polygons (level 1, 3)
        water_color : str
            Fill color for water polygons (level 2, 4)
        fill : bool
            Whether to fill polygons
        outline : bool
            Whether to draw polygon outlines
        outline_color : str
            Color for outlines
        outline_width : float
            Width of outlines
        transform : cartopy projection, optional
            Projection for coordinates (default: PlateCarree)
        """
        if transform is None:
            transform = ccrs.PlateCarree()
        
        from matplotlib.patches import Polygon as MplPolygon
        
        # Separate polygons by level
        land_patches = []  # levels 1, 3
        water_patches = []  # levels 2, 4
        
        for poly in polygons:
            coords = np.column_stack([poly['lon'], poly['lat']])
            
            if poly['level'] in [1, 3]:  # Land, Island in lake
                land_patches.append(coords)
            elif poly['level'] in [2, 4]:  # Lake, Pond in island
                water_patches.append(coords)
        
        # Plot filled polygons - draw them individually to ensure proper rendering
        if fill:
            # Draw land polygons first (level 1, 3)
            for coords in land_patches:
                poly = MplPolygon(coords, 
                                facecolor=land_color, 
                                edgecolor='none',
                                transform=transform, 
                                zorder=1,
                                closed=True)
                ax.add_patch(poly)
            
            # Draw water polygons on top (level 2, 4) - lakes should be water color
            for coords in water_patches:
                poly = MplPolygon(coords, 
                                facecolor=water_color,  # Lakes get water/ocean color
                                edgecolor='none',
                                transform=transform, 
                                zorder=2,
                                closed=True)
                ax.add_patch(poly)
        
        # Plot outlines if requested
        if outline:
            all_patches = land_patches + water_patches
            for coords in all_patches:
                poly = MplPolygon(coords, 
                                facecolor='none', 
                                edgecolor=outline_color,
                                linewidth=outline_width, 
                                transform=transform, 
                                zorder=3,
                                closed=True)
                ax.add_patch(poly)

def get_gshhs_resolution_file(base_path: str, resolution: str = 'f') -> str:
    """
    Get GSHHS filename for specified resolution
    
    Parameters:
    -----------
    base_path : str
        Base directory containing GSHHS files
    resolution : str
        Resolution code:
        'f' = full (highest resolution)
        'h' = high
        'i' = intermediate
        'l' = low
        'c' = crude (lowest resolution)
    
    Returns:
    --------
    str : Full path to GSHHS file
    """
    filename = Path(base_path) / f'gshhs_{resolution}.b'
    return str(filename)
