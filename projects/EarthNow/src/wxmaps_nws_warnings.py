"""
NWS Warnings Module
Read and plot NWS watches/warnings from shapefiles
Uses official NWS color scheme from map_warnings_assist.xml

Rendering Strategy:
- Two-pass rendering: Non-warnings first (thinner), then warnings (thicker) on top
- Z-order: Non-warnings at z=10/11, Warnings at z=12/13
- Line thickness: Warnings = 1.0, Watches/Advisories/Statements = 0.5
"""
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PolyCollection
from datetime import datetime
import xml.etree.ElementTree as ET


class NWSWarnings:
    """Handler for NWS watch/warning shapefiles"""
    
    # Official NWS VTEC Significance codes
    VTEC_SIGNIFICANCE = {
        "W": "Warning",
        "Y": "Advisory",
        "A": "Watch",
        "S": "Statement",
        "O": "Outlook",
        "N": "Synopsis",
        "F": "Forecast",
    }
    
    # Official NWS VTEC Phenomena codes
    # From https://www.nws.noaa.gov/directives/sym/pd01017003curr.pdf
    VTEC_PHENOMENA = {
        "AF": "Ashfall",
        "AS": "Air Stagnation",
        "BH": "Beach Hazard",
        "BS": "Blowing Snow",
        "BW": "Brisk Wind",
        "BZ": "Blizzard",
        "CF": "Coastal Flood",
        "CW": "Cold Weather",
        "DF": "Debris Flow",
        "DS": "Dust Storm",
        "DU": "Blowing Dust",
        "EC": "Extreme Cold",
        "EH": "Excessive Heat",
        "EW": "Extreme Wind",
        "FA": "Flood",
        "FF": "Flash Flood",
        "FG": "Dense Fog",
        "FL": "Flood",
        "FR": "Frost",
        "FW": "Red Flag",
        "FZ": "Freeze",
        "UP": "Freezing Spray",
        "GL": "Gale",
        "HF": "Hurricane Force Wind",
        "HI": "Inland Hurricane",
        "HS": "Heavy Snow",
        "HT": "Heat",
        "HU": "Hurricane",
        "HW": "High Wind",
        "HY": "Hydrologic",
        "HZ": "Hard Freeze",
        "IP": "Sleet",
        "IS": "Ice Storm",
        "LB": "Lake Effect Snow and Blowing Snow",
        "LE": "Lake Effect Snow",
        "LO": "Low Water",
        "LS": "Lakeshore Flood",
        "LW": "Lake Wind",
        "MA": "Marine",
        "MF": "Marine Dense Fog",
        "MH": "Marine Ashfall",
        "MS": "Marine Dense Smoke",
        "RB": "Small Craft for Rough",
        "RP": "Rip Currents",
        "SB": "Snow and Blowing",
        "SC": "Small Craft",
        "SE": "Hazardous Seas",
        "SI": "Small Craft for Winds",
        "SM": "Dense Smoke",
        "SN": "Snow",
        "SQ": "Snow Squall",
        "SR": "Storm",
        "SS": "Storm Surge",
        "SU": "High Surf",
        "SV": "Severe Thunderstorm",
        "SW": "Small Craft for Hazardous Seas",
        "TI": "Inland Tropical Storm",
        "TO": "Tornado",
        "TR": "Tropical Storm",
        "TS": "Tsunami",
        "TY": "Typhoon",
        "WC": "Wind Chill",
        "WI": "Wind",
        "WS": "Winter Storm",
        "WW": "Winter Weather",
        "XH": "Extreme Heat",  # March 2025
        "ZF": "Freezing Fog",
        "ZR": "Freezing Rain",
    }
    
    # Official NWS Colors from map_warnings_assist.xml
    # Taken from http://www.weather.gov/help-map
    NWS_COLORS = {
        "AF.W": "#A9A9A9",
        "AF.Y": "#696969",
        "AS.O": "#808080",
        "AS.Y": "#808080",
        "BH.S": "#40E0D0",
        "BW.Y": "#D8BFD8",
        "BZ.A": "#ADFF2F",
        "BZ.W": "#FF4500",
        "CF.A": "#66CDAA",
        "CF.S": "#6B8E23",
        "CF.W": "#228B22",
        "CF.Y": "#7CFC00",
        "DS.W": "#FFE4C4",
        "DS.Y": "#BDB76B",
        "DU.W": "#FFE4C4",
        "DU.Y": "#BDB76B",
        "EC.A": "#0000FF",
        "EC.W": "#0000FF",
        "EH.A": "#800000",
        "EH.W": "#C71585",
        "EH.Y": "#800000",
        "EW.W": "#FF8C00",
        "FA.A": "#2E8B57",
        "FA.W": "#00FF00",
        "FA.Y": "#00FF7F",
        "FF.A": "#2E8B57",
        "FF.S": "#8B0000",
        "FF.W": "#8B0000",
        "FG.Y": "#708090",
        "FL.A": "#2E8B57",
        "FL.S": "#00FF00",
        "FL.W": "#00FF00",
        "FL.Y": "#00FF7F",
        "FR.Y": "#6495ED",
        "FW.A": "#FFDEAD",
        "FW.W": "#FF1493",
        "FZ.A": "#00FFFF",
        "FZ.W": "#483D8B",
        "GL.A": "#FFC0CB",
        "GL.W": "#DDA0DD",
        "HF.A": "#9932CC",
        "HF.W": "#CD5C5C",
        "HT.Y": "#FF7F50",
        "HU.A": "#FF00FF",
        "HU.S": "#FFE4B5",
        "HU.W": "#DC143C",
        "HW.A": "#B8860B",
        "HW.W": "#DAA520",
        "HY.Y": "#00FF7F",
        "HZ.A": "#4169E1",
        "HZ.W": "#9400D3",
        "IS.W": "#8B008B",
        "LE.A": "#87CEFA",
        "LE.W": "#008B8B",
        "LE.Y": "#48D1CC",
        "LO.Y": "#A52A2A",
        "LS.A": "#66CDAA",
        "LS.S": "#6B8E23",
        "LS.W": "#228B22",
        "LS.Y": "#7CFC00",
        "LW.Y": "#D2B48C",
        "MA.S": "#FFDAB9",
        "MA.W": "#FFA500",
        "MF.Y": "#708090",
        "MH.Y": "#696969",
        "MS.Y": "#F0E68C",
        "RB.Y": "#D8BFD8",
        "RP.S": "#40E0D0",
        "SC.Y": "#D8BFD8",
        "SE.A": "#483D8B",
        "SE.W": "#D8BFD8",
        "SI.Y": "#D8BFD8",
        "SM.Y": "#F0E68C",
        "SQ.W": "#C71585",
        "SR.A": "#FFE4B5",
        "SR.W": "#9400D3",
        "SS.A": "#DB7FF7",
        "SS.W": "#C0C0C0",
        "SU.W": "#228B22",
        "SU.Y": "#BA55D3",
        "SV.A": "#DB7093",
        "SV.W": "#FFA500",
        "SW.Y": "#D8BFD8",
        "TO.A": "#FFFF00",
        "TO.W": "#FF0000",
        "TR.A": "#F08080",
        "TR.S": "#FFE4B5",
        "TR.W": "#B22222",
        "TS.A": "#FF00FF",
        "TS.W": "#FD6347",
        "TS.Y": "#D2691E",
        "TY.A": "#FF00FF",
        "TY.W": "#DC143C",
        "UP.A": "#4682B4",
        "UP.W": "#8B008B",
        "UP.Y": "#8B008B",
        "WC.A": "#5F9EA0",
        "WC.W": "#B0C4DE",
        "WC.Y": "#AFEEEE",
        "WI.Y": "#D2B48C",
        "WS.A": "#4682B4",
        "WS.W": "#FF69B4",
        "WW.Y": "#7B68EE",
        "XH.A": "#800000",  # maroon
        "XH.W": "#C71585",  # medium violet red
        "ZF.Y": "#008080",
        "ZR.Y": "#DA70D6",
    }
    
    def __init__(self, shapefile_path: str):
        """
        Initialize NWS warnings reader
        
        Parameters:
        -----------
        shapefile_path : str
            Path to NWS warnings shapefile
        """
        self.shapefile_path = shapefile_path
        self.gdf = None
        
    def load_shapefile(self):
        """Load the shapefile using geopandas"""
        if not Path(self.shapefile_path).exists():
            raise FileNotFoundError(f"Shapefile not found: {self.shapefile_path}")
        
        self.gdf = gpd.read_file(self.shapefile_path)
        print(f"Loaded {len(self.gdf)} warning polygons from shapefile")
        
        return self.gdf
    
    def filter_warnings(self, warning_type: Optional[str] = None,
                       warning_status: Optional[str] = None,
                       valid_time: Optional[datetime] = None) -> gpd.GeoDataFrame:
        """
        Filter warnings by type, status, and time
        
        Parameters:
        -----------
        warning_type : str, optional
            Warning type code (e.g., 'TO', 'SV', 'FF')
        warning_status : str, optional
            Warning status ('W'=Warning, 'A'=Watch, 'Y'=Advisory, 'S'=Statement)
        valid_time : datetime, optional
            Time to check validity (between ISSUED and EXPIRED)
        
        Returns:
        --------
        GeoDataFrame : Filtered warnings
        """
        if self.gdf is None:
            self.load_shapefile()
        
        filtered = self.gdf.copy()
        
        # Filter by warning type (column name is 'PHENOM')
        if warning_type is not None:
            if 'PHENOM' in filtered.columns:
                filtered = filtered[filtered['PHENOM'] == warning_type]
            elif 'PROD_TYPE' in filtered.columns:
                filtered = filtered[filtered['PROD_TYPE'].str.contains(warning_type, na=False)]
        
        # Filter by warning status (column name is 'SIG')
        if warning_status is not None:
            if 'SIG' in filtered.columns:
                filtered = filtered[filtered['SIG'] == warning_status]
            elif 'GTYPE' in filtered.columns:
                filtered = filtered[filtered['GTYPE'] == warning_status]
        
        # Filter by time validity
        if valid_time is not None:
            valid_time_int = int(valid_time.strftime('%Y%m%d%H%M'))
            
            # Try ISSUED/EXPIRED columns
            if 'ISSUED' in filtered.columns and 'EXPIRED' in filtered.columns:
                # Convert to numeric, coercing errors to NaN
                issued = pd.to_numeric(filtered['ISSUED'], errors='coerce')
                expired = pd.to_numeric(filtered['EXPIRED'], errors='coerce')
                
                # Filter where time is between issued and expired (and both are valid)
                filtered = filtered[
                    (issued <= valid_time_int) &
                    (expired >= valid_time_int) &
                    (issued.notna()) &
                    (expired.notna())
                ]
            # Try INIT_ISS/INIT_EXP columns as alternative
            elif 'INIT_ISS' in filtered.columns and 'INIT_EXP' in filtered.columns:
                init_iss = pd.to_numeric(filtered['INIT_ISS'], errors='coerce')
                init_exp = pd.to_numeric(filtered['INIT_EXP'], errors='coerce')
                
                filtered = filtered[
                    (init_iss <= valid_time_int) &
                    (init_exp >= valid_time_int) &
                    (init_iss.notna()) &
                    (init_exp.notna())
                ]
        
        print(f"Filtered to {len(filtered)} warnings")
        return filtered
    
    def _collect_polygons(self, geometries: List) -> List[np.ndarray]:
        """
        Convert shapefile geometries to polygon coordinate arrays
        
        Parameters:
        -----------
        geometries : list
            List of shapely geometries
        
        Returns:
        --------
        list : List of numpy arrays with polygon coordinates
        """
        polygons = []
        for geom in geometries:
            if geom.geom_type == 'Polygon':
                coords = np.array(geom.exterior.coords)
                polygons.append(coords)
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    coords = np.array(poly.exterior.coords)
                    polygons.append(coords)
        return polygons
    
    def _render_warning_group(self, ax, phenom_sig: str, geometries: List,
                              outline_width: float, fill: bool, alpha: float,
                              transform: ccrs.Projection, zorder_base: int):
        """
        Render a group of warnings with the same PHENOM.SIG combination
        
        This is the core rendering function used by both single and batch plotting.
        Line thickness is automatically adjusted:
        - Warnings (W): outline_width (default 1.0)
        - Non-warnings (A/Y/S/etc): outline_width * 0.5 (default 0.5)
        
        Parameters:
        -----------
        ax : GeoAxes
            Cartopy axes to plot on
        phenom_sig : str
            Warning identifier (e.g., 'TO.W', 'SV.A')
        geometries : list
            List of shapely geometries to plot
        outline_width : float
            Line thickness for warnings (non-warnings get 0.5x this value)
        fill : bool
            Whether to fill polygons
        alpha : float
            Transparency for fills (0-1)
        transform : cartopy projection
            Coordinate projection
        zorder_base : int
            Base z-order (fill will use zorder_base, outline will use zorder_base+1)
        """
        # Get official NWS color
        color = self.NWS_COLORS.get(phenom_sig, '#FF0000')
        
        # Parse phenom and sig for display
        parts = phenom_sig.split('.')
        if len(parts) != 2:
            return
        
        warning_type, warning_status = parts
        type_name = self.VTEC_PHENOMENA.get(warning_type, warning_type)
        status_name = self.VTEC_SIGNIFICANCE.get(warning_status, warning_status)
        
        # Adjust line thickness based on warning status
        # Warnings (W) get full thickness, others get half
        if warning_status == 'W':
            line_width = outline_width
        else:
            line_width = outline_width * 0.5
        
        # Collect polygon coordinates
        polygons = self._collect_polygons(geometries)
        
        if not polygons:
            return
        
        print(f"  Rendering {type_name} {status_name}: "
              f"{len(polygons)} polygons (color: {color}, width: {line_width:.1f}, z: {zorder_base}/{zorder_base+1})")
        
        # Plot filled polygons (if requested)
        if fill:
            collection = PolyCollection(
                polygons,
                facecolors=color,
                alpha=alpha,
                edgecolors='none',
                transform=transform,
                zorder=zorder_base
            )
            ax.add_collection(collection)
        
        # Always plot outlines (solid, no transparency)
        outline_collection = PolyCollection(
            polygons,
            facecolors='none',
            edgecolors=color,
            linewidths=line_width,
            alpha=1.0,
            transform=transform,
            zorder=zorder_base + 1
        )
        ax.add_collection(outline_collection)
    
    def plot_warnings(self, ax, warnings: Optional[gpd.GeoDataFrame] = None,
                     warning_type: Optional[str] = None,
                     warning_status: Optional[str] = None,
                     valid_time: Optional[datetime] = None,
                     outline_width: float = 1.0,
                     fill: bool = False,
                     alpha: float = 0.5,
                     transform: Optional[ccrs.Projection] = None):
        """
        Plot specific warnings on a Cartopy map
        
        Line thickness is automatically adjusted based on warning status:
        - Warnings (W): outline_width (e.g., 1.0)
        - Non-warnings (A/Y/S): outline_width * 0.5 (e.g., 0.5)
        
        Parameters:
        -----------
        ax : GeoAxes
            Cartopy GeoAxes to plot on
        warnings : GeoDataFrame, optional
            Pre-filtered warnings. If None, will filter using other parameters
        warning_type : str, optional
            Warning type to filter and plot (e.g., 'TO', 'SV')
        warning_status : str, optional
            Warning status to filter (e.g., 'W', 'A', 'Y', 'S')
        valid_time : datetime, optional
            Valid time to filter
        outline_width : float
            Line thickness for warnings (non-warnings automatically get 0.5x)
            Default: 1.0 (warnings=1.0, watches/advisories=0.5)
        fill : bool
            Whether to fill polygons (default: False - outline only)
        alpha : float
            Transparency for fills (0-1)
        transform : cartopy projection, optional
            Projection for coordinates (default: PlateCarree)
        """
        if transform is None:
            transform = ccrs.PlateCarree()
        
        # Get warnings to plot
        if warnings is None:
            warnings = self.filter_warnings(warning_type, warning_status, valid_time)
        
        if len(warnings) == 0:
            print(f"No warnings to plot")
            return
        
        # Group by PHENOM.SIG
        warning_groups = {}
        for idx, row in warnings.iterrows():
            phenom = row.get('PHENOM', None)
            sig = row.get('SIG', None)
            
            if phenom and sig:
                key = f"{phenom}.{sig}"
                if key not in warning_groups:
                    warning_groups[key] = []
                warning_groups[key].append(row.geometry)
        
        # Determine z-order based on whether this is a warning or not
        if warning_status == 'W':
            zorder_base = 3  # Warnings on top
        else:
            zorder_base = 1  # Non-warnings below
        
        # Render each group
        for phenom_sig, geometries in warning_groups.items():
            self._render_warning_group(
                ax, phenom_sig, geometries, outline_width,
                fill, alpha, transform, zorder_base
            )
    
    def plot_all_warnings(self, ax, valid_time: Optional[datetime] = None,
                         outline_width: float = 1.0,
                         fill: bool = False,
                         alpha: float = 0.5,
                         severe_only: bool = False,
                         transform: Optional[ccrs.Projection] = None):
        """
        Plot ALL warnings and watches at the given time with NWS official colors
        
        Uses two-pass rendering strategy (like IDL code):
        - Pass 1: Non-warnings (watches, advisories, statements) with 0.5 line width at z=10/11
        - Pass 2: Warnings only with 1.0 line width at z=12/13 (drawn on top)
        
        Line thickness is automatically adjusted based on warning status:
        - Warnings (W): outline_width (e.g., 1.0)
        - Non-warnings (A/Y/S): outline_width * 0.5 (e.g., 0.5)
        
        Parameters:
        -----------
        ax : GeoAxes
            Cartopy GeoAxes to plot on
        valid_time : datetime, optional
            Valid time to filter (if None, plots all)
        outline_width : float
            Line thickness for warnings (non-warnings automatically get 0.5x)
            Default: 1.0 (warnings=1.0, watches/advisories=0.5)
        fill : bool
            Whether to fill polygons (default: False - outline only)
        alpha : float
            Transparency for fills (0-1)
        severe_only : bool
            If True, only plot tornado and severe thunderstorm warnings
        transform : cartopy projection, optional
            Projection for coordinates (default: PlateCarree)
        """
        if transform is None:
            transform = ccrs.PlateCarree()
        
        # Filter by time only (get all types and statuses)
        warnings = self.filter_warnings(valid_time=valid_time)
        
        if len(warnings) == 0:
            print("No active warnings found")
            return
        
        # Group warnings by PHENOM.SIG combination
        warning_groups = {}
        
        for idx, row in warnings.iterrows():
            phenom = row.get('PHENOM', None)
            sig = row.get('SIG', None)
            
            if phenom and sig:
                # Filter for severe only if requested
                if severe_only:
                    if not ((phenom == 'TO' and sig == 'W') or
                           (phenom == 'SV' and sig == 'W')):
                        continue
                
                key = f"{phenom}.{sig}"
                if key not in warning_groups:
                    warning_groups[key] = []
                warning_groups[key].append(row.geometry)
        
        print(f"Found {len(warning_groups)} different warning types/statuses")
        
        # TWO-PASS RENDERING (like IDL code)
        # This ensures warnings are always drawn on top of watches/advisories
        
        # PASS 1: Plot non-warnings (watches, advisories, statements) at z=10/11
        print("Pass 1: Rendering non-warnings (watches, advisories, statements)...")
        for phenom_sig, geometries in warning_groups.items():
            parts = phenom_sig.split('.')
            if len(parts) == 2:
                _, sig = parts
                if sig != 'W':  # Not a warning
                    self._render_warning_group(
                        ax, phenom_sig, geometries, outline_width,
                        fill, alpha, transform, zorder_base=1
                    )
        
        # PASS 2: Plot warnings only at z=12/13 (on top)
        print("Pass 2: Rendering warnings (on top)...")
        for phenom_sig, geometries in warning_groups.items():
            parts = phenom_sig.split('.')
            if len(parts) == 2:
                _, sig = parts
                if sig == 'W':  # Warning only
                    self._render_warning_group(
                        ax, phenom_sig, geometries, outline_width,
                        fill, alpha, transform, zorder_base=3
                    )


def get_nws_shapefile_path(base_path: str, year: int) -> str:
    """
    Get NWS shapefile path for a given year
    
    Parameters:
    -----------
    base_path : str
        Base directory containing shapefiles
    year : int
        Year
    
    Returns:
    --------
    str : Path to shapefile
    """
    filename = f"wwa_{year}01010000_{year}12312359.shp"
    return str(Path(base_path) / filename)
