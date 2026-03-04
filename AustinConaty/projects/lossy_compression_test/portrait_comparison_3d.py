#!/usr/bin/env python3
"""
Script to compare two NetCDF files with identical variables
Creates three-panel plots in PORTRAIT mode for ALL variables and ALL levels
Handles 2D, 3D, and 4D data automatically
"""

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

def read_netcdf_file(filename):
    """
    Read NetCDF file and return dataset
    
    Parameters:
    -----------
    filename : str
        Path to NetCDF file
    
    Returns:
    --------
    dataset : netCDF4.Dataset
        Opened NetCDF dataset
    """
    try:
        dataset = nc.Dataset(filename, 'r')
        print(f"Successfully opened: {filename}")
        print(f"Available variables: {list(dataset.variables.keys())}")
        return dataset
    except Exception as e:
        print(f"Error opening file: {e}")
        return None

def get_plottable_variables(dataset):
    """
    Identify variables that can be plotted as horizontal slices
    
    Parameters:
    -----------
    dataset : netCDF4.Dataset
        Opened NetCDF dataset
    
    Returns:
    --------
    plottable_vars : list
        List of tuples (variable_name, variable_object, info_dict)
    """
    
    # Common coordinate/dimension variable names to skip
    coord_names = ['lat', 'latitude', 'lon', 'longitude', 'time', 't', 
                   'lev', 'level', 'height', 'altitude', 'depth', 'pressure',
                   'x', 'y', 'z', 'Lat', 'Lon', 'Time', 'LAT', 'LON', 'TIME',
                   'LATITUDE', 'LONGITUDE', 'nv', 'bnds', 'bounds']
    
    plottable_vars = []
    
    for var_name in dataset.variables:
        var = dataset.variables[var_name]
        
        # Skip coordinate variables
        if var_name in coord_names:
            continue
        
        # Skip variables with too few or too many dimensions
        if len(var.shape) < 2 or len(var.shape) > 4:
            continue
        
        # Skip 1D variables
        if var.ndim == 1:
            continue
        
        # Get variable metadata
        info = {
            'shape': var.shape,
            'dimensions': var.dimensions,
            'dtype': var.dtype,
            'long_name': getattr(var, 'long_name', var_name),
            'units': getattr(var, 'units', ''),
            'standard_name': getattr(var, 'standard_name', ''),
        }
        
        plottable_vars.append((var_name, var, info))
    
    return plottable_vars

def find_common_variables(dataset1, dataset2):
    """
    Find variables that exist in both datasets
    
    Parameters:
    -----------
    dataset1, dataset2 : netCDF4.Dataset
        Opened NetCDF datasets
    
    Returns:
    --------
    common_vars : list
        List of tuples (variable_name, info_from_dataset1)
    """
    
    vars1 = get_plottable_variables(dataset1)
    vars2 = get_plottable_variables(dataset2)
    
    var_names1 = {v[0] for v in vars1}
    var_names2 = {v[0] for v in vars2}
    
    common_names = var_names1.intersection(var_names2)
    
    common_vars = [(name, info) for name, var, info in vars1 if name in common_names]
    
    return common_vars

def get_dimension_info(dataset, var_name):
    """
    Get information about dimensions including number of time steps and levels
    
    Parameters:
    -----------
    dataset : netCDF4.Dataset
        Opened NetCDF dataset
    var_name : str
        Variable name
    
    Returns:
    --------
    dim_info : dict
        Dictionary with dimension information
    """
    
    var = dataset.variables[var_name]
    dims = var.dimensions
    
    # Common dimension names
    time_dims = ['time', 't', 'Time', 'TIME']
    lat_dims = ['lat', 'latitude', 'Lat', 'Latitude', 'LAT', 'LATITUDE', 'y']
    lon_dims = ['lon', 'longitude', 'Lon', 'Longitude', 'LON', 'LONGITUDE', 'x']
    level_dims = ['lev', 'level', 'z', 'height', 'altitude', 'depth', 'pressure', 'plev']
    
    dim_info = {
        'has_time': False,
        'has_level': False,
        'n_time': 1,
        'n_level': 1,
        'time_dim_name': None,
        'level_dim_name': None,
        'level_values': None
    }
    
    # Check for time dimension
    for dim in dims:
        if dim in time_dims:
            dim_info['has_time'] = True
            dim_info['time_dim_name'] = dim
            dim_info['n_time'] = len(dataset.dimensions[dim])
            break
    
    # Check for level dimension
    for dim in dims:
        if dim in level_dims:
            dim_info['has_level'] = True
            dim_info['level_dim_name'] = dim
            dim_info['n_level'] = len(dataset.dimensions[dim])
            # Get level values if available
            if dim in dataset.variables:
                dim_info['level_values'] = dataset.variables[dim][:]
            break
    
    return dim_info

def extract_horizontal_slice(dataset, variable_name, 
                             time_index=0, level_index=0):
    """
    Extract a 2D horizontal slice from a multi-dimensional variable
    
    Parameters:
    -----------
    dataset : netCDF4.Dataset
        Opened NetCDF dataset
    variable_name : str
        Name of variable to extract
    time_index : int
        Time index to extract (if time dimension exists)
    level_index : int
        Vertical level index to extract (if level dimension exists)
    
    Returns:
    --------
    data : numpy array
        2D array of extracted data
    lon : numpy array
        Longitude values
    lat : numpy array
        Latitude values
    """
    
    var = dataset.variables[variable_name]
    dims = var.dimensions
    
    # Common dimension names
    time_dims = ['time', 't', 'Time', 'TIME']
    lat_dims = ['lat', 'latitude', 'Lat', 'Latitude', 'LAT', 'LATITUDE', 'y']
    lon_dims = ['lon', 'longitude', 'Lon', 'Longitude', 'LON', 'LONGITUDE', 'x']
    level_dims = ['lev', 'level', 'z', 'height', 'altitude', 'depth', 'pressure', 'plev']
    
    # Extract data based on dimensions
    if len(var.shape) == 2:
        # Already 2D
        data = var[:]
    elif len(var.shape) == 3:
        # Could be (time, lat, lon) or (level, lat, lon)
        if any(d in time_dims for d in dims):
            data = var[time_index, :, :]
        elif any(d in level_dims for d in dims):
            data = var[level_index, :, :]
        else:
            data = var[0, :, :]
    elif len(var.shape) == 4:
        # Likely (time, level, lat, lon)
        data = var[time_index, level_index, :, :]
    else:
        raise ValueError(f"Cannot handle variable with {len(var.shape)} dimensions")
    
    # Extract lat/lon
    lon, lat = None, None
    
    for var_name in dataset.variables:
        if var_name.lower() in lon_dims:
            lon = dataset.variables[var_name][:]
        if var_name.lower() in lat_dims:
            lat = dataset.variables[var_name][:]
    
    # Handle 1D lat/lon (create 2D mesh if needed)
    if lon is not None and lat is not None:
        if lon.ndim == 1 and lat.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)
    
    return data, lon, lat

def get_appropriate_colormap(var_name, standard_name, long_name, is_difference=False):
    """
    Select an appropriate colormap based on variable type
    
    Parameters:
    -----------
    var_name : str
        Variable name
    standard_name : str
        Standard name attribute
    long_name : str
        Long name attribute
    is_difference : bool
        If True, return a diverging colormap for difference plots
    
    Returns:
    --------
    cmap : str
        Colormap name
    """
    
    if is_difference:
        # Use diverging colormaps for differences
        return 'RdBu_r'
    
    var_lower = f"{var_name} {standard_name} {long_name}".lower()
    
    # Temperature variables
    if any(word in var_lower for word in ['temp', 'sst', 'temperature']):
        return 'RdYlBu_r'
    
    # Precipitation/humidity
    elif any(word in var_lower for word in ['precip', 'rain', 'humidity', 'moisture']):
        return 'Blues'
    
    # Wind/velocity
    elif any(word in var_lower for word in ['wind', 'velocity', 'speed']):
        return 'YlOrRd'
    
    # Pressure
    elif any(word in var_lower for word in ['pressure', 'slp', 'mslp']):
        return 'RdPu'
    
    # Chlorophyll/vegetation
    elif any(word in var_lower for word in ['chlor', 'ndvi', 'vegetation', 'leaf']):
        return 'Greens'
    
    # Ocean color/sediment
    elif any(word in var_lower for word in ['sediment', 'turbidity']):
        return 'YlOrBr'
    
    # Anomalies (diverging)
    elif any(word in var_lower for word in ['anomaly', 'anomalies', 'difference']):
        return 'RdBu_r'
    
    # Default
    else:
        return 'viridis'

def format_level_info(level_value, level_dim_name, units=''):
    """
    Format level information for display
    
    Parameters:
    -----------
    level_value : float or int
        Level value
    level_dim_name : str
        Name of level dimension
    units : str
        Units of level
    
    Returns:
    --------
    level_str : str
        Formatted level string
    """
    
    if level_value is None:
        return ""
    
    # Determine if it's pressure, height, etc.
    if 'pressure' in level_dim_name.lower() or 'plev' in level_dim_name.lower():
        # Pressure levels - convert Pa to hPa if needed
        if level_value > 10000:  # Likely in Pa
            return f"Level {int(level_value/100)} hPa"
        else:
            return f"Level {int(level_value)} hPa"
    elif 'height' in level_dim_name.lower() or 'altitude' in level_dim_name.lower():
        return f"Height {level_value:.1f} m"
    elif 'depth' in level_dim_name.lower():
        return f"Depth {level_value:.1f} m"
    else:
        return f"Level {level_value}"

def plot_three_panel_comparison(data1, data2, lon=None, lat=None,
                                var_name="Variable",
                                title1="File 1", title2="File 2",
                                units="",
                                level_info="",
                                time_info="",
                                use_cartopy=True,
                                output_file=None,
                                symmetric_diff=True):
    """
    Create a three-panel comparison plot in PORTRAIT orientation (vertical)
    
    Parameters:
    -----------
    data1, data2 : numpy array
        2D arrays of data to plot and compare
    lon, lat : numpy array, optional
        Longitude and latitude values
    var_name : str
        Variable name
    title1, title2 : str
        Titles for first and second panels
    units : str
        Units for colorbar
    level_info : str
        Information about vertical level
    time_info : str
        Information about time step
    use_cartopy : bool
        Whether to use Cartopy for geographic projection
    output_file : str, optional
        If provided, save plot to this filename
    symmetric_diff : bool
        If True, use symmetric color scale for difference plot
    """
    
    # Calculate difference
    diff = data1 - data2
    
    # Mask invalid values
    data1 = np.ma.masked_invalid(data1)
    data2 = np.ma.masked_invalid(data2)
    diff = np.ma.masked_invalid(diff)
    
    # Get statistics
    valid_data1 = data1.compressed() if hasattr(data1, 'compressed') else data1[~np.isnan(data1)]
    valid_data2 = data2.compressed() if hasattr(data2, 'compressed') else data2[~np.isnan(data2)]
    valid_diff = diff.compressed() if hasattr(diff, 'compressed') else diff[~np.isnan(diff)]
    
    # Check for valid data
    if len(valid_data1) == 0 or len(valid_data2) == 0:
        print(f"  WARNING: No valid data, skipping plot...")
        return
    
    # Determine colormaps
    cmap_data = get_appropriate_colormap(var_name, "", "", is_difference=False)
    cmap_diff = 'RdBu_r'  # Always use diverging colormap for differences
    
    # Determine color limits
    # For data plots, use same scale for both files
    vmin_data = min(np.min(valid_data1), np.min(valid_data2))
    vmax_data = max(np.max(valid_data1), np.max(valid_data2))
    
    # For difference plot, make symmetric around zero
    if symmetric_diff and len(valid_diff) > 0:
        vmax_diff = max(abs(np.min(valid_diff)), abs(np.max(valid_diff)))
        vmin_diff = -vmax_diff
    else:
        vmin_diff = np.min(valid_diff) if len(valid_diff) > 0 else 0
        vmax_diff = np.max(valid_diff) if len(valid_diff) > 0 else 0
    
    # Create figure in portrait orientation (3 rows, 1 column)
    if use_cartopy and lon is not None and lat is not None:
        try:
            fig = plt.figure(figsize=(10, 16))
            
            # Panel 1: File 1 (top)
            ax1 = plt.subplot(3, 1, 1, projection=ccrs.PlateCarree())
            setup_cartopy_axis(ax1)
            im1 = ax1.pcolormesh(lon, lat, data1, 
                                transform=ccrs.PlateCarree(),
                                cmap=cmap_data, vmin=vmin_data, vmax=vmax_data,
                                shading='auto')
            
            panel_title1 = title1
            if level_info:
                panel_title1 += f" - {level_info}"
            if time_info:
                panel_title1 += f" - {time_info}"
            ax1.set_title(panel_title1, fontsize=12, fontweight='bold', pad=10)
            
            # Add colorbar for panel 1
            cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', 
                                pad=0.05, shrink=0.8)
            cbar1_label = f"{var_name}"
            if units:
                cbar1_label += f" ({units})"
            cbar1.set_label(cbar1_label, fontsize=10)
            
            # Add statistics text
            stats1 = f"min={np.min(valid_data1):.3e}, max={np.max(valid_data1):.3e}, mean={np.mean(valid_data1):.3e}"
            ax1.text(0.02, 0.98, stats1, transform=ax1.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Panel 2: File 2 (middle)
            ax2 = plt.subplot(3, 1, 2, projection=ccrs.PlateCarree())
            setup_cartopy_axis(ax2)
            im2 = ax2.pcolormesh(lon, lat, data2, 
                                transform=ccrs.PlateCarree(),
                                cmap=cmap_data, vmin=vmin_data, vmax=vmax_data,
                                shading='auto')
            
            panel_title2 = title2
            if level_info:
                panel_title2 += f" - {level_info}"
            if time_info:
                panel_title2 += f" - {time_info}"
            ax2.set_title(panel_title2, fontsize=12, fontweight='bold', pad=10)
            
            # Add colorbar for panel 2
            cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', 
                                pad=0.05, shrink=0.8)
            cbar2.set_label(cbar1_label, fontsize=10)
            
            # Add statistics text
            stats2 = f"min={np.min(valid_data2):.3e}, max={np.max(valid_data2):.3e}, mean={np.mean(valid_data2):.3e}"
            ax2.text(0.02, 0.98, stats2, transform=ax2.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Panel 3: Difference (bottom)
            ax3 = plt.subplot(3, 1, 3, projection=ccrs.PlateCarree())
            setup_cartopy_axis(ax3)
            im3 = ax3.pcolormesh(lon, lat, diff, 
                                transform=ccrs.PlateCarree(),
                                cmap=cmap_diff, vmin=vmin_diff, vmax=vmax_diff,
                                shading='auto')
            
            panel_title3 = 'Difference (File 1 - File 2)'
            if level_info:
                panel_title3 += f" - {level_info}"
            if time_info:
                panel_title3 += f" - {time_info}"
            ax3.set_title(panel_title3, fontsize=12, fontweight='bold', pad=10)
            
            # Add colorbar for difference panel
            cbar3 = plt.colorbar(im3, ax=ax3, orientation='horizontal', 
                                pad=0.05, shrink=0.8)
            cbar3.set_label(f"Difference ({units})" if units else "Difference", fontsize=10)
            
            # Add difference statistics text
            if len(valid_diff) > 0:
                rmse = np.sqrt(np.mean(valid_diff**2))
                stats3 = f"min={np.min(valid_diff):.3e}, max={np.max(valid_diff):.3e}, mean={np.mean(valid_diff):.3e}, rmse={rmse:.3e}"
            else:
                stats3 = "No valid data"
            ax3.text(0.02, 0.98, stats3, transform=ax3.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            axes = [ax1, ax2, ax3]
            
        except Exception as e:
            print(f"Warning: Cartopy plotting failed ({e}), falling back to simple plot")
            use_cartopy = False
    
    if not use_cartopy:
        # Simple plot without projection
        fig, axes = plt.subplots(3, 1, figsize=(10, 16))
        
        if lon is not None and lat is not None:
            im1 = axes[0].pcolormesh(lon, lat, data1, cmap=cmap_data, 
                                     vmin=vmin_data, vmax=vmax_data, shading='auto')
            im2 = axes[1].pcolormesh(lon, lat, data2, cmap=cmap_data, 
                                     vmin=vmin_data, vmax=vmax_data, shading='auto')
            im3 = axes[2].pcolormesh(lon, lat, diff, cmap=cmap_diff, 
                                     vmin=vmin_diff, vmax=vmax_diff, shading='auto')
            for ax in axes:
                ax.set_xlabel('Longitude', fontsize=10)
                ax.set_ylabel('Latitude', fontsize=10)
        else:
            im1 = axes[0].imshow(data1, cmap=cmap_data, vmin=vmin_data, vmax=vmax_data,
                                origin='lower', aspect='auto')
            im2 = axes[1].imshow(data2, cmap=cmap_data, vmin=vmin_data, vmax=vmax_data,
                                origin='lower', aspect='auto')
            im3 = axes[2].imshow(diff, cmap=cmap_diff, vmin=vmin_diff, vmax=vmax_diff,
                                origin='lower', aspect='auto')
            for ax in axes:
                ax.set_xlabel('X Index', fontsize=10)
                ax.set_ylabel('Y Index', fontsize=10)
        
        panel_title1 = title1
        if level_info:
            panel_title1 += f" - {level_info}"
        if time_info:
            panel_title1 += f" - {time_info}"
        axes[0].set_title(panel_title1, fontsize=12, fontweight='bold', pad=10)
        
        panel_title2 = title2
        if level_info:
            panel_title2 += f" - {level_info}"
        if time_info:
            panel_title2 += f" - {time_info}"
        axes[1].set_title(panel_title2, fontsize=12, fontweight='bold', pad=10)
        
        panel_title3 = 'Difference (File 1 - File 2)'
        if level_info:
            panel_title3 += f" - {level_info}"
        if time_info:
            panel_title3 += f" - {time_info}"
        axes[2].set_title(panel_title3, fontsize=12, fontweight='bold', pad=10)
        
        # Add colorbars
        cbar1 = plt.colorbar(im1, ax=axes[0], orientation='horizontal', 
                            pad=0.05, shrink=0.8)
        cbar1_label = f"{var_name}"
        if units:
            cbar1_label += f" ({units})"
        cbar1.set_label(cbar1_label, fontsize=10)
        
        cbar2 = plt.colorbar(im2, ax=axes[1], orientation='horizontal', 
                            pad=0.05, shrink=0.8)
        cbar2.set_label(cbar1_label, fontsize=10)
        
        cbar3 = plt.colorbar(im3, ax=axes[2], orientation='horizontal', 
                            pad=0.05, shrink=0.8)
        cbar3.set_label(f"Difference ({units})" if units else "Difference", fontsize=10)
        
        # Add statistics text boxes
        stats1 = f"min={np.min(valid_data1):.3e}, max={np.max(valid_data1):.3e}, mean={np.mean(valid_data1):.3e}"
        axes[0].text(0.02, 0.98, stats1, transform=axes[0].transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        stats2 = f"min={np.min(valid_data2):.3e}, max={np.max(valid_data2):.3e}, mean={np.mean(valid_data2):.3e}"
        axes[1].text(0.02, 0.98, stats2, transform=axes[1].transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if len(valid_diff) > 0:
            rmse = np.sqrt(np.mean(valid_diff**2))
            stats3 = f"min={np.min(valid_diff):.3e}, max={np.max(valid_diff):.3e}, mean={np.mean(valid_diff):.3e}, rmse={rmse:.3e}"
        else:
            stats3 = "No valid data"
        axes[2].text(0.02, 0.98, stats3, transform=axes[2].transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add super title
    fig.suptitle(f"{var_name}", fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"    Plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()

def setup_cartopy_axis(ax):
    """
    Setup a Cartopy axis with standard features
    
    Parameters:
    -----------
    ax : cartopy axis
        Axis to setup
    """
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAND, alpha=0.1)
    ax.add_feature(cfeature.OCEAN, alpha=0.1)
    
    gl = ax.gridlines(draw_labels=True, alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

def compare_two_files(dataset1, dataset2, 
                     file1_name="File 1", file2_name="File 2",
                     output_dir="comparison_plots",
                     time_index=0, level_index=None,
                     use_cartopy=True, show_plots=False,
                     symmetric_diff=True,
                     plot_all_levels=True,
                     plot_all_times=False):
    """
    Compare all common variables between two NetCDF files
    Handles 2D, 3D (with levels), and 4D (with time and levels) data
    
    Parameters:
    -----------
    dataset1, dataset2 : netCDF4.Dataset
        Opened NetCDF datasets to compare
    file1_name, file2_name : str
        Names/descriptions for the files
    output_dir : str
        Directory to save output plots
    time_index : int
        Time step to plot (if plot_all_times=False)
    level_index : int or None
        Vertical level to plot (if plot_all_levels=False). None means all levels.
    use_cartopy : bool
        Whether to use Cartopy for geographic projection
    show_plots : bool
        Whether to display plots interactively
    symmetric_diff : bool
        If True, use symmetric color scale for difference plots
    plot_all_levels : bool
        If True, plot all vertical levels for 3D/4D variables
    plot_all_times : bool
        If True, plot all time steps (can generate many plots!)
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Find common variables
    common_vars = find_common_variables(dataset1, dataset2)
    
    if not common_vars:
        print("No common plottable variables found between the two files!")
        return
    
    print(f"\nFound {len(common_vars)} common variable(s) to compare")
    print("="*70)
    
    total_plots = 0
    
    # Compare each variable
    for idx, (var_name, info) in enumerate(common_vars, 1):
        print(f"\n[{idx}/{len(common_vars)}] Comparing: {var_name}")
        print(f"  Long name: {info['long_name']}")
        print(f"  Units: {info['units']}")
        print(f"  Shape: {info['shape']}")
        print(f"  Dimensions: {info['dimensions']}")
        
        try:
            # Get dimension info from both datasets
            dim_info1 = get_dimension_info(dataset1, var_name)
            dim_info2 = get_dimension_info(dataset2, var_name)
            
            # Determine time indices to plot
            if plot_all_times and dim_info1['has_time']:
                time_indices = range(dim_info1['n_time'])
                print(f"  Will plot all {dim_info1['n_time']} time step(s)")
            else:
                time_indices = [time_index] if dim_info1['has_time'] else [0]
                if dim_info1['has_time'] and time_index < dim_info1['n_time']:
                    print(f"  Plotting time index {time_index} of {dim_info1['n_time']}")
                elif dim_info1['has_time']:
                    print(f"  WARNING: time_index {time_index} out of range (max: {dim_info1['n_time']-1}), using 0")
                    time_indices = [0]
            
            # Determine level indices to plot
            if plot_all_levels and dim_info1['has_level']:
                level_indices = range(dim_info1['n_level'])
                print(f"  Will plot all {dim_info1['n_level']} level(s)")
            elif level_index is not None and dim_info1['has_level']:
                level_indices = [level_index]
                print(f"  Plotting level index {level_index} of {dim_info1['n_level']}")
            elif dim_info1['has_level']:
                level_indices = range(dim_info1['n_level'])
                print(f"  plot_all_levels=True, will plot all {dim_info1['n_level']} level(s)")
            else:
                level_indices = [0]
            
            # Loop through time and level indices
            for t_idx in time_indices:
                for l_idx in level_indices:
                    
                    # Extract data from both files
                    data1, lon1, lat1 = extract_horizontal_slice(dataset1, var_name, t_idx, l_idx)
                    data2, lon2, lat2 = extract_horizontal_slice(dataset2, var_name, t_idx, l_idx)
                    
                    # Check if shapes match
                    if data1.shape != data2.shape:
                        print(f"  WARNING: Shape mismatch at time={t_idx}, level={l_idx}!")
                        print(f"           File 1: {data1.shape}, File 2: {data2.shape}")
                        print(f"  Skipping this slice...")
                        continue
                    
                    # Use lon/lat from first file (assuming they're the same)
                    lon, lat = lon1, lat1
                    
                    # Convert to numpy arrays and mask invalid values
                    data1 = np.ma.masked_invalid(np.array(data1))
                    data2 = np.ma.masked_invalid(np.array(data2))
                    
                    # Check for valid data
                    valid_data1 = data1.compressed() if hasattr(data1, 'compressed') else data1[~np.isnan(data1)]
                    valid_data2 = data2.compressed() if hasattr(data2, 'compressed') else data2[~np.isnan(data2)]
                    
                    if len(valid_data1) == 0 or len(valid_data2) == 0:
                        print(f"  WARNING: No valid data at time={t_idx}, level={l_idx}, skipping...")
                        continue
                    
                    # Create level info string
                    level_info = ""
                    if dim_info1['has_level']:
                        if dim_info1['level_values'] is not None:
                            level_val = dim_info1['level_values'][l_idx]
                            level_info = format_level_info(level_val, dim_info1['level_dim_name'])
                        else:
                            level_info = f"Level Index {l_idx}"
                    
                    # Create time info string
                    time_info = ""
                    if dim_info1['has_time'] and len(time_indices) > 1:
                        time_info = f"Time Index {t_idx}"
                    
                    # Create output filename
                    output_filename = f"{var_name}"
                    if dim_info1['has_time'] and len(time_indices) > 1:
                        output_filename += f"_t{t_idx:03d}"
                    if dim_info1['has_level']:
                        output_filename += f"_lev{l_idx:03d}"
                    output_filename += "_comparison.png"
                    
                    output_file = os.path.join(output_dir, output_filename)
                    
                    # Progress indicator
                    if dim_info1['has_level'] or len(time_indices) > 1:
                        level_str = f"level {l_idx+1}/{dim_info1['n_level']}" if dim_info1['has_level'] else ""
                        time_str = f"time {t_idx+1}/{len(time_indices)}" if len(time_indices) > 1 else ""
                        progress_str = ", ".join(filter(None, [time_str, level_str]))
                        print(f"    Processing {progress_str}...")
                    
                    # Create three-panel plot
                    plot_three_panel_comparison(data1, data2, lon, lat,
                                               var_name=var_name,
                                               title1=file1_name,
                                               title2=file2_name,
                                               units=info['units'],
                                               level_info=level_info,
                                               time_info=time_info,
                                               use_cartopy=use_cartopy,
                                               output_file=output_file,
                                               symmetric_diff=symmetric_diff)
                    
                    total_plots += 1
                    
                    # Optionally show plot interactively
                    if show_plots:
                        plot_three_panel_comparison(data1, data2, lon, lat,
                                                   var_name=var_name,
                                                   title1=file1_name,
                                                   title2=file2_name,
                                                   units=info['units'],
                                                   level_info=level_info,
                                                   time_info=time_info,
                                                   use_cartopy=use_cartopy,
                                                   output_file=None,
                                                   symmetric_diff=symmetric_diff)
            
            print(f"  ✓ Successfully compared {var_name}")
            
        except Exception as e:
            print(f"  ✗ Error comparing {var_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*70)
    print(f"Comparison complete!")
    print(f"Total plots generated: {total_plots}")
    print(f"Check '{output_dir}' directory for output files.")

def main():
    """
    Main function - compare two NetCDF files with support for 3D/4D data
    """
    
    # ============== CONFIGURATION ==============
    
    # Input files
#    netcdf_file1 = "GEOS.fp.fcst.inst3_3d_chm_Np.20251202_00+20251210_0600.V01.nc4"      # Path to first NetCDF file
#    netcdf_file2 = "GEOS.fp.fcst.inst3_3d_chm_Np.20251202_00+20251210_0600.compress.nc4"      # Path to second NetCDF file
    netcdf_file1 = "GEOS.fp.fcst.tavg3_3d_asm_Nv.20251202_00+20251211_1030.V01.nc4"      # Path to first NetCDF file
    netcdf_file2 = "GEOS.fp.fcst.tavg3_3d_asm_Nv.20251202_00+20251211_1030.compress.nc4"      # Path to second NetCDF file
    
    # Optional: Short names for the files (used in plot titles)
    file1_name = "Normal"          # e.g., "Model Run 1", "Observation", etc.
    file2_name = "Compressed"          # e.g., "Model Run 2", "Forecast", etc.
    
    # Output settings
    output_dir = "comparison_plots_3d"  # Directory to save plots
    
    # Data extraction settings
    time_index = 0                 # Time step (0 = first) - used if plot_all_times=False
    level_index = None             # Vertical level (None = all levels) - used if plot_all_levels=False
    
    # Control what to plot
    plot_all_levels = True         # Plot all vertical levels for 3D/4D variables
    plot_all_times = False         # Plot all time steps (WARNING: can create many files!)
    
    # Plot settings
    use_cartopy = True             # Set to False if cartopy not available
    show_plots = False             # Set to True to also display plots interactively
    symmetric_diff = True          # Use symmetric color scale for difference plots
    
    # ===========================================
    
    print("="*70)
    print("NetCDF File Comparison Tool - 3D/4D Support")
    print("="*70)
    
    # Read both NetCDF files
    print("\nOpening first file...")
    dataset1 = read_netcdf_file(netcdf_file1)
    
    print("\nOpening second file...")
    dataset2 = read_netcdf_file(netcdf_file2)
    
    if dataset1 is None or dataset2 is None:
        print("Failed to open one or both datasets. Exiting.")
        return
    
    # Compare files
    try:
        compare_two_files(dataset1, dataset2,
                         file1_name=file1_name,
                         file2_name=file2_name,
                         output_dir=output_dir,
                         time_index=time_index,
                         level_index=level_index,
                         use_cartopy=use_cartopy,
                         show_plots=show_plots,
                         symmetric_diff=symmetric_diff,
                         plot_all_levels=plot_all_levels,
                         plot_all_times=plot_all_times)
    except Exception as e:
        print(f"Error in comparison process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        dataset1.close()
        dataset2.close()
        print("\nDatasets closed.")

if __name__ == "__main__":
    main()
