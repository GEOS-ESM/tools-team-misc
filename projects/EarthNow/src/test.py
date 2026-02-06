#!/usr/bin/env python3
"""
Test script to debug thickness calculation
"""

from data_readers import DATA_READERS
import numpy as np

# Initialize reader
reader = DATA_READERS["geos_forward_processing"](exp_id="f5295_fp")

# Test dates
fdate = "20260122_00z"
pdate = "20260122_06z"

print("="*80)
print("TESTING PHIS READ")
print("="*80)
phis_data, phis_lats, phis_lons, phis_meta = reader.read_variable(
    fdate=fdate,
    pdate=pdate,
    variables=["PHIS"],
    var_type='inst'  # Let it auto-detect
)
print(f"\nPHIS Statistics:")
print(f"  Shape: {phis_data.shape}")
print(f"  Range: [{phis_data.min():.2f}, {phis_data.max():.2f}] {phis_meta['units']}")
print(f"  Mean: {phis_data.mean():.2f}")
print(f"  Collection: {phis_meta['collection']}")
print(f"  File: {phis_meta['file']}")

print("\n" + "="*80)
print("TESTING H500 READ")
print("="*80)
h500_data, h500_lats, h500_lons, h500_meta = reader.read_variable(
    fdate=fdate,
    pdate=pdate,
    variables=["H500"],
    var_type='inst'  # Let it auto-detect
)
print(f"\nH500 Statistics:")
print(f"  Shape: {h500_data.shape}")
print(f"  Range: [{h500_data.min():.2f}, {h500_data.max():.2f}] {h500_meta['units']}")
print(f"  Mean: {h500_data.mean():.2f}")
print(f"  Collection: {h500_meta['collection']}")
print(f"  File: {h500_meta['file']}")
if 'extracted_level' in h500_meta:
    print(f"  Extracted level: {h500_meta['extracted_level']} hPa")

print("\n" + "="*80)
print("TESTING H1000 READ")
print("="*80)
h1000_data, h1000_lats, h1000_lons, h1000_meta = reader.read_variable(
    fdate=fdate,
    pdate=pdate,
    variables=["H1000"],
    var_type='inst'  # Let it auto-detect
)
print(f"\nH1000 Statistics:")
print(f"  Shape: {h1000_data.shape}")
print(f"  Range: [{h1000_data.min():.2f}, {h1000_data.max():.2f}] {h1000_meta['units']}")
print(f"  Mean: {h1000_data.mean():.2f}")
print(f"  Collection: {h1000_meta['collection']}")
print(f"  File: {h1000_meta['file']}")
if 'extracted_level' in h1000_meta:
    print(f"  Extracted level: {h1000_meta['extracted_level']} hPa")

print(f"\nLongitude ranges:")
print(f"  PHIS lons: [{phis_lons.min():.2f}, {phis_lons.max():.2f}]")
print(f"  H500 lons: [{h500_lons.min():.2f}, {h500_lons.max():.2f}]")
print(f"  H1000 lons: [{h1000_lons.min():.2f}, {h1000_lons.max():.2f}]")

print("\n" + "="*80)
print("CALCULATING THICKNESS")
print("="*80)

# Convert PHIS (m^2/s^2) to geopotential height (m)
g = 9.80665  # m/s^2
phis_m = phis_data / g

print(f"\nPHIS in meters:")
print(f"  Range: [{phis_m.min():.2f}, {phis_m.max():.2f}] m")
print(f"  Mean: {phis_m.mean():.2f} m")

# Calculate thickness (should be in meters if H is in meters)
thickness = (h500_data - phis_m) - (h1000_data - phis_m)
# Simplifies to:
thickness_simple = h500_data - h1000_data

print(f"\nThickness (H500 - H1000):")
print(f"  Range: [{thickness_simple.min():.2f}, {thickness_simple.max():.2f}] m")
print(f"  Mean: {thickness_simple.mean():.2f} m")
print(f"  Expected range: ~4800-5800 m (reasonable for 1000-500 hPa thickness)")

# Check if units are consistent
print(f"\nUnit check:")
print(f"  PHIS units: {phis_meta['units']}")
print(f"  H500 units: {h500_meta['units']}")
print(f"  H1000 units: {h1000_meta['units']}")

# Sanity checks
print(f"\nSanity checks:")
print(f"  H500 > H1000? {np.all(h500_data > h1000_data)} (should be True)")
print(f"  Thickness > 0? {np.all(thickness_simple > 0)} (should be True)")
print(f"  Typical thickness at mid-latitudes: ~5400 m")
print(f"  Thickness range reasonable? {4800 < thickness_simple.mean() < 5800}")

# Additional check: Look at a specific location (e.g., center of domain)
mid_lat = len(h500_lats) // 2
mid_lon = len(h500_lons) // 2
print(f"\nSample point (center of domain):")
print(f"  Location: lat={h500_lats[mid_lat]:.2f}, lon={h500_lons[mid_lon]:.2f}")
print(f"  PHIS: {phis_m[mid_lat, mid_lon]:.2f} m")
print(f"  H1000: {h1000_data[mid_lat, mid_lon]:.2f} m")
print(f"  H500: {h500_data[mid_lat, mid_lon]:.2f} m")
print(f"  Thickness: {thickness_simple[mid_lat, mid_lon]:.2f} m")
