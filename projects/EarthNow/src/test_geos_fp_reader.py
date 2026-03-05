#!/usr/bin/env python3
"""
Test script for GEOS Forward Processing data reader
"""

import sys
from datetime import datetime
from data_readers import DATA_READERS

def test_reader_registration():
    """Test that reader is properly registered"""
    print("=" * 70)
    print("TEST 1: Reader Registration")
    print("=" * 70)
    
    print(f"Available readers: {list(DATA_READERS.keys())}")
    
    if "geos_forward_processing" in DATA_READERS:
        print("✓ geos_forward_processing reader is registered")
    else:
        print("✗ geos_forward_processing reader NOT registered")
        return False
    
    return True


def test_reader_initialization():
    """Test reader initialization"""
    print("\n" + "=" * 70)
    print("TEST 2: Reader Initialization")
    print("=" * 70)
    
    try:
        ReaderClass = DATA_READERS["geos_forward_processing"]
        reader = ReaderClass(
            base_path="/discover/nobackup/projects/gmao/gmao_ops/pub",
            exp_id="f5295_fp"
        )
        print("✓ Reader initialized successfully")
        return reader
    except Exception as e:
        print(f"✗ Reader initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_find_available_times(reader):
    """Test finding available forecast times"""
    print("\n" + "=" * 70)
    print("TEST 3: Find Available Times")
    print("=" * 70)
    
    fdate = "20260122_00z"
    
    try:
        # Test instantaneous (30-minute)
        print(f"\nSearching for instantaneous times (fdate={fdate})...")
        times_inst = reader.find_available_times(fdate, var_type='inst')
        print(f"✓ Found {len(times_inst)} instantaneous times")
        print(f"  First 5 times: {[t.strftime('%Y%m%d_%H%M') for t in times_inst[:5]]}")
        print(f"  Last 5 times:  {[t.strftime('%Y%m%d_%H%M') for t in times_inst[-5:]]}")
        
        # Test tavg (hourly)
        print(f"\nSearching for time-averaged times (fdate={fdate})...")
        times_tavg = reader.find_available_times(fdate, var_type='tavg')
        print(f"✓ Found {len(times_tavg)} tavg times")
        print(f"  First 5 times: {[t.strftime('%Y%m%d_%H%M') for t in times_tavg[:5]]}")
        
        # Test pressure level (3-hourly)
        print(f"\nSearching for pressure level times (fdate={fdate})...")
        times_pres = reader.find_available_times(fdate, var_type='pres')
        print(f"✓ Found {len(times_pres)} pressure level times")
        print(f"  First 5 times: {[t.strftime('%Y%m%d_%H%M') for t in times_pres[:5]]}")
        
        return times_inst[0] if times_inst else None
        
    except Exception as e:
        print(f"✗ Finding available times failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_get_file_path(reader):
    """Test file path resolution"""
    print("\n" + "=" * 70)
    print("TEST 4: File Path Resolution")
    print("=" * 70)
    
    fdate = "20260122_00z"
    pdate = "20260122_0030z"
    
    try:
        path = reader.get_file_path(fdate, pdate, var_type='inst')
        print(f"✓ Resolved file path:")
        print(f"  {path}")
        
        import os
        if os.path.exists(path):
            print(f"  ✓ File exists")
            file_size = os.path.getsize(path) / (1024**2)  # MB
            print(f"  File size: {file_size:.1f} MB")
        else:
            print(f"  ✗ File does not exist!")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ File path resolution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_available_variables(reader):
    """Test listing available variables"""
    print("\n" + "=" * 70)
    print("TEST 5: Available Variables")
    print("=" * 70)
    
    fdate = "20260122_00z"
    pdate = "20260122_0030z"
    
    try:
        # Test instantaneous collection
        print("\nInstantaneous collection variables:")
        vars_inst = reader.get_available_variables(fdate, pdate, var_type='inst')
        print(f"✓ Found {len(vars_inst)} variables")
        print(f"  First 10: {vars_inst[:10]}")
        
        # Test tavg collection
        print("\nTime-averaged collection variables:")
        vars_tavg = reader.get_available_variables(fdate, pdate, var_type='tavg')
        print(f"✓ Found {len(vars_tavg)} variables")
        print(f"  First 10: {vars_tavg[:10]}")
        
        return vars_inst
        
    except Exception as e:
        print(f"✗ Getting available variables failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_read_2d_variable(reader):
    """Test reading 2D surface variable"""
    print("\n" + "=" * 70)
    print("TEST 6: Read 2D Variable")
    print("=" * 70)
    
    fdate = "20260122_00z"
    pdate = "20260122_0030z"
    
    # Try common 2D variables
    test_vars = ["T2M", "U10M", "V10M", "PS", "SLP"]
    
    for varname in test_vars:
        try:
            print(f"\nTrying to read {varname}...")
            data, lats, lons, meta = reader.read_variable(
                fdate=fdate,
                pdate=pdate,
                variables=varname,
                var_type='inst'
            )
            
            print(f"✓ Successfully read {varname}")
            print(f"  Data shape: {data.shape}")
            print(f"  Lat range:  [{lats.min():.2f}, {lats.max():.2f}]")
            print(f"  Lon range:  [{lons.min():.2f}, {lons.max():.2f}]")
            print(f"  Data range: [{data.min():.4f}, {data.max():.4f}]")
            print(f"  Units:      {meta.get('units', 'N/A')}")
            print(f"  Long name:  {meta.get('long_name', 'N/A')}")
            print(f"  Collection: {meta.get('collection', 'N/A')}")
            print(f"  Grid:       {meta.get('grid', 'N/A')}")
            
            return True
            
        except FileNotFoundError:
            print(f"  Variable {varname} not found, trying next...")
            continue
        except Exception as e:
            print(f"✗ Reading {varname} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"✗ Could not read any test variables: {test_vars}")
    return False


def test_read_3d_variable(reader):
    """Test reading 3D pressure level variable"""
    print("\n" + "=" * 70)
    print("TEST 7: Read 3D Pressure Level Variable")
    print("=" * 70)
    
    fdate = "20260122_00z"
    pdate = "20260122_0300z"  # 3-hourly for pressure levels
    
    try:
        # First, get available pressure levels
        print("\nGetting available pressure levels...")
        levels = reader.get_pressure_levels(fdate, pdate)
        if levels is not None:
            print(f"✓ Found {len(levels)} pressure levels")
            print(f"  Levels: {levels}")
        else:
            print("✗ No pressure levels found")
            return False
        
        # Read temperature at 500 hPa
        print(f"\nReading temperature at 500 hPa...")
        data, lats, lons, meta = reader.read_variable(
            fdate=fdate,
            pdate=pdate,
            variables="T",
            var_type='pres',
            level=500.0
        )
        
        print(f"✓ Successfully read T at 500 hPa")
        print(f"  Data shape: {data.shape}")
        print(f"  Data range: [{data.min():.2f}, {data.max():.2f}] {meta.get('units', '')}")
        print(f"  Extracted level: {meta.get('extracted_level', 'N/A')} hPa")
        print(f"  Collection: {meta.get('collection', 'N/A')}")
        
        # Read full 3D temperature
        print(f"\nReading full 3D temperature field...")
        data_3d, lats_3d, lons_3d, meta_3d = reader.read_variable(
            fdate=fdate,
            pdate=pdate,
            variables="T",
            var_type='pres',
            level=None  # Get all levels
        )
        
        print(f"✓ Successfully read 3D T")
        print(f"  Data shape: {data_3d.shape} (lev x lat x lon)")
        print(f"  Is 3D: {meta_3d.get('is_3d', False)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Reading 3D variable failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_read_multiple_variables(reader):
    """Test reading multiple variables at once"""
    print("\n" + "=" * 70)
    print("TEST 8: Read Multiple Variables")
    print("=" * 70)
    
    fdate = "20260122_00z"
    pdate = "20260122_0030z"
    
    try:
        var_list = ["U10M", "V10M", "T2M"]
        
        print(f"\nReading multiple variables: {var_list}")
        results = reader.read_multiple_variables(
            fdate=fdate,
            pdate=pdate,
            variables=var_list,
            var_type='inst'
        )
        
        print(f"✓ Successfully read {len(results)} variables")
        
        for varname, (data, lats, lons, meta) in results.items():
            print(f"\n  {varname}:")
            print(f"    Shape: {data.shape}")
            print(f"    Range: [{data.min():.4f}, {data.max():.4f}] {meta.get('units', '')}")
            print(f"    Long name: {meta.get('long_name', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Reading multiple variables failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_list_available_cycles(reader):
    """Test listing available forecast cycles"""
    print("\n" + "=" * 70)
    print("TEST 9: List Available Cycles")
    print("=" * 70)
    
    try:
        print("\nSearching for cycles from 2026-01-20 to 2026-01-23...")
        cycles = reader.list_available_cycles(
            start_date="2026-01-20",
            end_date="2026-01-23"
        )
        
        print(f"✓ Found {len(cycles)} forecast cycles")
        if cycles:
            print(f"  First cycle: {cycles[0].strftime('%Y-%m-%d %H:%M')}")
            print(f"  Last cycle:  {cycles[-1].strftime('%Y-%m-%d %H:%M')}")
            print(f"  All cycles: {[c.strftime('%Y%m%d_%Hz') for c in cycles]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Listing cycles failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("GEOS Forward Processing Reader Test Suite")
    print("=" * 70)
    
    # Test 1: Registration
    if not test_reader_registration():
        print("\n✗ FATAL: Reader not registered. Exiting.")
        sys.exit(1)
    
    # Test 2: Initialization
    reader = test_reader_initialization()
    if reader is None:
        print("\n✗ FATAL: Reader initialization failed. Exiting.")
        sys.exit(1)
    
    # Test 3: Find times
    first_time = test_find_available_times(reader)
    
    # Test 4: File path
    test_get_file_path(reader)
    
    # Test 5: Available variables
    test_get_available_variables(reader)
    
    # Test 6: Read 2D variable
    test_read_2d_variable(reader)
    
    # Test 7: Read 3D variable
    test_read_3d_variable(reader)
    
    # Test 8: Multiple variables
    test_read_multiple_variables(reader)
    
    # Test 9: List cycles
    test_list_available_cycles(reader)
    
    print("\n" + "=" * 70)
    print("Test suite complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
