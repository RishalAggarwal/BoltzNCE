#!/usr/bin/env python3
"""
Simple test script to verify the W2 calculation script works.
"""

import numpy as np
import os
import sys

# Add current directory to path so we can import our script
sys.path.append('.')

try:
    from calculate_w2_metrics import calc_energy_w2, calc_torsion_w2
    print("✅ Successfully imported W2 calculation functions")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

def test_basic_w2_functions():
    """Test the basic W2 calculation functions with synthetic data."""
    print("\n🧪 Testing basic W2 functions...")
    
    # Test energy W2
    print("Testing energy W2...")
    gen_energies = np.random.normal(-100, 10, 1000)
    ref_energies = np.random.normal(-95, 12, 1000) 
    
    try:
        energy_w2 = calc_energy_w2(gen_energies, ref_energies)
        print(f"✅ Energy W2 calculation successful: {energy_w2:.6f}")
    except Exception as e:
        print(f"❌ Energy W2 calculation failed: {e}")
        return False
    
    # Test torsion W2
    print("Testing torsion W2...")
    gen_angles = np.random.uniform(-np.pi, np.pi, (1000, 2))
    ref_angles = np.random.uniform(-np.pi, np.pi, (1000, 2))
    
    try:
        torsion_w2 = calc_torsion_w2(gen_angles, ref_angles)
        print(f"✅ Torsion W2 calculation successful: {torsion_w2:.6f}")
    except Exception as e:
        print(f"❌ Torsion W2 calculation failed: {e}")
        return False
    
    return True

def create_test_data():
    """Create synthetic test data in the expected format."""
    print("\n📦 Creating test data...")
    
    # Create synthetic generated data
    n_samples = 5000
    samples = np.random.randn(n_samples, 66)  # Alanine dipeptide has 66 coordinates
    energies = np.random.normal(-100, 15, (n_samples, 1))
    log_w = np.random.normal(-50, 10, (n_samples, 1))
    
    np.savez('test_generated_data.npz', 
             samples=samples, 
             energies=energies, 
             log_w=log_w)
    print("✅ Created test_generated_data.npz")
    
    # Create synthetic reference data
    n_ref = 3000
    ref_samples = np.random.randn(n_ref, 22, 3)  # Reference format: (n_samples, 22, 3)
    ref_energies = np.random.normal(-95, 12, n_ref)
    
    np.save('test_reference_samples.npy', ref_samples)
    np.save('test_reference_energies.npy', ref_energies)
    print("✅ Created test reference data files")
    
    return True

if __name__ == "__main__":
    print("🧪 Testing W2 calculation script")
    print("=" * 50)
    
    # Test 1: Basic function imports and calculations
    if not test_basic_w2_functions():
        print("❌ Basic function tests failed")
        sys.exit(1)
    
    # Test 2: Create test data
    if not create_test_data():
        print("❌ Test data creation failed")
        sys.exit(1)
    
    print("\n✅ All tests passed!")
    print("\n📋 You can now test the full script with:")
    print("python calculate_w2_metrics.py \\")
    print("  --generated_data test_generated_data.npz \\")
    print("  --reference_samples test_reference_samples.npy \\")
    print("  --reference_energies test_reference_energies.npy \\")
    print("  --n_bootstrap 3 \\")
    print("  --subsample_size 1000 \\")
    print("  --output test_w2_results.npz")