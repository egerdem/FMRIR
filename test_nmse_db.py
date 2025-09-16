#!/usr/bin/env python3
"""
Quick test to verify NMSE calculation is now properly in dB.
"""

import torch
import numpy as np

def test_nmse_db_conversion():
    """Test NMSE calculation in dB with synthetic data."""
    
    # Create synthetic ground truth data
    torch.manual_seed(42)
    z_true = torch.randn(1, 20, 11, 11, 11)  # Similar shape to your model
    z_true_var = torch.var(z_true).item()
    
    # Test case 1: Perfect prediction (NMSE should be -inf dB)
    z_est_perfect = z_true.clone()
    mse_perfect = torch.mean((z_est_perfect - z_true) ** 2).item()
    nmse_linear_perfect = mse_perfect / z_true_var if z_true_var > 0 else float('inf')
    nmse_db_perfect = 10 * np.log10(nmse_linear_perfect) if nmse_linear_perfect > 0 and nmse_linear_perfect != float('inf') else float('inf')
    
    print("Perfect prediction:")
    print("  MSE: {:.6f}".format(mse_perfect))
    print("  Ground truth variance: {:.6f}".format(z_true_var))
    print("  NMSE (linear): {:.6f}".format(nmse_linear_perfect))
    print("  NMSE (dB): {:.2f}".format(nmse_db_perfect))
    print()
    
    # Test case 2: Noisy prediction (NMSE should be negative dB)
    noise = torch.randn_like(z_true) * 0.1  # Small noise
    z_est_noisy = z_true + noise
    mse_noisy = torch.mean((z_est_noisy - z_true) ** 2).item()
    nmse_linear_noisy = mse_noisy / z_true_var if z_true_var > 0 else float('inf')
    nmse_db_noisy = 10 * np.log10(nmse_linear_noisy) if nmse_linear_noisy > 0 and nmse_linear_noisy != float('inf') else float('inf')
    
    print("Noisy prediction (10% noise):")
    print("  MSE: {:.6f}".format(mse_noisy))
    print("  Ground truth variance: {:.6f}".format(z_true_var))
    print("  NMSE (linear): {:.6f}".format(nmse_linear_noisy))
    print("  NMSE (dB): {:.2f}".format(nmse_db_noisy))
    print()
    
    # Test case 3: Random prediction (NMSE should be around 3 dB for NMSE=2)
    z_est_random = torch.randn_like(z_true)
    mse_random = torch.mean((z_est_random - z_true) ** 2).item()
    nmse_linear_random = mse_random / z_true_var if z_true_var > 0 else float('inf')
    nmse_db_random = 10 * np.log10(nmse_linear_random) if nmse_linear_random > 0 and nmse_linear_random != float('inf') else float('inf')
    
    print("Random prediction:")
    print("  MSE: {:.6f}".format(mse_random))
    print("  Ground truth variance: {:.6f}".format(z_true_var))
    print("  NMSE (linear): {:.6f}".format(nmse_linear_random))
    print("  NMSE (dB): {:.2f}".format(nmse_db_random))
    print()
    
    # Verify expected behavior
    assert nmse_db_perfect == float('-inf') or nmse_db_perfect < -50, "Perfect prediction NMSE should be -inf or very negative dB, got {}".format(nmse_db_perfect)
    assert nmse_db_noisy < 0, "Noisy prediction NMSE should be negative dB, got {}".format(nmse_db_noisy)
    assert nmse_db_random > 0, "Random prediction NMSE should be positive dB, got {}".format(nmse_db_random)
    
    print("All NMSE dB calculations look reasonable!")
    print("- Perfect prediction: {} dB (should be -inf or very negative)".format(nmse_db_perfect))
    print("- Noisy prediction: {:.2f} dB (should be negative)".format(nmse_db_noisy))
    print("- Random prediction: {:.2f} dB (should be positive)".format(nmse_db_random))
    return True

if __name__ == "__main__":
    test_nmse_db_conversion()
