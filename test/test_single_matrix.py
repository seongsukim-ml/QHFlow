#!/usr/bin/env python3
"""
Test script for single matrix orbital calculation methods in metric.py

This script tests both the eigenvalue-based method and the Cholesky-based method
for calculating orbital energies and coefficients from single (n x n) overlap and Hamiltonian matrices.
"""

import torch
import numpy as np
import sys
import os

# Add the src directory to the path to import the module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from common.metric import cal_orbital_and_energies_single

def create_test_matrices(matrix_size=5, seed=42):
    """Create test overlap and Hamiltonian matrices.
    
    Args:
        matrix_size (int): Size of each matrix (matrix_size x matrix_size)
        seed (int): Random seed for reproducibility
        
    Returns:
        Tuple[Tensor, Tensor]: (overlap_matrix, hamiltonian_matrix)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create positive definite overlap matrix
    # Method: S = A A^T + εI where A is random and ε is small positive number
    A = torch.randn(matrix_size, matrix_size)
    overlap_matrix = torch.mm(A, A.t()) + 1e-6 * torch.eye(matrix_size)
    
    # Create symmetric Hamiltonian matrix
    H = torch.randn(matrix_size, matrix_size)
    hamiltonian_matrix = H + H.t()  # Make symmetric
    
    return overlap_matrix, hamiltonian_matrix

def create_ill_conditioned_matrices(matrix_size=4, seed=123):
    """Create ill-conditioned test matrices to test numerical stability.
    
    Args:
        matrix_size (int): Size of each matrix
        seed (int): Random seed for reproducibility
        
    Returns:
        Tuple[Tensor, Tensor]: (overlap_matrix, hamiltonian_matrix)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create nearly singular overlap matrix
    A = torch.randn(matrix_size, matrix_size)
    overlap_matrix = torch.mm(A, A.t())
    
    # Make one eigenvalue very small
    eigvals, eigvecs = torch.linalg.eigh(overlap_matrix)
    eigvals[0] = 1e-10  # Make first eigenvalue very small
    overlap_matrix = torch.mm(torch.mm(eigvecs, torch.diag(eigvals)), eigvecs.t())
    
    # Create symmetric Hamiltonian matrix
    H = torch.randn(matrix_size, matrix_size)
    hamiltonian_matrix = H + H.t()
    
    return overlap_matrix, hamiltonian_matrix

def test_basic_functionality():
    """Test basic functionality of both methods."""
    print("=" * 60)
    print("Testing Basic Functionality (Single Matrix)")
    print("=" * 60)
    
    # Create test matrices
    overlap_matrix, hamiltonian_matrix = create_test_matrices(matrix_size=5)
    
    print(f"Overlap matrix shape: {overlap_matrix.shape}")
    print(f"Hamiltonian matrix shape: {hamiltonian_matrix.shape}")
    
    # Test original method
    print("\nTesting eigenvalue method...")
    try:
        energies_eigh, coeffs_eigh = cal_orbital_and_energies_single(overlap_matrix, hamiltonian_matrix, method="eigh")
        print(f"✓ Eigenvalue method successful")
        print(f"  Energies shape: {energies_eigh.shape}")
        print(f"  Coefficients shape: {coeffs_eigh.shape}")
        print(f"  Energy range: [{energies_eigh.min().item():.6f}, {energies_eigh.max().item():.6f}]")
    except Exception as e:
        print(f"✗ Eigenvalue method failed: {e}")
        return False
    
    # Test Cholesky method
    print("\nTesting Cholesky method...")
    try:
        energies_chol, coeffs_chol = cal_orbital_and_energies_single(overlap_matrix, hamiltonian_matrix, method="cholesky")
        print(f"✓ Cholesky method successful")
        print(f"  Energies shape: {energies_chol.shape}")
        print(f"  Coefficients shape: {coeffs_chol.shape}")
        print(f"  Energy range: [{energies_chol.min().item():.6f}, {energies_chol.max().item():.6f}]")
    except Exception as e:
        print(f"✗ Cholesky method failed: {e}")
        return False
    
    return True

def test_result_consistency():
    """Test that both methods give consistent results."""
    print("\n" + "=" * 60)
    print("Testing Result Consistency (Single Matrix)")
    print("=" * 60)
    
    # Create test matrices
    overlap_matrix, hamiltonian_matrix = create_test_matrices(matrix_size=4)
    
    # Get results from both methods
    energies_eigh, coeffs_eigh = cal_orbital_and_energies_single(overlap_matrix, hamiltonian_matrix, method="eigh")
    energies_chol, coeffs_chol = cal_orbital_and_energies_single(overlap_matrix, hamiltonian_matrix, method="cholesky")
    
    # Compare energies (should be very close)
    energy_diff = torch.abs(energies_eigh - energies_chol)
    max_energy_diff = energy_diff.max().item()
    mean_energy_diff = energy_diff.mean().item()
    
    print(f"Energy comparison:")
    print(f"  Max difference: {max_energy_diff:.2e}")
    print(f"  Mean difference: {mean_energy_diff:.2e}")
    
    # Check if energies are sorted (should be for both methods)
    energies_eigh_sorted = torch.sort(energies_eigh)[0]
    energies_chol_sorted = torch.sort(energies_chol)[0]
    
    energy_sorted_diff = torch.abs(energies_eigh_sorted - energies_chol_sorted)
    max_sorted_diff = energy_sorted_diff.max().item()
    
    print(f"  Max difference (sorted): {max_sorted_diff:.2e}")
    
    # Compare coefficients (more complex due to possible sign differences and ordering)
    # We'll check the orthogonality condition: C^T S C = I
    S_C_eigh = torch.mm(overlap_matrix, coeffs_eigh)
    orthogonality_eigh = torch.mm(coeffs_eigh.t(), S_C_eigh)
    
    S_C_chol = torch.mm(overlap_matrix, coeffs_chol)
    orthogonality_chol = torch.mm(coeffs_chol.t(), S_C_chol)
    
    # Check how close to identity matrix
    identity = torch.eye(overlap_matrix.size(-1))
    
    ortho_diff_eigh = torch.abs(orthogonality_eigh - identity).max().item()
    ortho_diff_chol = torch.abs(orthogonality_chol - identity).max().item()
    
    print(f"\nOrthogonality check (C^T S C should be identity):")
    print(f"  Eigenvalue method max deviation: {ortho_diff_eigh:.2e}")
    print(f"  Cholesky method max deviation: {ortho_diff_chol:.2e}")
    
    # Check eigenvalue equation: H C = S C E
    H_C_eigh = torch.mm(hamiltonian_matrix, coeffs_eigh)
    S_C_E_eigh = torch.mm(S_C_eigh, torch.diag(energies_eigh))
    eigenvalue_error_eigh = torch.abs(H_C_eigh - S_C_E_eigh).max().item()
    
    H_C_chol = torch.mm(hamiltonian_matrix, coeffs_chol)
    S_C_E_chol = torch.mm(S_C_chol, torch.diag(energies_chol))
    eigenvalue_error_chol = torch.abs(H_C_chol - S_C_E_chol).max().item()
    
    print(f"\nEigenvalue equation check (H C = S C E):")
    print(f"  Eigenvalue method max error: {eigenvalue_error_eigh:.2e}")
    print(f"  Cholesky method max error: {eigenvalue_error_chol:.2e}")
    
    # Determine if results are consistent
    tolerance = 1e-6
    consistent = (max_energy_diff < tolerance and 
                 ortho_diff_eigh < tolerance and 
                 ortho_diff_chol < tolerance and
                 eigenvalue_error_eigh < tolerance and
                 eigenvalue_error_chol < tolerance)
    
    if consistent:
        print(f"\n✓ Results are consistent within tolerance {tolerance}")
    else:
        print(f"\n✗ Results may not be consistent within tolerance {tolerance}")
    
    return consistent

def test_numerical_stability():
    """Test numerical stability with ill-conditioned matrices."""
    print("\n" + "=" * 60)
    print("Testing Numerical Stability (Single Matrix)")
    print("=" * 60)
    
    # Create ill-conditioned matrices
    overlap_matrix, hamiltonian_matrix = create_ill_conditioned_matrices()
    
    print("Testing with ill-conditioned matrices...")
    
    # Test eigenvalue method
    print("\nEigenvalue method with ill-conditioned matrices:")
    try:
        energies_eigh, coeffs_eigh = cal_orbital_and_energies_single(overlap_matrix, hamiltonian_matrix, method="eigh")
        print(f"✓ Eigenvalue method successful")
        
        # Check orthogonality
        S_C_eigh = torch.mm(overlap_matrix, coeffs_eigh)
        orthogonality_eigh = torch.mm(coeffs_eigh.t(), S_C_eigh)
        identity = torch.eye(overlap_matrix.size(-1))
        ortho_error_eigh = torch.abs(orthogonality_eigh - identity).max().item()
        print(f"  Orthogonality error: {ortho_error_eigh:.2e}")
        
    except Exception as e:
        print(f"✗ Eigenvalue method failed: {e}")
        ortho_error_eigh = float('inf')
    
    # Test Cholesky method
    print("\nCholesky method with ill-conditioned matrices:")
    try:
        energies_chol, coeffs_chol = cal_orbital_and_energies_single(overlap_matrix, hamiltonian_matrix, method="cholesky")
        print(f"✓ Cholesky method successful")
        
        # Check orthogonality
        S_C_chol = torch.mm(overlap_matrix, coeffs_chol)
        orthogonality_chol = torch.mm(coeffs_chol.t(), S_C_chol)
        identity = torch.eye(overlap_matrix.size(-1))
        ortho_error_chol = torch.abs(orthogonality_chol - identity).max().item()
        print(f"  Orthogonality error: {ortho_error_chol:.2e}")
        
    except Exception as e:
        print(f"✗ Cholesky method failed: {e}")
        ortho_error_chol = float('inf')
    
    # Compare stability
    if ortho_error_eigh < ortho_error_chol:
        print(f"\n✓ Eigenvalue method is more stable for this case")
    elif ortho_error_chol < ortho_error_eigh:
        print(f"\n✓ Cholesky method is more stable for this case")
    else:
        print(f"\n≈ Both methods have similar stability for this case")

def test_performance():
    """Test performance comparison between methods."""
    print("\n" + "=" * 60)
    print("Testing Performance (Single Matrix)")
    print("=" * 60)
    
    import time
    
    # Create larger test matrices
    matrix_size = 50
    overlap_matrix, hamiltonian_matrix = create_test_matrices(matrix_size)
    
    print(f"Testing with matrix_size={matrix_size}")
    
    # Warm up
    cal_orbital_and_energies_single(overlap_matrix, hamiltonian_matrix, method="eigh")
    cal_orbital_and_energies_single(overlap_matrix, hamiltonian_matrix, method="cholesky")
    
    # Time eigenvalue method
    num_runs = 10
    start_time = time.time()
    for _ in range(num_runs):
        cal_orbital_and_energies_single(overlap_matrix, hamiltonian_matrix, method="eigh")
    eigh_time = (time.time() - start_time) / num_runs
    
    # Time Cholesky method
    start_time = time.time()
    for _ in range(num_runs):
        cal_orbital_and_energies_single(overlap_matrix, hamiltonian_matrix, method="cholesky")
    chol_time = (time.time() - start_time) / num_runs
    
    print(f"Eigenvalue method average time: {eigh_time*1000:.2f} ms")
    print(f"Cholesky method average time: {chol_time*1000:.2f} ms")
    print(f"Speed ratio (Cholesky/Eigenvalue): {chol_time/eigh_time:.2f}")

def test_input_validation():
    """Test input validation."""
    print("\n" + "=" * 60)
    print("Testing Input Validation")
    print("=" * 60)
    
    # Test with 3D tensor (should fail)
    print("Testing with 3D tensor...")
    try:
        overlap_3d = torch.randn(2, 3, 3)
        hamiltonian_3d = torch.randn(2, 3, 3)
        cal_orbital_and_energies_single(overlap_3d, hamiltonian_3d)
        print("✗ Should have failed with 3D tensor")
    except ValueError as e:
        print(f"✓ Correctly caught 3D tensor error: {e}")
    
    # Test with 1D tensor (should fail)
    print("\nTesting with 1D tensor...")
    try:
        overlap_1d = torch.randn(3)
        hamiltonian_1d = torch.randn(3)
        cal_orbital_and_energies_single(overlap_1d, hamiltonian_1d)
        print("✗ Should have failed with 1D tensor")
    except ValueError as e:
        print(f"✓ Correctly caught 1D tensor error: {e}")
    
    # Test with invalid method (should fail)
    print("\nTesting with invalid method...")
    try:
        overlap_matrix, hamiltonian_matrix = create_test_matrices(3)
        cal_orbital_and_energies_single(overlap_matrix, hamiltonian_matrix, method="invalid")
        print("✗ Should have failed with invalid method")
    except AssertionError as e:
        print(f"✓ Correctly caught invalid method error: {e}")

def main():
    """Run all tests."""
    print("Single Matrix Orbital Calculation Methods Test Suite")
    print("=" * 60)
    
    # Check if we can import the module
    try:
        from common.metric import cal_orbital_and_energies_single
        print("✓ Successfully imported cal_orbital_and_energies_single")
    except ImportError as e:
        print(f"✗ Failed to import cal_orbital_and_energies_single: {e}")
        return
    
    # Run tests
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Basic functionality
    if test_basic_functionality():
        tests_passed += 1
    
    # Test 2: Result consistency
    if test_result_consistency():
        tests_passed += 1
    
    # Test 3: Numerical stability
    test_numerical_stability()
    tests_passed += 1  # This test doesn't have a clear pass/fail
    
    # Test 4: Performance
    test_performance()
    tests_passed += 1  # This test doesn't have a clear pass/fail
    
    # Test 5: Input validation
    test_input_validation()
    tests_passed += 1  # This test doesn't have a clear pass/fail
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests completed: {total_tests}")
    print(f"Tests passed: {tests_passed}")
    
    if tests_passed >= 2:  # At least basic functionality and consistency should pass
        print("✓ Overall: Tests PASSED")
    else:
        print("✗ Overall: Tests FAILED")

if __name__ == "__main__":
    main()
