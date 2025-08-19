"""
Test suite for get_feasible_irrep function to verify o3 and cue compatibility
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# import pytest  # Not required for basic testing
import torch
import numpy as np
from e3nn import o3
import cuequivariance as cue

from models.QHNet_cue import get_feasible_irrep, as_o3_irreps, as_cue_irreps, O3_e3nn


class TestGetFeasibleIrrep:
    """Test class for get_feasible_irrep function"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Define test irreps in different formats
        self.o3_irrep_1 = o3.Irreps("1x0e + 1x1o")
        self.o3_irrep_2 = o3.Irreps("1x0e + 1x1e + 1x2e")
        self.o3_cutoff = o3.Irreps("2x0e + 2x1o + 1x1e + 1x2e + 1x2o")
        
        # Convert to cue irreps
        self.cue_irrep_1 = as_cue_irreps(self.o3_irrep_1)
        self.cue_irrep_2 = as_cue_irreps(self.o3_irrep_2)
        self.cue_cutoff = as_cue_irreps(self.o3_cutoff)
        
        # String representations
        self.str_irrep_1 = "1x0e + 1x1o"
        self.str_irrep_2 = "1x0e + 1x1e + 1x2e"
        self.str_cutoff = "2x0e + 2x1o + 1x1e + 1x2e + 1x2o"
    
    def test_o3_inputs_return_o3(self):
        """Test with o3.Irreps inputs returning o3 format"""
        irrep_mid, instructions = get_feasible_irrep(
            self.o3_irrep_1, 
            self.o3_irrep_2, 
            self.o3_cutoff,
            return_cue=False
        )
        
        # Check return types
        assert isinstance(irrep_mid, o3.Irreps), f"Expected o3.Irreps, got {type(irrep_mid)}"
        assert isinstance(instructions, list), f"Expected list, got {type(instructions)}"
        
        # Check instructions format
        if instructions:
            ins = instructions[0]
            assert len(ins) == 6, f"Expected instruction length 6, got {len(ins)}"
            assert isinstance(ins[0], int), "First element should be int (i_in1)"
            assert isinstance(ins[1], int), "Second element should be int (i_in2)"
            assert isinstance(ins[2], int), "Third element should be int (i_out)"
            assert isinstance(ins[3], str), "Fourth element should be str (mode)"
            assert isinstance(ins[4], bool), "Fifth element should be bool (train)"
            assert isinstance(ins[5], float), "Sixth element should be float (alpha)"
    
    def test_o3_inputs_return_cue(self):
        """Test with o3.Irreps inputs returning cue format"""
        irrep_mid, instructions = get_feasible_irrep(
            self.o3_irrep_1, 
            self.o3_irrep_2, 
            self.o3_cutoff,
            return_cue=True
        )
        
        # Check return types
        assert isinstance(irrep_mid, cue.Irreps), f"Expected cue.Irreps, got {type(irrep_mid)}"
        assert isinstance(instructions, list), f"Expected list, got {type(instructions)}"
    
    def test_cue_inputs_return_o3(self):
        """Test with cue.Irreps inputs returning o3 format"""
        irrep_mid, instructions = get_feasible_irrep(
            self.cue_irrep_1, 
            self.cue_irrep_2, 
            self.cue_cutoff,
            return_cue=False
        )
        
        # Check return types
        assert isinstance(irrep_mid, o3.Irreps), f"Expected o3.Irreps, got {type(irrep_mid)}"
        assert isinstance(instructions, list), f"Expected list, got {type(instructions)}"
    
    def test_cue_inputs_return_cue(self):
        """Test with cue.Irreps inputs returning cue format"""
        irrep_mid, instructions = get_feasible_irrep(
            self.cue_irrep_1, 
            self.cue_irrep_2, 
            self.cue_cutoff,
            return_cue=True
        )
        
        # Check return types
        assert isinstance(irrep_mid, cue.Irreps), f"Expected cue.Irreps, got {type(irrep_mid)}"
        assert isinstance(instructions, list), f"Expected list, got {type(instructions)}"
    
    def test_string_inputs_return_o3(self):
        """Test with string inputs returning o3 format"""
        irrep_mid, instructions = get_feasible_irrep(
            self.str_irrep_1, 
            self.str_irrep_2, 
            self.str_cutoff,
            return_cue=False
        )
        
        # Check return types
        assert isinstance(irrep_mid, o3.Irreps), f"Expected o3.Irreps, got {type(irrep_mid)}"
        assert isinstance(instructions, list), f"Expected list, got {type(instructions)}"
    
    def test_string_inputs_return_cue(self):
        """Test with string inputs returning cue format"""
        irrep_mid, instructions = get_feasible_irrep(
            self.str_irrep_1, 
            self.str_irrep_2, 
            self.str_cutoff,
            return_cue=True
        )
        
        # Check return types
        assert isinstance(irrep_mid, cue.Irreps), f"Expected cue.Irreps, got {type(irrep_mid)}"
        assert isinstance(instructions, list), f"Expected list, got {type(instructions)}"
    
    def test_mixed_inputs_consistency(self):
        """Test that mixed input types give consistent results"""
        # Test o3 vs string inputs
        irrep_mid_o3, ins_o3 = get_feasible_irrep(
            self.o3_irrep_1, self.o3_irrep_2, self.o3_cutoff, return_cue=False
        )
        irrep_mid_str, ins_str = get_feasible_irrep(
            self.str_irrep_1, self.str_irrep_2, self.str_cutoff, return_cue=False
        )
        
        # Check consistency
        assert str(irrep_mid_o3) == str(irrep_mid_str), "O3 and string inputs should give same result"
        assert len(ins_o3) == len(ins_str), "Instruction lists should have same length"
        
        # Test cue vs o3 inputs
        irrep_mid_cue, ins_cue = get_feasible_irrep(
            self.cue_irrep_1, self.cue_irrep_2, self.cue_cutoff, return_cue=False
        )
        
        assert str(irrep_mid_o3) == str(irrep_mid_cue), "O3 and cue inputs should give same result"
        assert len(ins_o3) == len(ins_cue), "Instruction lists should have same length"
    
    def test_different_tp_modes(self):
        """Test different tensor product modes"""
        tp_modes = ["uvu", "uvw", "uuu", "uvv", "uuw"]
        
        for mode in tp_modes:
            irrep_mid, instructions = get_feasible_irrep(
                self.o3_irrep_1, 
                self.o3_irrep_2, 
                self.o3_cutoff,
                tp_mode=mode,
                return_cue=False
            )
            
            assert isinstance(irrep_mid, o3.Irreps)
            assert isinstance(instructions, list)
            
            # Check that mode is correctly set in instructions
            for ins in instructions:
                assert ins[3] == mode, f"Expected mode {mode}, got {ins[3]}"
    
    def test_conversion_functions(self):
        """Test the conversion functions as_o3_irreps and as_cue_irreps"""
        # Test o3 to cue and back
        cue_converted = as_cue_irreps(self.o3_irrep_1)
        o3_converted_back = as_o3_irreps(cue_converted)
        
        assert str(self.o3_irrep_1) == str(o3_converted_back), "Round-trip conversion should preserve irreps"
        
        # Test cue to o3 and back
        o3_converted = as_o3_irreps(self.cue_irrep_1)
        cue_converted_back = as_cue_irreps(o3_converted)
        
        assert str(self.cue_irrep_1) == str(cue_converted_back), "Round-trip conversion should preserve irreps"
    
    def test_empty_inputs(self):
        """Test with empty irreps"""
        empty_o3 = o3.Irreps("")
        empty_cue = as_cue_irreps(empty_o3)
        
        irrep_mid, instructions = get_feasible_irrep(
            empty_o3, 
            self.o3_irrep_2, 
            self.o3_cutoff,
            return_cue=False
        )
        
        # Should handle empty inputs gracefully
        assert isinstance(irrep_mid, o3.Irreps)
        assert isinstance(instructions, list)
    
    def test_large_irreps(self):
        """Test with larger irreps"""
        large_irrep_1 = o3.Irreps("5x0e + 3x1o + 2x2e + 1x3o")
        large_irrep_2 = o3.Irreps("4x0e + 2x1e + 3x2o + 1x3e")
        large_cutoff = o3.Irreps("10x0e + 8x1o + 8x1e + 6x2e + 6x2o + 4x3o + 4x3e + 2x4e + 2x4o + 1x5o + 1x5e + 1x6e")
        
        irrep_mid, instructions = get_feasible_irrep(
            large_irrep_1, 
            large_irrep_2, 
            large_cutoff,
            return_cue=False
        )
        
        assert isinstance(irrep_mid, o3.Irreps)
        assert isinstance(instructions, list)
        assert len(instructions) > 0, "Should generate instructions for large irreps"
    
    def test_normalization_coefficients(self):
        """Test that normalization coefficients are reasonable"""
        irrep_mid, instructions = get_feasible_irrep(
            self.o3_irrep_1, 
            self.o3_irrep_2, 
            self.o3_cutoff,
            return_cue=False
        )
        
        for ins in instructions:
            alpha = ins[5]  # normalization coefficient
            assert isinstance(alpha, float), "Normalization coefficient should be float"
            assert alpha > 0, "Normalization coefficient should be positive"
            assert not np.isnan(alpha), "Normalization coefficient should not be NaN"
            assert not np.isinf(alpha), "Normalization coefficient should not be infinite"


def run_comprehensive_tests():
    """Run comprehensive tests without pytest"""
    print("=" * 60)
    print("COMPREHENSIVE TEST SUITE FOR get_feasible_irrep")
    print("=" * 60)
    
    test_instance = TestGetFeasibleIrrep()
    test_instance.setup_method()
    
    tests = [
        ("O3 inputs return O3", test_instance.test_o3_inputs_return_o3),
        ("O3 inputs return CUE", test_instance.test_o3_inputs_return_cue),
        ("CUE inputs return O3", test_instance.test_cue_inputs_return_o3),
        ("CUE inputs return CUE", test_instance.test_cue_inputs_return_cue),
        ("String inputs return O3", test_instance.test_string_inputs_return_o3),
        ("String inputs return CUE", test_instance.test_string_inputs_return_cue),
        ("Mixed inputs consistency", test_instance.test_mixed_inputs_consistency),
        ("Different TP modes", test_instance.test_different_tp_modes),
        ("Conversion functions", test_instance.test_conversion_functions),
        ("Empty inputs", test_instance.test_empty_inputs),
        ("Large irreps", test_instance.test_large_irreps),
        ("Normalization coefficients", test_instance.test_normalization_coefficients),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Testing: {test_name}")
            test_func()
            print(f"✅ PASSED: {test_name}")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name}")
            print(f"   Error: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! The function supports both O3 and CUE inputs.")
        print("\n📋 SUMMARY:")
        print("✓ Accepts o3.Irreps, cue.Irreps, and string inputs")
        print("✓ Can return either o3.Irreps or cue.Irreps format")
        print("✓ Conversion functions work bidirectionally")
        print("✓ All tensor product modes supported")
        print("✓ Handles edge cases (empty, large irreps)")
        print("✓ Normalization coefficients are valid")
    else:
        print(f"⚠️  {failed} tests failed. Check the implementation.")
    
    return failed == 0


def test_run_basic_functionality():
    """Simple test that can be run without pytest"""
    print("Running basic functionality test...")
    
    # Create test irreps
    irrep_1 = o3.Irreps("1x0e + 1x1o")
    irrep_2 = o3.Irreps("1x0e + 1x1e")
    cutoff = o3.Irreps("2x0e + 2x1o + 1x1e")
    
    # Test o3 inputs
    print("Testing o3 inputs...")
    irrep_mid_o3, ins_o3 = get_feasible_irrep(irrep_1, irrep_2, cutoff, return_cue=False)
    print(f"O3 result: {irrep_mid_o3}")
    print(f"O3 instructions count: {len(ins_o3)}")
    
    # Test cue inputs
    print("Testing cue inputs...")
    cue_irrep_1 = as_cue_irreps(irrep_1)
    cue_irrep_2 = as_cue_irreps(irrep_2)
    cue_cutoff = as_cue_irreps(cutoff)
    
    irrep_mid_cue, ins_cue = get_feasible_irrep(cue_irrep_1, cue_irrep_2, cue_cutoff, return_cue=True)
    print(f"CUE result: {irrep_mid_cue}")
    print(f"CUE instructions count: {len(ins_cue)}")
    
    # Test string inputs
    print("Testing string inputs...")
    irrep_mid_str, ins_str = get_feasible_irrep("1x0e + 1x1o", "1x0e + 1x1e", "2x0e + 2x1o + 1x1e", return_cue=False)
    print(f"String result: {irrep_mid_str}")
    print(f"String instructions count: {len(ins_str)}")
    
    print("Basic functionality test completed successfully!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--comprehensive":
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    else:
        test_run_basic_functionality()
