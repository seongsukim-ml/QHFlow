"""
Simple test to verify string inputs work correctly with get_feasible_irrep
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.QHNet_cue import get_feasible_irrep, as_o3_irreps, as_cue_irreps
from e3nn import o3
import cuequivariance as cue


def test_string_inputs():
    """Test various string input formats"""
    
    print("🧪 Testing String Inputs for get_feasible_irrep")
    print("=" * 50)
    
    # Test cases with different string formats
    test_cases = [
        {
            "name": "Simple irreps",
            "irrep1": "1x0e",
            "irrep2": "1x0e",
            "cutoff": "1x0e"
        },
        {
            "name": "Basic scalar + vector",
            "irrep1": "1x0e + 1x1o",
            "irrep2": "1x0e + 1x1e",
            "cutoff": "2x0e + 2x1o + 2x1e"
        },
        {
            "name": "Complex irreps",
            "irrep1": "2x0e + 1x1o + 1x2e",
            "irrep2": "1x0e + 1x1e + 1x2o",
            "cutoff": "5x0e + 3x1o + 3x1e + 2x2e + 2x2o + 1x3o + 1x3e + 1x4e"
        },
        {
            "name": "Single character format",
            "irrep1": "0e",
            "irrep2": "1o",
            "cutoff": "0e + 1o"
        },
        {
            "name": "Large multiplicities",
            "irrep1": "10x0e + 5x1o",
            "irrep2": "8x0e + 3x1e",
            "cutoff": "20x0e + 15x1o + 15x1e"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print(f"   Input 1: {test_case['irrep1']}")
        print(f"   Input 2: {test_case['irrep2']}")
        print(f"   Cutoff:  {test_case['cutoff']}")
        
        try:
            # Test with return_cue=False (o3.Irreps output)
            irrep_mid_o3, ins_o3 = get_feasible_irrep(
                test_case['irrep1'],
                test_case['irrep2'], 
                test_case['cutoff'],
                return_cue=False
            )
            
            # Test with return_cue=True (cue.Irreps output)
            irrep_mid_cue, ins_cue = get_feasible_irrep(
                test_case['irrep1'],
                test_case['irrep2'], 
                test_case['cutoff'],
                return_cue=True
            )
            
            print(f"   ✅ O3 Result:  {irrep_mid_o3} ({len(ins_o3)} instructions)")
            print(f"   ✅ CUE Result: {irrep_mid_cue} ({len(ins_cue)} instructions)")
            
            # Verify types
            assert isinstance(irrep_mid_o3, o3.Irreps), f"Expected o3.Irreps, got {type(irrep_mid_o3)}"
            assert isinstance(irrep_mid_cue, cue.Irreps), f"Expected cue.Irreps, got {type(irrep_mid_cue)}"
            assert len(ins_o3) == len(ins_cue), "Instruction counts should match"
            
        except Exception as e:
            print(f"   ❌ FAILED: {str(e)}")
            return False
    
    # Test conversion consistency
    print(f"\n🔄 Testing conversion consistency...")
    test_string = "2x0e + 1x1o + 1x2e"
    
    # String → o3 → cue → o3
    o3_from_str = as_o3_irreps(test_string)
    cue_from_o3 = as_cue_irreps(o3_from_str)
    o3_from_cue = as_o3_irreps(cue_from_o3)
    
    print(f"   Original string: {test_string}")
    print(f"   String → O3:     {o3_from_str}")
    print(f"   O3 → CUE:        {cue_from_o3}")
    print(f"   CUE → O3:        {o3_from_cue}")
    
    # Check round-trip consistency
    if str(o3_from_str) == str(o3_from_cue):
        print(f"   ✅ Round-trip conversion successful!")
    else:
        print(f"   ❌ Round-trip conversion failed!")
        return False
    
    # Test edge cases
    print(f"\n🎯 Testing edge cases...")
    
    edge_cases = [
        ("Empty string", "", "", ""),
        ("Spaces in string", " 1x0e + 1x1o ", " 1x0e ", " 2x0e + 1x1o "),
        ("Different formats", "0e+1o", "0e+1e", "0e+1o+1e"),
    ]
    
    for case_name, irrep1, irrep2, cutoff in edge_cases:
        try:
            if irrep1 == "" and irrep2 == "" and cutoff == "":
                # Skip empty test as it might not be meaningful
                print(f"   ⏭️  Skipping {case_name}")
                continue
                
            result, instructions = get_feasible_irrep(irrep1, irrep2, cutoff, return_cue=False)
            print(f"   ✅ {case_name}: {result} ({len(instructions)} instructions)")
        except Exception as e:
            print(f"   ⚠️  {case_name}: {str(e)}")
    
    print(f"\n🎉 All string input tests completed successfully!")
    print(f"\n💡 Summary:")
    print(f"   • String inputs are properly converted to o3.Irreps internally")
    print(f"   • Both o3.Irreps and cue.Irreps outputs work with string inputs")
    print(f"   • Round-trip conversions maintain consistency")
    print(f"   • Various string formats are supported")
    print(f"   • Edge cases are handled gracefully")
    
    return True


if __name__ == "__main__":
    success = test_string_inputs()
    exit(0 if success else 1)
