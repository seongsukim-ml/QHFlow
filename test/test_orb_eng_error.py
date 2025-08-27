import torch
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the QHFlow directory to the path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'QHFlow', 'src'))
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pl_module.base_module import LitModel
from experiment import get_test_conf_qh9, get_test_conf_md17
from common.setup import setup_paths, get_root_path
# from dataset_module.ori_dataset import MD17_DFT, random_split, get_mask
from common.qh9_utils import load_qh9_dataset, _create_qh9_data_loaders, _create_qh9_dataset
from common.data_utils import _create_md17_data_loaders, load_md17_dataset, _create_md17_dataset
from torch_geometric.loader import DataLoader
import warnings
warnings.filterwarnings("ignore")


# setup_paths()
class TestOrbAndEngError:
    """Test class for _orb_and_eng_error function with qh9 and md17 datasets."""
    def __init__(self):
        self.qh9_conf = get_test_conf_qh9()
        self.md17_conf = get_test_conf_md17()
        
        self.qh9_model = LitModel(self.qh9_conf)
        self.md17_model = LitModel(self.md17_conf)
        root_path = "/home/holywater2/25DFT/QHFlow"
        self.qh9_data = load_qh9_dataset(self.qh9_conf, root_path)
        self.qh9_train_dataset, self.qh9_valid_dataset, self.qh9_test_dataset = _create_qh9_dataset(self.qh9_data, self.qh9_conf)
        # self.qh9_train_loader, self.qh9_val_loader, self.qh9_test_loader = _create_qh9_data_loaders(self.qh9_train_dataset, self.qh9_valid_dataset, self.qh9_test_dataset, self.qh9_conf)
        self.qh9_train_loader = DataLoader(self.qh9_train_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
        self.qh9_train_loader_b1 = DataLoader(self.qh9_train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        
        self.md17_data = load_md17_dataset(self.md17_conf, root_path)
        self.md17_train_dataset, self.md17_valid_dataset, self.md17_test_dataset = _create_md17_dataset(self.md17_data, self.md17_conf)
        self.md17_train_loader = DataLoader(self.md17_train_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
        self.md17_train_loader_b1 = DataLoader(self.md17_train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        
        self.qh9_model.set(torch.device('cpu'))
        self.md17_model.set(torch.device('cpu'))
        
    def test_orb_and_eng_error_qh9_success(self):
        """Test _orb_and_eng_error with qh9 dataset."""
        
        qh9_batch = next(iter(self.qh9_train_loader)) # batch size 4
        qh9_batch_b1 = next(iter(self.qh9_train_loader_b1))
        
        md17_batch = next(iter(self.md17_train_loader)) # batch size 16
        md17_batch_b1 = next(iter(self.md17_train_loader_b1))
                
        ## md17
        md17_orb, md17_eng = self.md17_model.cal_orbital_and_energies(md17_batch.overlap, md17_batch.hamiltonian)
        md17_orb_b1, md17_eng_b1 = self.md17_model.cal_orbital_and_energies(md17_batch_b1.overlap, md17_batch_b1.hamiltonian)
        
        orb_diff = md17_orb[0] - md17_orb_b1[0]
        eng_diff = md17_eng[0] - md17_eng_b1[0]
        
        # print(f"md17_orb diff/sum/allclose: {orb_diff}, {orb_diff.abs().sum()}, {torch.allclose(md17_orb[0], md17_orb_b1[0])}")
        # print(f"md17_eng diff/sum/allclose: {eng_diff}, {eng_diff.abs().sum()}, {torch.allclose(md17_eng[0], md17_eng_b1[0])}")
        

        ## qh9
        qh9_ham_list = self.qh9_model.build_final_matrix(qh9_batch, qh9_batch.diagonal_hamiltonian, qh9_batch.non_diagonal_hamiltonian)
        qh9_ham_b1_list = self.qh9_model.build_final_matrix(qh9_batch_b1, qh9_batch_b1.diagonal_hamiltonian, qh9_batch_b1.non_diagonal_hamiltonian)
        
        qh9_ham = qh9_ham_list[0]
        qh9_ham_b1 = qh9_ham_b1_list[0]
        
        qh9_ham_diff = qh9_ham - qh9_ham_b1
        qh9_ham_diff_sum = qh9_ham_diff.abs().sum()
        qh9_ham_diff_allclose = torch.allclose(qh9_ham, qh9_ham_b1)
        
        print(f"qh9_ham diff/sum/allclose: {qh9_ham_diff}, {qh9_ham_diff_sum}, {qh9_ham_diff_allclose}")
        
        qh9_overlap_list = self.qh9_model.build_final_matrix(qh9_batch, qh9_batch.diagonal_overlap, qh9_batch.non_diagonal_overlap)
        qh9_overlap_list2 = self.qh9_model.build_final_matrix(qh9_batch, qh9_batch.diagonal_overlap, qh9_batch.non_diagonal_overlap, transform=False)
        qh9_overlap_b1_list = self.qh9_model.build_final_matrix(qh9_batch_b1, qh9_batch_b1.diagonal_overlap, qh9_batch_b1.non_diagonal_overlap)
        
        qh9_overlap = qh9_overlap_list[0]
        qh9_overlap_b1 = qh9_overlap_b1_list[0]
        
        qh9_overlap_diff = qh9_overlap - qh9_overlap_b1
        # print(f"qh9_overlap_b1_t_cuda_diff/sum/allclose: {qh9_overlap_b1_t_cuda_diff}, {qh9_overlap_b1_t_cuda_diff.abs().sum()}, {torch.allclose(qh9_overlap_b1_t_cuda, qh9_overlap_b1_t_cpu)}")
        # print(f"qh9_overlap_t_cuda_diff/sum/allclose: {qh9_overlap_t_cuda_diff}, {qh9_overlap_t_cuda_diff.abs().sum()}, {torch.allclose(qh9_overlap_t_cuda, qh9_overlap_t_cpu)}")        
        
        # import pdb; pdb.set_trace()

        qh9_orb_b1, qh9_eng_b1 = self.qh9_model.cal_orbital_and_energies(qh9_overlap_b1_list, qh9_ham_b1_list)
        qh9_orb, qh9_eng = self.qh9_model.cal_orbital_and_energies(qh9_overlap_list, qh9_ham_list)
        
        import pdb; pdb.set_trace()
        
        qh9_orb_diff = qh9_orb[0] - qh9_orb_b1[0]
        qh9_eng_diff = qh9_eng[0] - qh9_eng_b1[0]
        print(f"qh9_orb_diff/sum/allclose: {qh9_orb_diff}, {qh9_orb_diff.abs().sum()}, {torch.allclose(qh9_orb[0], qh9_orb_b1[0])}")
        print(f"qh9_eng_diff/sum/allclose: {qh9_eng_diff}, {qh9_eng_diff.abs().sum()}, {torch.allclose(qh9_eng[0], qh9_eng_b1[0])}")
        
        import pdb; pdb.set_trace()
        
        
        
def run_comprehensive_test():
    """Run comprehensive tests for _orb_and_eng_error function."""
    print("Starting comprehensive tests for _orb_and_eng_error function...")
    
    # Create test instance
    test_instance = TestOrbAndEngError()
    
    # Test cases
    test_cases = [
        ("qh9_success", test_instance.test_orb_and_eng_error_qh9_success),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in test_cases:
        try:
            test_func()
            print(f"✅ {test_name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {str(e)}")
            failed += 1
    
    print(f"\nTest Summary:")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    return passed, failed


if __name__ == "__main__":
    run_comprehensive_test()
