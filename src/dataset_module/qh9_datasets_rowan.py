import numpy as np
import lmdb
import pickle
import os
from common.custom_logger import setup_global_logger, get_logger
import json
import gdown
import torch
# from tqdm.rich import tqdm
from tqdm import tqdm
import random
import time
from typing import Union, List

from common.metric import cal_orbital_and_energies
from common.matrix_transforms import pack_upper_triangle, unpack_upper_triangle, _matrix_transform_single, get_convention_dict, _cut_matrix_3d, _cut_matrix_3d_last
from dataset_module.lmdb_shard import LMDBShard_maker_db
from common.dft_utils import calc_overlap_and_init_hamiltonian, calc_dm0

from torch_geometric.data import InMemoryDataset, Data
from utils import AOData, Onsite_3idx_Overlap_Integral, build_molecule, build_AO_index
from common.units import *

logger = get_logger(__file__)

GoogleDriveLink = (
    "https://drive.google.com/drive/u/0/folders/1LXTC8uaOQzmb76FsuGfwSocAbK5Hshfj"
)

########################################################
# QH9Stable
########################################################

# 0-1: finding number of orbitals
import pyscf.gto

orbital_list = {}

def get_orbital_list(atomic_symbol, basis="def2tzvppd"):
    if basis not in orbital_list:   
        orbital_list[basis] = {}
    if atomic_symbol in orbital_list[basis]:
        return orbital_list[basis][atomic_symbol]
    else:
        orbital_data = pyscf.gto.basis.load(basis, atomic_symbol)
        orb_list = []
        orbstr_list = []
        orbstr = "spdfghi"
        for orb in orbital_data:
            orb_rank = orb[0]
            orb_list.append(orb_rank + 1)
            orb_idx = orb_list.count(orb_rank + 1) + orb_rank - 1
            orbstr_list.append(f"{orb_idx+1}{orbstr[orb_rank]}")
        orbital_list[atomic_symbol] = orb_list, orbstr_list
        return orb_list, orbstr_list

def get_orbital_list_from_atomic_symbol(atomic_symbols, basis="def2tzvppd"):
    tot_orb_list = []
    tot_orbstr_list = []
    for sym in atomic_symbols:
        orb_list, orbstr_list = get_orbital_list(sym, basis)
        tot_orb_list.extend(orb_list)
        tot_orbstr_list.extend(orbstr_list)
    return tot_orb_list, tot_orbstr_list

class QM9Rowan_shard(LMDBShard_maker_db):
    def __init__(
        self,
        root_path: str,
        shard_num: int,
        save_path=None,
        max_workers=8,
        use_parallel=True,
        processd_dir_name="processed",
        shard_dir_name="lmdbs",
        split="random",
        *args,
        **kwargs,
    ):
        super().__init__(
            root_path=root_path,
            shard_num=shard_num,
            save_path=save_path,
            max_workers=max_workers,
            use_parallel=use_parallel,
            processd_dir_name=processd_dir_name,
            shard_dir_name=shard_dir_name,
            *args,
            **kwargs,
        )
        self.split = split
        assert self.split in ["random", "size_ood"], f"Split {self.split} for QM9Rowan is not supported"
        if self.split == "random":
            self.split_path = os.path.join(self.save_path, "processed_QM9Rowan_random_12.json")
        elif self.split == "size_ood":
            self.split_path = os.path.join(self.save_path, "processed_QM9Rowan_size_ood.json")

        self.convention_dict = get_convention_dict()
        # if cal_orbital_and_energies is True, data will contain orbital energies and coefficients

    def _make_split_info(self):
        if os.path.exists(self.split_path):
            pass
        else:
            if self.split == "random":
                logger.info("Making random split")
                self.split_info = {
                    "train": 0.8,
                    "val": 0.1,
                    "test": 0.1,
                }
                total_len = self._get_total_len()
                train_len = int(total_len * self.split_info["train"])
                val_len = int(total_len * self.split_info["val"])
                test_len = total_len - train_len - val_len
                indices = np.random.RandomState(seed=43).permutation(total_len)
                train_indices = indices[:train_len]
                val_indices = indices[train_len:train_len+val_len]
                test_indices = indices[train_len+val_len:]

            # elif self.split == "size_ood":
            #     """ 
            #     size OOD split:
            #         train: N <= 20
            #         val: 21 <= N <= 22
            #         test: N >= 23
            #     """
            #     logger.info("Making size OOD split...")
            #     assert self.all_data is not None, "all_data is not set"
            #     num_nodes_list = [row[1] for row in self.all_data]
            #     num_nodes_array = np.array(num_nodes_list)
            #     total_len = len(num_nodes_array)
            #     train_indices = np.where(num_nodes_array <= 20)
            #     val_indices = np.where(np.logical_and(num_nodes_array >= 21, num_nodes_array <= 22))
            #     test_indices = np.where(num_nodes_array >= 23)

            assert len(train_indices) + len(val_indices) + len(test_indices) == total_len, f"Total length {total_len} is not equal to the sum of train, val, and test length {len(train_indices) + len(val_indices) + len(test_indices)}"
            self.split_info = {
                "train": sorted(train_indices.tolist()),
                "val": sorted(val_indices.tolist()),
                "test": sorted(test_indices.tolist()),
            }
            with open(self.split_path, "w") as f:
                json.dump(self.split_info, f)

    def process_data(self, key_data_pair):
        """
        Process a single row from the database.
        
        Args:
            data: Database row containing molecular data
            
        Returns:
            tuple: (key, data_dict) for LMDB storage

        Note:
            QM9Rowan.db data format:
                Column 0: id (INTEGER) NULL PRIMARY KEY
                Column 1: molecule_id (TEXT) NULL 
                Column 2: filename (TEXT) NULL 
                Column 3: num_atoms (INTEGER) NULL 
                Column 4: atomic_numbers (BLOB) NULL 
                Column 5: atomic_symbols (TEXT) NULL 
                Column 6: positions (BLOB) NULL 
                Column 7: units (TEXT) NULL 
                Column 8: energy (REAL) NULL 
                Column 9: forces (BLOB) NULL 
                Column 10: total_time (REAL) NULL 
                Column 11: energy_time (REAL) NULL 
                Column 12: gradient_time (REAL) NULL 
                Column 13: hamiltonian (BLOB) NULL 
                Column 14: overlap (BLOB) NULL 
                Column 15: coefficients (BLOB) NULL 
                Column 16: epsilon (BLOB) NULL 
                Column 17: density_matrix (BLOB) NULL 
                Column 18: matrix_dim (INTEGER) NULL 
                Column 19: num_electrons (INTEGER) NULL 
                Column 20: full_data_blob (BLOB) NULL 
                
                # 0: id (INTEGER, np.int32) (PRIMARY KEY) NOT NULL index
                # 1: N (INTEGER, np.int32) number of atoms
                # 2: Z (BLOB, np.int32) atomic numbers
                # 3: pos (BLOB, np.float64) atomic positions / unit: angstrom 
                # 4: Ham (BLOB, np.float64) hamiltonian matrix / unit: hartree / pyscf convention
      """

        data, data_idx = key_data_pair
        key = int(data_idx).to_bytes(length=4, byteorder="big") # real key is not used
        # atoms = np.frombuffer(data[4], np.int32)
        # pos = np.frombuffer(data[6], np.float64)
        atoms = pickle.loads(data[4])
        pos = pickle.loads(data[6])

        ovlp, init_ham, mf = calc_overlap_and_init_hamiltonian(
            atoms,
            pos.reshape(-1, 3),
            unit="ang",
            out_mf=True,
            basis="def2-tzvppd",
            # xc="ωB97M-D3BJ",
            xc="wb97m-v",
        )
        hamiltonian = pickle.loads(data[13])
        hamiltonian = _matrix_transform_single(
            torch.from_numpy(hamiltonian),
            atoms,
            self.convention_dict["psi4_def2-tzvppd_to_pyscf"]
            ).numpy()

        ovlp = pickle.loads(data[14])
        
        # Debugging
        # transformed_overlap = _matrix_transform_single(torch.from_numpy(overlap), atoms, self.convention_dict["psi4_def2-tzvppd_to_pyscf"])
        # transformed_overlap = transformed_overlap.numpy()
        # print(abs(ovlp - transformed_overlap).mean())
                
        h_dim = ovlp.shape[0]
        hamiltonian = hamiltonian.reshape(h_dim, h_dim)
        orbital_energies, orbital_coefficients = cal_orbital_and_energies(
            torch.from_numpy(ovlp).unsqueeze(0).to(torch.float64), 
            torch.from_numpy(hamiltonian).unsqueeze(0).to(torch.float64)
            )
        orbital_coefficients = orbital_coefficients.squeeze()
        dm0 = calc_dm0(atoms, orbital_coefficients)
        
        orbital_coefficients = orbital_coefficients.cpu().numpy()
        orbital_energies = orbital_energies.cpu().numpy()
        # dft_energy = mf.energy_tot(dm0)

        # # Calculate DFT forces
        # grad_frame = mf.nuc_grad_method()
        # mo_occ = mf.get_occ(orbital_energies.squeeze(), orbital_coefficients.squeeze())
        # dft_forces = -grad_frame.kernel(mo_energy=orbital_energies.squeeze(), mo_coeff=orbital_coefficients.squeeze(), mo_occ=mo_occ)

        # dft_energy = pickle.loads(data[8])
        dft_energy = data[8]
        dft_forces = pickle.loads(data[9])
        
        packed_hamiltonian, h_dim = self.pack_upper_triangle(hamiltonian) # h_dim is the dimension of the hamiltonian matrix
        packed_ovlp, _ = self.pack_upper_triangle(ovlp)
        packed_init_ham, _ = self.pack_upper_triangle(init_ham)
        # packed_dm0, _ = self.pack_upper_triangle(dm0)
        
        # orbital_coefficients is not symmetric, so we do not pack it
        # packed_orbital_coefficients, _ = self.pack_upper_triangle(orbital_coefficients)

        ori_data_dict = {
            "id": data[0],
            "num_nodes": data[3],
            "atoms": atoms.tobytes(),
            "pos": pos.tobytes(),  # unit: angstrom
            "dft_energy": dft_energy, # unit: Eh
            "dft_forces": dft_forces.tobytes(), # unit: Eh/Bohr
            "h_dim": h_dim,
            "packed_hamiltonian": packed_hamiltonian.tobytes(), # unit: Eh
            "packed_overlap": packed_ovlp.tobytes(),
            "packed_initial_hamiltonian": packed_init_ham.tobytes(), # unit: Eh
            "orbital_energies": orbital_energies.tobytes(), # unit: Eh
            # "packed_orbital_coefficients": packed_orbital_coefficients.tobytes(),
            "orbital_coefficients": orbital_coefficients.tobytes(),
            "dm0": dm0.tobytes(),
            "pos_unit": "Angstrom",
            "basis": "def2-tzvppd",
            "xc": "wb97m-D3BJ",
            "convention": "pyscf",
            # "packed_dm0": packed_dm0.tobytes(),
        }
        data_dict = pickle.dumps(ori_data_dict)
        return key, data_dict

class QM9Rowan(InMemoryDataset):
    url = "https://drive.google.com/file/d/1LcEJGhB8VUGkuyb0oQ_9ANJdSkky9xMS/view?usp=sharing"

    def __init__(
        self,
        root="datasets/",
        split="random",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        prefix="",
        shard_num=60,
        shard_idx=-1,
        max_workers_preprocess=8,
        use_parallel_preprocess=False,
        return_orbital_and_energies=False,
    ):
        
        self.folder = os.path.join(root, "QM9Rowan" + prefix)
        self.split = split
        self.processd_dir_name = "processed"
        self.shard_dir_name = "lmdbs"
        self._processed_path = os.path.join(self.folder, self.processd_dir_name)

        self.shard_num = shard_num
        self.shard_idx = shard_idx
        self.max_workers_preprocess = max_workers_preprocess
        self.use_parallel_preprocess = use_parallel_preprocess

        self.return_orbital_and_energies = return_orbital_and_energies

        assert self.split in ["random", "size_ood"], f"Split {self.split} for QM9Rowan is not supported"
        if self.split == "random":
            self.split_path = os.path.join(self.folder, "processed_QM9Rowan_random_12.json")
        elif self.split == "size_ood":
            self.split_path = os.path.join(self.folder, "processed_QM9Rowan_size_ood.json")

        self.lmdb_path_list = [os.path.join(self._processed_path,self.shard_dir_name, f"shard_{i:03d}.lmdb") for i in range(self.shard_num)]
        self.atom_list = ["H", "C", "N", "O", "F"]
        self.atom_num = {
            "H": 1,
            "C": 6,
            "N": 7,
            "O": 8,
            "F": 9,
        }
        
        self.full_orbitals = 14
        self.orbital_mask = {}

        # Define orbital indices for 1s, 2s, and 2p orbitals (indices 0(1s), 1(2s), 3(2p), 4(2p), 5(2p))
        # This corresponds to the minimal basis set for light elements (H, He)
        
        # Create orbital masks for different atomic numbers
        # orbital_mask_line1: minimal orbital set for hydrogen and helium (atomic numbers 1-2)
        # orbital_mask_line2: full orbital set for heavier elements (atomic numbers 3-10)
        # orbital indices
        # 1s [0]
        # 2s [1], 2p [3, 4, 5]
        # 3s [2], 3p [6, 7, 8], 3d [9, 10, 11, 12, 13]        
        orbital_mask_line1 = torch.tensor([0, 1, 3, 4, 5])
        orbital_mask_line2 = torch.arange(self.full_orbitals)
        
        # Assign appropriate orbital masks based on atomic number
        self.orbital_mask[self.atom_num["H"]] = orbital_mask_line1 # H: 1s, 2s, 2p
        self.orbital_mask[self.atom_num["C"]] = orbital_mask_line2 # C: 1s, 2s, 2p, 3s, 3p, 3d
        self.orbital_mask[self.atom_num["N"]] = orbital_mask_line2 # N: 1s, 2s, 2p, 3s, 3p, 3d
        self.orbital_mask[self.atom_num["O"]] = orbital_mask_line2 # O: 1s, 2s, 2p, 3s, 3p, 3d
        self.orbital_mask[self.atom_num["F"]] = orbital_mask_line2 # F: 1s, 2s, 2p, 3s, 3p, 3d

        self.Q_dict = Onsite_3idx_Overlap_Integral(atom_list=self.atom_list, basis="def2-svp").Q_table()
        self.convention_dict = get_convention_dict()
        
        # LMDB connection caching for performance optimization
        self._db_envs = {}  # Cache for LMDB environments by shard index
        self.shard_idx_list = [] # Mapping from data index to shard index
        
        super(QM9Rowan, self).__init__(self.folder, transform, pre_transform, pre_filter)

        self._load_split_info()
        self._load_index_info()

    def _load_split_info(self):
        # load the json split info
        with open(self.processed_paths[0], "r") as f:
            self.split_info = json.load(f)  
    
        self.train_mask = torch.tensor(self.split_info["train"], dtype=torch.int64)
        self.val_mask = torch.tensor(self.split_info["val"], dtype=torch.int64)
        self.test_mask = torch.tensor(self.split_info["test"], dtype=torch.int64)

        self.slices = {
            "id": torch.arange(
                self.train_mask.shape[0]
                + self.val_mask.shape[0]
                + self.test_mask.shape[0]
                + 1
            )
        }

    def _load_index_info(self):
        with open(os.path.join(self._processed_path, "index.json"), "r") as f:
            self.index_info = json.load(f)
        self.index_info = self.index_info["index"]
        self.shard_idx_list = []
        self.shard_data_idx_list = []
        for idx, index_info in enumerate(self.index_info):
            shard_idx, cur_idx, shard_data_idx = index_info
            assert cur_idx == idx, f"Shard index {cur_idx} is not equal to the index {idx}"
            self.shard_idx_list.append(shard_idx)
            self.shard_data_idx_list.append(shard_data_idx)
        self.shard_idx_list = torch.tensor(self.shard_idx_list, dtype=torch.int64)
        max_shard_idx = torch.max(self.shard_idx_list)
        assert max_shard_idx == self.shard_num - 1, f"Max shard index {max_shard_idx} is not equal to the number of shards {self.shard_num}"

    @property
    def raw_file_names(self):
        return [f"QM9Rowan.db"]

    @property
    def processed_file_names(self):
        if self.split == "random":
            return [
                "processed_QM9Rowan_random_12.json",
                "ALL_SHARDS_COMPLETED.txt",
                "index.json",
            ]
        elif self.split == "size_ood":
            return [
                "processed_QM9Rowan_size_ood.json",
                "ALL_SHARDS_COMPLETED.txt",
                "index.json",
            ]

    def download(self):
        try:
            print(f"Downloading the QM9Rowan dataset to through {self.url}")
            gdown.download(self.url, output=self.raw_paths[0], fuzzy=True)
        except:
            print(
                f"Downloading failed! Please download the QM9Rowan dataset to {self.raw_paths[0]} through {self.url}"
            )
            print(f"Or you can try to download the zip file through {GoogleDriveLink}")
            raise FileNotFoundError(f"QM9Rowan needs to be downloaded.")

    def process(self):
        self.QM9Rowan_shard = QM9Rowan_shard(
            root_path=self.raw_paths[0],
            shard_num=self.shard_num,
            save_path=self.folder,
            max_workers=self.max_workers_preprocess,
            processd_dir_name=self.processd_dir_name,
            shard_dir_name=self.shard_dir_name,
            use_parallel=self.use_parallel_preprocess,
            split=self.split,
        )
        if self.shard_idx == -1 or self.shard_idx is None:
            logger.info(f"Processing all shards of QM9Rowan dataset")
            self.QM9Rowan_shard.process()
        else:
            logger.info(f"Processing QM9Rowan dataset with shard_idx: {self.shard_idx}")
            self.QM9Rowan_shard.process(self.shard_idx)
            
    def _get_shard_db_env(self, idx):
        """Get LMDB environment with caching for performance optimization."""
        shard_idx = self.shard_idx_list[idx]
        
        # Return cached environment if available
        if shard_idx in self._db_envs:
            try:
                # Test if the environment is still valid
                with self._db_envs[shard_idx].begin() as txn:
                    txn.stat()  # This will raise an exception if the env is invalid
                return self._db_envs[shard_idx]
            except Exception:
                # Environment is invalid, remove it from cache
                try:
                    self._db_envs[shard_idx].close()
                except:
                    pass
                del self._db_envs[shard_idx]
        
        # Create new environment and cache it
        db_env = lmdb.open(
            self.lmdb_path_list[shard_idx], 
            readonly=True, 
            lock=False,
            max_readers=1024,  # Increase max readers
            readahead=False    # Disable readahead for better concurrent access
        )
        self._db_envs[shard_idx] = db_env
        return db_env
    
    def get(self, idx):
        """Optimized data loading: Reuse LMDB connection and minimize unnecessary operations."""
        try:
            return self._get(idx)
        except Exception as e:
            # If there's an error, try to refresh the LMDB environment
            logger.warning(f"Error accessing LMDB for idx {idx}: {e}. Attempting to refresh environment.")
            shard_idx = self.shard_idx_list[idx]
            if shard_idx in self._db_envs:
                try:
                    self._db_envs[shard_idx].close()
                except:
                    pass
                del self._db_envs[shard_idx]
            
            # Retry with fresh environment
            return self._get(idx)
    
    def _get(self, idx):
        # Get cached LMDB environment (no need for context manager since we're reusing connections)        
        db_env = self._get_shard_db_env(idx)
        with db_env.begin() as txn:
            key = int(idx).to_bytes(length=4, byteorder="big")
            data_dict = txn.get(key)
            
            if data_dict is None:
                raise KeyError(f"Index idx {idx}, shard_data_idx {self.shard_data_idx_list[idx]} not found in database {self.shard_idx_list[idx]}")
                
            data_dict = pickle.loads(data_dict)
            # data_dict, save = self._fix_data_dict(data_dict)
            # if save:
            #     txn.put(key, pickle.dumps(data_dict))
            
            data = self.get_mol(data_dict, orb_energy_and_coeff=self.return_orbital_and_energies)
        return data

    def _fix_data_dict(self, data_dict):
        # Fix the data dict if the orbital_coefficients is not in the data dict
        save = False
        if "orbital_coefficients" not in data_dict.keys():
            save = True
            h_dim = data_dict["h_dim"] # sum of orbital dimensions
            packed_hamiltonian = np.frombuffer(data_dict["packed_hamiltonian"], np.float64)
            packed_overlap = np.frombuffer(data_dict["packed_overlap"], np.float64)
            hamiltonian = torch.from_numpy(self.unpack_upper_triangle(packed_hamiltonian, h_dim)).to(torch.float64)
            overlap_matrix = torch.from_numpy(self.unpack_upper_triangle(packed_overlap, h_dim)).to(torch.float64)

            orbital_energies, orbital_coefficients = cal_orbital_and_energies(
                overlap_matrix.unsqueeze(0),
                hamiltonian.unsqueeze(0),
            )
            data_dict["orbital_coefficients"] = orbital_coefficients.numpy().tobytes()
        return data_dict, save
    
    
    def get_key_list(self, idx):
        """Get the key list of the shard (for debugging)"""
        db_env = self._get_shard_db_env(idx)
        with db_env.begin() as txn:
            key_list = []
            for key, value in txn.cursor():
                int_key = int.from_bytes(key, byteorder="big")
                key_list.append(int_key)
        return key_list
    
    def _close_db_envs(self):
        """Safely close all cached LMDB environments."""
        for shard_idx, db_env in list(self._db_envs.items()):
            try:
                db_env.close()
            except Exception as e:
                logger.warning(f"Error closing LMDB environment for shard {shard_idx}: {e}")
        self._db_envs.clear()
    
    def __del__(self):
        """Destructor: Clean up all LMDB environments."""
        self._close_db_envs()
    
    def __enter__(self):
        """Context manager entry: Initialize LMDB environments."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: Clean up all LMDB environments."""
        self._close_db_envs()
    
    @staticmethod
    def unpack_upper_triangle(packed: np.ndarray, h_dim: int):
        return unpack_upper_triangle(packed, h_dim)
    
    def matrix_transform(self, hamiltonian, atoms, convention="pyscf_def2-tzvppd_to_e3nn"):
        return _matrix_transform_single(hamiltonian, atoms, self.convention_dict[convention])
      
    def get_mol(self, data_dict, orb_energy_and_coeff=False):
        num_nodes = torch.tensor(data_dict["num_nodes"], dtype=torch.int64)
        atoms = torch.tensor(np.frombuffer(data_dict["atoms"], np.int32), dtype=torch.int64)
        pos = torch.tensor(np.frombuffer(data_dict["pos"], np.float64).reshape(-1, 3), dtype=torch.float64)
        dft_energy = torch.tensor(data_dict["dft_energy"], dtype=torch.float64)
        dft_forces = torch.tensor(np.frombuffer(data_dict["dft_forces"], np.float64).reshape(-1, 3), dtype=torch.float64)
        h_dim = data_dict["h_dim"] # sum of orbital dimensions
        
        # Optimize memory usage: unpack matrices directly to tensors without intermediate numpy arrays
        packed_hamiltonian = np.frombuffer(data_dict["packed_hamiltonian"], np.float64)
        packed_overlap = np.frombuffer(data_dict["packed_overlap"], np.float64)
        packed_initial_hamiltonian = np.frombuffer(data_dict["packed_initial_hamiltonian"], np.float64)
        # packed_dm0 = np.frombuffer(data_dict["packed_dm0"], np.float64) # [h_dim, h_dim]
        
        # Direct tensor creation from unpacked matrices to reduce memory overhead
        hamiltonian = torch.from_numpy(self.unpack_upper_triangle(packed_hamiltonian, h_dim)).to(torch.float64)
        overlap_matrix = torch.from_numpy(self.unpack_upper_triangle(packed_overlap, h_dim)).to(torch.float64)
        initial_hamiltonian = torch.from_numpy(self.unpack_upper_triangle(packed_initial_hamiltonian, h_dim)).to(torch.float64)
        # dm0 = torch.from_numpy(self.unpack_upper_triangle(packed_dm0, h_dim)).to(torch.float64)
        
        convention = "pyscf_def2svp_to_e3nn"
        # stack in 0th dimension
        concat_feat = torch.stack([hamiltonian, overlap_matrix, initial_hamiltonian], dim=0)
        transformed_concat_feat = self.matrix_transform(concat_feat, atoms, convention=convention)
        concat_diagonal_blocks, concat_non_diagonal_blocks, concat_diagonal_masks, concat_non_diagonal_masks, edge_index_full = _cut_matrix_3d(transformed_concat_feat, atoms, self.orbital_mask, self.full_orbitals)

        # slice the concat_diagonal_blocks and concat_non_diagonal_blocks
        diagonal_hamiltonian = concat_diagonal_blocks[:,0]
        non_diagonal_hamiltonian = concat_non_diagonal_blocks[:,0]
        diagonal_hamiltonian_mask = concat_diagonal_masks[:,0]
        non_diagonal_hamiltonian_mask = concat_non_diagonal_masks[:,0]
      
        diagonal_overlap = concat_diagonal_blocks[:,1]
        non_diagonal_overlap = concat_non_diagonal_blocks[:,1]
  
        diagonal_init_ham = concat_diagonal_blocks[:,2]
        non_diagonal_init_ham = concat_non_diagonal_blocks[:,2]
  
                
        # Optimize Q tensor creation: use list comprehension for better memory efficiency
        Q_blocks = []
        for l in range(60):
            block_diag_components = [self.Q_dict[z.item()][l] for z in atoms]
            Q_blocks.append(torch.block_diag(*block_diag_components))
        
        Q = torch.stack(Q_blocks)  # [60, h_dim, h_dim]
        Q = self.matrix_transform(Q, atoms, convention="pyscf_def2svp_to_e3nn").permute(1, 2, 0) #[h_dim, h_dim, 60]
        
        # Optimize tensor reshaping: use in-place operations where possible
        Q_slice = Q[:, :, 16:40]
        Q[:, :, 16:40] = Q_slice.reshape(h_dim, h_dim, -1, 3)[:, :, :, [1, 2, 0]].reshape(h_dim, h_dim, 24)
        
        diagonal_Q, non_diagonal_Q, _, _, _ = _cut_matrix_3d_last(Q, atoms, self.orbital_mask, self.full_orbitals)

        data = Data(
            pos=pos,
            atoms=atoms.view(-1, 1),
            diagonal_hamiltonian=diagonal_hamiltonian,
            non_diagonal_hamiltonian=non_diagonal_hamiltonian,
            diagonal_hamiltonian_mask=diagonal_hamiltonian_mask,
            non_diagonal_hamiltonian_mask=non_diagonal_hamiltonian_mask,
            diagonal_init_ham=diagonal_init_ham,
            non_diagonal_init_ham=non_diagonal_init_ham,
            diagonal_overlap=diagonal_overlap,
            non_diagonal_overlap=non_diagonal_overlap,
            diagonal_Q=diagonal_Q,
            non_diagonal_Q=non_diagonal_Q,
            edge_index_full=edge_index_full,
            dft_energy=dft_energy.view(1, 1),
            dft_forces=dft_forces,
            num_nodes=num_nodes,
            h_dim=torch.tensor(h_dim, dtype=torch.int64),
        )
        
        if orb_energy_and_coeff:    # orbital_coefficients was not symmetric, so saved data is corrupted
            # Optimize orbital data loading: direct tensor creation from buffer
            orbital_energies = np.frombuffer(data_dict["orbital_energies"], np.float64)
            data.orbital_energies = torch.from_numpy(orbital_energies.copy()).to(torch.float64)


            if "orbital_coefficients" in data_dict.keys():
                orbital_coefficients = np.frombuffer(data_dict["orbital_coefficients"], np.float64).reshape(h_dim, h_dim)
            else:
                orbital_energies, orbital_coefficients = cal_orbital_and_energies(
                    overlap_matrix.unsqueeze(0),
                    hamiltonian.unsqueeze(0),
                )
                orbital_coefficients = orbital_coefficients.numpy().reshape(h_dim, h_dim)

            data.orbital_coefficients = orbital_coefficients

            # packed_orbital_coefficients = np.frombuffer(data_dict["packed_orbital_coefficients"], np.float64)
            # orbital_coefficients = self.unpack_upper_triangle(packed_orbital_coefficients, h_dim)
            # orbital_coefficients = torch.from_numpy(orbital_coefficients.copy()).to(torch.float64).unsqueeze(0)
            # diagonal_orbital_coefficients, non_diagonal_orbital_coefficients, _, _, _ = _cut_matrix_3d(orbital_coefficients, atoms, self.orbital_mask, self.full_orbitals)
            # data.diagonal_orbital_coefficients = diagonal_orbital_coefficients[:,0]
            # data.non_diagonal_orbital_coefficients = non_diagonal_orbital_coefficients[:,0]

            # data.orbital_coefficients = orbital_coefficients.numpy()
            # data.np_hamiltonian = hamiltonian.numpy()

        return data
    
########################################################
# Main function & Argument parser
########################################################

def parse_shard_idx(shard_idx_str):
    """Parse shard_idx string into a list of integers"""
    if shard_idx_str == "-1":
        return int(-1)
    elif ',' in shard_idx_str:
        # Comma-separated list: "0,1,2,3"
        return [int(x.strip()) for x in shard_idx_str.split(',')]
    elif '-' in shard_idx_str and shard_idx_str.count('-') == 1:
        # Range: "0-5"
        start, end = map(int, shard_idx_str.split('-'))
        return list(range(start, end + 1))
    else:
        # Single integer: "0"
        return [int(shard_idx_str)]

if __name__ == "__main__":
    setup_global_logger()

    import argparse

    parser = argparse.ArgumentParser(description="Generation")
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset"),
    )
    parser.add_argument("--name", type=str, default="QM9Rowan")
    parser.add_argument("--version", type=str, default="300k")
    parser.add_argument("--split", type=str, default="random")
    parser.add_argument("--dynamic_split", type=str, default="geometry")
    parser.add_argument("--shard_num", type=int, default=10, help="Number of shards for the dataset, for example, 10 shards for QM9Rowan dataset")
    parser.add_argument("--shard_idx", type=str, default="0", help="-1 for all shards, Comma-separated list of shard indices or single index ex) 0,1,2,3 or 0-5")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--pdb", action="store_true", default=False)

    args = parser.parse_args()
    args.shard_idx = parse_shard_idx(args.shard_idx)
    torch.set_num_threads(4)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"

    logger.info(f"OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS']}")
    logger.info(f"MKL_NUM_THREADS: {os.environ['MKL_NUM_THREADS']}")
    logger.info(f"NUMEXPR_NUM_THREADS: {os.environ['NUMEXPR_NUM_THREADS']}")
    logger.info(f"OPENBLAS_NUM_THREADS: {os.environ['OPENBLAS_NUM_THREADS']}")

    logger.info(f"dataset path: {args.root}")

    logger.info(f"Processing {args.name} dataset with shard_idx: {args.shard_idx}")

    assert args.name in ["QM9Rowan"]
    if args.name == "QM9Rowan":
        assert args.split in ["random", "size_ood"]
        dataset = QM9Rowan(
            root=args.root,
            split=args.split,
            shard_num=args.shard_num,
            shard_idx=args.shard_idx,
            prefix=args.prefix,
        )
        
    # Compare with split dataset
    if args.pdb:
        import pdb
        pdb.set_trace()

    print(len(dataset))
    print(dataset[0])
    print(dataset[-1])
    for i in range(len(dataset)):
        data = dataset[i]