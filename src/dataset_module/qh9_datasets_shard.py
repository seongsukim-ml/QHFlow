import numpy as np
import lmdb
import pickle
import os
import logging
import json
import gdown
import torch
from tqdm import tqdm
import random
from typing import Union, List

from common.metric import cal_orbital_and_energies
from common.matrix_transforms import pack_upper_triangle, unpack_upper_triangle, _matrix_transform_single, get_convention_dict, _cut_matrix_3d, _cut_matrix_3d_last
from dataset_module.lmdb_shard import LMDBShard_maker_db
from dataset_module.data_dft_utils import calc_overlap_and_init_hamiltonian, calc_dm0

from torch_geometric.data import InMemoryDataset, Data
from utils import AOData, Onsite_3idx_Overlap_Integral, build_molecule, build_AO_index
#

# Conversion factor from Bohr to Angstrom
ANG2BOHR = 1.8897259886
BOHR2ANG = 1 / ANG2BOHR # 0.52917721067 - Bohr to Angstrom conversion (MD17)

GoogleDriveLink = (
    "https://drive.google.com/drive/u/0/folders/1LXTC8uaOQzmb76FsuGfwSocAbK5Hshfj"
)

########################################################
# QH9Stable
########################################################


class QH9Stable_shard(LMDBShard_maker_db):
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
        cal_orbital_and_energies=False,
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
        assert self.split in ["random", "size_ood"], f"Split {self.split} for QH9Stable is not supported"
        if self.split == "random":
            self.split_path = os.path.join(self.save_path, "processed_QH9Stable_random_12.json")
        elif self.split == "size_ood":
            self.split_path = os.path.join(self.save_path, "processed_QH9Stable_size_ood.json")

        # if cal_orbital_and_energies is True, data will contain orbital energies and coefficients
        self.cal_orbital_and_energies = cal_orbital_and_energies

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

            elif self.split == "size_ood":
                """ 
                size OOD split:
                    train: N <= 20
                    val: 21 <= N <= 22
                    test: N >= 23
                """
                logger.info("Making size OOD split...")
                assert self.all_data is not None, "all_data is not set"
                num_nodes_list = [row[1] for row in self.all_data]
                num_nodes_array = np.array(num_nodes_list)
                total_len = len(num_nodes_array)
                train_indices = np.where(num_nodes_array <= 20)
                val_indices = np.where(np.logical_and(num_nodes_array >= 21, num_nodes_array <= 22))
                test_indices = np.where(num_nodes_array >= 23)

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
            QH9Stable.db data format:
                0: id (INTEGER, np.int32) (PRIMARY KEY) NOT NULL index
                1: N (INTEGER, np.int32) number of atoms
                2: Z (BLOB, np.int32) atomic numbers
                3: pos (BLOB, np.float64) atomic positions / unit: angstrom 
                4: Ham (BLOB, np.float64) hamiltonian matrix / unit: hartree / pyscf convention
      """
        data, data_idx = key_data_pair
        key = int(data_idx).to_bytes(length=4, byteorder="big") # real key is not used
        atoms = np.frombuffer(data[2], np.int32)
        pos = np.frombuffer(data[3], np.float64)

        ovlp, init_ham, mf = calc_overlap_and_init_hamiltonian(atoms, pos.reshape(-1, 3), out_mf=True)
        hamiltonian = np.frombuffer(data[4], np.float64) # flattened hamiltonian matrix
        h_dim = ovlp.shape[0]
        hamiltonian = hamiltonian.reshape(h_dim, h_dim)
        orbital_energies, orbital_coefficients = cal_orbital_and_energies(torch.from_numpy(ovlp).unsqueeze(0).to(torch.float64), torch.from_numpy(init_ham).unsqueeze(0).to(torch.float64))
        orbital_coefficients = orbital_coefficients.squeeze()
        dm0 = calc_dm0(atoms, orbital_coefficients)
        
        orbital_coefficients = orbital_coefficients.cpu().numpy()
        orbital_energies = orbital_energies.cpu().numpy()
        dft_energy = mf.energy_tot(dm0)
        
        packed_hamiltonian, h_dim = self.pack_upper_triangle(hamiltonian) # h_dim is the dimension of the hamiltonian matrix
        packed_ovlp, _ = self.pack_upper_triangle(ovlp)
        packed_init_ham, _ = self.pack_upper_triangle(init_ham)
        packed_dm0, _ = self.pack_upper_triangle(dm0)
        packed_orbital_coefficients, _ = self.pack_upper_triangle(orbital_coefficients)

        ori_data_dict = {
            "id": data[0],
            "num_nodes": data[1],
            "atoms": data[2],
            "pos": data[3],  # angstrom
            "dft_energy": dft_energy,
            "h_dim": h_dim,
            "packed_hamiltonian": packed_hamiltonian.tobytes(),
            "packed_overlap": packed_ovlp.tobytes(),
            "packed_initial_hamiltonian": packed_init_ham.tobytes(),
            "orbital_energies": orbital_energies.tobytes(),
            "packed_orbital_coefficients": packed_orbital_coefficients.tobytes(),
            "packed_dm0": packed_dm0.tobytes(),
        }
        data_dict = pickle.dumps(ori_data_dict)
        return key, data_dict

class QH9Stable(InMemoryDataset):
    url = "https://drive.google.com/file/d/1LcEJGhB8VUGkuyb0oQ_9ANJdSkky9xMS/view?usp=sharing"

    def __init__(
        self,
        root="datasets/",
        split="random",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        prefix="",
        shard_num=30,
        shard_idx=-1,
        max_workers_preprocess=8,
        use_parallel_preprocess=False,
    ):
        
        self.folder = os.path.join(root, "QH9Stable" + prefix)
        self.split = split
        self.processd_dir_name = "processed"
        self.shard_dir_name = "lmdbs"
        self._processed_path = os.path.join(self.folder, self.processd_dir_name)

        self.shard_num = shard_num
        self.shard_idx = shard_idx
        self.max_workers_preprocess = max_workers_preprocess
        self.use_parallel_preprocess = use_parallel_preprocess

        assert self.split in ["random", "size_ood"], f"Split {self.split} for QH9Stable is not supported"
        if self.split == "random":
            self.split_path = os.path.join(self.folder, "processed_QH9Stable_random_12.json")
        elif self.split == "size_ood":
            self.split_path = os.path.join(self.folder, "processed_QH9Stable_size_ood.json")

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
        idx_1s_2s_2p = torch.tensor([0, 1, 3, 4, 5])
        
        # Create orbital masks for different atomic numbers
        # orbital_mask_line1: minimal orbital set for hydrogen and helium (atomic numbers 1-2)
        # orbital_mask_line2: full orbital set for heavier elements (atomic numbers 3-10)
        # orbital indices
        # 1s [0]
        # 2s [1], 2p [3, 4, 5]
        # 3s [2], 3p [6, 7, 8], 3d [9, 10, 11, 12, 13]        
        orbital_mask_line1 = idx_1s_2s_2p
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
        
        super(QH9Stable, self).__init__(self.folder, transform, pre_transform, pre_filter)

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
        return [f"QH9Stable.db"]

    @property
    def processed_file_names(self):
        if self.split == "random":
            return [
                "processed_QH9Stable_random_12.json",
                "ALL_SHARDS_COMPLETED.txt",
                "index.json",
            ]
        elif self.split == "size_ood":
            return [
                "processed_QH9Stable_size_ood.json",
                "ALL_SHARDS_COMPLETED.txt",
                "index.json",
            ]

    def download(self):
        try:
            print(f"Downloading the QH9Stable dataset to through {self.url}")
            gdown.download(self.url, output=self.raw_paths[0], fuzzy=True)
        except:
            print(
                f"Downloading failed! Please download the QH9Stable dataset to {self.raw_paths[0]} through {self.url}"
            )
            print(f"Or you can try to download the zip file through {GoogleDriveLink}")
            raise FileNotFoundError(f"QH9Stable needs to be downloaded.")

    def process(self):
        self.QH9Stable_shard = QH9Stable_shard(
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
            logger.info(f"Processing all shards of QH9Stable dataset")
            self.QH9Stable_shard.process()
        else:
            logger.info(f"Processing QH9Stable dataset with shard_idx: {self.shard_idx}")
            self.QH9Stable_shard.process(self.shard_idx)
            
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
                print(self.get_key_list(idx))
                raise KeyError(f"Index idx: {idx}, shard_data_idx: {self.shard_data_idx_list[idx]} not found in database {self.shard_idx_list[idx]}")
                
            data_dict = pickle.loads(data_dict)
            data = self.get_mol(data_dict, orb_energy_and_coeff=True)
        return data
    
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
    
    def matrix_transform(self, hamiltonian, atoms, convention="pyscf_def2svp_to_e3nn"):
        return _matrix_transform_single(hamiltonian, atoms, self.convention_dict[convention])
      
    def get_mol(self, data_dict, orb_energy_and_coeff=False):
        num_nodes = torch.tensor(data_dict["num_nodes"], dtype=torch.int64)
        atoms = torch.tensor(np.frombuffer(data_dict["atoms"], np.int32), dtype=torch.int64)
        pos = torch.tensor(np.frombuffer(data_dict["pos"], np.float64).reshape(-1, 3), dtype=torch.float64)
        dft_energy = torch.tensor(data_dict["dft_energy"], dtype=torch.float64)
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
            num_nodes=num_nodes.view(1, 1),
            h_dim=torch.tensor(h_dim, dtype=torch.int64).view(1, 1),
        )
        
        if orb_energy_and_coeff:    
            # Optimize orbital data loading: direct tensor creation from buffer
            orbital_energies = np.frombuffer(data_dict["orbital_energies"], np.float64)
            packed_orbital_coefficients = np.frombuffer(data_dict["packed_orbital_coefficients"], np.float64)
            orbital_coefficients = self.unpack_upper_triangle(packed_orbital_coefficients, h_dim)
            data.orbital_energies = torch.from_numpy(orbital_energies).to(torch.float64)
            data.orbital_coefficients = torch.from_numpy(orbital_coefficients).to(torch.float64).unsqueeze(0)

        return data
    
########################################################
# QH9Dynamic
########################################################

class QH9Dynamic_shard(LMDBShard_maker_db):
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
        cal_orbital_and_energies=False,
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
        assert self.split in ["geometry", "mol"], f"Split {self.split} for QH9Dynamic is not supported"
        if self.split == "geometry":
            self.split_path = os.path.join(self.save_path, "processed_QH9Dynamic_geometry.json")
        elif self.split == "mol":
            self.split_path = os.path.join(self.save_path, "processed_QH9Dynamic_mol.json")

        # if cal_orbital_and_energies is True, data will contain orbital energies and coefficients
        self.cal_orbital_and_energies = cal_orbital_and_energies

    def _make_split_info(self):
        if os.path.exists(self.split_path):
            pass
        else:
            num_geometry_per_mol = 100
            num_train_geometry_per_mol = 80
            num_val_geometry_per_mol = 10
            num_mol = 2998

            total_len = self._get_total_len()
            if self.split == "geometry":
                """
                geometry-wise split:
                    train: 80%
                    val: 10%
                    test: 10%
                """
                logger.info("Making geometry-wise split...")
                train_indices = np.array([], dtype=np.int64)
                val_indices = np.array([], dtype=np.int64)
                test_indices = np.array([], dtype=np.int64)
                cur_index = 0
                
                with self._get_db_env() as db_env:
                    cursor = db_env.cursor()
                    data = cursor.execute("select * from data").fetchall()
                    for ind, row in enumerate(tqdm(data)):
                        if (
                            ind + 1
                        ) % num_geometry_per_mol == 0:  # Finish traversing one molecule
                            indices = np.random.RandomState(seed=ind).permutation(
                                num_geometry_per_mol
                            )  # Different random split for different molecules
                            train_indices_cur_mol = (
                                cur_index + indices[:num_train_geometry_per_mol]
                            )
                            val_indices_cur_mol = (
                                cur_index
                                + indices[
                                    num_train_geometry_per_mol : num_train_geometry_per_mol
                                    + num_val_geometry_per_mol
                                ]
                            )
                            test_indices_cur_mol = (
                                cur_index
                                + indices[
                                    num_train_geometry_per_mol + num_val_geometry_per_mol :
                                ]
                            )
                            train_indices = np.concatenate((train_indices, train_indices_cur_mol))
                            val_indices = np.concatenate((val_indices, val_indices_cur_mol))
                            test_indices = np.concatenate((test_indices, test_indices_cur_mol))
                            cur_index += num_geometry_per_mol


            elif self.split == "mol":
                """ 
                molecule-wise split:
                    train: 80%
                    val: 10%
                    test: 10%
                """
                mol_id_list = [
                    i for i in range(num_mol) for _ in range(num_geometry_per_mol)
                ]
                data_ratio = [0.8, 0.1, 0.1]
                index_list = [i for i in range(max(mol_id_list) + 1)]
                random.seed(43)
                random.shuffle(index_list)
                train_mol_ids = np.array(index_list[: int(len(index_list) * data_ratio[0])])
                val_mol_ids = np.array(
                    index_list[
                        int(len(index_list) * data_ratio[0]) : int(
                            len(index_list) * data_ratio[0]
                        )
                        + int(len(index_list) * data_ratio[1])
                    ]
                )
                test_mol_ids = np.array(
                    index_list[
                        int(len(index_list) * data_ratio[0])
                        + int(len(index_list) * data_ratio[1]) :
                    ]
                )
                mol_id_array = np.array(mol_id_list)
                train_indices = np.where(np.isin(mol_id_array, train_mol_ids))[0].astype(np.int64)
                val_indices = np.where(np.isin(mol_id_array, val_mol_ids))[0].astype(np.int64)
                test_indices = np.where(np.isin(mol_id_array, test_mol_ids))[0].astype(np.int64)
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
            Dynamic.db Data
                0: id         (INTEGER) (PRIMARY KEY) NOT NULL 
                1: geo_id     (INTEGER) id that indicates the geometry (whether it is from the same molecule)
                2: N          (INTEGER) number of atoms
                3: Z          (BLOB, np.int32) atomic numbers
                4: pos        (BLOB, np.float64) atomic positions / unit: bohr 
                5: ekin       (FLOAT) kinetic energy / unit: hartree ? (Not used)
                6: epot       (FLOAT) potential energy / unit: hartree ? (Not used)
                7: etot       (FLOAT) total energy / unit: hartree ? (Not used)
                8: time       (FLOAT) time / unit : ? (Not used)
                9: Ham        (BLOB, np.float64) NULL 
                10: converged (INT, np.int) NULL (Not used)
      """
        data, data_idx = key_data_pair
        key = int(data_idx).to_bytes(length=4, byteorder="big")
        geo_id = data[1]
        atoms = np.frombuffer(data[3], np.int32)
        pos = np.frombuffer(data[4], np.float64) * BOHR2ANG # convert from Bohr to Angstrom

        ovlp, init_ham, mf = calc_overlap_and_init_hamiltonian(atoms, pos.reshape(-1, 3), out_mf=True)
        hamiltonian = np.frombuffer(data[9], np.float64) # flattened hamiltonian matrix
        h_dim = ovlp.shape[0]
        hamiltonian = hamiltonian.reshape(h_dim, h_dim)
        orbital_energies, orbital_coefficients = cal_orbital_and_energies(torch.from_numpy(ovlp).unsqueeze(0).to(torch.float64), torch.from_numpy(init_ham).unsqueeze(0).to(torch.float64))
        orbital_coefficients = orbital_coefficients.squeeze()
        dm0 = calc_dm0(atoms, orbital_coefficients)
        
        orbital_coefficients = orbital_coefficients.cpu().numpy()
        orbital_energies = orbital_energies.cpu().numpy()
        dft_energy = mf.energy_tot(dm0)
        
        packed_hamiltonian, h_dim = self.pack_upper_triangle(hamiltonian) # h_dim is the dimension of the hamiltonian matrix
        packed_ovlp, _ = self.pack_upper_triangle(ovlp)
        packed_init_ham, _ = self.pack_upper_triangle(init_ham)
        packed_dm0, _ = self.pack_upper_triangle(dm0)
        packed_orbital_coefficients, _ = self.pack_upper_triangle(orbital_coefficients)

        # Here, atoms and pos are not converted to "numpy array", since we have to convert them (byte stream) to "numpy array" in the get method
        ori_data_dict = {
            "id": data[0],
            "geo_id": geo_id,
            "num_nodes": data[2],
            "atoms": data[3],
            "pos": data[4],  # angstrom
            "dft_energy": dft_energy,
            "h_dim": h_dim,
            "packed_hamiltonian": packed_hamiltonian.tobytes(),
            "packed_overlap": packed_ovlp.tobytes(),
            "packed_initial_hamiltonian": packed_init_ham.tobytes(),
            "orbital_energies": orbital_energies.tobytes(),
            "packed_orbital_coefficients": packed_orbital_coefficients.tobytes(),
            "packed_dm0": packed_dm0.tobytes(),
        }
        data_dict = pickle.dumps(ori_data_dict)
        return key, data_dict

########################################################
# QH9Dynamic
########################################################

class QH9Dynamic(InMemoryDataset):
    url = {
        "100k": "https://drive.google.com/file/d/1SNWk0GD6Nt96qNAJJU2uedwWDQ4bbB1w/view?usp=sharing",
        "300k": "https://drive.google.com/file/d/1sbf-sFhh3ZmhXgTcN2ke_la39MaG0Yho/view?usp=sharing",
    }
    def __init__(
        self,
        root="datasets/",
        split="geometry",
        version="300k",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        prefix="",
        shard_num=30,
        shard_idx=-1,
        max_workers_preprocess=8,
        use_parallel_preprocess=False,
    ):
        self.version = version
        assert self.version in ["100k", "300k"], f"Version {self.version} for QH9Dynamic is not supported"
        self.split = split
        assert self.split in ["geometry", "mol"], f"Split {self.split} for QH9Dynamic is not supported"
        self.full_dataname = f"QH9Dynamic_{self.version}"

        self.folder = os.path.join(root, self.full_dataname + prefix)
        self.processd_dir_name = "processed"
        self.shard_dir_name = "lmdbs"
        self._processed_path = os.path.join(self.folder, self.processd_dir_name)

        self.shard_num = shard_num
        self.shard_idx = shard_idx
        self.max_workers_preprocess = max_workers_preprocess
        self.use_parallel_preprocess = use_parallel_preprocess

        assert self.split in ["geometry", "mol"], f"Split {self.split} for QH9Dynamic is not supported"
        if self.split == "geometry":
            self.split_path = os.path.join(self.folder, "processed_QH9Dynamic_geometry.json")
        elif self.split == "mol":
            self.split_path = os.path.join(self.folder, "processed_QH9Dynamic_mol.json")

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
        idx_1s_2s_2p = torch.tensor([0, 1, 3, 4, 5])
        
        # Create orbital masks for different atomic numbers
        # orbital_mask_line1: minimal orbital set for hydrogen and helium (atomic numbers 1-2)
        # orbital_mask_line2: full orbital set for heavier elements (atomic numbers 3-10)
        # orbital indices
        # 1s [0]
        # 2s [1], 2p [3, 4, 5]
        # 3s [2], 3p [6, 7, 8], 3d [9, 10, 11, 12, 13]        
        orbital_mask_line1 = idx_1s_2s_2p
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
        
        super(QH9Dynamic, self).__init__(self.folder, transform, pre_transform, pre_filter)

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
        if self.version == "100k":
            return [f"QH9Dynamic_100k.db"]
        elif self.version == "300k":
            return [f"QH9Dynamic_300k.db"]

    @property
    def processed_file_names(self):
        if self.split == "geometry":
            return [
                "processed_QH9Dynamic_geometry.json",
                "ALL_SHARDS_COMPLETED.txt",
                "index.json",
            ]
        elif self.split == "mol":
            return [
                "processed_QH9Dynamic_mol.json",
                "ALL_SHARDS_COMPLETED.txt",
                "index.json",
            ]

    def download(self):
        try:
            print(
                f"Downloading the QH9Dynamic_{self.version} dataset through {self.url}"
            )
            gdown.download(self.url[self.version], output=self.raw_paths[0], fuzzy=True)
        except:
            print(
                f"Downloading failed! Please download the QH9Dynamic_{self.version} dataset to {self.raw_paths[0]} "
                f"through {self.url[self.version]}"
            )
            print(f"Or you can try to download the zip file through {GoogleDriveLink}")
            raise FileNotFoundError(
                f"QH9Dynamic_{self.version} needs to be downloaded."
            )

    def process(self):
        self.QH9Dynamic_shard = QH9Dynamic_shard(
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
            logger.info(f"Processing all shards of QH9Dynamic_{self.version} dataset")
            self.QH9Dynamic_shard.process()
        else:
            logger.info(f"Processing QH9Dynamic_{self.version} dataset with shard_idx: {self.shard_idx}")
            self.QH9Dynamic_shard.process(self.shard_idx)
            
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
                raise KeyError(f"Index idx{idx}, shard_data_idx{self.shard_data_idx_list[idx]} not found in database {self.shard_idx_list[idx]}")
                
            data_dict = pickle.loads(data_dict)
            data = self.get_mol(data_dict, orb_energy_and_coeff=True)
        return data
    
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
    
    def matrix_transform(self, hamiltonian, atoms, convention="pyscf_def2svp_to_e3nn"):
        return _matrix_transform_single(hamiltonian, atoms, self.convention_dict[convention])
      
    def get_mol(self, data_dict, orb_energy_and_coeff=False):
        num_nodes = torch.tensor(data_dict["num_nodes"], dtype=torch.int64)
        atoms = torch.tensor(np.frombuffer(data_dict["atoms"], np.int32), dtype=torch.int64)
        pos = torch.tensor(np.frombuffer(data_dict["pos"], np.float64).reshape(-1, 3), dtype=torch.float64)
        dft_energy = torch.tensor(data_dict["dft_energy"], dtype=torch.float64)
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
            num_nodes=num_nodes.view(1, 1),
            h_dim=torch.tensor(h_dim, dtype=torch.int64).view(1, 1),
        )
        
        if orb_energy_and_coeff:    
            # Optimize orbital data loading: direct tensor creation from buffer
            orbital_energies = np.frombuffer(data_dict["orbital_energies"], np.float64)
            packed_orbital_coefficients = np.frombuffer(data_dict["packed_orbital_coefficients"], np.float64)
            orbital_coefficients = self.unpack_upper_triangle(packed_orbital_coefficients, h_dim)
            data.orbital_energies = torch.from_numpy(orbital_energies).to(torch.float64)
            data.orbital_coefficients = torch.from_numpy(orbital_coefficients).to(torch.float64).unsqueeze(0)

        return data

########################################################
# Main function & Argument parser
########################################################

def parse_shard_idx(shard_idx_str):
    """Parse shard_idx string into a list of integers"""
    if ',' in shard_idx_str:
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler('qh9_datasets.log')  # File output
        ]
    )

    logger = logging.getLogger(__name__)

    import argparse

    parser = argparse.ArgumentParser(description="Generation")
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset"),
    )
    parser.add_argument("--name", type=str, default="QH9Stable")
    parser.add_argument("--version", type=str, default="300k")
    parser.add_argument("--split", type=str, default="random")
    parser.add_argument("--dynamic_split", type=str, default="geometry")
    parser.add_argument("--shard_num", type=int, default=10, help="Number of shards for the dataset, for example, 10 shards for QH9Stable dataset")
    parser.add_argument("--shard_idx", type=str, default="0", help="-1 for all shards, Comma-separated list of shard indices or single index ex) 0,1,2,3 or 0-5")
    parser.add_argument("--prefix", type=str, default="_shard")
    parser.add_argument("--pdb", action="store_true", default=False)

    args = parser.parse_args()
    args.shard_idx = parse_shard_idx(args.shard_idx)
    torch.set_num_threads(4)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    logger.info(f"dataset path: {args.root}")

    logger.info(f"Processing {args.name} dataset with shard_idx: {args.shard_idx}")

    assert args.name in ["QH9Stable", "QH9Dynamic"]
    if args.name == "QH9Stable":
        assert args.split in ["random", "size_ood"]
        dataset = QH9Stable(
            root=args.root,
            split=args.split,
            shard_num=args.shard_num,
            shard_idx=args.shard_idx,
            prefix=args.prefix,
        )
    elif args.name == "QH9Dynamic":
        assert args.version in ["300k"], "Only 300k version is supported for QH9Dynamic currently"
        assert args.dynamic_split in ["geometry", "mol"]
        dataset = QH9Dynamic(
            root=args.root,
            version=args.version,
            split=args.dynamic_split,
            shard_num=args.shard_num,
            shard_idx=args.shard_idx,
            prefix=args.prefix,
        )
    if args.pdb:
        import pdb

        pdb.set_trace()

        num_total = len(dataset)
        chunk_size = (num_total + (args.num_chunks - 1)) // args.num_chunks
        idx = []
        for i in range(args.num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, num_total)
            idx.append((start_idx, end_idx))

        pdb.set_trace()

    print(len(dataset))
    print(dataset[0])
    print(dataset[-1])
    for i in range(len(dataset)):
        data = dataset[i]