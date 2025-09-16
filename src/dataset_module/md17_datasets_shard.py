import numpy as np
import lmdb
import pickle
import os
from common.custom_logger import setup_global_logger, get_logger
import json
import gdown
import torch
from tqdm.rich import tqdm
import random
import time
import tarfile
from typing import Union, List

from common.metric import cal_orbital_and_energies
from common.matrix_transforms import pack_upper_triangle, unpack_upper_triangle, _matrix_transform_single, get_convention_dict, _cut_matrix_3d, _cut_matrix_3d_last
from dataset_module.lmdb_shard import LMDBShard_maker_db
from common.dft_utils import calc_overlap_and_init_hamiltonian, calc_dm0

from torch_geometric.data import InMemoryDataset, Data, download_url
from utils import AOData, Onsite_3idx_Overlap_Integral, build_molecule, build_AO_index
from common.units import *

logger = get_logger(__file__)

########################################################
# MD17
########################################################

class MD17_shard(LMDBShard_maker_db):
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

    def _make_split_info(self,seed=43):
        if self.split is None:
            return
        else:
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
                    indices = np.random.RandomState(seed=seed).permutation(total_len)
                    train_indices = indices[:train_len]
                    val_indices = indices[train_len:train_len+val_len]
                    test_indices = indices[train_len+val_len:]

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
            QH9Stable.db data format: (To be updated)
                0: id (INTEGER, np.int32) (PRIMARY KEY) NOT NULL index
                1: N (INTEGER, np.int32) number of atoms
                2: Z (BLOB, np.int32) atomic numbers
                3: pos (BLOB, np.float64) atomic positions / unit: angstrom 
                4: Ham (BLOB, np.float64) hamiltonian matrix / unit: hartree / pyscf convention
      """
        data, data_idx = key_data_pair      
        key = int(data_idx).to_bytes(length=4, byteorder="big") # real key is not used
        pos    = torch.tensor(data["positions"], dtype=torch.float64) # unit: angstrom
        atoms  = torch.tensor(data["numbers"], dtype=torch.int64).view(-1, 1)
        energy = torch.tensor(data["energy"], dtype=torch.float64).item()
        force  = torch.tensor(data["forces"], dtype=torch.float64).numpy().tobytes()
        data_hamiltonian = data["hamiltonian"]
        # overlap     = data["overlap"]

        ovlp, init_ham, mf = calc_overlap_and_init_hamiltonian(atoms, pos.reshape(-1, 3), out_mf=True)
        mf.kernel()
        hamiltonian = mf.get_fock(dm=mf.make_rdm1())
        # hamiltonian = np.frombuffer(data[4], np.float64) # flattened hamiltonian matrix
        h_dim = ovlp.shape[0]
        hamiltonian = hamiltonian.reshape(h_dim, h_dim)
        orbital_energies, orbital_coefficients = cal_orbital_and_energies(torch.from_numpy(ovlp).unsqueeze(0).to(torch.float64), torch.from_numpy(init_ham).unsqueeze(0).to(torch.float64))
        orbital_coefficients = orbital_coefficients.squeeze()
        dm0 = calc_dm0(atoms, orbital_coefficients)
        
        orbital_coefficients = orbital_coefficients.cpu().numpy()
        orbital_energies = orbital_energies.cpu().numpy()
        dft_energy = mf.energy_tot(dm0)

        # Calculate DFT forces
        grad_frame = mf.nuc_grad_method()
        mo_occ = mf.get_occ(orbital_energies.squeeze(), orbital_coefficients.squeeze())
        dft_forces = -grad_frame.kernel(mo_energy=orbital_energies.squeeze(), mo_coeff=orbital_coefficients.squeeze(), mo_occ=mo_occ)
        
        packed_hamiltonian, h_dim = self.pack_upper_triangle(hamiltonian) # h_dim is the dimension of the hamiltonian matrix
        packed_data_hamiltonian, _ = self.pack_upper_triangle(data_hamiltonian)
        packed_ovlp, _ = self.pack_upper_triangle(ovlp)
        packed_init_ham, _ = self.pack_upper_triangle(init_ham)
        packed_dm0, _ = self.pack_upper_triangle(dm0)
        packed_orbital_coefficients, _ = self.pack_upper_triangle(orbital_coefficients)

        ori_data_dict = {
            # "id": data[0],
            "num_nodes": pos.shape[0],
            "atoms": atoms.numpy().tobytes(),
            "pos": pos.numpy().tobytes(),  # unit: angstrom
            "energy": energy.item(),
            "force": force.tobytes(),
            "dft_energy": dft_energy, # unit: Eh
            "dft_forces": dft_forces.tobytes(), # unit: Eh/Bohr
            "h_dim": h_dim,
            "packed_hamiltonian": packed_hamiltonian.tobytes(), # unit: Eh
            "packed_data_hamiltonian": packed_data_hamiltonian.tobytes(), # unit: Eh
            "packed_overlap": packed_ovlp.tobytes(),
            "packed_initial_hamiltonian": packed_init_ham.tobytes(), # unit: Eh
            "orbital_energies": orbital_energies.tobytes(), # unit: Eh
            "packed_orbital_coefficients": packed_orbital_coefficients.tobytes(),
            "packed_dm0": packed_dm0.tobytes(),
        }
        data_dict = pickle.dumps(ori_data_dict)
        return key, data_dict

class MD17_DFT_Shard(InMemoryDataset):

    url = "http://quantum-machine.org/data/schnorb_hamiltonian"
    chemical_symbols = ["n", "H", "He", "Li", "Be", "B", "C", "N", "O"]

    def __init__(
        self,
        root="datasets/",
        name="water",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        prefix="",
        shard_num=30,
        shard_idx=-1,
        max_workers_preprocess=8,
        use_parallel_preprocess=False,
    ):
        self.name = name
        self.folder = os.path.join(root, self.name)
        self.processd_dir_name = "processed"
        self.shard_dir_name = "lmdbs"
        self._processed_path = os.path.join(self.folder, self.processd_dir_name)

        self.shard_num = shard_num
        self.shard_idx = shard_idx
        self.max_workers_preprocess = max_workers_preprocess
        self.use_parallel_preprocess = use_parallel_preprocess

        self.lmdb_path_list = [os.path.join(self._processed_path,self.shard_dir_name, f"shard_{i:02d}.lmdb") for i in range(self.shard_num)]
        
        self.full_orbitals = 14
        self.orbital_mask = {}
        
        orbitals_ref = {}
        orbitals_ref[1] = np.array([0, 0, 1])  # H: 2s 1p
        orbitals_ref[6] = np.array([0, 0, 0, 1, 1, 2])  # C: 3s 2p 1d
        orbitals_ref[7] = np.array([0, 0, 0, 1, 1, 2])  # N: 3s 2p 1d
        orbitals_ref[8] = np.array([0, 0, 0, 1, 1, 2])  # O: 3s 2p 1d
        self.orbitals_ref = orbitals_ref

        orbitals = []
        assert name in ["water", "ethanol", "malondialdehyde", "uracil", "aspirin", "malonaldehyde"]
        if name == "malonaldehyde":
            logger.info("Malonaldehyde is renamed to Malondialdehyde for compatibility")
            name = "malondialdehyde"
        if name == "water":
            self.atoms = [8, 1, 1]
            self.atom_list = ["O", "H"]
            self.hamiltonian_size = 24
        elif name == "ethanol":
            self.atoms = [6, 6, 8, 1, 1, 1, 1, 1, 1]
            self.atom_list = ["C", "O", "H"]
            self.hamiltonian_size = 72
        elif name == "malondialdehyde":
            self.atoms = [6, 6, 6, 8, 8, 1, 1, 1, 1]
            self.atom_list = ["C", "O", "H"]
            self.hamiltonian_size = 90
        elif name == "uracil":
            self.atoms = [6, 6, 7, 6, 7, 6, 8, 8, 1, 1, 1, 1]
            self.atom_list = ["C", "N", "O", "H"]
            self.hamiltonian_size = 132
        elif name == "aspirin":
            self.atoms = [6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1]
            self.atom_list = ["C", "O", "H"]
            raise NotImplementedError
        
        self.Q_dict = Onsite_3idx_Overlap_Integral(atom_list=self.atom_list, basis="def2-svp").Q_table()
        self.convention_dict = get_convention_dict()
        self.setup_Q()
  
        for Z in self.atoms:
            orbitals.append(tuple((int(Z), int(l)) for l in self.orbitals_ref[Z]))
        
        self.orbitals = tuple(orbitals)

        super().__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self):
        if self.name == "ethanol":
            return [
                f"schnorb_hamiltonian_{self.name}_dft.tgz",
                f"schnorb_hamiltonian_{self.name}_dft.db",
            ]
        elif self.name == "aspirin":
            return [
                f"schnorb_hamiltonian_{self.name}_quambo.db",
                f"schnorb_hamiltonian_{self.name}_quambo.db",
            ]
        else: # water, malondialdehyde, uracil
            return [
                f"schnorb_hamiltonian_{self.name}.tgz",
                f"schnorb_hamiltonian_{self.name}.db",
            ]
    @property
    def processed_file_names(self):
        if self.split == "random":
            return [
                "random_split.json",
                "ALL_SHARDS_COMPLETED.txt",
                "index.json",
            ]
        else:
            return [
                "ALL_SHARDS_COMPLETED.txt",
                "index.json",
            ]

    def download(self):
        if self.name == "ethanol":
            url = f"{self.url}/schnorb_hamiltonian_{self.name}" + "_dft.tgz"
        else:
            url = f"{self.url}/schnorb_hamiltonian_{self.name}" + ".tgz"
        download_url(url, self.raw_dir)
        extract_path = self.raw_dir
        tar = tarfile.open(os.path.join(self.raw_dir, self.raw_file_names[0]), "r")
        for item in tar:
            tar.extract(item, extract_path)

    def matrix_transform(self, hamiltonian, atoms, convention="pyscf_def2svp_to_e3nn"):
        return _matrix_transform_single(hamiltonian, atoms, self.convention_dict[convention])


    @staticmethod
    def construct_orbital_l_index(AO_lm_index):
        idx = 0
        AO_l_index = []
        while True:
            if idx >= len(AO_lm_index):
                break
            AO_l_index.append(AO_lm_index[idx].item())
            idx += 2 * AO_lm_index[idx] + 1
        return torch.tensor(AO_l_index)
    
    def setup_Q(self):
        self.Q = (
            torch.stack([
                    torch.block_diag(*[self.Q_dict[z][l] for z in self.atoms])
                    for l in range(60)]
            ).double().numpy()
        )
        self.Q = (
            torch.from_numpy(
                self.matrix_transform(self.Q, self.atoms, convention="pyscf_def2svp_to_e3nn")
            ).double().permute(1, 2, 0)
        )
        self.Q[:, :, 16:40] = (
            self.Q[:, :, 16:40]
            .reshape(self.hamiltonian_size, self.hamiltonian_size, -1, 3)[:, :, :, [1, 2, 0]]
            .reshape(self.hamiltonian_size, self.hamiltonian_size, 24)
        )
    
    def process(self):
        self.MD17_DFT_Shard = MD17_DFT_Shard(
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
            logger.info(f"Processing all shards of MD17_DFT dataset")
            self.MD17_DFT_Shard.process()
        else:
            logger.info(f"Processing MD17_DFT dataset with shard_idx: {self.shard_idx}")
            self.MD17_DFT_Shard.process(self.shard_idx)
    
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
    
    
    def __getitem__(self, idx):
        return self.get(idx)
    
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
    
    def get_mol(self, data_dict, orb_energy_and_coeff=False):
        num_nodes = torch.tensor(data_dict["num_nodes"], dtype=torch.int64)
        atoms = torch.tensor(np.frombuffer(data_dict["atoms"], np.int32), dtype=torch.int64)
        pos = torch.tensor(np.frombuffer(data_dict["pos"], np.float64).reshape(-1, 3), dtype=torch.float64)
        energy = torch.tensor(data_dict["energy"], dtype=torch.float64)
        force = torch.tensor(np.frombuffer(data_dict["force"], np.float64).reshape(-1, 3), dtype=torch.float64)
        dft_energy = torch.tensor(data_dict["dft_energy"], dtype=torch.float64)
        dft_forces = torch.tensor(np.frombuffer(data_dict["dft_forces"], np.float64).reshape(-1, 3), dtype=torch.float64)
        h_dim = data_dict["h_dim"] # sum of orbital dimensions
        
        packed_hamiltonian = np.frombuffer(data_dict["packed_hamiltonian"], np.float64)
        # packed_data_hamiltonian = np.frombuffer(data_dict["packed_data_hamiltonian"], np.float64)
        packed_ovlp = np.frombuffer(data_dict["packed_overlap"], np.float64)
        packed_init_ham = np.frombuffer(data_dict["packed_initial_hamiltonian"], np.float64)
        # packed_dm0 = np.frombuffer(data_dict["packed_dm0"], np.float64)
        
        hamiltonian = torch.from_numpy(self.unpack_upper_triangle(packed_hamiltonian, h_dim)).to(torch.float64)
        # data_hamiltonian = torch.from_numpy(self.unpack_upper_triangle(packed_data_hamiltonian, h_dim)).to(torch.float64)
        overlap_matrix = torch.from_numpy(self.unpack_upper_triangle(packed_ovlp, h_dim)).to(torch.float64)
        initial_hamiltonian = torch.from_numpy(self.unpack_upper_triangle(packed_init_ham, h_dim)).to(torch.float64)
        # dm0 = torch.from_numpy(self.unpack_upper_triangle(packed_dm0, h_dim)).to(torch.float64)
        
        convention = "pyscf_def2svp_to_e3nn"
        # stack in 0th dimension
        
        hamiltonian = self.matrix_transform(hamiltonian, atoms, convention=convention)
        overlap_matrix = self.matrix_transform(overlap_matrix, atoms, convention=convention)
        initial_hamiltonian = self.matrix_transform(initial_hamiltonian, atoms, convention=convention)
        
        AO_index = build_AO_index(build_molecule(atoms, pos), "def2-svp")
        AO_l_index = self.construct_orbital_l_index(AO_index[1])
        
        return AOData(
            pos=pos,
            atoms=atoms.view(-1, 1),
            dft_energy=dft_energy.view(1, 1),
            dft_forces=dft_forces,
            energy=energy.view(1, 1),
            force=force,
            hamiltonian=hamiltonian.reshape(1, h_dim, h_dim),
            overlap=overlap_matrix.reshape(1, h_dim, h_dim),
            init_ham=initial_hamiltonian.reshape(1, h_dim, h_dim),
            AO_index=AO_index,
            AO_l_index=AO_l_index,
            AO_l_index_len=torch.tensor(len(AO_l_index), dtype=torch.int64).view(1, 1),
            num_atoms=num_nodes.view(1, 1),
            Q=self.Q,
            h_dim=torch.tensor(h_dim, dtype=torch.int64).view(1, 1),
        )


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
    # get arguments
    import argparse

    parser = argparse.ArgumentParser(description="Generation")
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset"),
    )
    parser.add_argument("--name", type=str, default="water")
    parser.add_argument("--pdb", action="store_true")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--shard_num", type=int, default=10)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="_shard")

    args = parser.parse_args()
    args.shard_idx = parse_shard_idx(args.shard_idx)
    torch.set_num_threads(4)

    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"

    args = parser.parse_args()
    dataset = MD17_DFT_Shard(
        root=args.root,
        name=args.name,
        shard_idx=args.shard_idx,
        shard_num=args.shard_num,
        split=args.split,
        prefix=args.prefix,
    )
    print(len(dataset))
    print(dataset[0])
    print(dataset[-1])
    if args.pdb:
        import pdb

        pdb.set_trace()
    logger.info("Finished")