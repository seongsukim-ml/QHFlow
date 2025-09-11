"""
QH9 Dataset Module for Quantum Chemistry Calculations

This module provides optimized dataset classes for QH9Stable and QH9Dynamic datasets,
which contain molecular geometries with precomputed quantum chemistry properties.
The datasets are optimized for efficient data loading using LMDB storage.

Classes:
    QH9Stable: Dataset for stable molecular geometries
    QH9Dynamic: Dataset for dynamic molecular conformations

Functions:
    matrix_transform: Transform matrices according to orbital conventions
    cal_orbital_and_energies: Calculate orbital energies and coefficients
    calc_ovlp_and_ham_init: Calculate overlap and initial Hamiltonian matrices
"""

import os
import lmdb
import random
import torch
import numpy as np
import os.path as osp
from argparse import Namespace
import pickle
import gdown
import pyscf
from pyscf import gto
from tqdm.rich import tqdm
from apsw import Connection
from torch_geometric.data import InMemoryDataset, Data
from pyscf import gto, scf, dft
import concurrent.futures

from utils import AOData, Onsite_3idx_Overlap_Integral, build_molecule, build_AO_index

# Conversion factor from Bohr to Angstrom
BOHR2ANG = 1.8897259886

# Google Drive link for dataset downloads
GoogleDriveLink = (
    "https://drive.google.com/drive/u/0/folders/1LXTC8uaOQzmb76FsuGfwSocAbK5Hshfj"
)

# Orbital convention mappings for different basis sets
convention_dict = {
    "pyscf_631G": Namespace(
        atom_to_orbitals_map={1: "ss", 6: "ssspp", 7: "ssspp", 8: "ssspp", 9: "ssspp"},
        orbital_idx_map={"s": [0], "p": [2, 0, 1], "d": [0, 1, 2, 3, 4]},
        orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1],
            6: [0, 1, 2, 3, 4],
            7: [0, 1, 2, 3, 4],
            8: [0, 1, 2, 3, 4],
            9: [0, 1, 2, 3, 4],
        },
    ),
    "pyscf_def2svp": Namespace(
        atom_to_orbitals_map={
            1: "ssp",
            6: "sssppd",
            7: "sssppd",
            8: "sssppd",
            9: "sssppd",
        },
        orbital_idx_map={"s": [0], "p": [1, 2, 0], "d": [0, 1, 2, 3, 4]},
        orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2],
            6: [0, 1, 2, 3, 4, 5],
            7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5],
            9: [0, 1, 2, 3, 4, 5],
        },
    ),
    "back2pyscf": Namespace(
        atom_to_orbitals_map={
            1: "ssp",
            6: "sssppd",
            7: "sssppd",
            8: "sssppd",
            9: "sssppd",
        },
        orbital_idx_map={"s": [0], "p": [2, 0, 1], "d": [0, 1, 2, 3, 4]},
        orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2],
            6: [0, 1, 2, 3, 4, 5],
            7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5],
            9: [0, 1, 2, 3, 4, 5],
        },
    ),
}

# Atomic reference energies for different properties
atomrefs = {
    6: [0.0, 0.0, 0.0, 0.0, 0.0],
    7: [-13.61312172, -1029.86312267, -1485.30251237, -2042.61123593, -2713.48485589],
    8: [-13.5745904, -1029.82456413, -1485.26398105, -2042.5727046, -2713.44632457],
    9: [-13.54887564, -1029.79887659, -1485.2382935, -2042.54701705, -2713.42063702],
    10: [-13.90303183, -1030.25891228, -1485.71166277, -2043.01812778, -2713.88796536],
    11: [0.0, 0.0, 0.0, 0.0, 0.0],
}

# Conversion factors for energy units
HAR2EV = 27.211386246  # Hartree to electron volts
KCALMOL2EV = 0.04336414  # kcal/mol to electron volts

# Unit conversion tensor for different properties
conversion = torch.tensor(
    [
        1.0,
        1.0,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        1.0,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        HAR2EV,
        1.0,
        KCALMOL2EV,
        KCALMOL2EV,
        KCALMOL2EV,
        KCALMOL2EV,
        1.0,
        1.0,
        1.0,
    ]
)

# Tensor version of atomic reference energies
atomrefs_tensor = torch.zeros(5, 19)
atomrefs_tensor[:, 7] = torch.tensor(atomrefs[7])
atomrefs_tensor[:, 8] = torch.tensor(atomrefs[8])
atomrefs_tensor[:, 9] = torch.tensor(atomrefs[9])
atomrefs_tensor[:, 10] = torch.tensor(atomrefs[10])


def matrix_transform(matrices, atoms, convention="pyscf_631G"):
    """
    Transform matrices according to orbital convention.
    
    Args:
        matrices: Input matrices to transform
        atoms: Atomic numbers for the molecule
        convention: Orbital convention to use ('pyscf_631G', 'pyscf_def2svp', 'back2pyscf')
    
    Returns:
        Transformed matrices according to the specified convention
    """
    conv = convention_dict[convention]
    orbitals = ""
    orbitals_order = []
    for a in atoms:
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[a]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a]]

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        map_sign = conv.orbital_sign_map[orb]
        transform_indices.append(np.array(map_idx) + offset)
        transform_signs.append(np.array(map_sign))

    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = np.concatenate(transform_indices).astype(np.int32)
    transform_signs = np.concatenate(transform_signs)

    matrices_new = matrices[..., transform_indices, :]
    matrices_new = matrices_new[..., :, transform_indices]
    matrices_new = matrices_new * transform_signs[:, None]
    matrices_new = matrices_new * transform_signs[None, :]
    return matrices_new


class QH9Stable(InMemoryDataset):
    """
    QH9Stable dataset for quantum chemistry calculations.
    
    This dataset contains stable molecular geometries with precomputed 
    Hamiltonian matrices, overlap matrices, and initial Hamiltonian matrices.
    The dataset supports both random and size-based out-of-distribution splits.
    
    Attributes:
        url (str): Google Drive URL for dataset download
        folder (str): Local folder path for dataset storage
        split (str): Dataset split type ('random' or 'size_ood')
        cal_orbital_and_energies (bool): Whether to calculate orbital energies
        _db_env: LMDB environment for optimized data access
        _lmdb_path: Cached path to LMDB database
    """
    url = "https://drive.google.com/file/d/1LcEJGhB8VUGkuyb0oQ_9ANJdSkky9xMS/view?usp=sharing"

    def __init__(
        self,
        root="datasets/",
        split="random",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        prefix="",
        num_chunks=10,
        chunk_idx=0,
        cal_orbital_and_energies=False,
    ):
        self.folder = osp.join(root, "QH9Stable" + prefix)
        self.split = split
        self.db_dir = os.path.join(self.folder, "processed")
        self.full_orbitals = 14
        self.orbital_mask = {}
        self.cal_orbital_and_energies = cal_orbital_and_energies
        
        # Variables for LMDB connection optimization
        self._db_env = None
        self._lmdb_path = None

        idx_1s_2s_2p = torch.tensor([0, 1, 3, 4, 5])
        orbital_mask_line1 = idx_1s_2s_2p
        orbital_mask_line2 = torch.arange(self.full_orbitals)
        for i in range(1, 11):
            self.orbital_mask[i] = orbital_mask_line1 if i <= 2 else orbital_mask_line2

        atom_list = ["H", "C", "N", "O", "F"]

        self.Q_dict = Onsite_3idx_Overlap_Integral(
            atom_list=atom_list, basis="def2-svp"
        ).Q_table()
        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        assert chunk_idx < num_chunks
        super(QH9Stable, self).__init__(
            self.folder, transform, pre_transform, pre_filter
        )
        self.train_mask, self.val_mask, self.test_mask = torch.load(
            self.processed_paths[0]
        )
        self.slices = {
            "id": torch.arange(
                self.train_mask.shape[0]
                + self.val_mask.shape[0]
                + self.test_mask.shape[0]
                + 1
            )
        }

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

    @property
    def raw_file_names(self):
        return [f"QH9Stable.db"]

    @property
    def processed_file_names(self):
        if self.split == "random":
            return ["processed_QH9Stable_random_12.pt", "QH9Stable.lmdb/data.mdb"]
        elif self.split == "size_ood":
            return ["processed_QH9Stable_size_ood.pt", "QH9Stable.lmdb/data.mdb"]

    def merge_lmbdbs(self, chunk_paths, merged_lmdb_path):
        print("Merging lmdb databases...", merged_lmdb_path)
        merge_env = lmdb.open(merged_lmdb_path, map_size=1048576000000)
        for ppath in chunk_paths:
            print(f"Merging {ppath}")
            part_env = lmdb.open(ppath, readonly=True)
            with part_env.begin() as part_txn, merge_env.begin(write=True) as merge_txn:
                cursor = part_txn.cursor()
                for key, value in cursor:
                    merge_txn.put(key, value)
        merge_env.close()
        print("Merging lmdb databases done.")

    def process_row(self, row):
        """
        Process a single row from the database.
        
        Args:
            row: Database row containing molecular data
            
        Returns:
            tuple: (key, data_dict) for LMDB storage
        """
        atoms = np.frombuffer(row[2], np.int32)
        pos = np.frombuffer(row[3], np.float64)
        ovlp, init_ham = calc_ovlp_and_ham_init(atoms, pos.reshape(-1, 3))
        ori_data_dict = {
            "id": row[0],
            "num_nodes": row[1],
            "atoms": row[2],
            "pos": row[3],  # angstrom
            "Ham": row[4],
            "ovlp": ovlp.tobytes(),
            "init_ham": init_ham.tobytes(),
        }
        key = ori_data_dict["id"].to_bytes(length=4, byteorder="big")
        data_dict = pickle.dumps(ori_data_dict)
        return key, data_dict

    def process(self):
        # assert pyscf.__version__ == "2.2.1"  # pyscf (2.2.1) (QH9-Stable)

        if self.split == "size_ood":
            num_nodes_list = []

        all_data = []
        for raw_file_name in self.raw_file_names:
            connection = Connection(os.path.join(self.root, "raw", raw_file_name))
            cursor = connection.cursor()
            data = cursor.execute("select * from data").fetchall()
            connection.close()
            all_data.extend(data)

        num_total = len(all_data)
        chunk_size = (num_total + (self.num_chunks - 1)) // self.num_chunks
        start_idx = self.chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, num_total)

        assert start_idx < end_idx and end_idx <= num_total
        assert start_idx < num_total and start_idx >= 0
        data_chunk = all_data[start_idx:end_idx]
        print(f"Total data points: {num_total}")
        print(
            f"Processing chunk {self.chunk_idx+1}/{self.num_chunks} with {len(data_chunk)} data points."
        )
        print(f"Data range: {start_idx} ~ {end_idx}")

        lmdb_chunk_name = f"QH9Stable_{self.chunk_idx}.lmdb"
        lmdb_completed_chunk_name = f"QH9Stable_{self.chunk_idx}_completed.lmdb"
        lmdb_chunk_path = os.path.join(self.db_dir, lmdb_chunk_name)
        lmbd_completed_chunk_path = os.path.join(self.db_dir, lmdb_completed_chunk_name)
        if not os.path.isdir(lmbd_completed_chunk_path):
            db_env = lmdb.open(lmdb_chunk_path, map_size=1048576000000)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                results = list(
                    tqdm(
                        executor.map(self.process_row, data_chunk),
                        total=len(data_chunk),
                    )
                )
            with db_env.begin(write=True) as txn:
                for key, data_dict in results:
                    txn.put(key, data_dict)
            db_env.close()
            os.rename(
                lmdb_chunk_path, os.path.join(self.db_dir, lmdb_completed_chunk_name)
            )
        else:
            print(f"Chunk {self.chunk_idx} already exists. Skip the processing.")
        db_env = lmdb.open(lmbd_completed_chunk_path, map_size=1048576000000)

        merged_path = os.path.join(self.db_dir, "QH9Stable.lmdb")
        if not os.path.isdir(merged_path):
            ppath_list = [
                os.path.join(self.db_dir, f"QH9Stable_{i}_completed.lmdb")
                for i in range(self.num_chunks)
            ]
            ppath_list = [ppath for ppath in ppath_list if os.path.exists(ppath)]
            if len(ppath_list) == self.num_chunks:
                self.merge_lmbdbs(ppath_list, merged_path)
            else:
                print(
                    f"Chunk {self.chunk_idx} is missing. Please check the processing."
                )
                print(f"Missing chunks: {ppath_list}")
                print(f"Total chunks: {self.num_chunks}")
                raise FileNotFoundError("Missing chunks.")
        else:
            print("Merged lmdb database exists. Jump the merging step.")

        if self.split == "random":
            print("Random splitting...")
            data_ratio = [0.8, 0.1, 0.1]
            data_split = [
                int(len(all_data) * data_ratio[0]),
                int(len(all_data) * data_ratio[1]),
            ]
            data_split.append(len(all_data) - sum(data_split))
            indices = np.random.RandomState(seed=43).permutation(len(all_data))
            train_mask = indices[: data_split[0]]
            val_mask = indices[data_split[0] : data_split[0] + data_split[1]]
            test_mask = indices[data_split[0] + data_split[1] :]

        elif self.split == "size_ood":
            print("Size OOD splitting...")
            num_nodes_list = [row[1] for row in all_data]
            num_nodes_array = np.array(num_nodes_list)
            train_indices = np.where(num_nodes_array <= 20)
            val_condition = np.logical_and(num_nodes_array >= 21, num_nodes_array <= 22)
            val_indices = np.where(val_condition)
            test_indices = np.where(num_nodes_array >= 23)
            train_mask = train_indices[0].astype(np.int64)
            val_mask = val_indices[0].astype(np.int64)
            test_mask = test_indices[0].astype(np.int64)

        torch.save((train_mask, val_mask, test_mask), self.processed_paths[0])

    def cut_matrix(self, matrix, atoms):
        all_diagonal_matrix_blocks = []
        all_non_diagonal_matrix_blocks = []
        all_diagonal_matrix_block_masks = []
        all_non_diagonal_matrix_block_masks = []
        col_idx = 0
        edge_index_full = []
        for idx_i, atom_i in enumerate(atoms):  # (src)
            row_idx = 0
            atom_i = atom_i.item()
            mask_i = self.orbital_mask[atom_i]
            for idx_j, atom_j in enumerate(atoms):  # (dst)
                if idx_i != idx_j:
                    edge_index_full.append([idx_j, idx_i])
                atom_j = atom_j.item()
                mask_j = self.orbital_mask[atom_j]

                if len(matrix.size()) == 2:
                    matrix_block = torch.zeros(
                        self.full_orbitals, self.full_orbitals
                    ).type(torch.float64)
                    matrix_block_mask = torch.zeros(
                        self.full_orbitals, self.full_orbitals
                    ).type(torch.float64)
                else:
                    len_ = matrix.size(-1)
                    matrix_block = torch.zeros(
                        self.full_orbitals, self.full_orbitals, len_
                    ).type(torch.float64)
                    matrix_block_mask = torch.zeros(
                        self.full_orbitals, self.full_orbitals, len_
                    ).type(torch.float64)

                extracted_matrix = matrix[
                    row_idx : row_idx + len(mask_j), col_idx : col_idx + len(mask_i)
                ]

                # for matrix_block
                tmp = matrix_block[mask_j]
                tmp[:, mask_i] = extracted_matrix
                matrix_block[mask_j] = tmp

                tmp = matrix_block_mask[mask_j]
                tmp[:, mask_i] = 1
                matrix_block_mask[mask_j] = tmp

                if idx_i == idx_j:
                    all_diagonal_matrix_blocks.append(matrix_block)
                    all_diagonal_matrix_block_masks.append(matrix_block_mask)
                else:
                    all_non_diagonal_matrix_blocks.append(matrix_block)
                    all_non_diagonal_matrix_block_masks.append(matrix_block_mask)
                row_idx = row_idx + len(mask_j)
            col_idx = col_idx + len(mask_i)
        return (
            torch.stack(all_diagonal_matrix_blocks, dim=0),
            torch.stack(all_non_diagonal_matrix_blocks, dim=0),
            torch.stack(all_diagonal_matrix_block_masks, dim=0),
            torch.stack(all_non_diagonal_matrix_block_masks, dim=0),
            torch.tensor(edge_index_full).transpose(-1, -2),
        )

    def get_mol(self, atoms, pos, Ham, ovlp, _init_ham):
        if self.cal_orbital_and_energies:
            e0, c0 = cal_orbital_and_energies(
                torch.tensor(ovlp).unsqueeze(0), torch.tensor(_init_ham).unsqueeze(0)
            )
            ef, cf = cal_orbital_and_energies(
                torch.tensor(ovlp).unsqueeze(0), torch.tensor(Ham).unsqueeze(0)
            )
        hamiltonian = torch.tensor(
            matrix_transform(Ham, atoms, convention="pyscf_def2svp"),
            dtype=torch.float64,
        )
        (
            diagonal_hamiltonian,
            non_diagonal_hamiltonian,
            diagonal_hamiltonian_mask,
            non_diagonal_hamiltonian_mask,
            edge_index_full,
        ) = self.cut_matrix(hamiltonian, atoms)

        init_ham = torch.tensor(
            matrix_transform(_init_ham, atoms, convention="pyscf_def2svp"),
            dtype=torch.float64,
        )

        (
            diagonal_init_ham,
            non_diagonal_init_ham,
            diagonal_init_ham_mask,
            non_diagonal_init_ham_mask,
            _,
        ) = self.cut_matrix(init_ham, atoms)

        overlap = torch.tensor(
            matrix_transform(ovlp, atoms, convention="pyscf_def2svp"),
            dtype=torch.float64,
        )
        diagonal_overlap, non_diagonal_overlap, _, _, _ = self.cut_matrix(
            overlap, atoms
        )

        Q = (
            torch.stack(
                [
                    torch.block_diag(*[self.Q_dict[z][l] for z in atoms])
                    for l in range(60)
                ]
            )
            .double()
            .numpy()
        )
        Q = (
            torch.from_numpy(matrix_transform(Q, atoms, convention="pyscf_def2svp"))
            .double()
            .permute(1, 2, 0)
        )
        H_size = hamiltonian.size(-1)
        Q[:, :, 16:40] = (
            Q[:, :, 16:40]
            .reshape(H_size, H_size, -1, 3)[:, :, :, [1, 2, 0]]
            .reshape(H_size, H_size, 24)
        )
        diagonal_Q, non_diagonal_Q, _, _, _ = self.cut_matrix(Q, atoms)

        data = Data(
            pos=torch.tensor(pos, dtype=torch.float64),
            atoms=torch.tensor(atoms, dtype=torch.int64).view(-1, 1),
            diagonal_hamiltonian=diagonal_hamiltonian,
            non_diagonal_hamiltonian=non_diagonal_hamiltonian,
            diagonal_hamiltonian_mask=diagonal_hamiltonian_mask,
            non_diagonal_hamiltonian_mask=non_diagonal_hamiltonian_mask,
            diagonal_init_ham=diagonal_init_ham,
            non_diagonal_init_ham=non_diagonal_init_ham,
            diagonal_overlap=diagonal_overlap,
            non_diagonal_overlap=non_diagonal_overlap,
            overlap=torch.tensor(ovlp, dtype=torch.float64).flatten(),  # pyscf
            overlap_dim=torch.tensor(ovlp.shape[0]).long().view(1, 1),
            diagonal_Q=diagonal_Q,
            non_diagonal_Q=non_diagonal_Q,
            edge_index_full=edge_index_full,
        )

        if self.cal_orbital_and_energies:
            data.e0 = e0.numpy()
            data.c0 = c0.numpy()
            data.ef = ef.numpy()
            data.cf = cf.numpy()
            data.h = Ham
            data.ovlp = ovlp
        return data

    def _get_db_env(self):
        """Lazy initialization and reuse of LMDB environment."""
        if self._db_env is None:
            self._lmdb_path = os.path.join(self.processed_dir, "QH9Stable.lmdb")
            self._db_env = lmdb.open(
                self._lmdb_path,
                readonly=True,
                lock=False,
            )
        return self._db_env

    def _close_db_env(self):
        """Safely close LMDB environment."""
        if self._db_env is not None:
            self._db_env.close()
            self._db_env = None

    def __enter__(self):
        """Context manager entry: Initialize LMDB environment."""
        self._get_db_env()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: Clean up LMDB environment."""
        self._close_db_env()

    def __del__(self):
        """Destructor: Clean up LMDB environment."""
        self._close_db_env()

    def get(self, idx):
        """Optimized data loading: Reuse LMDB connection and minimize unnecessary operations."""
        db_env = self._get_db_env()
        with db_env.begin() as txn:
            # Pre-create key for reuse
            key = int(idx).to_bytes(length=4, byteorder="big")
            data_dict = txn.get(key)
            
            if data_dict is None:
                raise KeyError(f"Index {idx} not found in database")
                
            data_dict = pickle.loads(data_dict)
            
            # Process data extraction and transformation in one step
            _, num_nodes, atoms, pos, Ham, ovlp, init_ham = (
                data_dict["id"],
                data_dict["num_nodes"],
                np.frombuffer(data_dict["atoms"], np.int32),
                np.frombuffer(data_dict["pos"], np.float64),
                np.frombuffer(data_dict["Ham"], np.float64),
                np.frombuffer(data_dict["ovlp"], np.float64),
                np.frombuffer(data_dict["init_ham"], np.float64),
            )
            
            # Memory-efficient reshape operation
            pos = pos.reshape(num_nodes, 3)
            num_orbitals = sum([5 if atom <= 2 else 14 for atom in atoms])
            Ham = Ham.reshape(num_orbitals, num_orbitals)
            init_ham = init_ham.reshape(num_orbitals, num_orbitals)
            ovlp = ovlp.reshape(num_orbitals, num_orbitals)

            data = self.get_mol(atoms, pos, Ham, ovlp, init_ham)
        return data




class QH9Dynamic(InMemoryDataset):
    """
    QH9Dynamic dataset for quantum chemistry calculations with dynamic geometries.
    
    This dataset contains molecular geometries with dynamic conformations and 
    precomputed Hamiltonian matrices, overlap matrices, and initial Hamiltonian matrices.
    The dataset supports both geometry-wise and molecule-wise splits.
    
    Attributes:
        url (dict): Google Drive URLs for different dataset versions
        version (str): Dataset version ('100k' or '300k')
        split (str): Dataset split type ('geometry' or 'mol')
        cal_orbital_and_energies (bool): Whether to calculate orbital energies
        _db_env: LMDB environment for optimized data access
        _lmdb_path: Cached path to LMDB database
    """
    url = {
        "100k": "https://drive.google.com/file/d/1SNWk0GD6Nt96qNAJJU2uedwWDQ4bbB1w/view?usp=sharing",
        "300k": "https://drive.google.com/file/d/1sbf-sFhh3ZmhXgTcN2ke_la39MaG0Yho/view?usp=sharing",
    }

    def __init__(
        self,
        root="datasets/",
        task="",
        split="geometry",
        version="300k",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        prefix="",
        num_chunks=10,
        chunk_idx=0,
        cal_orbital_and_energies=False,
    ):
        assert task in [""]
        self.version = version
        if self.version == "300k":
            self.folder = osp.join(root, "QH9Dynamic_300k" + prefix)
        elif self.version == "100k":
            self.folder = osp.join(root, "QH9Dynamic_100k" + prefix)
        else:
            print(
                f"Current version parameter is {version}, which is not included in [100k, 300k]."
            )
            print(f"Using 300k version instead...")
            self.folder = osp.join(root, "QH9Dynamic_300k" + prefix)
            self.version = "300k"

        self.split = split
        self.db_dir = os.path.join(self.folder, "processed")
        self.full_orbitals = 14
        self.orbital_mask = {}
        self.cal_orbital_and_energies = cal_orbital_and_energies
        
        # Variables for LMDB connection optimization
        self._db_env = None
        self._lmdb_path = None
        
        idx_1s_2s_2p = torch.tensor([0, 1, 3, 4, 5])
        orbital_mask_line1 = idx_1s_2s_2p
        orbital_mask_line2 = torch.arange(self.full_orbitals)
        for i in range(1, 11):
            self.orbital_mask[i] = orbital_mask_line1 if i <= 2 else orbital_mask_line2

        atom_list = ["H", "C", "N", "O", "F"]

        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        assert chunk_idx < num_chunks

        self.Q_dict = Onsite_3idx_Overlap_Integral(
            atom_list=atom_list, basis="def2-svp"
        ).Q_table()
        super(QH9Dynamic, self).__init__(
            self.folder, transform, pre_transform, pre_filter
        )
        self.train_mask, self.val_mask, self.test_mask = torch.load(
            self.processed_paths[0]
        )
        self.slices = {
            "id": torch.arange(
                self.train_mask.shape[0]
                + self.val_mask.shape[0]
                + self.test_mask.shape[0]
                + 1
            )
        }

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

    @property
    def raw_file_names(self):
        if self.version == "300k":
            return [f"QH9Dynamic_300k.db"]
        elif self.version == "100k":
            return [f"QH9Dynamic_100k.db"]

    @property
    def processed_file_names(self):
        if self.split == "geometry":
            return ["processed_QH9Dynamic_geometry.pt", "QH9Dynamic.lmdb/data.mdb"]
        elif self.split == "mol":
            return ["processed_QH9Dynamic_mol.pt", "QH9Dynamic.lmdb/data.mdb"]

    def process_row(self, data_chunk_idx):
        """
        Process a single row from the database chunk.
        
        Args:
            data_chunk_idx: Tuple of (row, index) from database chunk
            
        Returns:
            tuple: (key, data_dict) for LMDB storage
        """
        row, indice = data_chunk_idx
        atoms = np.frombuffer(row[3], np.int32)
        pos = np.frombuffer(row[4], np.float64) / BOHR2ANG
        ovlp, init_ham = calc_ovlp_and_ham_init(atoms, pos.reshape(-1, 3))
        ori_data_dict = {
            "id": int(np.frombuffer(row[0], dtype=np.int64)),
            "geo_id": row[1],
            "num_nodes": row[2],
            "atoms": row[3],
            "pos": row[4],  # Bohr
            "Ham": row[9],
            "ovlp": ovlp.tobytes(),
            "init_ham": init_ham.tobytes(),
        }
        data_dict = pickle.dumps(ori_data_dict)
        key = indice.item().to_bytes(length=4, byteorder="big")
        return key, data_dict

    def merge_lmbdbs(self, chunk_paths, merged_lmdb_path):
        print("Merging lmdb databases...", merged_lmdb_path)
        merge_env = lmdb.open(merged_lmdb_path, map_size=1048576000000)
        for ppath in chunk_paths:
            print(f"Merging {ppath}")
            part_env = lmdb.open(ppath, readonly=True)
            with part_env.begin() as part_txn, merge_env.begin(write=True) as merge_txn:
                cursor = part_txn.cursor()
                for key, value in cursor:
                    merge_txn.put(key, value)
        merge_env.close()
        print("Merging lmdb databases done.")

    def process(
        self,
        num_geometry_per_mol=100,
        num_train_geometry_per_mol=80,
        num_val_geometry_per_mol=10,
        num_mol=2998,
    ):
        if self.version == "300k":
            # assert pyscf.__version__ == "2.2.1"  # pyscf (2.2.1) (QH9-Dynamic-300k)
            # num_mol = 2698 
            num_mol = 2998 # Missing 100 molecules
        elif self.version == "100k":
            # assert pyscf.__version__ == "2.3.0"  # pyscf (2.3.0) (QH9-Dynamic-100k)
            num_mol = 999

        lmdb_chunk_name = f"QH9Dynamic_{self.chunk_idx}.lmdb"
        lmdb_completed_chunk_name = f"QH9Dynamic_{self.chunk_idx}_completed.lmdb"
        lmdb_chunk_path = os.path.join(self.db_dir, lmdb_chunk_name)
        lmdb_completed_chunk_path = os.path.join(self.db_dir, lmdb_completed_chunk_name)

        if not os.path.isdir(lmdb_completed_chunk_path):
            all_data = []
            for raw_file_name in self.raw_file_names:
                connection = Connection(os.path.join(self.root, "raw", raw_file_name))
                cursor = connection.cursor()
                data = cursor.execute("select * from data").fetchall()
                connection.close()
                all_data.extend(data)

            num_total = len(all_data)
            chunk_size = (num_total + (self.num_chunks - 1)) // self.num_chunks
            start_idx = self.chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_total)

            assert start_idx < end_idx and end_idx <= num_total
            assert start_idx < num_total and start_idx >= 0
            data_chunk = all_data[start_idx:end_idx]
            # fmt: off
            print(f"Total data points: {num_total}")
            print(f"Processing chunk {self.chunk_idx+1}/{self.num_chunks} with {len(data_chunk)} data points.")
            print(f"Data range: {start_idx} ~ {end_idx}")
            # fmt: on

            db_env = lmdb.open(lmdb_chunk_path, map_size=1048576000000)
            indices = np.arange(start_idx, end_idx)
            data_chunk_idx = list(zip(data_chunk, indices))
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                results = list(
                    tqdm(
                        executor.map(self.process_row, data_chunk_idx),
                        total=len(data_chunk),
                    )
                )
            with db_env.begin(write=True) as txn:
                for key, data_dict in results:
                    txn.put(key, data_dict)
            db_env.close()
            os.rename(
                lmdb_chunk_path, os.path.join(self.db_dir, lmdb_completed_chunk_name)
            )
        else:
            print(f"Chunk {self.chunk_idx} already exists. Skip the processing.")

        merged_path = os.path.join(self.db_dir, "QH9Dynamic.lmdb")
        if not os.path.isdir(merged_path):
            ppath_list = [
                os.path.join(self.db_dir, f"QH9Dynamic_{i}_completed.lmdb")
                for i in range(self.num_chunks)
            ]
            ppath_list = [ppath for ppath in ppath_list if os.path.exists(ppath)]
            if len(ppath_list) == self.num_chunks:
                self.merge_lmbdbs(ppath_list, merged_path)
            else:
                print(
                    f"Chunk {self.chunk_idx} is missing. Please check the processing."
                )
                print(f"Missing chunks: {ppath_list}")
                print(f"Total chunks: {self.num_chunks}")
                raise FileNotFoundError("Missing chunks.")
        else:
            print("Merged lmdb database exists. Jump the merging step.")

        db_env = lmdb.open(
            os.path.join(self.processed_dir, "QH9Dynamic.lmdb"), map_size=1048576000000
        )

        if self.split == "geometry":
            print("Geometry-wise splitting...")
            train_mask = np.array([], dtype=np.int64)
            val_mask = np.array([], dtype=np.int64)
            test_mask = np.array([], dtype=np.int64)
            cur_index = 0
            for raw_file_name in self.raw_file_names:
                connection = Connection(os.path.join(self.root, "raw", raw_file_name))
                cursor = connection.cursor()
                data = cursor.execute("select * from data").fetchall()
                for ind, row in enumerate(tqdm(data)):
                    if (
                        ind + 1
                    ) % num_geometry_per_mol == 0:  # Finish traversing one molecule
                        indices = np.random.RandomState(seed=ind).permutation(
                            num_geometry_per_mol
                        )  # Different random split for different molecules
                        train_mask_cur_mol = (
                            cur_index + indices[:num_train_geometry_per_mol]
                        )
                        val_mask_cur_mol = (
                            cur_index
                            + indices[
                                num_train_geometry_per_mol : num_train_geometry_per_mol
                                + num_val_geometry_per_mol
                            ]
                        )
                        test_mask_cur_mol = (
                            cur_index
                            + indices[
                                num_train_geometry_per_mol + num_val_geometry_per_mol :
                            ]
                        )
                        train_mask = np.concatenate((train_mask, train_mask_cur_mol))
                        val_mask = np.concatenate((val_mask, val_mask_cur_mol))
                        test_mask = np.concatenate((test_mask, test_mask_cur_mol))
                        cur_index += num_geometry_per_mol

        elif self.split == "mol":
            print("Molecule-wise splitting...")
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
            train_mask = np.where(np.isin(mol_id_array, train_mol_ids))[0].astype(
                np.int64
            )
            val_mask = np.where(np.isin(mol_id_array, val_mol_ids))[0].astype(np.int64)
            test_mask = np.where(np.isin(mol_id_array, test_mol_ids))[0].astype(
                np.int64
            )
        print("Saving...")

        torch.save((train_mask, val_mask, test_mask), self.processed_paths[0])

    def cut_matrix(self, matrix, atoms):
        all_diagonal_matrix_blocks = []
        all_non_diagonal_matrix_blocks = []
        all_diagonal_matrix_block_masks = []
        all_non_diagonal_matrix_block_masks = []
        col_idx = 0
        edge_index_full = []
        for idx_i, atom_i in enumerate(atoms):  # (src)
            row_idx = 0
            atom_i = atom_i.item()
            mask_i = self.orbital_mask[atom_i]
            for idx_j, atom_j in enumerate(atoms):  # (dst)
                atom_j = atom_j.item()
                if idx_i != idx_j:
                    edge_index_full.append([idx_j, idx_i])
                mask_j = self.orbital_mask[atom_j]

                if len(matrix.size()) == 2:
                    matrix_block = torch.zeros(
                        self.full_orbitals, self.full_orbitals
                    ).type(torch.float64)
                    matrix_block_mask = torch.zeros(
                        self.full_orbitals, self.full_orbitals
                    ).type(torch.float64)
                else:
                    len_ = matrix.size(-1)
                    matrix_block = torch.zeros(
                        self.full_orbitals, self.full_orbitals, len_
                    ).type(torch.float64)
                    matrix_block_mask = torch.zeros(
                        self.full_orbitals, self.full_orbitals, len_
                    ).type(torch.float64)

                extracted_matrix = matrix[
                    row_idx : row_idx + len(mask_j), col_idx : col_idx + len(mask_i)
                ]

                # for matrix_block
                tmp = matrix_block[mask_j]
                tmp[:, mask_i] = extracted_matrix
                matrix_block[mask_j] = tmp

                tmp = matrix_block_mask[mask_j]
                tmp[:, mask_i] = 1
                matrix_block_mask[mask_j] = tmp

                if idx_i == idx_j:
                    all_diagonal_matrix_blocks.append(matrix_block)
                    all_diagonal_matrix_block_masks.append(matrix_block_mask)
                else:
                    all_non_diagonal_matrix_blocks.append(matrix_block)
                    all_non_diagonal_matrix_block_masks.append(matrix_block_mask)
                row_idx = row_idx + len(mask_j)
            col_idx = col_idx + len(mask_i)
        return (
            torch.stack(all_diagonal_matrix_blocks, dim=0),
            torch.stack(all_non_diagonal_matrix_blocks, dim=0),
            torch.stack(all_diagonal_matrix_block_masks, dim=0),
            torch.stack(all_non_diagonal_matrix_block_masks, dim=0),
            torch.tensor(edge_index_full).transpose(-1, -2),
        )

    def get_mol(self, atoms, pos, Ham, ovlp, _init_ham):
        if self.cal_orbital_and_energies:
            e0, c0 = cal_orbital_and_energies(
                torch.tensor(ovlp).unsqueeze(0), torch.tensor(_init_ham).unsqueeze(0)
            )
            ef, cf = cal_orbital_and_energies(
                torch.tensor(ovlp).unsqueeze(0), torch.tensor(Ham).unsqueeze(0)
            )

        hamiltonian = torch.tensor(
            matrix_transform(Ham, atoms, convention="pyscf_def2svp"),
            dtype=torch.float64,
        )
        (
            diagonal_hamiltonian,
            non_diagonal_hamiltonian,
            diagonal_hamiltonian_mask,
            non_diagonal_hamiltonian_mask,
            edge_index_full,
        ) = self.cut_matrix(hamiltonian, atoms)

        init_ham = torch.tensor(
            matrix_transform(_init_ham, atoms, convention="pyscf_def2svp"),
            dtype=torch.float64,
        )

        (
            diagonal_init_ham,
            non_diagonal_init_ham,
            diagonal_init_ham_mask,
            non_diagonal_init_ham_mask,
            _,
        ) = self.cut_matrix(init_ham, atoms)

        overlap = torch.tensor(
            matrix_transform(ovlp, atoms, convention="pyscf_def2svp"),
            dtype=torch.float64,
        )
        diagonal_overlap, non_diagonal_overlap, _, _, _ = self.cut_matrix(
            overlap, atoms
        )

        Q = (
            torch.stack(
                [
                    torch.block_diag(*[self.Q_dict[z][l] for z in atoms])
                    for l in range(60)
                ]
            )
            .double()
            .numpy()
        )
        Q = (
            torch.from_numpy(matrix_transform(Q, atoms, convention="pyscf_def2svp"))
            .double()
            .permute(1, 2, 0)
        )
        H_size = hamiltonian.size(-1)
        Q[:, :, 16:40] = (
            Q[:, :, 16:40]
            .reshape(H_size, H_size, -1, 3)[:, :, :, [1, 2, 0]]
            .reshape(H_size, H_size, 24)
        )
        diagonal_Q, non_diagonal_Q, _, _, _ = self.cut_matrix(Q, atoms)

        data = Data(
            pos=torch.tensor(pos, dtype=torch.float64),
            atoms=torch.tensor(atoms, dtype=torch.int64).view(-1, 1),
            diagonal_hamiltonian=diagonal_hamiltonian,
            non_diagonal_hamiltonian=non_diagonal_hamiltonian,
            diagonal_hamiltonian_mask=diagonal_hamiltonian_mask,
            non_diagonal_hamiltonian_mask=non_diagonal_hamiltonian_mask,
            diagonal_init_ham=diagonal_init_ham,
            non_diagonal_init_ham=non_diagonal_init_ham,
            diagonal_init_ham_mask=diagonal_init_ham_mask,
            non_diagonal_init_ham_mask=non_diagonal_init_ham_mask,
            diagonal_overlap=diagonal_overlap,
            non_diagonal_overlap=non_diagonal_overlap,
            overlap=torch.tensor(ovlp, dtype=torch.float64).flatten(),  # pyscf
            overlap_dim=torch.tensor(ovlp.shape[0]).long().view(1, 1),
            diagonal_Q=diagonal_Q,
            non_diagonal_Q=non_diagonal_Q,
            edge_index_full=edge_index_full,
        )
        if self.cal_orbital_and_energies:
            data.e0 = e0.numpy()
            data.c0 = c0.numpy()
            data.ef = ef.numpy()
            data.cf = cf.numpy()
            data.h = Ham
            data.ovlp = ovlp

        return data

    def _get_db_env(self):
        """Lazy initialization and reuse of LMDB environment."""
        if self._db_env is None:
            self._lmdb_path = os.path.join(self.processed_dir, "QH9Dynamic.lmdb")
            self._db_env = lmdb.open(
                self._lmdb_path,
                readonly=True,
                lock=False,
            )
        return self._db_env

    def _close_db_env(self):
        """Safely close LMDB environment."""
        if self._db_env is not None:
            self._db_env.close()
            self._db_env = None

    def __enter__(self):
        """Context manager entry: Initialize LMDB environment."""
        self._get_db_env()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: Clean up LMDB environment."""
        self._close_db_env()

    def __del__(self):
        """Destructor: Clean up LMDB environment."""
        self._close_db_env()

    def get(self, idx):
        """Optimized data loading: Reuse LMDB connection and minimize unnecessary operations."""
        db_env = self._get_db_env()
        with db_env.begin() as txn:
            # Pre-generate key for reuse
            key = int(idx).to_bytes(length=4, byteorder="big")
            data_dict = txn.get(key)
            
            if data_dict is None:
                raise KeyError(f"Index {idx} not found in database")
                
            data_dict = pickle.loads(data_dict)
            
            # Process data extraction and transformation in one go
            (_, num_nodes, atoms, pos, Ham, ovlp, init_ham) = (
                data_dict["id"],
                data_dict["num_nodes"],
                np.frombuffer(data_dict["atoms"], np.int32),
                np.frombuffer(data_dict["pos"], np.float64),
                np.frombuffer(data_dict["Ham"], np.float64),
                np.frombuffer(data_dict["ovlp"], np.float64),
                np.frombuffer(data_dict["init_ham"], np.float64),
            )
            
            # Memory-efficient reshape operations and unit conversion
            pos = pos.reshape(num_nodes, 3)
            pos = pos / BOHR2ANG  # Convert from Bohr to Angstrom
            num_orbitals = sum([5 if atom <= 2 else 14 for atom in atoms])
            Ham = Ham.reshape(num_orbitals, num_orbitals)
            init_ham = init_ham.reshape(num_orbitals, num_orbitals)
            ovlp = ovlp.reshape(num_orbitals, num_orbitals)
            
            data = self.get_mol(atoms, pos, Ham, ovlp, init_ham)
        return data


def cal_orbital_and_energies(overlap_matrix, full_hamiltonian, EPS=1e-8):
    """
    Calculate orbital energies and coefficients from overlap and Hamiltonian matrices.
    
    Args:
        overlap_matrix: Overlap matrix (S)
        full_hamiltonian: Full Hamiltonian matrix (H)
        EPS: Small value to avoid numerical issues with eigenvalues
    
    Returns:
        tuple: (orbital_energies, orbital_coefficients)
    """
    eigvals, eigvecs = torch.linalg.eigh(overlap_matrix)
    eps = EPS * torch.ones_like(eigvals)
    eigvals = torch.where(eigvals > EPS, eigvals, eps)
    frac_overlap = eigvecs / torch.sqrt(eigvals).unsqueeze(-2)

    Fs = torch.bmm(
        torch.bmm(frac_overlap.transpose(-1, -2), full_hamiltonian), frac_overlap
    )
    orbital_energies, orbital_coefficients = torch.linalg.eigh(Fs)
    orbital_coefficients = torch.bmm(frac_overlap, orbital_coefficients)
    return orbital_energies, orbital_coefficients





def calc_ovlp_and_ham_init(atoms, pos, init="minao"):
    """
    Calculate overlap matrix and initial Hamiltonian matrix using PySCF.
    
    Args:
        atoms: Atomic numbers for the molecule
        pos: Atomic positions in Angstrom
        init: Initial guess method ('minao' or '1e')
    
    Returns:
        tuple: (overlap_matrix, initial_hamiltonian_matrix)
    """
    mol = gto.Mole()
    if len(pos.shape) == 1:
        pos = pos.reshape(-1, 3)
    t = [[atoms[atom_idx], pos[atom_idx]] for atom_idx in range(len(atoms))]
    mol.build(verbose=0, atom=t, basis="def2svp", unit="ang")

    ovlp = mol.intor("int1e_ovlp")
    mf_hf = dft.RKS(mol)
    mf_hf.xc = "b3lyp"
    mf_hf.basis = "def2svp"
    if init == "minao":
        init_dm = mf_hf.init_guess_by_minao()
    elif init == "1e":
        init_dm = mf_hf.init_guess_by_1e()
    # init_ham = scf.hf.get_fock(mf_hf, dm=init_dm)
    init_ham = mf_hf.get_fock(dm=init_dm)

    return ovlp.astype("float64"), init_ham.astype("float64")





if __name__ == "__main__":
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
    parser.add_argument("--num_chunks", type=int, default=10)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--pdb", action="store_true", default=False)

    print(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset"))
    args = parser.parse_args()
    assert args.name in ["QH9Stable", "QH9Dynamic"]
    if args.name == "QH9Stable":
        assert args.split in ["random", "size_ood"]
        dataset = QH9Stable(
            root=args.root,
            split=args.split,
            num_chunks=args.num_chunks,
            chunk_idx=args.chunk_idx,
            prefix=args.prefix,
        )
    elif args.name == "QH9Dynamic":
        assert args.version in ["100k", "300k"]

        dataset = QH9Dynamic(
            root=args.root,
            version=args.version,
            split=args.split,
            num_chunks=args.num_chunks,
            chunk_idx=args.chunk_idx,
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
