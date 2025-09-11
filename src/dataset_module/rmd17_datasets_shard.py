import numpy as np
import lmdb
import pickle
import os
import logging
import json
import gdown
import torch
from tqdm.rich import tqdm
import random
from typing import Union, List

from common.metric import cal_orbital_and_energies
from common.matrix_transforms import pack_upper_triangle, unpack_upper_triangle, _matrix_transform_single, get_convention_dict, _cut_matrix_3d, _cut_matrix_3d_last
from dataset_module.lmdb_shard import LMDBShard_maker_db
from common.dft_utils import calc_overlap_and_init_hamiltonian, calc_dm0

from torch_geometric.data import InMemoryDataset, Data
from utils import AOData, Onsite_3idx_Overlap_Integral, build_molecule, build_AO_index

# Conversion factor from Bohr to Angstrom
ANG2BOHR = 1.8897259886
BOHR2ANG = 1 / ANG2BOHR # 0.52917721067 - Bohr to Angstrom conversion (MD17)

rMD17_url = (
    "https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038"
)

# Under development