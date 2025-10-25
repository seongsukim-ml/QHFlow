"""
Matrix Transforms Module

This module provides utility functions for Hamiltonian and overlap matrix transformations,
including orbital transformations and matrix transformations.
"""

import torch
import numpy as np
from argparse import Namespace

# Periodic Table of Elements
# -----------------------------------------------------------------------------------------------
#   │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │ 8  │ 9  │ 10 │ 11 │ 12 │ 13 │ 14 │ 15 │ 16 │ 17 │ 18 │
#   ┌────┐                                                                               ┌────┐
# 1 │ H  │ 2                                                      13   14   15   16   17 │ He │
#   │ 1  │                                                                               │ 2  │
#   ├────┼────┐                                                 ┌────┬────┬────┬────┬────┼────┤
# 2 │ Li │ Be │                                                 │ B  │ C  │ N  │ O  │ F  │ Ne │
#   │ 3  │ 4  │                                                 │ 5  │ 6  │ 7  │ 8  │ 9  │ 10 │
#   ├────┼────┤                                                 ├────┼────┼────┼────┼────┼────┤
# 3 │ Na │ Mg │ 3    4    5    6    7    8    9    10   11   12 │ Al │ Si │ P  │ S  │ Cl │ Ar │
#   │ 11 │ 12 │                                                 │ 13 │ 14 │ 15 │ 16 │ 17 │ 18 │
#   ├────┼────┼────┬────┬────┬────┬────┬────┬────┬────┬────┬────┼────┼────┼────┼────┼────┼────┤
# 4 │ K  │ Ca │ Sc │ Ti │ V  │ Cr │ Mn │ Fe │ Co │ Ni │ Cu │ Zn │ Ga │ Ge │ As │ Se │ Br │ Kr │
#   │ 19 │ 20 │ 21 │ 22 │ 23 │ 24 │ 25 │ 26 │ 27 │ 28 │ 29 │ 30 │ 31 │ 32 │ 33 │ 34 │ 35 │ 36 │
#   ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
# 5 │ Rb │ Sr │ Y  │ Zr │ Nb │ Mo │ Tc │ Ru │ Rh │ Pd │ Ag │ Cd │ In │ Sn │ Sb │ Te │ I  │ Xe │
#   │ 37 │ 38 │ 39 │ 40 │ 41 │ 42 │ 43 │ 44 │ 45 │ 46 │ 47 │ 48 │ 49 │ 50 │ 51 │ 52 │ 53 │ 54 │
#   ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
# 6 │ Cs │ Ba │ L* │ Hf │ Ta │ W  │ Re │ Os │ Ir │ Pt │ Au │ Hg │ Tl │ Pb │ Bi │ Po │ At │ Rn │
#   │ 55 │ 56 │ -- │ 72 │ 73 │ 74 │ 75 │ 76 │ 77 │ 78 │ 79 │ 80 │ 81 │ 82 │ 83 │ 84 │ 85 │ 86 │
#   ├────┼────┼────┼────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
# 7 │ Fr │ Ra │ A* │
#   │ 87 │ 88 │ -- │
#   └────┴────┴────┘
# ----------------------------------------------------------------------------------------------
# L* (Lanthanide)
#   ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
# 6 │ La │ Ce │ Pr │ Nd │ Pm │ Sm │ Eu │ Gd │ Tb │ Dy │ Ho │ Er │ Tm │ Yb │ Lu │
#   │ 57 │ 58 │ 59 │ 60 │ 61 │ 62 │ 63 │ 64 │ 65 │ 66 │ 67 │ 68 │ 69 │ 70 │ 71 │
#   └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
# A* (Actinide)
#   ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
# 7 │ Ac │ Th │ Pa │ U  │ Np │ Pu │ Am │ Cm │ Bk │ Cf │ Es │ Fm │ Md │ No │ Lr │
#   │ 89 │ 90 │ 91 │ 92 │ 93 │ 94 │ 95 │ 96 │ 97 │ 98 │ 99 │ 100│ 101│ 102│ 103│
#   └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘


# Atomic numbers 1 to 103
CHEMICAL_SYMBOLS = [
    "n",
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At","Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"
    ] 

convention_dict = {
    "psi4_def2-tzvppd_to_pyscf": Namespace(
        atom_to_orbitals_map={
            1:  'ssspppd', # H (3s + 3p + 1d) = 3 + 9 + 5 = 17
            6:  'sssssspppdddf', # C  (6s + 3p + 3d + 1f) = 6 + 9 + 15 + 7 = 37
            7:  'sssssspppdddf', # N  (6s + 3p + 3d + 1f) = 6 + 9 + 15 + 7 = 37
            8:  'ssssssppppdddf', # O (6s + 4p + 3d + 1f) = 6 + 12 + 15 + 7 = 40
            9:  'ssssssppppdddf', # F (6s + 4p + 3d + 1f) = 6 + 12 + 15 + 7 = 40
        },
        orbital_idx_map={
            's': [0],
            'p': [1, 2, 0],
            'd': [4, 2, 0, 1, 3],
            'f': [6, 4, 2, 0, 1, 3, 5]
        },
        orbital_sign_map={
            's': [1],
            'p': [1, 1, 1],
            'd': [1, 1, 1, 1, 1],
            'f': [1, 1, 1, 1, 1, 1, 1],
        },
        orbital_order_map={
            1: list(range(7)),
            6: list(range(13)),
            7: list(range(13)),
            8: list(range(14)),
            9: list(range(14)),
        },
        max_block_size= 40, # 6s + 4p + 3d + 1f = 6 + 12 + 15 + 7 = 40
    ),
    'pyscf_def2-tzvp_to_e3nn': Namespace(
        atom_to_orbitals_map={
            1:  'sssp', # H
            6:  'ssssspppddf', # C
            7:  'ssssspppddf', # N
            8:  'ssssspppddf', # O
            9:  'ssssspppddf', # F
            15: 'ssssspppppddf', # P
            16: 'ssssspppppddf', # S
            17: 'ssssspppppddf', # Cl
        },      
        orbital_idx_map={
            's': [0],
            'p': [1, 2, 0],
            'd': [0, 1, 2, 3, 4],
            'f': [0, 1, 2, 3, 4, 5, 6],
        },
        orbital_sign_map={
            's': [1],
            'p': [1, 1, 1],
            'd': [1, 1, 1, 1, 1],
            'f': [1, 1, 1, 1, 1, 1, 1],
        },
        orbital_order_map={
            1:  [0, 1, 2, 3],   
            6:  list(range(11)),
            7:  list(range(11)),
            8:  list(range(11)),
            9:  list(range(11)),
            15: list(range(13)),
            16: list(range(13)),
            17: list(range(13)),
        },
        max_block_size= 37, # 5s + 5p + 2d + 1f = 5 + 15 + 10 + 7 = 37
    ),
    "pyscf_631G_to_e3nn": Namespace(
        # 6-31G basis set convention used by PySCF
        # p orbitals: [px, py, pz] -> reordered to [pz, px, py] for compatibility
        atom_to_orbitals_map={
            1: "ss",
            6: "ssspp",
            7: "ssspp",
            8: "ssspp",
            9: "ssspp",
        },
        orbital_idx_map={
            "s": [0],
            "p": [1, 2, 0],   # p: [pz, px, py]
            "d": [0, 1, 2, 3, 4]
            },
        orbital_sign_map={
            "s": [1],
            "p": [1, 1, 1],
            "d": [1, 1, 1, 1, 1],
        },
        orbital_order_map={
            1: [0, 1],           # H: 2 orbitals (s, s)
            6: [0, 1, 2, 3, 4],  # C: 5 orbitals (s, s, s, p, p)
            7: [0, 1, 2, 3, 4],  # N: 5 orbitals (s, s, s, p, p)
            8: [0, 1, 2, 3, 4],  # O: 5 orbitals (s, s, s, p, p)
            9: [0, 1, 2, 3, 4],  # F: 5 orbitals (s, s, s, p, p)
        },
        max_block_size=9, # 3s + 2p = 3 + 6 = 9
    ),
    "pyscf_def2svp_to_e3nn": Namespace(
        # def2-SVP basis set convention used by PySCF
        # p orbitals: [px, py, pz] -> reordered to [py, pz, px] for compatibility
        atom_to_orbitals_map={
            1: "ssp",      # H: 3 orbitals (s, s, p)
            6: "sssppd",   # C: 6 orbitals (s, s, s, p, p, d)
            7: "sssppd",   # N: 6 orbitals (s, s, s, p, p, d)
            8: "sssppd",   # O: 6 orbitals (s, s, s, p, p, d)
            9: "sssppd",   # F: 6 orbitals (s, s, s, p, p, d)
        },
        orbital_idx_map={
            "s": [0],
            "p": [1, 2, 0],
            "d": [0, 1, 2, 3, 4],
        },  # p: [py, pz, px]
        orbital_sign_map={
            "s": [1],
            "p": [1, 1, 1],
            "d": [1, 1, 1, 1, 1],
        },
        orbital_order_map={
            1: [0, 1, 2],      # H: 3 orbitals (s, s, p)
            6: [0, 1, 2, 3, 4, 5],  # C: 6 orbitals (s, s, s, p, p, d)
            7: [0, 1, 2, 3, 4, 5],  # N: 6 orbitals (s, s, s, p, p, d)
            8: [0, 1, 2, 3, 4, 5],  # O: 6 orbitals (s, s, s, p, p, d)
            9: [0, 1, 2, 3, 4, 5],  # F: 6 orbitals (s, s, s, p, p, d)
        },
        max_block_size= 14, # 3s + 2p + 1d = 3 + 6 + 5 = 14
    ),
    'e3nn_to_pyscf_def2-tzvp': Namespace(
        atom_to_orbitals_map={
            1:  'sssp', # H
            6:  'ssssspppddf', # C
            7:  'ssssspppddf', # N
            8:  'ssssspppddf', # O
            9:  'ssssspppddf', # F
            15: 'ssssspppppddf', # P
            16: 'ssssspppppddf', # S
            17: 'ssssspppppddf', # Cl
        },
        orbital_idx_map={
            's': [0],
            'p': [2, 0, 1],
            'd': [0, 1, 2, 3, 4],
            'f': [0, 1, 2, 3, 4, 5, 6],
        },
        orbital_sign_map={
            's': [1],
            'p': [1, 1, 1],
            'd': [1, 1, 1, 1, 1],
            'f': [1, 1, 1, 1, 1, 1, 1],
        },
        orbital_order_map={
            1:  [0, 1, 2, 3],   
            6:  list(range(11)),
            7:  list(range(11)),
            8:  list(range(11)),
            9:  list(range(11)),
            15: list(range(13)),
            16: list(range(13)),
            17: list(range(13)),
        },
        max_block_size= 37, # 5s + 5p + 2d + 1f = 5 + 15 + 10 + 7 = 37
    ),
    "e3nn_to_pyscf_def2svp": Namespace(
        # Special convention to convert back to PySCF's native orbital ordering
        # This is used when you have matrices in a different convention and need to
        # convert them back to PySCF's expected format for further processing
        # 
        # Key differences from pyscf_def2svp:
        # - Same orbital types (def2-SVP basis) but different p-orbital ordering
        # - p orbitals: [py, pz, px] -> reordered to [px, py, pz] (PySCF native)
        # - This allows seamless integration with PySCF calculations
        #
        # Use case: Convert matrices from other software (e.g., ORCA, Gaussian) 
        # back to PySCF format for density matrix calculations or further analysis
        atom_to_orbitals_map={
            1: "ssp",      # H: 3 orbitals (s, s, p)
            6: "sssppd",   # C: 6 orbitals (s, s, s, p, p, d)
            7: "sssppd",   # N: 6 orbitals (s, s, s, p, p, d)
            8: "sssppd",   # O: 6 orbitals (s, s, s, p, p, d)
            9: "sssppd",   # F: 6 orbitals (s, s, s, p, p, d)
        },
        orbital_idx_map={
            "s": [0],
            "p": [2, 0, 1],
            "d": [0, 1, 2, 3, 4],
        },  # p: [px, py, pz] (PySCF native)
        orbital_sign_map={
            "s": [1],
            "p": [1, 1, 1],
            "d": [1, 1, 1, 1, 1],
        },
        orbital_order_map={
            1: [0, 1, 2],           # H: 3 orbitals (s, s, p)
            6: [0, 1, 2, 3, 4, 5],  # C: 6 orbitals (s, s, s, p, p, d)
            7: [0, 1, 2, 3, 4, 5],  # N: 6 orbitals (s, s, s, p, p, d)
            8: [0, 1, 2, 3, 4, 5],  # O: 6 orbitals (s, s, s, p, p, d)
            9: [0, 1, 2, 3, 4, 5],  # F: 6 orbitals (s, s, s, p, p, d)
        },
        max_block_size= 14,
    ),
}

# Alias for compatibility
convention_dict["back2pyscf"]      = convention_dict["e3nn_to_pyscf_def2svp"]
convention_dict["pyscf_def2svp"]   = convention_dict["pyscf_def2svp_to_e3nn"]
convention_dict["pyscf_631G"]      = convention_dict["pyscf_631G_to_e3nn"]
convention_dict["pyscf_def2-tzvp"] = convention_dict["pyscf_def2-tzvp_to_e3nn"]

# For MD17 dataset. Original MD17 dataset is from ORCA. (Following the convention of QHNet)
convention_dict["orca_to_e3nn"]    = convention_dict["e3nn_to_pyscf_def2svp"]

def get_convention_dict():
    """Get the dictionary of orbital convention mappings.
    
    Returns:
        dict: Dictionary containing orbital convention rules for different
              basis sets and software packages.
    """
    return convention_dict

def _get_orbital_mask(basis = "def2-svp"):
    """Get orbital masks for different atomic numbers.
    
    Args:
        ORBITAL_1S_2S_INDICES (torch.Tensor, optional): Indices for 1s and 2s orbitals.
            Defaults to [0, 1].
        ORBITAL_2P_INDICES (torch.Tensor, optional): Indices for 2p orbitals.
            Defaults to [3, 4, 5].
        ORBITAL_MASK_SIZE_LINE2 (int, optional): Size of orbital mask for line 2 elements.
            Defaults to 14.
    
    Returns:
        dict: Dictionary mapping atomic numbers to their orbital masks.
    """
    assert basis in ["def2-svp", "def2-tzvp"], f"Invalid basis: {basis}, only def2-svp and def2-tzvp are supported now"
    orbital_mask = {}
    
    if basis == "631G":
        pass

    elif basis == "def2-svp":        
        MAX_ORBITAL_LENGTH = 14
        MAX_ATOMIC_NUMBER = 9
        DEFAULT_ORBITAL_INDICES = torch.arange(MAX_ORBITAL_LENGTH)
        orbital_mask[1] = torch.tensor([0, 1, 3, 4, 5]) # ssp
        orbital_mask[6] = DEFAULT_ORBITAL_INDICES
        orbital_mask[7] = DEFAULT_ORBITAL_INDICES
        orbital_mask[8] = DEFAULT_ORBITAL_INDICES
        orbital_mask[9] = DEFAULT_ORBITAL_INDICES
        
    elif basis == "def2-tzvp":
        MAX_ORBITAL_LENGTH = 37
        MAX_ATOMIC_NUMBER = 17
        DEFAULT_ORBITAL_INDICES_1 = torch.tensor([
            0, 1, 2, 3, 4,      # 1s-5s
            5, 6, 7,            # 2p
            8, 9, 10,           # 3p
            11, 12, 13,         # 4p (skip 5p,6p)
            20, 21, 22, 23, 24, # 3d
            25, 26, 27, 28, 29, # 4d
            30, 31, 32, 33, 34, 35, 36 # 4f
        ])
        DEFAULT_ORBITAL_INDICES_2 = torch.arange(MAX_ORBITAL_LENGTH)
        orbital_mask[1] = torch.tensor([0, 1, 2, 5, 6, 7]) # sssp
        orbital_mask[6] = DEFAULT_ORBITAL_INDICES_1
        orbital_mask[7] = DEFAULT_ORBITAL_INDICES_1
        orbital_mask[8] = DEFAULT_ORBITAL_INDICES_1
        orbital_mask[9] = DEFAULT_ORBITAL_INDICES_1
        orbital_mask[15] = DEFAULT_ORBITAL_INDICES_2
        orbital_mask[16] = DEFAULT_ORBITAL_INDICES_2
        orbital_mask[17] = DEFAULT_ORBITAL_INDICES_2

    return orbital_mask

def _build_final_matrix(
    data,
    diagonal_matrix,
    non_diagonal_matrix,
    orbital_mask,
):
    """Build final matrix from diagonal and non-diagonal blocks.
    
    Args:
        data: PyG Data object containing graph information.
        diagonal_matrix: Diagonal matrix blocks.
        non_diagonal_matrix: Non-diagonal matrix blocks.
        orbital_mask: Dictionary mapping atomic numbers to orbital indices.
    
    Returns:
        list: List of final matrices, one per graph in the batch.
    """
    final_matrix = []
    if hasattr(data, "full_edge_index"):
        dst, src = data.full_edge_index
    else:
        dst, src = data.edge_index_full
    for graph_idx in range(data.ptr.shape[0] - 1):
        matrix_block_col = []
        for src_idx in range(data.ptr[graph_idx], data.ptr[graph_idx + 1]):
            matrix_col = []
            for dst_idx in range(data.ptr[graph_idx], data.ptr[graph_idx + 1]):
                if src_idx == dst_idx:
                    matrix_col.append(
                        diagonal_matrix[src_idx]
                        .index_select(
                            -2, orbital_mask[data.atoms[dst_idx].item()]
                        )
                        .index_select(
                            -1, orbital_mask[data.atoms[src_idx].item()]
                        )
                    )
                else:
                    mask1 = src == src_idx
                    mask2 = dst == dst_idx
                    index = torch.where(mask1 & mask2)[0].item()

                    matrix_col.append(
                        non_diagonal_matrix[index]
                        .index_select(
                            -2, orbital_mask[data.atoms[dst_idx].item()]
                        )
                        .index_select(
                            -1, orbital_mask[data.atoms[src_idx].item()]
                        )
                    )
            matrix_block_col.append(torch.cat(matrix_col, dim=-2))
        mat_res = torch.cat(matrix_block_col, dim=-1)
        final_matrix.append(mat_res)
    return final_matrix


def _matrix_transform_list(hamiltonian_list, data, convention_rule):
    """Transform matrix between different orbital conventions - CUDA optimized version.
    
    This function transforms a list of Hamiltonian matrices between different orbital conventions,
    optimized for CUDA execution. It handles the transformation for each graph in a batch separately.
    
    Args:
        hamiltonian_list (list): List of Hamiltonian matrices to transform, one per graph.
        data: PyG Data object containing graph information like atoms and batch indices.
        convention_rule (Namespace): Orbital convention to use:
            - 'pyscf_def2-tzvp': def2-TZVP basis set convention (p: [pz, px, py])
            - 'pyscf_631G': 6-31G basis set convention (p: [pz, px, py])
            - 'pyscf_def2svp': def2-SVP basis set convention (p: [py, pz, px])
            - 'back2pyscf': Convert back to PySCF native convention (p: [pz, px, py])
              * Use this when you have matrices from other software and need to
                convert them back to PySCF format for density matrix calculations
              * Same basis as def2-SVP but with PySCF's native p-orbital ordering
    
    Returns:
        list: List of transformed Hamiltonian matrices, one per graph in the batch.
    """
    final_matrix_list = []
    
    for graph_idx in range(data.ptr.shape[0] - 1):
        hamiltonian = hamiltonian_list[graph_idx]
        atoms = data.atoms[data.batch == graph_idx]
        mat_res = _matrix_transform_single(hamiltonian, atoms, convention_rule)
        final_matrix_list.append(mat_res)
        
    return final_matrix_list

def _matrix_transform_single(hamiltonian, atoms, convention_rule):
    """Transform matrices according to orbital convention using PyTorch.
    
    This function reorders and applies sign changes to orbital matrices based on
    different quantum chemistry software conventions. Different software packages
    use different orbital ordering and sign conventions.
    
    Example:
        Transform from 6-31G to def2-SVP convention:
        - 6-31G: p orbitals ordered as [px, py, pz] 
        - def2-SVP: p orbitals ordered as [py, pz, px]
        - This function handles the reordering and sign changes
    
    Args:
        hamiltonian (torch.Tensor): Input matrices to transform, shape (..., n_orb, n_orb).
        atoms (torch.Tensor): Atomic numbers for the molecule (e.g., [6, 1, 1, 1] for CH3).
        convention_rule (Namespace): Orbital convention to use:
            - 'pyscf_def2-tzvp': def2-TZVP basis set convention (p: [pz, px, py])
            - 'pyscf_631G': 6-31G basis set convention (p: [pz, px, py])
            - 'pyscf_def2svp': def2-SVP basis set convention (p: [py, pz, px])
            - 'back2pyscf': Convert back to PySCF native convention (p: [pz, px, py])
              * Use this when you have matrices from other software and need to
                convert them back to PySCF format for density matrix calculations
              * Same basis as def2-SVP but with PySCF's native p-orbital ordering
    
    Returns:
        torch.Tensor: Transformed matrices with reordered orbitals and applied sign changes.
    """
    conv = convention_rule
    
    # Get device from hamiltonian tensor
    device = hamiltonian.device
    dtype = hamiltonian.dtype
    
    orbitals = ""
    orbitals_order = []
    for a in atoms:
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[a.item()]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a.item()]]

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        map_sign = conv.orbital_sign_map[orb]
        # Convert to torch tensors directly on the correct device
        transform_indices.append(torch.tensor(map_idx, device=device, dtype=torch.long) + offset)
        transform_signs.append(torch.tensor(map_sign, device=device, dtype=dtype))

    # Reorder according to orbitals_order
    transform_indices = [transform_indices[idx] for idx in orbitals_order]
    transform_signs = [transform_signs[idx] for idx in orbitals_order]
    
    # Concatenate using torch.cat instead of np.concatenate
    transform_indices = torch.cat(transform_indices)
    transform_signs = torch.cat(transform_signs)

    # Apply transformation using torch indexing
    hamiltonian_new = hamiltonian[..., transform_indices, :]
    hamiltonian_new = hamiltonian_new[..., :, transform_indices]
    
    # Apply signs using torch operations
    hamiltonian_new = hamiltonian_new * transform_signs.unsqueeze(-1)
    hamiltonian_new = hamiltonian_new * transform_signs.unsqueeze(-2)

    return hamiltonian_new

def matrix_transform_single(hamiltonian, atoms, convention="pyscf_def2svp"):
    """Transform matrix between different orbital conventions - CUDA optimized version.
    
    This function reorders and transforms the hamiltonian matrix according to the
    specified orbital convention, handling different basis set orderings.
    
    Args:
        hamiltonian (torch.Tensor): Hamiltonian matrix to transform.
        atoms (torch.Tensor): Atomic numbers tensor.
        convention (str): Orbital convention to use (default: "pyscf_def2svp").
        
    Returns:
        torch.Tensor: Transformed hamiltonian matrix.
        
    Raises:
        AssertionError: If convention is not in CONVENTION_DICT.
    """
    assert convention in get_convention_dict(), f"Invalid convention: {convention}"
    conv = get_convention_dict()[convention]
    return _matrix_transform_single(hamiltonian, atoms, conv)

def cut_matrix(matrix, atoms, orbital_mask, full_orbitals, last_dim=False):
    """Cut matrix into atomic blocks with optimized performance.
    
    This function takes a molecular orbital matrix and splits it into atomic blocks.
    Each block represents interactions between specific atoms. The matrix is divided
    into diagonal blocks (same atom interactions) and non-diagonal blocks (different atom interactions).
    
    Algorithm Overview:
        Input matrix structure for CH3 molecule (C=5 orbitals, H=2 orbitals each):
        
        Full Matrix (11x11):
        ┌─────────────────────────────────────────┐
        │ C-C │ C-H │ C-H │ C-H │  ← C interactions
        ├─────┼─────┼─────┼─────┤
        │ H-C │ H-H │ H-H │ H-H │  ← H1 interactions  
        ├─────┼─────┼─────┼─────┤
        │ H-C │ H-H │ H-H │ H-H │  ← H2 interactions
        ├─────┼─────┼─────┼─────┤
        │ H-C │ H-H │ H-H │ H-H │  ← H3 interactions
        └─────────────────────────────────────────┘
        
        Output blocks:
        - Diagonal: [C-C(5x5), H-H(2x2), H-H(2x2), H-H(2x2)]
        - Non-diagonal: [C-H(5x2), C-H(5x2), C-H(5x2), H-C(2x5), H-C(2x5), H-C(2x5), ...]
    
    Example:
        For a molecule with atoms [C, H, H, H], the matrix is split into:
        - Diagonal blocks: C-C, H-H, H-H, H-H interactions
        - Non-diagonal blocks: C-H, H-C, H-H (different atoms) interactions
    
    Args:
        matrix (torch.Tensor): Input matrix tensor of shape (n_orb, n_orb) or (n_orb, n_orb, n_features).
            - 2D: Single property matrix (e.g., Hamiltonian, overlap)
            - 3D: Multiple property matrices stacked along last dimension
        atoms (torch.Tensor): Atomic numbers tensor (e.g., [6, 1, 1, 1] for CH3).
        orbital_mask (dict): Dictionary mapping atomic numbers to orbital indices.
            e.g., {6: [0,1,2,3,4], 1: [0,1]} for C(5 orbitals) and H(2 orbitals).
        full_orbitals (int): Maximum number of orbitals per atom (used for padding).
        last_dim (bool): If True, features are stacked along the last dimension.
        
    Returns:
        tuple: (diagonal_blocks, non_diagonal_blocks, diagonal_masks, non_diagonal_masks, edge_index).
            - diagonal_blocks: Blocks for same-atom interactions
            - non_diagonal_blocks: Blocks for different-atom interactions  
            - diagonal_masks: Binary masks indicating valid orbital positions in diagonal blocks
            - non_diagonal_masks: Binary masks indicating valid orbital positions in non-diagonal blocks
            - edge_index: Graph connectivity (which atoms interact with which)
    """
    # Fast dispatch based on matrix dimensions (no type checking for performance)
    # This avoids runtime overhead of isinstance() and shape validation
    if len(matrix.shape) == 2:
        return _cut_matrix_2d(matrix, atoms, orbital_mask, full_orbitals)
    else:
        if last_dim:
            return _cut_matrix_3d_last(matrix, atoms, orbital_mask, full_orbitals)
        else:
            return _cut_matrix_3d(matrix, atoms, orbital_mask, full_orbitals)

def _cut_matrix_2d(matrix, atoms, orbital_mask, full_orbitals):
    """Optimized 2D matrix cutting - no runtime checks for maximum performance.
    
    This function processes 2D matrices (single property like Hamiltonian or overlap matrix).
    It's separated from 3D case to avoid conditional checks in the hot loop.
    
    Args:
        matrix (torch.Tensor): 2D input matrix tensor.
        atoms (torch.Tensor): Atomic numbers tensor.
        orbital_mask (dict): Dictionary mapping atomic numbers to orbital indices.
        full_orbitals (int): Maximum number of orbitals per atom.
    
    Returns:
        tuple: (diagonal_blocks, non_diagonal_blocks, diagonal_masks, non_diagonal_masks, edge_index).
    """
    # Get tensor properties once to avoid repeated access
    device = matrix.device
    dtype = matrix.dtype
    
    # Pre-allocate lists for better memory efficiency
    # These will store the final atomic blocks
    diagonal_blocks = []      # Same-atom interactions (e.g., C-C, H-H)
    non_diagonal_blocks = []  # Different-atom interactions (e.g., C-H, H-C)
    diagonal_masks = []       # Binary masks for diagonal blocks
    non_diagonal_masks = []   # Binary masks for non-diagonal blocks
    edge_indices = []         # Graph connectivity information
    
    # Pre-compute values to avoid repeated .item() calls in loops
    # This is a key optimization - .item() is expensive when called repeatedly
    atom_values = [atom.item() for atom in atoms]
    orbital_masks = [orbital_mask[atom_val] for atom_val in atom_values]
    orbital_lengths = [len(mask) for mask in orbital_masks]
    
    # Matrix cutting algorithm:
    # We iterate through all atom pairs (src_idx, dst_idx) and extract
    # the corresponding submatrix from the full orbital matrix
    col_idx = 0  # Column index in the full matrix
    
    for src_idx, (src_mask, src_length) in enumerate(zip(orbital_masks, orbital_lengths)):
        row_idx = 0  # Row index in the full matrix
        
        for dst_idx, (dst_mask, dst_length) in enumerate(zip(orbital_masks, orbital_lengths)):
            # Build edge index for graph representation
            # Only non-diagonal pairs create edges (different atoms)
            if src_idx != dst_idx:
                edge_indices.append([dst_idx, src_idx])  # [source, target] format
            
            # Create empty blocks with proper shape and device/dtype
            # full_orbitals is used for padding to ensure all blocks have same size
            matrix_block = torch.zeros((full_orbitals, full_orbitals), device=device, dtype=dtype)
            matrix_block_mask = torch.zeros((full_orbitals, full_orbitals), device=device, dtype=dtype)
            
            # Extract the relevant submatrix from the full matrix
            # This is the actual orbital interaction data between src and dst atoms
            extracted_matrix = matrix[row_idx:row_idx + dst_length, col_idx:col_idx + src_length]
            
            # Fill the block using orbital masks
            # dst_mask and src_mask specify which orbitals are actually present
            matrix_block[dst_mask, src_mask] = extracted_matrix
            matrix_block_mask[dst_mask, src_mask] = 1  # Mark valid positions
            
            # Store blocks based on whether they're diagonal or not
            if src_idx == dst_idx:
                # Same atom interactions (diagonal blocks)
                diagonal_blocks.append(matrix_block)
                diagonal_masks.append(matrix_block_mask)
            else:
                # Different atom interactions (non-diagonal blocks)
                non_diagonal_blocks.append(matrix_block)
                non_diagonal_masks.append(matrix_block_mask)
            
            # Move to next row block
            row_idx += dst_length
        
        # Move to next column block
        col_idx += src_length
    
    # Convert edge indices to tensor format expected by PyTorch Geometric
    # Transpose to get [2, n_edges] format: [[source_nodes], [target_nodes]]
    if edge_indices:
        edge_index_tensor = torch.tensor(edge_indices, device=device).transpose(-1, -2)
    else:
        # Handle edge case of no edges (shouldn't happen in practice)
        edge_index_tensor = torch.empty((2, 0), device=device, dtype=torch.long)
    
    # Stack all blocks into tensors for efficient batch processing
    return (
        torch.stack(diagonal_blocks, dim=0),      # [n_atoms, full_orbitals, full_orbitals]
        torch.stack(non_diagonal_blocks, dim=0),  # [n_edges, full_orbitals, full_orbitals]
        torch.stack(diagonal_masks, dim=0),       # [n_atoms, full_orbitals, full_orbitals]
        torch.stack(non_diagonal_masks, dim=0),   # [n_edges, full_orbitals, full_orbitals]
        edge_index_tensor,                        # [2, n_edges]
    )

def _cut_matrix_3d(matrix, atoms, orbital_mask, full_orbitals):
    """Optimized 3D matrix cutting - no runtime checks for maximum performance.
    
    This function processes 3D matrices where multiple properties are stacked
    along the first dimension (e.g., [Hamiltonian, overlap, kinetic_energy]).
    The algorithm is identical to 2D case but handles the extra dimension.
    
    Args:
        matrix (torch.Tensor): 3D input matrix tensor with features along first dimension.
        atoms (torch.Tensor): Atomic numbers tensor.
        orbital_mask (dict): Dictionary mapping atomic numbers to orbital indices.
        full_orbitals (int): Maximum number of orbitals per atom.
    
    Returns:
        tuple: (diagonal_blocks, non_diagonal_blocks, diagonal_masks, non_diagonal_masks, edge_index).
    """
    # Get tensor properties once to avoid repeated access
    device = matrix.device
    dtype = matrix.dtype
    n_features = matrix.shape[0]  # Number of properties stacked along first dimension
    
    # Pre-allocate lists for better memory efficiency
    # These will store the final atomic blocks
    diagonal_blocks = []      # Same-atom interactions (e.g., C-C, H-H)
    non_diagonal_blocks = []  # Different-atom interactions (e.g., C-H, H-C)
    diagonal_masks = []       # Binary masks for diagonal blocks
    non_diagonal_masks = []   # Binary masks for non-diagonal blocks
    edge_indices = []         # Graph connectivity information
    
    # Pre-compute values to avoid repeated .item() calls in loops
    # This is a key optimization - .item() is expensive when called repeatedly
    atom_values = [atom.item() for atom in atoms]
    orbital_masks = [orbital_mask[atom_val] for atom_val in atom_values]
    orbital_lengths = [len(mask) for mask in orbital_masks]
    
    # Matrix cutting algorithm (same as 2D but with extra dimension):
    # We iterate through all atom pairs (src_idx, dst_idx) and extract
    # the corresponding submatrix from the full orbital matrix
    col_idx = 0  # Column index in the full matrix
    
    for src_idx, (src_mask, src_length) in enumerate(zip(orbital_masks, orbital_lengths)):
        row_idx = 0  # Row index in the full matrix
        
        for dst_idx, (dst_mask, dst_length) in enumerate(zip(orbital_masks, orbital_lengths)):
            # Build edge index for graph representation
            # Only non-diagonal pairs create edges (different atoms)
            if src_idx != dst_idx:
                edge_indices.append([dst_idx, src_idx])  # [source, target] format
            
            # Create empty blocks with proper shape and device/dtype
            # Note: 3D blocks have shape (n_features, full_orbitals, full_orbitals)
            matrix_block = torch.zeros((n_features, full_orbitals, full_orbitals), device=device, dtype=dtype)
            matrix_block_mask = torch.zeros((n_features, full_orbitals, full_orbitals), device=device, dtype=dtype)
            
            # Extract the relevant submatrix from the full matrix
            # This extracts all properties for the interaction between src and dst atoms
            extracted_matrix = matrix[:, row_idx:row_idx + dst_length, col_idx:col_idx + src_length]
            
            # Fill the block using orbital masks
            # dst_mask and src_mask specify which orbitals are actually present
            # The : at the beginning preserves all feature dimensions
            # Use advanced indexing to properly assign values
            matrix_block[:, dst_mask[:, None], src_mask] = extracted_matrix
            matrix_block_mask[:, dst_mask[:, None], src_mask] += 1  # Mark valid positions
            
            # Store blocks based on whether they're diagonal or not
            if src_idx == dst_idx:
                # Same atom interactions (diagonal blocks)
                diagonal_blocks.append(matrix_block)
                diagonal_masks.append(matrix_block_mask)
            else:
                # Different atom interactions (non-diagonal blocks)
                non_diagonal_blocks.append(matrix_block)
                non_diagonal_masks.append(matrix_block_mask)
            
            # Move to next row block
            row_idx += dst_length
        
        # Move to next column block
        col_idx += src_length
    
    # Convert edge indices to tensor format expected by PyTorch Geometric
    # Transpose to get [2, n_edges] format: [[source_nodes], [target_nodes]]
    if edge_indices:
        edge_index_tensor = torch.tensor(edge_indices, device=device).transpose(-1, -2)
    else:
        # Handle edge case of no edges (shouldn't happen in practice)
        edge_index_tensor = torch.empty((2, 0), device=device, dtype=torch.long)
    
    # Stack all blocks into tensors for efficient batch processing
    return (
        torch.stack(diagonal_blocks, dim=0),      # [n_atoms, n_features, full_orbitals, full_orbitals]
        torch.stack(non_diagonal_blocks, dim=0),  # [n_edges, n_features, full_orbitals, full_orbitals]
        torch.stack(diagonal_masks, dim=0),       # [n_atoms, n_features, full_orbitals, full_orbitals]
        torch.stack(non_diagonal_masks, dim=0),   # [n_edges, n_features, full_orbitals, full_orbitals]
        edge_index_tensor,                        # [2, n_edges]
    )

def _cut_matrix_3d_last(matrix, atoms, orbital_mask, full_orbitals):
    """Optimized 3D matrix cutting - no runtime checks for maximum performance.
    
    This function processes 3D matrices where multiple properties are stacked
    along the last dimension (e.g., [Hamiltonian, overlap, kinetic_energy]).
    The algorithm is identical to 2D case but handles the extra dimension.
    
    Args:
        matrix (torch.Tensor): 3D input matrix tensor with features along last dimension.
        atoms (torch.Tensor): Atomic numbers tensor.
        orbital_mask (dict): Dictionary mapping atomic numbers to orbital indices.
        full_orbitals (int): Maximum number of orbitals per atom.
    
    Returns:
        tuple: (diagonal_blocks, non_diagonal_blocks, diagonal_masks, non_diagonal_masks, edge_index).
    """
    # Get tensor properties once to avoid repeated access
    device = matrix.device
    dtype = matrix.dtype
    n_features = matrix.shape[-1]  # Number of properties stacked along last dimension
    
    # Pre-allocate lists for better memory efficiency
    # These will store the final atomic blocks
    diagonal_blocks = []      # Same-atom interactions (e.g., C-C, H-H)
    non_diagonal_blocks = []  # Different-atom interactions (e.g., C-H, H-C)
    diagonal_masks = []       # Binary masks for diagonal blocks
    non_diagonal_masks = []   # Binary masks for non-diagonal blocks
    edge_indices = []         # Graph connectivity information
    
    # Pre-compute values to avoid repeated .item() calls in loops
    # This is a key optimization - .item() is expensive when called repeatedly
    atom_values = [atom.item() for atom in atoms]
    orbital_masks = [orbital_mask[atom_val] for atom_val in atom_values]
    orbital_lengths = [len(mask) for mask in orbital_masks]
    
    # Matrix cutting algorithm (same as 2D but with extra dimension):
    # We iterate through all atom pairs (src_idx, dst_idx) and extract
    # the corresponding submatrix from the full orbital matrix
    col_idx = 0  # Column index in the full matrix
    
    for src_idx, (src_mask, src_length) in enumerate(zip(orbital_masks, orbital_lengths)):
        row_idx = 0  # Row index in the full matrix
        
        for dst_idx, (dst_mask, dst_length) in enumerate(zip(orbital_masks, orbital_lengths)):
            # Build edge index for graph representation
            # Only non-diagonal pairs create edges (different atoms)
            if src_idx != dst_idx:
                edge_indices.append([dst_idx, src_idx])  # [source, target] format
            
            # Create empty blocks with proper shape and device/dtype
            # Note: 3D blocks have shape (n_features, full_orbitals, full_orbitals)
            matrix_block = torch.zeros((full_orbitals, full_orbitals, n_features), device=device, dtype=dtype)
            matrix_block_mask = torch.zeros((full_orbitals, full_orbitals, n_features), device=device, dtype=dtype)
            
            # Extract the relevant submatrix from the full matrix
            # This extracts all properties for the interaction between src and dst atoms
            extracted_matrix = matrix[row_idx:row_idx + dst_length, col_idx:col_idx + src_length, :]
            
            # Fill the block using orbital masks
            # dst_mask and src_mask specify which orbitals are actually present
            # The : at the beginning preserves all feature dimensions
            # Use advanced indexing to properly assign values
            matrix_block[dst_mask[:, None], src_mask, :] = extracted_matrix
            matrix_block_mask[dst_mask[:, None], src_mask, :] = 1  # Mark valid positions
            
            # Store blocks based on whether they're diagonal or not
            if src_idx == dst_idx:
                # Same atom interactions (diagonal blocks)
                diagonal_blocks.append(matrix_block)
                diagonal_masks.append(matrix_block_mask)
            else:
                # Different atom interactions (non-diagonal blocks)
                non_diagonal_blocks.append(matrix_block)
                non_diagonal_masks.append(matrix_block_mask)
            
            # Move to next row block
            row_idx += dst_length
        
        # Move to next column block
        col_idx += src_length
    
    # Convert edge indices to tensor format expected by PyTorch Geometric
    # Transpose to get [2, n_edges] format: [[source_nodes], [target_nodes]]
    if edge_indices:
        edge_index_tensor = torch.tensor(edge_indices, device=device).transpose(-1, -2)
    else:
        # Handle edge case of no edges (shouldn't happen in practice)
        edge_index_tensor = torch.empty((2, 0), device=device, dtype=torch.long)
    
    # Stack all blocks into tensors for efficient batch processing
    return (
        torch.stack(diagonal_blocks, dim=0),      # [n_atoms, full_orbitals, full_orbitals, n_features]
        torch.stack(non_diagonal_blocks, dim=0),  # [n_edges, full_orbitals, full_orbitals, n_features]
        torch.stack(diagonal_masks, dim=0),       # [n_atoms, full_orbitals, full_orbitals, n_features]
        torch.stack(non_diagonal_masks, dim=0),   # [n_edges, full_orbitals, full_orbitals, n_features]
        edge_index_tensor,                        # [2, n_edges]
    )


def pack_upper_triangle(M: np.ndarray):
    """Pack upper triangle of a symmetric matrix into a 1D array.
    
    Args:
        M (np.ndarray): 2D symmetric matrix to pack.
    
    Returns:
        tuple: (packed_array, matrix_size) where packed_array contains the upper
               triangle elements and matrix_size is the original matrix dimension.
    
    Raises:
        AssertionError: If matrix is not 2D or not square.
    """
    assert M.ndim == 2 and M.shape[0] == M.shape[1]
    n = M.shape[0]
    idx = np.triu_indices(n)
    return M[idx].astype(np.float64), n

def unpack_upper_triangle(packed: np.ndarray, n: int):
    """Unpack upper triangle array back into a symmetric matrix.
    
    Args:
        packed (np.ndarray): 1D array containing upper triangle elements.
        n (int): Size of the original square matrix.
    
    Returns:
        np.ndarray: Reconstructed symmetric matrix.
    """
    M = np.zeros((n,n), dtype=packed.dtype)
    iu = np.triu_indices(n)
    M[iu] = packed
    M[(iu[1], iu[0])] = packed  # mirror
    return M

