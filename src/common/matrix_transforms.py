import torch
import numpy as np
from argparse import Namespace

convention_dict = {
    "pyscf_631G_to_e3nn": Namespace(
        # 6-31G basis set convention used by PySCF
        # p orbitals: [px, py, pz] -> reordered to [pz, px, py] for compatibility
        atom_to_orbitals_map={1: "ss", 6: "ssspp", 7: "ssspp", 8: "ssspp", 9: "ssspp"},
        orbital_idx_map={"s": [0], "p": [2, 0, 1], "d": [0, 1, 2, 3, 4]},  # p: [pz, px, py]
        orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1],      # H: 2 orbitals (s, s)
            6: [0, 1, 2, 3, 4],  # C: 5 orbitals (s, s, s, p, p)
            7: [0, 1, 2, 3, 4],  # N: 5 orbitals (s, s, s, p, p)
            8: [0, 1, 2, 3, 4],  # O: 5 orbitals (s, s, s, p, p)
            9: [0, 1, 2, 3, 4],  # F: 5 orbitals (s, s, s, p, p)
        },
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
        orbital_idx_map={"s": [0], "p": [1, 2, 0], "d": [0, 1, 2, 3, 4]},  # p: [py, pz, px]
        orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2],      # H: 3 orbitals (s, s, p)
            6: [0, 1, 2, 3, 4, 5],  # C: 6 orbitals (s, s, s, p, p, d)
            7: [0, 1, 2, 3, 4, 5],  # N: 6 orbitals (s, s, s, p, p, d)
            8: [0, 1, 2, 3, 4, 5],  # O: 6 orbitals (s, s, s, p, p, d)
            9: [0, 1, 2, 3, 4, 5],  # F: 6 orbitals (s, s, s, p, p, d)
        },
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
        orbital_idx_map={"s": [0], "p": [2, 0, 1], "d": [0, 1, 2, 3, 4]},  # p: [px, py, pz] (PySCF native)
        orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2],      # H: 3 orbitals (s, s, p)
            6: [0, 1, 2, 3, 4, 5],  # C: 6 orbitals (s, s, s, p, p, d)
            7: [0, 1, 2, 3, 4, 5],  # N: 6 orbitals (s, s, s, p, p, d)
            8: [0, 1, 2, 3, 4, 5],  # O: 6 orbitals (s, s, s, p, p, d)
            9: [0, 1, 2, 3, 4, 5],  # F: 6 orbitals (s, s, s, p, p, d)
        },
    ),
}

convention_dict["back2pyscf"] = convention_dict["e3nn_to_pyscf_def2svp"]
convention_dict["pyscf_def2svp"] = convention_dict["pyscf_def2svp_to_e3nn"]
convention_dict["pyscf_631G"] = convention_dict["pyscf_631G_to_e3nn"]

def get_convetion_dict():
    return convention_dict

def _get_orbital_mask(ORBITAL_1S_2S_INDICES = None, ORBITAL_2P_INDICES = None, ORBITAL_MASK_SIZE_LINE2 = None):
    """Get orbital masks for different atomic numbers."""
    if ORBITAL_1S_2S_INDICES is None:
        ORBITAL_1S_2S_INDICES = torch.tensor([0, 1])
    if ORBITAL_2P_INDICES is None:
        ORBITAL_2P_INDICES = torch.tensor([3, 4, 5])
    if ORBITAL_MASK_SIZE_LINE2 is None:
        ORBITAL_MASK_SIZE_LINE2 = 14
    orbital_mask_line1 = torch.cat([ORBITAL_1S_2S_INDICES, ORBITAL_2P_INDICES])
    orbital_mask_line2 = torch.arange(ORBITAL_MASK_SIZE_LINE2)
    orbital_mask = {}
    for i in range(1, 11):
        orbital_mask[i] = orbital_mask_line1 if i <= 2 else orbital_mask_line2
    return orbital_mask

def _build_final_matrix(
    data,
    diagonal_matrix,
    non_diagonal_matrix,
    orbital_mask,
):
    """Build final matrix from diagonal and non-diagonal blocks."""
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
    """Transform matrix between different orbital conventions - CUDA optimized version."""    
    final_matrix_list = []
    
    for graph_idx in range(data.ptr.shape[0] - 1):
        hamiltonian = hamiltonian_list[graph_idx]
        atoms = data.atoms[data.batch == graph_idx]
        mat_res = _matrix_transform_single(hamiltonian, atoms, convention_rule)
        final_matrix_list.append(mat_res)
        
    return final_matrix_list

def _matrix_transform_single(hamiltonian, atoms, convention_rule):
    """
    Transform matrices according to orbital convention using NumPy.
    
    This function reorders and applies sign changes to orbital matrices based on
    different quantum chemistry software conventions. Different software packages
    use different orbital ordering and sign conventions.
    
    Example:
        Transform from 6-31G to def2-SVP convention:
        - 6-31G: p orbitals ordered as [px, py, pz] 
        - def2-SVP: p orbitals ordered as [py, pz, px]
        - This function handles the reordering and sign changes
    
    Args:
        matrices: Input matrices to transform, shape (..., n_orb, n_orb)
        atoms: Atomic numbers for the molecule (e.g., [6, 1, 1, 1] for CH3)
        convention: Orbital convention to use:
            - 'pyscf_631G': 6-31G basis set convention (p: [pz, px, py])
            - 'pyscf_def2svp': def2-SVP basis set convention (p: [py, pz, px])
            - 'back2pyscf': Convert back to PySCF native convention (p: [pz, px, py])
              * Use this when you have matrices from other software and need to
                convert them back to PySCF format for density matrix calculations
              * Same basis as def2-SVP but with PySCF's native p-orbital ordering
    
    Returns:
        Transformed matrices with reordered orbitals and applied sign changes
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
