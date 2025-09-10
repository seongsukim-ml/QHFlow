from pyscf import gto, scf, dft
from argparse import Namespace
import numpy as np
import torch
from common.metric import cal_orbital_and_energies
from common.matrix_transforms import get_convention_dict

convention_dict = get_convention_dict()

# Energy unit is Eh in pyscf
ANG2BOHR = 1.8897261258369282     # Angstrom to Bohr conversion
BOHR2ANG = 1.0 / ANG2BOHR         # Bohr to Angstrom conversion
HA2meV = 27.211396641308 * 1000   # Hartree to meV conversion
KCALPM2meV = 43.36410424180094    # kcal/mol to meV conversion
HA2KCALPM = 627.5094740628942     # Hartree to kcal/mol
KCALPM2HA = 1.0 / HA2KCALPM       # kcal/mol to Hartree

# Force unit is Eh/Bohr in pyscf
HA_BOHR_2_KCALPM_ANG = HA2KCALPM / BOHR2ANG      # Hartree/Bohr to kcal/mol/Angstrom
KCALPM_ANG_2_HA_BOHR = 1.0 / HA_BOHR_2_KCALPM_ANG  # kcal/mol/Angstrom to Hartree/Bohr
HA_BOHR_2_meV_ANG = HA2meV / BOHR2ANG      # Hartree/Bohr to meV/Angstrom
meV_ANG_2_HA_BOHR = 1.0 / HA_BOHR_2_meV_ANG  # meV/Angstrom to Hartree/Bohr

def init_pyscf_mol(atoms, pos, init="minao"):
    """
    Initialize PySCF Molecule object.
    """
    mol = gto.Mole()
    if len(pos.shape) == 1:
        pos = pos.reshape(-1, 3)
    atom_conf = [[atoms[atom_idx], pos[atom_idx]] for atom_idx in range(len(atoms))]
    mol.build(verbose=0, atom=atom_conf, basis="def2svp", unit="ang")
    return mol

def calc_overlap_and_init_hamiltonian(
    atoms,
    pos,
    init="minao",
    basis="def2svp",
    xc="b3lyp",
    out_mf=False,
):
    """
    Calculate overlap matrix and initial Hamiltonian matrix using PySCF.
    
    Args:
        atoms: Atomic numbers for the molecule
        pos: Atomic positions in Angstrom
        init: Initial guess method ('minao' or '1e')
        basis: Basis set name (default: "def2svp")
        xc: Exchange-correlation functional (default: "b3lyp")
    Returns:
        tuple: (overlap_matrix, initial_hamiltonian_matrix)
    """
    mol = init_pyscf_mol(atoms, pos, init)

    overlap_matrix = mol.intor("int1e_ovlp")
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.basis = basis

    density_matrix_init = mf.get_init_guess(key=init)
    init_hamiltonian = mf.get_fock(dm=density_matrix_init)
    
    if out_mf:
        return overlap_matrix.astype("float64"), init_hamiltonian.astype("float64"), mf
    else:
        return overlap_matrix.astype("float64"), init_hamiltonian.astype("float64")
    
def calc_dm0_from_ham(atoms, overlap, hamiltonian, transform=True, convention="back2pyscf", output_res=True):
    """
    Calculate density matrix from Hamiltonian.
    
    This function computes the density matrix by solving the eigenvalue problem
    and constructing the density matrix from occupied orbitals.
    
    Args:
        atoms (torch.Tensor): Atomic numbers
        overlap (torch.Tensor): Overlap matrix
        hamiltonian (torch.Tensor): Hamiltonian matrix
        transform (bool): Whether to transform matrices (default: True)
        convention (str): Orbital convention for transformation (default: "back2pyscf")
            - "back2pyscf": Convert matrices back to PySCF native format for density matrix calculation
            - This is the default because density matrix calculations typically require PySCF format
        output_res (bool): Whether to return additional results (default: True)
        
    Returns:
        tuple or numpy.ndarray: Density matrix and optionally additional results
    """    
    # Calculate orbital energies and coefficients
    orbital_energies, orbital_coefficients = cal_orbital_and_energies(
        overlap, hamiltonian
    )
    
    # Number of occupied orbitals (half of total electrons)
    num_orb = int(atoms.sum() / 2)    
    orbital_coefficients = orbital_coefficients.squeeze()

    # Construct density matrix from occupied orbitals
    sliced_orbital_coefficients = orbital_coefficients[:, :num_orb]
    dm0 = sliced_orbital_coefficients.matmul(sliced_orbital_coefficients.T)* 2
    dm0 = dm0.cpu().numpy()

    if output_res:
        res = {
            "dm0": dm0,
            "orbital_energies": orbital_energies,
            "orbital_coefficients": orbital_coefficients,
            "num_orb": num_orb,
            "overlap": overlap,
            "hamiltonian": hamiltonian,
        }
        return dm0, res
    else:
        return dm0

def calc_dm0(atoms, orbital_coefficients):
    """
    Calculate density matrix from orbital coefficients.
    orbital_coefficients can be a tensor of shape (B, N, N) or (N, N)
    """
    num_orb = int(atoms.sum() / 2)
    if orbital_coefficients.ndim == 3:    
        orbital_sliced = orbital_coefficients[:,:, :num_orb]
        dm0 = orbital_sliced.bmm(orbital_sliced.transpose(-1, -2))* 2
    else:
        orbital_sliced = orbital_coefficients[:, :num_orb]
        dm0 = orbital_sliced.matmul(orbital_sliced.T)* 2
    dm0 = dm0.cpu().numpy()
    return dm0

def cut_matrix(matrix, atoms, orbital_mask, full_orbitals):
    """
    Cut matrix into atomic blocks with optimized performance.
    
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
        matrix: Input matrix tensor of shape (n_orb, n_orb) or (n_orb, n_orb, n_features)
               - 2D: Single property matrix (e.g., Hamiltonian, overlap)
               - 3D: Multiple property matrices stacked along last dimension
        atoms: Atomic numbers tensor (e.g., [6, 1, 1, 1] for CH3)
        orbital_mask: Dictionary mapping atomic numbers to orbital indices
                     e.g., {6: [0,1,2,3,4], 1: [0,1]} for C(5 orbitals) and H(2 orbitals)
        full_orbitals: Maximum number of orbitals per atom (used for padding)
        
    Returns:
        tuple: (diagonal_blocks, non_diagonal_blocks, diagonal_masks, non_diagonal_masks, edge_index)
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
        return _cut_matrix_3d(matrix, atoms, orbital_mask, full_orbitals)


def _cut_matrix_2d(matrix, atoms, orbital_mask, full_orbitals):
    """
    Optimized 2D matrix cutting - no runtime checks for maximum performance.
    
    This function processes 2D matrices (single property like Hamiltonian or overlap matrix).
    It's separated from 3D case to avoid conditional checks in the hot loop.
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
    """
    Optimized 3D matrix cutting - no runtime checks for maximum performance.
    
    This function processes 3D matrices where multiple properties are stacked
    along the last dimension (e.g., [Hamiltonian, overlap, kinetic_energy]).
    The algorithm is identical to 2D case but handles the extra dimension.
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
            # Note: 3D blocks have shape (full_orbitals, full_orbitals, n_features)
            matrix_block = torch.zeros((full_orbitals, full_orbitals, n_features), device=device, dtype=dtype)
            matrix_block_mask = torch.zeros((full_orbitals, full_orbitals, n_features), device=device, dtype=dtype)
            
            # Extract the relevant submatrix from the full matrix
            # This extracts all properties for the interaction between src and dst atoms
            extracted_matrix = matrix[row_idx:row_idx + dst_length, col_idx:col_idx + src_length]
            
            # Fill the block using orbital masks
            # dst_mask and src_mask specify which orbitals are actually present
            # The : at the end preserves all feature dimensions
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
        torch.stack(diagonal_blocks, dim=0),      # [n_atoms, full_orbitals, full_orbitals, n_features]
        torch.stack(non_diagonal_blocks, dim=0),  # [n_edges, full_orbitals, full_orbitals, n_features]
        torch.stack(diagonal_masks, dim=0),       # [n_atoms, full_orbitals, full_orbitals, n_features]
        torch.stack(non_diagonal_masks, dim=0),   # [n_edges, full_orbitals, full_orbitals, n_features]
        edge_index_tensor,                        # [2, n_edges]
    )