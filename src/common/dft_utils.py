from pyscf import gto, scf, dft
from common.metric import cal_orbital_and_energies
from common.matrix_transforms import get_convention_dict, matrix_transform_single
from common.units import *

# =============================================================================
# PYSCF INTEGRATION FUNCTIONS
# =============================================================================

def init_pyscf_mol(atoms, pos, unit="ang", basis="def2svp"):
    """
    Initialize PySCF Molecule object.
    
    Args:
        atoms (list): List of atomic numbers
        pos (array): Atomic positions in angstrom
        unit (str): Unit of position (default: "ang")
    """
    if unit.lower() == "ang" or unit.lower() == "angstrom" or unit.lower() == "a":
        pos_factor = 1.0
    elif unit.lower() == "bohr":
        pos_factor = BOHR2ANG
    else:
        raise ValueError(f"Invalid unit: {unit}")
    return init_pyscf_mol_(atoms, pos, pos_factor=pos_factor, basis=basis)

def init_pyscf_mol_(atoms, pos, pos_factor=1.0, basis="def2svp"):
    """
    Initialize PySCF Molecule object.
    
    Args:
        atoms (list): List of atomic numbers
        pos (array): Atomic positions in angstrom
        unit (str): Unit of position (default: "ang")
    """
    pos = pos * pos_factor
    mol = gto.Mole()
    mol_conf = [[atoms[atom_idx], pos[atom_idx]] for atom_idx in range(len(atoms))]
    mol.build(verbose=0, atom=mol_conf, basis=basis, unit="ang")
    return mol

def init_pyscf_mf_(atoms, pos, pos_factor=1.0, xc="pbe", basis="def2svp"):
    """
    Initialize PySCF Molecule object.
    
    Args:
        atoms (list): List of atomic numbers
        pos (array): Atomic positions in angstrom
        pos_factor (float): Position scaling factor (default: 1.0)
        xc (str): Exchange-correlation functional (default: "pbe")
        basis (str): Basis set name (default: "def2svp")
        gpu (bool): Whether to use GPU acceleration (default: False)
        
    Returns:
        pyscf.dft.RKS: PySCF RKS object
    """
    mol = init_pyscf_mol_(atoms, pos, pos_factor, basis)
    mf = dft.RKS(mol)
    
    mf.xc = xc
    mf.basis = basis
    
    return mf

def init_pyscf_mf(atoms, pos, unit="ang", xc="pbe", basis="def2svp"):
    """
    Initialize PySCF Molecule object.
    
    Args:
        atoms (list): List of atomic numbers
        pos (array): Atomic positions in angstrom
        pos_factor (float): Position scaling factor (default: 1.0)
        xc (str): Exchange-correlation functional (default: "pbe")
        basis (str): Basis set name (default: "def2svp")
        gpu (bool): Whether to use GPU acceleration (default: False)
        
    Returns:
        pyscf.dft.RKS: PySCF RKS object
    """
    if unit.lower() == "ang" or unit.lower() == "angstrom" or unit.lower() == "a":
        pos_factor = 1.0
    elif unit.lower() == "bohr":
        pos_factor = BOHR2ANG
    else:
        raise ValueError(f"Invalid unit: {unit}")
    return init_pyscf_mf_(atoms, pos, pos_factor=pos_factor, xc=xc, basis=basis)

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
    
def calc_init_ham(atoms=None, pos=None, init="minao", pos_factor=1.0, 
                  xc="pbe", basis="def2svp", mf=None):
    """
    Calculate initial Hamiltonian using PySCF.
    
    Args:
        atoms (list, optional): List of atomic numbers
        pos (array, optional): Atomic positions
        init (str): Initial guess method ("minao" or "1e") (default: "minao")
        pos_factor (float): Position scaling factor (default: 1.0)
        xc (str): Exchange-correlation functional (default: "pbe")
        basis (str): Basis set name (default: "def2svp")
        mf (pyscf.dft.RKS, optional): Pre-initialized PySCF object
        
    Returns:
        tuple: (initial_hamiltonian, initial_density_matrix, mf_object)
        
    Raises:
        ValueError: If init method is not implemented
        AssertionError: If atoms and pos are not provided when mf is None
    """
    if mf is None:
        assert atoms is not None and pos is not None, "atoms and pos must be provided if mf is not provided"
        mf = init_pyscf_mf_(atoms, pos, pos_factor, xc, basis)

    # Choose initial guess method
    if init == "minao":
        init_dm = mf.init_guess_by_minao()
    elif init == "1e":
        init_dm = mf.init_guess_by_1e()
    else:
        raise ValueError(f"init {init} not implemented")

    # Calculate initial Hamiltonian
    init_ham = scf.hf.get_fock(mf, dm=init_dm)
    return init_ham, init_dm, mf

def calc_dm0_from_ham_(atoms, overlap, hamiltonian, output_res=True):
    """
    Calculate density matrix from Hamiltonian.
    
    This function computes the density matrix by solving the eigenvalue problem
    and constructing the density matrix from occupied orbitals.
    
    Args:
        atoms (torch.Tensor): Atomic numbers
        overlap (torch.Tensor): Overlap matrix
        hamiltonian (torch.Tensor): Hamiltonian matrix
        output_res (bool): Whether to return additional results (default: True)
        
    Returns:
        tuple or numpy.ndarray: Density matrix and optionally additional results
    """    
    # Calculate orbital energies and coefficients
    if overlap.dim() == 2:
        overlap = overlap.unsqueeze(0)
    if hamiltonian.dim() == 2:
        hamiltonian = hamiltonian.unsqueeze(0)
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
        output_res (bool): Whether to return additional results (default: True)
        
    Returns:
        tuple or numpy.ndarray: Density matrix and optionally additional results
    """
    if transform:
        # Ensure tensors have batch dimension
        if overlap.dim() == 2:
            overlap = overlap.unsqueeze(0)
        if hamiltonian.dim() == 2:
            hamiltonian = hamiltonian.unsqueeze(0)
            
        # Apply matrix transformations
        overlap = matrix_transform_single(overlap, atoms, convention=convention)
        hamiltonian = matrix_transform_single(hamiltonian, atoms, convention=convention)
    
    # Calculate orbital energies and coefficients
    orbital_energies, orbital_coefficients = cal_orbital_and_energies(
        overlap, hamiltonian
    )
    return orbital_energies, orbital_coefficients

def calc_dm0(atoms, orbital_coefficients):
    """
    Calculate density matrix from orbital coefficients.
    
    Args:
        atoms (torch.Tensor): Atomic numbers
        orbital_coefficients (torch.Tensor): Orbital coefficients with shape (B, N, N) or (N, N)
            where B is batch size, N is number of basis functions
            
    Returns:
        numpy.ndarray: Density matrix with shape (B, N, N) or (N, N)
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
