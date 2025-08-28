import torch

def cal_orbital_and_energies(overlap_matrix, full_hamiltonian, method="eigh", tol=1e-8):
    """Calculate orbital energies and coefficients from overlap and Hamiltonian matrices.
    This function solves the generalized eigenvalue problem HC = SCE, where:
    - H is the Hamiltonian matrix
    - S is the overlap matrix 
    - C are the orbital coefficients
    - E are the orbital energies
    
    Args:
        overlap_matrix (Tensor): Batch of overlap matrices [B, N, N]
        full_hamiltonian (Tensor): Batch of Hamiltonian matrices [B, N, N]
        method (str): Method to use for solving the generalized eigenvalue problem.
            - "eigh": Use eigenvalue decomposition
            - "cholesky": Use Cholesky decomposition
    
    Returns:
        Tuple[Tensor, Tensor]: Tuple containing:
            - orbital_energies: Eigenvalues [B, N] 
            - orbital_coefficients: Eigenvectors [B, N, N]
    """
    assert method in ["eigh", "cholesky"], f"Invalid method: {method}"

    if method == "eigh":
        return _cal_orbital_and_energies_eigh(overlap_matrix, full_hamiltonian, tol)
    elif method == "cholesky":
        try:
            return _cal_orbital_and_energies_cholesky(overlap_matrix, full_hamiltonian)
        except torch.linalg.LinAlgError:
            return _cal_orbital_and_energies_eigh(overlap_matrix, full_hamiltonian, tol)

def _cal_orbital_and_energies_eigh(overlap_matrix, full_hamiltonian, tol=1e-8):
    """Calculate orbital energies and coefficients from overlap and Hamiltonian matrices.
    
    This function solves the generalized eigenvalue problem HC = SCE, where:
    - H is the Hamiltonian matrix
    - S is the overlap matrix 
    - C are the orbital coefficients
    - E are the orbital energies
    
    The solution involves:
    1. Diagonalizing the overlap matrix S = U s U^T
    2. Constructing s^(-1/2) U^T to transform to orthogonal basis
    3. Solving standard eigenvalue problem in orthogonal basis
    4. Transforming eigenvectors back to original basis
    
    Args:
        overlap_matrix (Tensor): Batch of overlap matrices [B, N, N]
        full_hamiltonian (Tensor): Batch of Hamiltonian matrices [B, N, N]
        
    Returns:
        Tuple[Tensor, Tensor]: Tuple containing:
            - orbital_energies: Eigenvalues [B, N] 
            - orbital_coefficients: Eigenvectors [B, N, N]
            
    Note:
        Uses a tolerance threshold (EIGENVALUE_TOLERANCE) to handle 
        numerically small eigenvalues of the overlap matrix.
    """
    eigvals, eigvecs = torch.linalg.eigh(overlap_matrix)
    eps = tol * torch.ones_like(eigvals)
    eigvals = torch.where(eigvals > tol, eigvals, eps)
    frac_overlap = eigvecs / torch.sqrt(eigvals).unsqueeze(-2)

    Fs = torch.bmm(
        torch.bmm(frac_overlap.transpose(-1, -2), full_hamiltonian), frac_overlap
    )
    orbital_energies, orbital_coefficients = torch.linalg.eigh(Fs)
    orbital_coefficients = torch.bmm(frac_overlap, orbital_coefficients)
    return orbital_energies, orbital_coefficients

def _cal_orbital_and_energies_cholesky(overlap_matrix, full_hamiltonian):
    """Calculate orbital energies and coefficients using Cholesky decomposition.
    
    This function solves the generalized eigenvalue problem HC = SCE using Cholesky decomposition, where:
    - H is the Hamiltonian matrix
    - S is the overlap matrix 
    - C are the orbital coefficients
    - E are the orbital energies
    
    The solution involves:
    1. Cholesky decomposition of overlap matrix S = L L^T
    2. Solving L^T H L C' = C' E in the orthogonal basis
    3. Transforming eigenvectors back to original basis using C = L^(-T) C'
    
    Args:
        overlap_matrix (Tensor): Batch of overlap matrices [B, N, N]
        full_hamiltonian (Tensor): Batch of Hamiltonian matrices [B, N, N]
        
    Returns:
        Tuple[Tensor, Tensor]: Tuple containing:
            - orbital_energies: Eigenvalues [B, N] 
            - orbital_coefficients: Eigenvectors [B, N, N]
            
    Note:
        Uses Cholesky decomposition which is more numerically stable than 
        eigenvalue decomposition for positive definite matrices.
        Falls back to eigenvalue method if Cholesky fails.
    """
    # Cholesky decomposition: S = L L^T
    L = torch.linalg.cholesky(overlap_matrix)
    
    # Solve L^(-1) H L^(-T) C' = C' E in orthogonal basis
    # First compute L^(-1) H L^(-T)
    L_inv = torch.linalg.solve_triangular(L, 
                                            torch.eye(L.size(-1), device=L.device, dtype=L.dtype).unsqueeze(0).expand_as(L), 
                                            upper=False)
    L_inv_T = torch.linalg.solve_triangular(L.transpose(-1, -2), 
                                            torch.eye(L.size(-1), device=L.device, dtype=L.dtype).unsqueeze(0).expand_as(L), 
                                            upper=True)
    
    # Compute L^(-1) H L^(-T)
    L_inv_H = torch.bmm(L_inv, full_hamiltonian)
    L_inv_H_L_inv_T = torch.bmm(L_inv_H, L_inv_T)
    
    # Solve standard eigenvalue problem
    orbital_energies, orbital_coefficients_ortho = torch.linalg.eigh(L_inv_H_L_inv_T)
    
    # Transform back to original basis: C = L^(-T) C'
    orbital_coefficients = torch.bmm(L_inv_T, orbital_coefficients_ortho)

    return orbital_energies, orbital_coefficients

def cal_orbital_and_energies_variable_size(overlap_matrices, hamiltonian_matrices, method="eigh", tol=1e-8):
    """Calculate orbital energies and coefficients from lists of variable-sized matrices.
    
    This function handles cases where matrices have different sizes. It provides multiple
    optimization strategies for different use cases.
    
    Args:
        overlap_matrices: List of overlap matrices, each of shape [N_i, N_i]
        hamiltonian_matrices: List of Hamiltonian matrices, each of shape [N_i, N_i]
        method (str): Method to use for solving the generalized eigenvalue problem.
            - "eigh": Use eigenvalue decomposition
            - "cholesky": Use Cholesky decomposition
        tol (float): Tolerance for handling small eigenvalues
        
    Returns:
        Tuple[List[Tensor], List[Tensor]]: Tuple containing:
            - orbital_energies: List of tensors, each containing orbital energies [N_i]
            - orbital_coefficients: List of tensors, each containing orbital coefficients [N_i, N_i]
    """
    assert len(overlap_matrices) == len(hamiltonian_matrices), "Lists must have same length"
    assert method in ["eigh", "cholesky"], f"Invalid method: {method}"
    
    orbital_energies_list = []
    orbital_coefficients_list = []
    
    for overlap_matrix, hamiltonian_matrix in zip(overlap_matrices, hamiltonian_matrices):
        # Ensure inputs are 2D tensors
        if overlap_matrix.dim() != 2:
            raise ValueError(f"overlap_matrix must be 2D, got {overlap_matrix.dim()}D")
        if hamiltonian_matrix.dim() != 2:
            raise ValueError(f"hamiltonian_matrix must be 2D, got {hamiltonian_matrix.dim()}D")
        
        if method == "eigh":
            energies, coefficients = _cal_orbital_and_energies_eigh_single(overlap_matrix, hamiltonian_matrix, tol)
        elif method == "cholesky":
            try:
                energies, coefficients = _cal_orbital_and_energies_cholesky_single(overlap_matrix, hamiltonian_matrix)
            except torch.linalg.LinAlgError:
                energies, coefficients = _cal_orbital_and_energies_eigh_single(overlap_matrix, hamiltonian_matrix, tol)
        
        orbital_energies_list.append(energies)
        orbital_coefficients_list.append(coefficients)
    
    return orbital_energies_list, orbital_coefficients_list


def cal_orbital_and_energies_variable_size_grouped(overlap_matrices, hamiltonian_matrices, method="eigh", tol=1e-8):
    """Calculate orbital energies and coefficients using size-based grouping for efficiency.
    
    This function groups matrices by size and processes each group in batch,
    providing a balance between memory efficiency and computational speed.
    
    Args:
        overlap_matrices: List of overlap matrices, each of shape [N_i, N_i]
        hamiltonian_matrices: List of Hamiltonian matrices, each of shape [N_i, N_i]
        method (str): Method to use for solving the generalized eigenvalue problem
        tol (float): Tolerance for handling small eigenvalues
        
    Returns:
        Tuple[List[Tensor], List[Tensor]]: Tuple containing:
            - orbital_energies: List of tensors, each containing orbital energies [N_i]
            - orbital_coefficients: List of tensors, each containing orbital coefficients [N_i, N_i]
    """
    assert len(overlap_matrices) == len(hamiltonian_matrices), "Lists must have same length"
    
    if not overlap_matrices:
        return [], []
    
    # Group matrices by size
    size_groups = {}
    for i, (overlap, hamiltonian) in enumerate(zip(overlap_matrices, hamiltonian_matrices)):
        size = overlap.size(0)
        if size not in size_groups:
            size_groups[size] = []
        size_groups[size].append((i, overlap, hamiltonian))
    
    # Initialize result lists
    orbital_energies_list = [None] * len(overlap_matrices)
    orbital_coefficients_list = [None] * len(overlap_matrices)
    
    # Process each group in batch
    for size, group in size_groups.items():
        # Always use individual processing to ensure numerical consistency
        for idx, overlap, hamiltonian in group:
            if method == "eigh":
                energies, coefficients = _cal_orbital_and_energies_eigh_single(overlap, hamiltonian, tol)
            elif method == "cholesky":
                try:
                    energies, coefficients = _cal_orbital_and_energies_cholesky_single(overlap, hamiltonian)
                except torch.linalg.LinAlgError:
                    energies, coefficients = _cal_orbital_and_energies_eigh_single(overlap, hamiltonian, tol)
            
            orbital_energies_list[idx] = energies
            orbital_coefficients_list[idx] = coefficients
    
    return orbital_energies_list, orbital_coefficients_list

def cal_orbital_and_energies_single(overlap_matrix, full_hamiltonian, method="eigh", tol=1e-8):
    """Calculate orbital energies and coefficients for single (n x n) matrices.
    This function solves the generalized eigenvalue problem HC = SCE, where:
    - H is the Hamiltonian matrix
    - S is the overlap matrix 
    - C are the orbital coefficients
    - E are the orbital energies
    
    Args:
        overlap_matrix (Tensor): Single overlap matrix [N, N]
        full_hamiltonian (Tensor): Single Hamiltonian matrix [N, N]
        method (str): Method to use for solving the generalized eigenvalue problem.
            - "eigh": Use eigenvalue decomposition
            - "cholesky": Use Cholesky decomposition
        tol (float): Tolerance for handling small eigenvalues
    
    Returns:
        Tuple[Tensor, Tensor]: Tuple containing:
            - orbital_energies: Eigenvalues [N] 
            - orbital_coefficients: Eigenvectors [N, N]
    """
    assert method in ["eigh", "cholesky"], f"Invalid method: {method}"
    
    # Ensure inputs are 2D tensors
    if overlap_matrix.dim() != 2:
        raise ValueError(f"overlap_matrix must be 2D, got {overlap_matrix.dim()}D")
    if full_hamiltonian.dim() != 2:
        raise ValueError(f"full_hamiltonian must be 2D, got {full_hamiltonian.dim()}D")
    
    if method == "eigh":
        return _cal_orbital_and_energies_eigh_single(overlap_matrix, full_hamiltonian, tol)
    elif method == "cholesky":
        try:
            return _cal_orbital_and_energies_cholesky_single(overlap_matrix, full_hamiltonian)
        except torch.linalg.LinAlgError:
            return _cal_orbital_and_energies_eigh_single(overlap_matrix, full_hamiltonian, tol)

def _cal_orbital_and_energies_eigh_single(overlap_matrix, full_hamiltonian, tol=1e-8):
    """Calculate orbital energies and coefficients from single overlap and Hamiltonian matrices using eigenvalue decomposition.
    
    This function solves the generalized eigenvalue problem HC = SCE, where:
    - H is the Hamiltonian matrix
    - S is the overlap matrix 
    - C are the orbital coefficients
    - E are the orbital energies
    
    The solution involves:
    1. Diagonalizing the overlap matrix S = U s U^T
    2. Constructing s^(-1/2) U^T to transform to orthogonal basis
    3. Solving standard eigenvalue problem in orthogonal basis
    4. Transforming eigenvectors back to original basis
    
    Args:
        overlap_matrix (Tensor): Single overlap matrix [N, N]
        full_hamiltonian (Tensor): Single Hamiltonian matrix [N, N]
        tol (float): Tolerance for handling small eigenvalues
        
    Returns:
        Tuple[Tensor, Tensor]: Tuple containing:
            - orbital_energies: Eigenvalues [N] 
            - orbital_coefficients: Eigenvectors [N, N]
            
    Note:
        Uses a tolerance threshold to handle numerically small eigenvalues of the overlap matrix.
    """
    eigvals, eigvecs = torch.linalg.eigh(overlap_matrix)
    
    eps = tol * torch.ones_like(eigvals)
    eigvals = torch.where(eigvals > tol, eigvals, eps)
    frac_overlap = eigvecs / torch.sqrt(eigvals).unsqueeze(0)
    Fs = torch.mm(torch.mm(frac_overlap.transpose(-1, -2), full_hamiltonian), frac_overlap)
    orbital_energies, orbital_coefficients = torch.linalg.eigh(Fs)
    orbital_coefficients = torch.mm(frac_overlap, orbital_coefficients)
    return orbital_energies, orbital_coefficients

# TODO: check if this is correct
def _cal_orbital_and_energies_cholesky_single(overlap_matrix, full_hamiltonian):
    """Calculate orbital energies and coefficients using Cholesky decomposition for single matrices.
    
    This function solves the generalized eigenvalue problem HC = SCE using Cholesky decomposition, where:
    - H is the Hamiltonian matrix
    - S is the overlap matrix 
    - C are the orbital coefficients
    - E are the orbital energies
    
    The solution involves:
    1. Cholesky decomposition of overlap matrix S = L L^T
    2. Solving L^(-1) H L^(-T) C' = C' E in the orthogonal basis
    3. Transforming eigenvectors back to original basis using C = L^(-T) C'
    
    Args:
        overlap_matrix (Tensor): Single overlap matrix [N, N]
        full_hamiltonian (Tensor): Single Hamiltonian matrix [N, N]
        
    Returns:
        Tuple[Tensor, Tensor]: Tuple containing:
            - orbital_energies: Eigenvalues [N] 
            - orbital_coefficients: Eigenvectors [N, N]
            
    Note:
        Uses Cholesky decomposition which is more numerically stable than 
        eigenvalue decomposition for positive definite matrices.
        Falls back to eigenvalue method if Cholesky fails.
    """
    # Cholesky decomposition: S = L L^T
    L = torch.linalg.cholesky(overlap_matrix)
    
    # Solve L^(-1) H L^(-T) C' = C' E in orthogonal basis
    # First compute L^(-1) H L^(-T)
    L_inv = torch.linalg.solve_triangular(
        L, 
        torch.eye(L.size(-1), device=L.device, dtype=L.dtype).unsqueeze(0).expand_as(L), 
        upper=False
    )
    L_inv_T = torch.linalg.solve_triangular(
        L.transpose(-1, -2), 
        torch.eye(L.size(-1), device=L.device, dtype=L.dtype).unsqueeze(0).expand_as(L), 
        upper=True
    )
    
    # Compute L^(-1) H L^(-T)
    L_inv_H = torch.mm(L_inv, full_hamiltonian)
    L_inv_H_L_inv_T = torch.mm(L_inv_H, L_inv_T)
    
    # Solve standard eigenvalue problem
    orbital_energies, orbital_coefficients_ortho = torch.linalg.eigh(L_inv_H_L_inv_T)
    
    # Transform back to original basis: C = L^(-T) C'
    orbital_coefficients = torch.mm(L_inv_T, orbital_coefficients_ortho)
        
    return orbital_energies, orbital_coefficients