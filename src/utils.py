import warnings
import torch
from torch import Tensor
from torch_geometric.data import Data
from pyscf import gto
from typing import Optional, List
from tqdm.rich import tqdm
from copy import deepcopy
from ase.data import chemical_symbols, atomic_numbers
import psi4
from e3nn import o3
from torch import nn
import time
import pyscf
from pyscf import dft
import numpy as np


# fmt : off
def get_total_cycles(envs):
    setattr(envs["mf"], "total_cycle", envs["cycle"])
    setattr(envs["mf"], "_dm_last", envs["dm"])

    if envs["mf"].gt is not None:
        # print(np.mean(np.abs(envs["fock"] - envs["mf"].gt)), envs["mf"].error_level)
        if (
            np.mean(np.abs(envs["fock"] - envs["mf"].gt)) < envs["mf"].error_level
            and envs["mf"].achieve_error_flag is False
        ):
            setattr(envs["mf"], "achieve_error_flag", True)
            setattr(envs["mf"], "achieve_error_cycle", envs["cycle"])
    if envs["mf"].e_tot_gt is not None:
        # print(np.abs(envs["e_tot"] - envs["mf"].e_tot_gt), envs["mf"].e_tot_error_level)
        if (
            np.abs(envs["e_tot"] - envs["mf"].e_tot_gt) < envs["mf"].e_tot_error_level
            and envs["mf"].e_tot_achieve_error_flag is False
        ):
            setattr(envs["mf"], "e_tot_achieve_error_flag", True)
            setattr(envs["mf"], "e_tot_achieve_error_cycle", envs["cycle"])
            # print(envs["mf"].e_tot_achieve_error_cycle)
    # print(envs["cycle"], envs["e_tot"])
    # info = {
    #     "fock": envs["fock"],
    #     "dm": envs["dm"],
    #     "mo_coeff": envs["mo_coeff"],
    #     "mo_energy": envs["mo_energy"],
    #     "e_tot": envs["e_tot"],
    # }
    # getattr(envs["mf"], "info").append(info)


# fmt : on


# PyTorch compatibility fix for _accumulate function
# In newer PyTorch versions (> 2.2.0), torch._utils._accumulate may not be available
def _accumulate(iterable, fn=lambda x, y: x + y):
    """Return running totals
    
    PyTorch Compatibility Note:
    This function provides a fallback implementation for torch._utils._accumulate
    which was moved or removed in newer PyTorch versions. It replicates the 
    behavior of itertools.accumulate with a custom function.
    
    Examples:
        _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
        _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    """
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


# Ensure torch._utils._accumulate is available for compatibility
if not hasattr(torch._utils, '_accumulate'):
    torch._utils._accumulate = _accumulate


def build_matrix(
    mol,
    dm0=None,
    error_level=None,
    Hamiltonian_gt=None,
    max_cycle=50,
    e_tot_error_level=None,
    e_tot_gt=None,
    qh9=False,
):
    start_time = time.time()
    scf_eng = dft.RKS(mol)
    scf_eng.info = []
    scf_eng.total_cycle = None
    scf_eng.max_cycle = max_cycle

    scf_eng.gt = Hamiltonian_gt
    scf_eng.error_level = error_level
    scf_eng.achieve_error_cycle = None
    scf_eng.achieve_error_flag = False

    scf_eng.e_tot_gt = e_tot_gt
    scf_eng.e_tot_error_level = e_tot_error_level
    scf_eng.e_tot_achieve_error_cycle = None
    scf_eng.e_tot_achieve_error_flag = False

    scf_eng.basis = "def2svp"
    if qh9:
        scf_eng.xc = "b3lyp"
    else:
        scf_eng.xc = "pbe, pbe"
        scf_eng.grids.level = 3

    scf_eng.callback = get_total_cycles
    if dm0 is not None:
        dm0 = dm0.astype("float64")
    scf_eng.kernel(dm0=dm0)
    num_cycle = scf_eng.total_cycle
    if hasattr(scf_eng, "achieve_error_cycle"):
        achieve_error_cycle = scf_eng.achieve_error_cycle
    else:
        achieve_error_cycle = None
    if hasattr(scf_eng, "e_tot_achieve_error_cycle"):
        e_tot_achieve_error_cycle = scf_eng.e_tot_achieve_error_cycle
    else:
        e_tot_achieve_error_cycle = None

    return {
        "cycle": num_cycle,
        "fock": scf_eng.get_fock(dm=scf_eng._dm_last),
        "achieve_error_cycle": achieve_error_cycle,
        "dm": scf_eng._dm_last,
        "total_time": time.time() - start_time,
        "e_tot": scf_eng.e_tot,
        "e_tot_achieve_error_cycle": e_tot_achieve_error_cycle,
    }


def prod(x):
    """Compute the product of a sequence."""
    out = 1
    for a in x:
        out *= a

    return out


class Expansion(nn.Module):
    def __init__(self, irrep_in, irrep_out_1, irrep_out_2):
        super(Expansion, self).__init__()
        self.irrep_in = irrep_in
        self.irrep_out_1 = irrep_out_1
        self.irrep_out_2 = irrep_out_2
        self.instructions = self.get_expansion_path(irrep_in, irrep_out_1, irrep_out_2)

    def forward(self, x_in):
        batch_num = x_in.shape[0]
        x_in_s = [
            x_in[:, i].reshape(batch_num, mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(self.irrep_in.slices(), self.irrep_in)
        ]

        outputs = {}
        for ins in self.instructions:
            mul_ir_in = self.irrep_in[ins[0]]
            mul_ir_out1 = self.irrep_out_1[ins[1]]
            mul_ir_out2 = self.irrep_out_2[ins[2]]
            x1 = x_in_s[ins[0]]
            x1 = x1.reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim)
            w3j_matrix = (
                o3.wigner_3j(ins[1], ins[2], ins[0]).to(x_in.device).type(x1.type())
            )
            result = torch.einsum(
                f"wuv, ijk, bwk-> buivj",
                torch.ones(ins[-1]).type(x1.type()).to(x_in.device),
                w3j_matrix,
                x1.reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim),
            )

            result = result.reshape(batch_num, mul_ir_out1.dim, mul_ir_out2.dim)
            key = (ins[1], ins[2])
            if key in outputs.keys():
                outputs[key] = outputs[key] + result
            else:
                outputs[key] = result

        rows = []
        for i in range(len(self.irrep_out_1)):
            blocks = []
            for j in range(len(self.irrep_out_2)):
                if (i, j) not in outputs.keys():
                    blocks += [
                        torch.zeros(
                            (
                                x_in.shape[0],
                                self.irrep_out_1[i].dim,
                                self.irrep_out_2[j].dim,
                            ),
                            device=x_in.device,
                        ).type(x_in.type())
                    ]
                else:
                    blocks += [outputs[(i, j)]]
            rows.append(torch.cat(blocks, dim=-1))
        output = torch.cat(rows, dim=-2)
        return output

    def get_expansion_path(self, irrep_in, irrep_out_1, irrep_out_2):
        instructions = []
        for i, (num_in, ir_in) in enumerate(irrep_in):
            for j, (num_out1, ir_out1) in enumerate(irrep_out_1):
                for k, (num_out2, ir_out2) in enumerate(irrep_out_2):
                    if ir_in in ir_out1 * ir_out2:
                        instructions.append(
                            [i, j, k, True, 1.0, [num_in, num_out1, num_out2]]
                        )
        return instructions


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def _matrix_to_euler_angles(R, convention="ZYZ"):
    if type(R) != torch.Tensor:
        R = torch.tensor(R, dtype=torch.float64)
    return matrix_to_euler_angles(R.transpose(-1, -2), convention=convention)


def construct_orbital_l_index(AO_lm_index):
    idx = 0
    AO_l_index = []
    while True:
        if idx >= len(AO_lm_index):
            break
        AO_l_index.append(AO_lm_index[idx].item())
        idx += 2 * AO_lm_index[idx] + 1
    return torch.tensor(AO_l_index)


def WDs(data, R):
    AO_l_index = construct_orbital_l_index(data.AO_index[1])
    if type(R) != torch.Tensor:
        R = torch.tensor(R, dtype=torch.float64)
    a, b, c = _matrix_to_euler_angles(R)
    # print(a.item(), b.item(), c.item())
    WDs = []
    for l in range(max(AO_l_index) + 1):
        if l == 1:
            WDs.append(R.T)
        else:
            WDs.append(o3.wigner_D(l, a, b, c))
    WD_stack = [WDs[l] for l in AO_l_index]
    WD_diag = torch.block_diag(*WD_stack)  # Left mul
    return WD_diag


def WDs_batch(data, R):
    if getattr(data, "AO_index", None) is not None:
        AO_l_index = data[0].AO_l_index
    else:
        AO_l_index = torch.tensor([0, 0, 0, 1, 1, 2]).to(R.device)
    if type(R) != torch.Tensor:
        R = torch.tensor(R, dtype=torch.float64)
    angles = _matrix_to_euler_angles(R)
    a, b, c = angles[:, 0], angles[:, 1], angles[:, 2]
    # print(a.item(), b.item(), c.item())
    WDs = []
    WDs_len = 0
    for l in range(max(AO_l_index) + 1):
        if l == 1:
            WDs.append(R.transpose(-1, -2))
        else:
            WDs.append(o3.wigner_D(l, a, b, c))
    WD_stack = [WDs[l] for l in AO_l_index]
    WDs_len = (2 * AO_l_index + 1).sum().item()
    if R.dim() == 2:
        WD_diag = torch.block_diag(*WD_stack)  # Left mul
    else:
        WD_diag = torch.zeros(R.shape[0], WDs_len, WDs_len).to(R.device)
        idx_col, idx_row = 0, 0
        for i in range(len(WD_stack)):
            col_len = 2 * AO_l_index[i] + 1
            row_len = 2 * AO_l_index[i] + 1
            WD_diag[:, idx_row : idx_row + row_len, idx_col : idx_col + col_len] = (
                WD_stack[i]
            )
            idx_col += col_len
            idx_row += row_len

    # WD_diag = torch.block_diag(*WD_stack)  # Left mul
    return WD_diag


def build_molecule(Z, pos):
    res = ""
    for i in range(len(Z)):
        res += f"{chemical_symbols[Z[i]]} {pos[i][0]} {pos[i][1]} {pos[i][2]};"

    return res[:-1]


def build_AO_index(atom, basis):
    r"""`AO_index` is a (2, |AO|)-shape tensor, which means AO per atom.
    For example, the AO index of H2 molecule is
    AO_index = [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    where [0, 0, 0, 1, 1, 1] means each H atom uses three AOs respectively, and [0, 0, 0, 0, 0, 0] means all these AOs belong to a molecule.
    """
    ao_map = {"s": 0, "p": 1, "d": 2, "f": 3}
    mol = gto.M(atom=atom, basis=basis)
    AO_index = torch.tensor(
        [
            [int(i.split()[0]) for i in mol.ao_labels()],
            [ao_map[i.split()[-1][1]] for i in mol.ao_labels()],
            [0 for _ in range(mol.nao)],
        ]
    ).long()
    return AO_index


def AO2Irreps(AO: List[int]):
    ao_map1 = {0: 1, 1: 3, 2: 5, 3: 7}
    ao_map2 = {0: "e", 1: "o", 2: "e", 3: "o"}

    irreps = ""
    count = 1
    for i in range(1, len(AO)):
        if AO[i] == AO[i - 1]:
            count += 1
        else:
            irreps += f"{count // ao_map1[AO[i-1]]}x{AO[i-1]}{ao_map2[AO[i-1]]}+"
            count = 1

    irreps += f"{count // ao_map1[AO[i-1]]}x{AO[i-1]}{ao_map2[AO[i-1]]}"
    return irreps


class AOData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "AO_index":
            return torch.tensor([[self.num_nodes], [0], [1]])
        else:
            return super(AOData, self).__inc__(key, value, *args, **kwargs)


class Onsite_3idx_Overlap_Integral:
    r"""The on-site three-index overlap integral :math:`\tilde{Q}` from the
    `"Informing geometric deep learning with electronic interactionsto accelerate quantum chemistry"
    <https://www.pnas.org/doi/epdf/10.1073/pnas.2205221119>`_ paper

    .. math::
        \tilde{Q}^{n,l,m}^{\mu, \nu} = \int_{r\in\mathbb{R}^3} (\Phi_A^{n_1, l_1, m_1}(r))^*\Phi_A^{n_2, l_2, m_3}(r)
        \tilde{\Phi}_A^{n, l, m} (r) dr
    """

    def __init__(self, atom_list: Optional[List[str]] = None, basis: str = "def2-svp"):
        self.atom_list = (
            chemical_symbols[1:58] if atom_list is None else atom_list
        )  # `def2-svp` only support from H to La

        if basis != "def2-svp":
            warnings.warn(
                f"The class is only tested for `def2-svp` rather than {basis}. Be careful!!!"
            )

        self.basis = basis

    def calc_Q(self, atom: str):
        psi4.core.be_quiet()

        # NOTE: prevent Psi4 from moving the molecule in space.
        # Refer to https://forum.psicode.org/t/how-to-align-the-atomic-orbitals-between-pyscf-and-psi4/3025/2
        mol = psi4.geometry(
            f"""
            {atom} 0 0 0
            nocom
            noreorient
        """
        )

        # Basis Set
        psi4.set_options({"basis": self.basis})

        # Build new wavefunction
        wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("basis"))

        # Initialize MintsHelper with wavefunction's basis set
        mints = psi4.core.MintsHelper(wfn.basisset())

        # Refer to https://psicode.org/psi4manual/4.0b5/quickaddbasis.html
        psi4.set_options({"basis": "auxiliary"})

        # Build new wavefunction
        wfn_aux = psi4.core.Wavefunction.build(
            mol, psi4.core.get_global_option("basis")
        )

        Q = mints.ao_3coverlap(wfn.basisset(), wfn.basisset(), wfn_aux.basisset())
        return torch.from_numpy(Q.np.T).double()

    def Q_table(self):
        Q_dict = {}
        for atom in tqdm(
            self.atom_list, desc="Building on-site three-index overlap integral table"
        ):
            Q = self.calc_Q(atom)
            Q_dict[atomic_numbers[atom]] = self.transform_Q(Q, atom)

        return Q_dict

    @property
    def AO_transform_row(self):  # psi4 -> pyscf
        """The on-site three-index overlap integral is calculated with `psi4`, while Hamiltonian obtained from `psycf`,
        which represents Hamiltonian under AOs with different ordering from that in `psi4`.

        Refer to https://psicode.org/psi4manual/master/prog_blas.html#how-to-name-orbital-bases-e-g-ao-so:
        If Spherical Harmonics are used, :math: `2L + 1` real combinations of the spherical harmonics are built
        from the :math: `(L+1)(L+2)/2` Cartesian Gaussians. Unlike Cartesian functions, these functions are all strictly normalized.
        Note that in PSI4, the real combinations of spherical harmonic functions are ordered as: :math: `0, 1+, 1-, 2+, 2-, \cdots`.

        Refer to https://github.com/pyscf/pyscf/blob/master/pyscf/lib/parameters.py#L68-L76:
        In `pyscf`,
            REAL_SPHERIC = (
                ('',),
                ('x', 'y', 'z'),
                ('xy', 'yz', 'z^2', 'xz', 'x2-y2',),
                ('-3', '-2', '-1', '+0', '+1', '+2', '+3'),
                ('-4', '-3', '-2', '-1', '+0', '+1', '+2', '+3', '+4'),
                ('-5', '-4', '-3', '-2', '-1', '+0', '+1', '+2', '+3', '+4', '+5'),
                ('-6', '-5', '-4', '-3', '-2', '-1', '+0', '+1', '+2', '+3', '+4', '+5', '+6'),
            )
        Refer to https://github.com/pyscf/pyscf/issues/2123#issuecomment-1985951880:
        `p` functions are special in pyscf, as they don't follow the same order as functions and higher.
        Refer to https://github.com/pyscf/pyscf/blob/master/pyscf/symm/Dmatrix.py#L29:
        ```python
        if reorder_p and l == 1:
            D = D[[2,0,1]][:,[2,0,1]]
        ```
        """
        return {
            "s": torch.tensor([[1.0]]).double(),
            "p": torch.tensor(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
            ).double(),
            "d": torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                ]
            ).double(),
        }

    def parse_basis(self, atom: str):
        AOs, AO_slices = [], []
        tmp = 0
        for info in gto.format_basis({atom: self.basis})[atom]:
            AO_slices.append(AO_slices[-1] + tmp if len(AO_slices) != 0 else 0)

            if info[0] == 0:
                AOs.append("s")
                tmp = 1
            elif info[0] == 1:
                AOs.append("p")
                tmp = 3
            elif info[0] == 2:
                AOs.append("d")
                tmp = 5
            else:
                raise NotImplementedError

        AO_slices.append(AO_slices[-1] + tmp if len(AO_slices) != 0 else 0)
        return AOs, AO_slices

    def transform_Q(self, Q: Tensor, atom: str):
        AOs, AO_slices = self.parse_basis(atom)
        Q_aligned_orb = deepcopy(Q)
        for i, mu in enumerate(AOs):
            for j, nu in enumerate(AOs):
                Q_aligned_orb[
                    :, AO_slices[i] : AO_slices[i + 1], AO_slices[j] : AO_slices[j + 1]
                ] = torch.einsum(
                    "hik, kl -> hil",
                    torch.einsum(
                        "ij, hjk -> hik",
                        self.AO_transform_row[mu],
                        Q[
                            :,
                            AO_slices[i] : AO_slices[i + 1],
                            AO_slices[j] : AO_slices[j + 1],
                        ],
                    ),
                    self.AO_transform_row[nu].T,  # transpose for column
                )

        Q_aligned_aux = deepcopy(Q_aligned_orb)

        for i in range(8):
            Q_aligned_aux[16 + i * 3 : 16 + (i + 1) * 3] = torch.einsum(
                "ij, jkl -> ikl",
                self.AO_transform_row["p"],
                Q_aligned_orb[16 + i * 3 : 16 + (i + 1) * 3],
            )

        for i in range(4):
            Q_aligned_aux[40 + i * 5 : 40 + (i + 1) * 5] = torch.einsum(
                "ij, jkl -> ikl",
                self.AO_transform_row["d"],
                Q_aligned_orb[40 + i * 5 : 40 + (i + 1) * 5],
            )

        return Q_aligned_aux


"""
Generalized QR Decomposition in PyTorch
============

Author:
-------
Yuchao Lin

"""
import torch


@torch.jit.script
def find_independent_vectors_cuda(P):
    """Find rank(P) linearly independent vectors from P"""
    n = P.size(0)
    r = int(torch.linalg.matrix_rank(P.to(torch.float32)))

    indices = torch.arange(r)
    done = False
    while not done:
        subset = P[indices, :]
        if torch.linalg.matrix_rank(subset.to(torch.float32)) == r:
            return subset
        done = True
        for i in range(r - 1, -1, -1):
            if indices[i] != i + n - r:
                indices[i] += 1
                for j in range(i + 1, r):
                    indices[j] = indices[j - 1] + 1
                done = False
                break

    return None


@torch.jit.script
def find_independent_vectors_complex_cuda(P):
    """Find rank(P) linearly independent vectors from P"""
    n = P.size(0)
    r = int(torch.linalg.matrix_rank(P.to(torch.complex64)))

    indices = torch.arange(r)
    done = False
    while not done:
        subset = P[indices, :]
        if torch.linalg.matrix_rank(subset.to(torch.complex64)) == r:
            return subset
        done = True
        for i in range(r - 1, -1, -1):
            if indices[i] != i + n - r:
                indices[i] += 1
                for j in range(i + 1, r):
                    indices[j] = indices[j - 1] + 1
                done = False
                break

    return None


@torch.jit.script
def qr_decomposition(coords):
    """QR decomposition on induced set"""
    vecs = find_independent_vectors_cuda(coords)
    assert vecs is not None
    Q, R = torch.linalg.qr(vecs.transpose(0, 1), mode="complete")
    for j in range(R.size(1)):
        if R[j, j] < 0:
            R[j, :] *= -1
            Q[:, j] *= -1
    return Q, R


@torch.jit.script
def qr_decomposition_complex(coords):
    """Complex QR decomposition on induced set"""
    vecs = find_independent_vectors_complex_cuda(coords)
    assert vecs is not None
    Q, R = torch.linalg.qr(vecs.H, mode="complete")
    for j in range(R.size(1)):
        if R[j, j].real < 0:
            R[j, :] *= -1
            Q[:, j] *= -1
    return Q, R


@torch.jit.script
def inner_product(u, v, eta):
    return torch.dot(u, torch.mv(eta, v))


@torch.jit.script
def project(u, v, eta):
    norm_sq = inner_product(u, u, eta)
    assert norm_sq != 0.0
    coeff = inner_product(u, v, eta) / norm_sq
    return coeff * u


@torch.jit.script
def gram_schmidt(A, eta):
    """Generalized Gram-Schmidt orthogonalization"""
    m, n = A.size()
    metric_length = eta.size(0)
    Q = torch.zeros((m, metric_length), dtype=A.dtype).to(A.device)
    R = torch.zeros((metric_length, n), dtype=A.dtype).to(A.device)
    eta_c = torch.zeros(metric_length, dtype=A.dtype).to(A.device)

    for j in range(n):
        v = A[:, j]

        for i in range(j):
            v -= project(Q[:, i], A[:, j], eta)
        norm_sq = inner_product(v, v, eta)
        norm_sq = torch.sqrt(torch.abs(norm_sq))
        assert norm_sq != 0.0
        Q[:, j] = v / norm_sq
        Rjj = inner_product(Q[:, j], A[:, j], eta)
        if Rjj < 0:
            Q[:, j] = -Q[:, j]
            Rjj = -Rjj
        R[j, j] = Rjj
        eta_c[j] = torch.sign(inner_product(Q[:, j], Q[:, j], eta))
        for i in range(j):
            R[i, j] = inner_product(Q[:, i], A[:, j], eta)

    return Q, eta_c, R


@torch.jit.script
def generate_permutation(eta_c, eta, vecs):
    """Algorithm 2"""
    n, d = vecs.size()
    S = torch.eye(d).to(vecs.dtype)
    a = eta_c
    b = torch.diag(eta)

    for i in range(d):
        if a[i] != 0 and a[i] != b[i]:
            for j in range(i + 1, d):
                if a[j] != b[j] and a[j] == b[i]:
                    S_prime = torch.eye(d).to(vecs.dtype)
                    S_prime[i, i] = 0
                    S_prime[j, j] = 0
                    S_prime[i, j] = 1
                    S_prime[j, i] = 1
                    a = torch.matmul(a, S_prime)
                    S = torch.matmul(S, S_prime)
                    break
    return S.T


@torch.jit.script
def generate_minkowski_permutation_matrix(Q, eta):
    """Algorithm 2 but only for O(1,d-1)/SO(1,d-1)"""
    diag_elements = torch.diag(torch.matmul(Q.T, torch.matmul(eta, Q)))
    swap_index = int(torch.argmax(diag_elements).item())
    P = torch.eye(len(diag_elements)).to(Q.dtype).to(Q.device)
    P[0, 0] = 0.0
    P[swap_index, swap_index] = 0.0
    P[0, swap_index] = 1.0
    P[swap_index, 0] = 1.0
    return P


@torch.jit.script
def generalized_qr_decomposition(coords, eta):
    """Generalized QR decomposition"""
    vecs = find_independent_vectors_cuda(coords)
    assert vecs is not None
    Q, eta_c, R = gram_schmidt(vecs.transpose(0, 1), eta)
    # P = generate_minkowski_permutation_matrix(Q, eta)
    P = generate_permutation(eta_c, eta, vecs)
    Q = Q @ P
    R = P.T @ torch.diag(eta_c) @ R
    return Q, R
