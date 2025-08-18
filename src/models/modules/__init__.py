"""
from PhiSNet SE(3)-equivariant prediction of molecular wavefunctions and electronic densities 
<https://arxiv.org/abs/2106.02347>
"""
from .exponential_bernstein_radial_basis_functions import *

def prod(x):
    """Compute the product of a sequence."""
    out = 1
    for a in x:
        out *= a

    return out


def ShiftedSoftPlus(x):
    return torch.nn.functional.softplus(x) - math.log(2.0)


def softplus_inverse(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    return x + torch.log(-torch.expm1(-x))


def get_nonlinear(nonlinear: str):
    if nonlinear.lower() == "ssp":
        return ShiftedSoftPlus
    elif nonlinear.lower() == "silu":
        return F.silu
    elif nonlinear.lower() == "tanh":
        return F.tanh
    elif nonlinear.lower() == "abs":
        return torch.abs
    else:
        raise NotImplementedError