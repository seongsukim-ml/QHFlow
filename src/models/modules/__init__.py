"""
Utility functions and modules for molecular property prediction.

This module contains various mathematical functions, activation functions,
and utility functions used in molecular neural networks, particularly for
SE(3)-equivariant models.

Based on PhiSNet: SE(3)-equivariant prediction of molecular wavefunctions 
and electronic densities <https://arxiv.org/abs/2106.02347>
"""

import math
import torch
import torch.nn.functional as F


# Mathematical constants
_LOG2 = math.log(2.0)


# ============================================================================
# Utility Functions
# ============================================================================

def prod(x):
    """Compute the product of a sequence.
    
    Args:
        x: Sequence of numbers to multiply
        
    Returns:
        Product of all elements in the sequence
    """
    result = 1
    for element in x:
        result *= element
    return result


def softplus_inverse(x):
    """Inverse of the softplus function.
    
    This function is useful for initialization of parameters that are
    constrained to be positive, as it allows setting the pre-activation
    value that will yield a desired positive output after softplus.
    
    Args:
        x: Input values (should be positive)
        
    Returns:
        Inverse softplus: log(exp(x) - 1)
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    
    return x + torch.log(-torch.expm1(-x))


# ============================================================================
# Activation Functions
# ============================================================================

def shifted_softplus(x):
    """Shifted softplus activation function.
    
    This function shifts the standard softplus so that f(0) = 0.
    
    Args:
        x: Input tensor
        
    Returns:
        Shifted softplus activation: softplus(x) - log(2)
    """
    return F.softplus(x) - _LOG2


def get_nonlinear(nonlinear: str):
    """Get activation function by name.
    
    Args:
        nonlinear: Name of the activation function
            - "ssp": Shifted softplus
            - "silu": SiLU (Swish) activation
            - "tanh": Hyperbolic tangent
            - "abs": Absolute value
            
    Returns:
        Activation function
        
    Raises:
        NotImplementedError: If activation function is not recognized
    """
    nonlinear = nonlinear.lower()
    
    activation_map = {
        "ssp": shifted_softplus,
        "silu": F.silu,
        "tanh": F.tanh,
        "abs": torch.abs,
    }
    
    if nonlinear in activation_map:
        return activation_map[nonlinear]
    else:
        available_activations = ", ".join(activation_map.keys())
        raise NotImplementedError(
            f"Activation function '{nonlinear}' not implemented. "
            f"Available options: {available_activations}"
        )


# ============================================================================
# Cutoff and Switch Functions
# ============================================================================

def cutoff_function(x, cutoff):
    """Smooth cutoff function that goes from 1 to 0 in interval [0, cutoff].
    
    This function has infinitely many smooth derivatives and is used for
    smooth distance-based cutoffs in molecular models.
    
    IMPORTANT: This function has numerical singularities at the cutoff point
    that can cause NaN gradients. Input masking may be required for stable
    automatic differentiation.
    
    Args:
        x: Input distances
        cutoff: Cutoff distance
        
    Returns:
        Cutoff values: 1 for x=0, 0 for x>=cutoff, smooth transition between
    """
    zeros = torch.zeros_like(x)
    x_masked = torch.where(x < cutoff, x, zeros)
    
    # Compute exp(-x²/((cutoff-x)(cutoff+x))) for x < cutoff
    denominator = (cutoff - x_masked) * (cutoff + x_masked)
    exponential = torch.exp(-x_masked**2 / denominator)
    
    return torch.where(x < cutoff, exponential, zeros)


def _switch_component(x, ones, zeros):
    """Helper function for switch_function with numerical stability.
    
    This implementation is numerically more stable than a simplified version.
    DO NOT CHANGE without careful numerical analysis.
    
    Args:
        x: Input tensor
        ones: Tensor of ones with same shape as x
        zeros: Tensor of zeros with same shape as x
        
    Returns:
        Switch component: exp(-1/x) for x > 0, 0 for x <= 0
    """
    x_masked = torch.where(x <= 0, ones, x)
    return torch.where(x <= 0, zeros, torch.exp(-ones / x_masked))


def switch_function(x, cuton, cutoff):
    """Smooth switch function that transitions from 1 to 0 between cuton and cutoff.
    
    This function has infinitely many smooth derivatives and provides a smooth
    transition between two regimes:
    - Returns 1 for x <= cuton
    - Returns 0 for x >= cutoff  
    - Smooth transition in between
    
    Note: If cutoff < cuton, the function goes from 0 to 1 instead.
    
    Args:
        x: Input values
        cuton: Start of transition region
        cutoff: End of transition region
        
    Returns:
        Switch values with smooth transition between cuton and cutoff
    """
    # Normalize x to [0, 1] interval
    x_normalized = (x - cuton) / (cutoff - cuton)
    
    ones = torch.ones_like(x_normalized)
    zeros = torch.zeros_like(x_normalized)
    
    # Compute switch components
    fp = _switch_component(x_normalized, ones, zeros)
    fm = _switch_component(1 - x_normalized, ones, zeros)
    
    # Combine components with appropriate conditions
    return torch.where(
        x_normalized <= 0, 
        ones,
        torch.where(
            x_normalized >= 1, 
            zeros, 
            fm / (fp + fm)
        )
    )


# ============================================================================
# Legacy aliases (for backward compatibility)
# ============================================================================

# Alias for consistency with other parts of the codebase
ShiftedSoftPlus = shifted_softplus


# ============================================================================
# Import RBF classes after defining utility functions (to avoid circular imports)
# ============================================================================

from .exponential_bernstein_radial_basis_functions import (
    Envelope,
    DistanceEmbedding,
    ExponentialBernsteinRadialBasisFunctions,
    # Legacy aliases
    dist_emb,
)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Core classes
    "Envelope",
    "DistanceEmbedding", 
    "ExponentialBernsteinRadialBasisFunctions",
    
    # Activation functions
    "shifted_softplus",
    "get_nonlinear",
    
    # Cutoff and switch functions
    "cutoff_function",
    "switch_function",
    
    # Utility functions
    "prod",
    "softplus_inverse",
    
    # Legacy aliases
    "ShiftedSoftPlus",
    "dist_emb",
]