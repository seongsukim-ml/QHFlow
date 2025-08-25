"""
Radial basis functions for molecular neural networks.

This module implements exponential Bernstein radial basis functions and related
distance embedding functions for SE(3)-equivariant molecular property prediction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define utility functions locally to avoid circular imports
def cutoff_function(x, cutoff):
    """Smooth cutoff function that goes from 1 to 0 in interval [0, cutoff]."""
    zeros = torch.zeros_like(x)
    x_masked = torch.where(x < cutoff, x, zeros)
    
    # Compute exp(-x²/((cutoff-x)(cutoff+x))) for x < cutoff
    denominator = (cutoff - x_masked) * (cutoff + x_masked)
    exponential = torch.exp(-x_masked**2 / denominator)
    
    return torch.where(x < cutoff, exponential, zeros)


def softplus_inverse(x):
    """Inverse of the softplus function."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    
    return x + torch.log(-torch.expm1(-x))


class Envelope(nn.Module):
    """Envelope function for distance-based features.
    
    This envelope function provides a smooth decay for distance-based interactions
    and is commonly used in molecular neural networks.
    
    Args:
        exponent (int): Exponent parameter controlling the envelope shape
    """
    
    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.exponent = exponent
        self.p = exponent + 1
        
        # Precompute polynomial coefficients
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        """Apply envelope function.
        
        Args:
            x (torch.Tensor): Input distances
            
        Returns:
            torch.Tensor: Envelope values
        """
        p = self.p
        
        # Compute powers efficiently
        x_pow_p_minus_1 = x.pow(p - 1)
        x_pow_p = x_pow_p_minus_1 * x
        x_pow_p_plus_1 = x_pow_p * x
        
        # Envelope function: 1/x + a*x^(p-1) + b*x^p + c*x^(p+1)
        return (1.0 / x + 
                self.a * x_pow_p_minus_1 + 
                self.b * x_pow_p + 
                self.c * x_pow_p_plus_1)


class DistanceEmbedding(nn.Module):
    """Distance embedding using sinusoidal functions with envelope.
    
    This module creates distance embeddings by combining sinusoidal functions
    with an envelope function for smooth distance-based features.
    
    Args:
        num_radial (int): Number of radial basis functions
        cutoff (float): Cutoff distance (default: 5.0)
        envelope_exponent (int): Exponent for envelope function (default: 5)
    """
    
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super(DistanceEmbedding, self).__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        
        # Learnable frequencies for sinusoidal functions
        self.frequencies = nn.Parameter(torch.Tensor(num_radial))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize frequencies with π, 2π, 3π, ..."""
        with torch.no_grad():
            self.frequencies.data = torch.arange(
                1, self.frequencies.numel() + 1, dtype=torch.float32
            ) * math.pi

    def forward(self, distances):
        """Compute distance embeddings.
        
        Args:
            distances (torch.Tensor): Input distances
            
        Returns:
            torch.Tensor: Distance embeddings of shape [..., num_radial]
        """
        # Normalize distances by cutoff
        normalized_distances = distances.unsqueeze(-1) / self.cutoff
        
        # Compute envelope * sin(frequency * normalized_distance)
        sinusoidal_features = torch.sin(self.frequencies * normalized_distances)
        envelope_values = self.envelope(normalized_distances)
        
        return envelope_values * sinusoidal_features


class ExponentialBernsteinRadialBasisFunctions(nn.Module):
    """Exponential Bernstein radial basis functions.
    
    This module implements radial basis functions using exponential Bernstein
    polynomials, which provide a flexible and smooth representation for
    distance-based interactions in molecular systems.
    
    The basis functions have the form:
    RBF_v(r) = cutoff(r) * C(n,v) * exp(-α*r)^n * (1-exp(-α*r))^v
    
    where C(n,v) is the binomial coefficient.
    
    Args:
        num_basis_functions (int): Number of radial basis functions
        cutoff (float): Cutoff distance
        ini_alpha (float): Initial value for α parameter (default: 0.5)
        fix_alpha (bool): Whether to fix α or make it learnable (default: True)
    """
    
    def __init__(self, num_basis_functions, cutoff, ini_alpha=0.5, fix_alpha=True):
        super(ExponentialBernsteinRadialBasisFunctions, self).__init__()
        
        self.num_basis_functions = num_basis_functions
        self.ini_alpha = ini_alpha
        self.fix_alpha = fix_alpha
        
        # Precompute binomial coefficients and indices
        self._precompute_coefficients(num_basis_functions)
        
        # Register cutoff as buffer
        self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float32))
        
        # Alpha parameter (learnable or fixed)
        self.register_parameter(
            '_alpha', 
            nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        )
        
        self.reset_parameters()

    def _precompute_coefficients(self, num_basis_functions):
        """Precompute logarithmic binomial coefficients for numerical stability."""
        # Compute log(factorial) values
        log_factorial = np.zeros(num_basis_functions)
        for i in range(2, num_basis_functions):
            log_factorial[i] = log_factorial[i-1] + np.log(i)
        
        # Compute indices for Bernstein polynomials
        v_indices = np.arange(0, num_basis_functions)  # polynomial degree
        n_indices = (num_basis_functions - 1) - v_indices  # complementary degree
        
        # Compute log binomial coefficients: log(C(n+v, v))
        log_binomial = (log_factorial[-1] - 
                       log_factorial[v_indices] - 
                       log_factorial[n_indices])
        
        # Register as buffers
        self.register_buffer('log_binomial_coeff', 
                           torch.tensor(log_binomial, dtype=torch.float32))
        self.register_buffer('n_indices', 
                           torch.tensor(n_indices, dtype=torch.float32))
        self.register_buffer('v_indices', 
                           torch.tensor(v_indices, dtype=torch.float32))

    def reset_parameters(self):
        """Initialize alpha parameter using softplus inverse for proper scaling."""
        with torch.no_grad():
            self._alpha.data.fill_(softplus_inverse(self.ini_alpha))

    @property
    def alpha(self):
        """Get the current alpha value (fixed or learnable)."""
        if self.fix_alpha:
            return 1.0
        else:
            return F.softplus(self._alpha)

    def forward(self, distances):
        """Compute exponential Bernstein radial basis functions.
        
        Args:
            distances (torch.Tensor): Input distances
            
        Returns:
            torch.Tensor: RBF values of shape [..., num_basis_functions]
        """
        # Get alpha parameter
        alpha = self.alpha
        
        # Compute exponential terms
        negative_alpha_r = -alpha * distances
        exp_neg_alpha_r = torch.exp(negative_alpha_r)
        one_minus_exp = -torch.expm1(negative_alpha_r)  # 1 - exp(-αr), numerically stable
        
        # Compute log terms for numerical stability
        # log(RBF) = log_binomial + n*log(exp(-αr)) + v*log(1-exp(-αr))
        log_terms = (self.log_binomial_coeff + 
                    self.n_indices * negative_alpha_r + 
                    self.v_indices * torch.log(one_minus_exp))
        
        # Compute RBF values with cutoff
        rbf_values = torch.exp(log_terms)
        cutoff_values = cutoff_function(distances, self.cutoff)
        
        return cutoff_values * rbf_values


# Legacy alias for backward compatibility
dist_emb = DistanceEmbedding