"""
cuEquivariance Layers Module

This module provides cuEquivariance implementations of commonly used e3nn layers,
maintaining the same interface and functionality while leveraging cuEquivariance
for improved performance.

Key implementations:
- CueNorm: Equivalent to e3nn.o3.Norm
- CueElementwiseTensorProduct: Equivalent to e3nn.o3.ElementwiseTensorProduct
- Additional utility functions for conversion between e3nn and cuEquivariance
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Sequence

from torch.nn import functional as F
from torch_cluster import radius_graph
from torch_scatter import scatter, scatter_sum
import types
import itertools

from typing import Iterator
import numpy as np
import itertools


from e3nn import o3  # Keep for compatibility with some functions
from e3nn.nn import FullyConnectedNet  # Keep for now, may need to replace
from e3nn.math import normalize2mom, perm  # Keep for utility functions
from e3nn.util.jit import compile_mode  # Keep for utility functions

from .modules import *
from .layers import InnerProduct

# cuEquivariance imports
import cuequivariance as cue
import cuequivariance_torch as cuet




class CuEquivarianceConfig:
    """Configuration for cuequivariance acceleration"""

    layout: str = "mul_ir"  # One of: mul_ir, ir_mul
    layout_str: str = "mul_ir"
    group: str = "O3"
    optimize_all: bool = True  # Set to True to enable all optimizations
    optimize_linear: bool = True
    optimize_channelwise: bool = True
    optimize_symmetric: bool = True
    optimize_fctp: bool = True
    conv_fusion: bool = False  # Set to True to enable conv fusion

    def __post_init__(self):
        self.layout_str = self.layout
        self.layout = getattr(cue, self.layout)
        self.group = (
            O3_e3nn if self.group == "O3_e3nn" else getattr(cue, self.group)
        )

# # O3_e3nn class for compatibility
# class O3_e3nn(cue.O3):
#     def __mul__(self, rep2):
#         return [O3_e3nn(l=ir.l, p=ir.p) for ir in cue.O3.__mul__(self, rep2)]

#     @classmethod
#     def clebsch_gordan(cls, rep1, rep2, rep3):
#         rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)
#         if rep1.p * rep2.p == rep3.p:
#             return o3.wigner_3j(rep1.l, rep2.l, rep3.l).numpy()[None] * np.sqrt(rep3.dim)
#         return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

#     def __lt__(self, rep2):
#         rep2 = self._from(rep2)
#         return (self.l, self.p) < (rep2.l, rep2.p)

#     @classmethod
#     def iterator(cls):
#         for l in itertools.count(0):
#             yield O3_e3nn(l=l, p=1 * (-1) ** l)
#             yield O3_e3nn(l=l, p=-1 * (-1) ** l)

class O3_e3nn(cue.O3):
    def __mul__(rep1: "O3_e3nn", rep2: "O3_e3nn") -> Iterator["O3_e3nn"]:
        return [O3_e3nn(l=ir.l, p=ir.p) for ir in cue.O3.__mul__(rep1, rep2)]

    @classmethod
    def clebsch_gordan(
        cls, rep1: "O3_e3nn", rep2: "O3_e3nn", rep3: "O3_e3nn"
    ) -> np.ndarray:
        rep1, rep2, rep3 = cls._from(rep1), cls._from(rep2), cls._from(rep3)

        if rep1.p * rep2.p == rep3.p:
            return o3.wigner_3j(rep1.l, rep2.l, rep3.l).numpy()[None] * np.sqrt(rep3.dim)
        else:
            return np.zeros((0, rep1.dim, rep2.dim, rep3.dim))

    def __lt__(rep1: "O3_e3nn", rep2: "O3_e3nn") -> bool:
        rep2 = rep1._from(rep2)
        return (rep1.l, rep1.p) < (rep2.l, rep2.p)

    @classmethod
    def iterator(cls) -> Iterator["O3_e3nn"]:
        for l in itertools.count(0):
            yield O3_e3nn(l=l, p=1 * (-1) ** l)
            yield O3_e3nn(l=l, p=-1 * (-1) ** l)

CUE_LAYER = cue.mul_ir
CUE_GROUPS = O3_e3nn

def as_cue_irreps(irreps: o3.Irreps, group='O3'):
    """Convert e3nn irreps to cuEquivariance irreps"""
    if group == 'O3':
        return cue.Irreps(O3_e3nn, str(irreps))
    elif group == 'SO3':
        assert all(irrep.ir.p == 1 for irrep in irreps)
        return cue.Irreps('SO3', str(irreps).replace('e', ''))
    else:
        raise ValueError(f'Unknown group: {group}')

def as_o3_irreps(irreps, group='O3'):
    """Convert cuEquivariance irreps, string, or o3.Irreps to e3nn o3.Irreps"""
    # If already o3.Irreps, return as is
    if isinstance(irreps, o3.Irreps):
        return irreps
    
    # If string, convert directly to o3.Irreps
    if isinstance(irreps, str):
        if group == 'O3':
            return o3.Irreps(irreps)
        elif group == 'SO3':
            # Handle SO3 format like "0+1" -> "0e+1e" for e3nn
            import re
            # First handle single numbers like "0" -> "0e"
            result = re.sub(r'\b(\d+)\b', r'\1e', irreps)
            return o3.Irreps(result)
        else:
            raise ValueError(f'Unknown group: {group}')
    
    # If cue.Irreps, convert to string then to o3.Irreps
    if isinstance(irreps, cue.Irreps):
        if group == 'O3':
            return o3.Irreps(str(irreps))
        elif group == 'SO3':
            # For SO3, we need to add 'e' parity to each irrep
            irreps_str = str(irreps)
            # Handle SO3 format like "0+1" -> "0e+1e" for e3nn
            import re
            # First handle single numbers like "0" -> "0e"
            result = re.sub(r'\b(\d+)\b', r'\1e', irreps_str)
            return o3.Irreps(result)
        else:
            raise ValueError(f'Unknown group: {group}')
    
    # Try to convert other types to string first
    try:
        return o3.Irreps(str(irreps))
    except:
        raise ValueError(f'Cannot convert {type(irreps)} to o3.Irreps')

def ensure_o3_irreps(irrep):
    """Utility function to ensure input is converted to o3.Irreps object."""
    return irrep if isinstance(irrep, o3.Irreps) else o3.Irreps(irrep)

def get_feasible_irrep(irrep_in1, irrep_in2, cutoff_irrep_out, tp_mode="uvu", return_cue=False):
    irrep_mid = []
    instructions = []
    
    irrep_in1 = as_o3_irreps(irrep_in1)
    irrep_in2 = as_o3_irreps(irrep_in2)
    cutoff_irrep_out = as_o3_irreps(cutoff_irrep_out)

    for i, (_, ir_in) in enumerate(irrep_in1):
        for j, (_, ir_edge) in enumerate(irrep_in2):
            for ir_out in ir_in * ir_edge:
                if ir_out in cutoff_irrep_out:
                    if (cutoff_irrep_out.count(ir_out), ir_out) not in irrep_mid:
                        k = len(irrep_mid)
                        irrep_mid.append((cutoff_irrep_out.count(ir_out), ir_out))
                    else:
                        k = irrep_mid.index((cutoff_irrep_out.count(ir_out), ir_out))

                    instructions.append((i, j, k, tp_mode, True))

    irrep_mid = o3.Irreps(irrep_mid)
    normalization_coefficients = []
    for ins in instructions:
        ins_dict = {
            "uvw": (irrep_in1[ins[0]].mul * irrep_in2[ins[1]].mul),
            "uvu": irrep_in2[ins[1]].mul,
            "uvv": irrep_in1[ins[0]].mul,
            "uuw": irrep_in1[ins[0]].mul,
            "uuu": 1,
            "uvuv": 1,
            "uvu<v": 1,
            "u<vw": irrep_in1[ins[0]].mul * (irrep_in2[ins[1]].mul - 1) // 2,
        }
        alpha = irrep_mid[ins[2]].ir.dim
        x = sum([ins_dict[ins[3]] for ins in instructions])
        if x > 0.0:
            alpha /= x

        normalization_coefficients += [math.sqrt(alpha)]

    irrep_mid, p, _ = irrep_mid.sort()
    instructions = [
        (i_in1, i_in2, p[i_out], mode, train, alpha)
        for (i_in1, i_in2, i_out, mode, train), alpha in zip(
            instructions, normalization_coefficients
        )
    ]
    if return_cue:
        return as_cue_irreps(irrep_mid), instructions
    else:
        return as_o3_irreps(irrep_mid), instructions


class CueGatedActivation(nn.Module):
    """
    CuEquivariance implementation of gated activation.
    
    This layer applies a gated non-linearity where scalars are used to "gate" 
    the irreducible representations. Each irrep is multiplied by a corresponding 
    scalar gate value obtained through a sigmoid activation.
    
    Args:
        irreps: Input irreps (can be cue.Irreps or string)
        activation: Activation function to use for gates (default: torch.sigmoid)
        device: Device to place the layer on
    """
    
    def __init__(self, irreps, activation=torch.sigmoid, group=O3_e3nn, layer=CUE_LAYER):
        super().__init__()
      
        # Convert to cuEquivariance irreps if needed
        self.irreps = as_cue_irreps(irreps)
        self.activation = activation
        
        # Create scalars needed to "activate" all the irreps
        self.irreps_gates = self.irreps.num_irreps * cue.Irreps(group, "0e")
        
        # Linear layer to generate gate scalars
        self.gate_linear = cuet.Linear(
            self.irreps,
            self.irreps_gates,
            layout=CUE_LAYER,
        )
        
        # Set up elementwise tensor product for gating
        e = cue.descriptors.elementwise_tensor_product(self.irreps_gates, self.irreps)
        e = e.flatten_coefficient_modes()
        self.gate_mul = cuet.SegmentedPolynomial(e.polynomial, method="fused_tp")
    
    def forward(self, input_tensor):
        """
        Apply gated activation to input.
        
        Args:
            input_tensor: Input tensor with shape (..., irreps.dim)
            
        Returns:
            Output tensor with same irreps as input
        """
        # Generate gate scalars
        scalars = self.gate_linear(input_tensor)
        
        # Apply activation to gates
        gates = self.activation(scalars)
        
        # Apply gates to input via elementwise tensor product
        [output] = self.gate_mul([gates, input_tensor])
        
        return output
    
    def __repr__(self):
        return f"CueGatedActivation(irreps={self.irreps}, activation={self.activation.__name__})"


class CueLinear:
    """Returns either a cuet.Linear based on config"""

    def __new__(
        cls,
        irreps_in: cue.Irreps,
        irreps_out: cue.Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        biases: bool = False,
        group=O3_e3nn,
        layer=CUE_LAYER,
    ):
        return cuet.Linear(
            cue.Irreps(group, irreps_in),
            cue.Irreps(group, irreps_out),
            layout=layer,
            shared_weights=shared_weights,
            use_fallback=True,
            # Note: cuet.Linear doesn't have biases parameter, but we accept it for compatibility
        )

# https://github.com/ACEsuit/mace/blob/9d31ac2c86ebc88c7a843fa7a3dfe360b276f08b/mace/modules/wrapper_ops.py#L127
def with_cueq_conv_fusion(conv_tp: torch.nn.Module) -> torch.nn.Module:
    """Wraps a cuet.ConvTensorProduct to use conv fusion"""
    conv_tp.original_forward = conv_tp.forward
    num_segment = conv_tp.m.buffer_num_segments[0]
    num_operands = conv_tp.m.operand_extent
    conv_tp.weight_numel = num_segment * num_operands

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        tp_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        return self.original_forward(
            [tp_weights, node_feats, edge_attrs],
            {1: sender},
            {0: node_feats},
            {0: receiver},
        )[0]

    conv_tp.forward = types.MethodType(forward, conv_tp)
    return conv_tp

# https://github.com/ACEsuit/mace/blob/9d31ac2c86ebc88c7a843fa7a3dfe360b276f08b/mace/modules/wrapper_ops.py#L105
def with_scatter_sum(conv_tp: torch.nn.Module) -> torch.nn.Module:
    conv_tp.original_forward = conv_tp.forward

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        tp_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        mji = self.original_forward(node_feats[sender], edge_attrs, tp_weights)
        message = scatter_sum(src=mji, index=receiver, dim=0, dim_size=num_nodes)
        return message

    conv_tp.forward = types.MethodType(forward, conv_tp)
    return conv_tp

class CueTensorProduct:
    """Wrapper around o3.TensorProduct/cuet.ChannelwiseTensorProduct/oeq.TensorProduct followed by a scatter sum"""

    def __new__(
        cls,
        irreps_in1: cue.Irreps,
        irreps_in2: cue.Irreps,
        irreps_out: cue.Irreps,
        shared_weights: bool = False,
        internal_weights: bool = False,
        conv_fusion: bool = False,
        group=O3_e3nn,
        layer=CUE_LAYER,
    ):
        if conv_fusion:
            return with_cueq_conv_fusion(
                cuet.SegmentedPolynomial(
                    cue.descriptors.channelwise_tensor_product(
                        cue.Irreps(group, irreps_in1),
                        cue.Irreps(group, irreps_in2),
                        cue.Irreps(group, irreps_out),
                    )
                    .flatten_coefficient_modes()
                    .squeeze_modes()
                    .polynomial,
                    math_dtype=torch.get_default_dtype(),
                )
            )
        else:
            return cuet.ChannelWiseTensorProduct(
                cue.Irreps(group, irreps_in1),
                cue.Irreps(group, irreps_in2),
                cue.Irreps(group, irreps_out),
                layout=layer,
                shared_weights=shared_weights,
                internal_weights=internal_weights,
                dtype=torch.get_default_dtype(),
                math_dtype=torch.get_default_dtype(),
            )
            

class CueConvLayer(torch.nn.Module):
    def __init__(
        self,
        irrep_in_node,
        irrep_hidden,
        irrep_out,
        sh_irrep,
        edge_attr_dim,
        node_attr_dim,
        invariant_layers=1,
        invariant_neurons=32,
        avg_num_neighbors=None,
        nonlinear="ssp",
        use_norm_gate=True,
        edge_wise=False,
        group=O3_e3nn,
        layer=CUE_LAYER,
    ):
        super(CueConvLayer, self).__init__()
        self.avg_num_neighbors = avg_num_neighbors
        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.edge_wise = edge_wise

        self.irrep_in_node = as_o3_irreps(irrep_in_node)
        self.irrep_hidden = as_o3_irreps(irrep_hidden)
        self.irrep_out = as_o3_irreps(irrep_out)
        self.sh_irrep = as_o3_irreps(sh_irrep)
        
        self.irrep_in_node_cue = as_cue_irreps(irrep_in_node)
        self.irrep_hidden_cue = as_cue_irreps(irrep_hidden)
        self.irrep_out_cue = as_cue_irreps(irrep_out)
        self.sh_irrep_cue = as_cue_irreps(sh_irrep)
        
        self.nonlinear_layer = get_nonlinear(nonlinear)

        self.irrep_tp_out_node, _ = get_feasible_irrep(
            self.irrep_in_node, self.sh_irrep, self.irrep_hidden, tp_mode="uvu",
        )
        
        self.irrep_tp_out_node_cue = as_cue_irreps(self.irrep_tp_out_node)
        
        self.tp_node = CueTensorProduct(
            self.irrep_in_node_cue,
            self.sh_irrep,
            self.irrep_tp_out_node_cue,
            shared_weights=False,
            internal_weights=False,
            group=group,
            layer=layer,
        )

        self.fc_node = FullyConnectedNet(
            [self.edge_attr_dim]
            + invariant_layers * [invariant_neurons]
            + [self.tp_node.weight_numel],
            self.nonlinear_layer,
        )

        num_mul = 0
        for mul, ir in self.irrep_in_node:
            num_mul = num_mul + mul

        self.layer_l0 = FullyConnectedNet(
            [num_mul + self.irrep_in_node[0][0]]
            + invariant_layers * [invariant_neurons]
            + [self.tp_node.weight_numel],
            self.nonlinear_layer,
        )

        self.linear_out = CueLinear(
            irreps_in=self.irrep_tp_out_node_cue,
            irreps_out=self.irrep_out_cue,
            internal_weights=True,
            shared_weights=True,
            biases=True,
            group=group,
            layer=layer,
        )

        self.use_norm_gate = use_norm_gate
        self.norm_gate = CueGatedActivation(self.irrep_in_node_cue, group=group, layer=layer)
        self.irrep_linear_out, _ = get_feasible_irrep(
            self.irrep_in_node, o3.Irreps("0e"), self.irrep_in_node
        )
        
        self.irrep_linear_out_cue = as_cue_irreps(self.irrep_linear_out)
        
        
        self.linear_node = CueLinear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_linear_out,
            internal_weights=True,
            shared_weights=True,
            biases=True,
            group=group,
            layer=layer,
        )
        self.linear_node_pre = CueLinear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_linear_out,
            internal_weights=True,
            shared_weights=True,
            biases=True,
            group=group,
            layer=layer,
        )
        self.inner_product = InnerProduct(self.irrep_in_node)

    def forward(self, data, x):
        edge_dst, edge_src = data.edge_index[0], data.edge_index[1]

        if self.use_norm_gate:
            pre_x = self.linear_node_pre(x)
            s0 = self.inner_product(pre_x[edge_dst], pre_x[edge_src])[
                :, self.irrep_in_node.slices()[0].stop :
            ]
            s0 = torch.cat(
                [
                    pre_x[edge_dst][:, self.irrep_in_node.slices()[0]],
                    pre_x[edge_src][:, self.irrep_in_node.slices()[0]],
                    s0,
                ],
                dim=-1,
            )
            x = self.norm_gate(x)
            x = self.linear_node(x)
        else:
            s0 = self.inner_product(x[edge_dst], x[edge_src])[
                :, self.irrep_in_node.slices()[0].stop :
            ]
            s0 = torch.cat(
                [
                    x[edge_dst][:, self.irrep_in_node.slices()[0]],
                    x[edge_src][:, self.irrep_in_node.slices()[0]],
                    s0,
                ],
                dim=-1,
            )

        self_x = x

        edge_features = self.tp_node(
            x[edge_src], data.edge_sh, self.fc_node(data.edge_attr) * self.layer_l0(s0)
        )

        if self.edge_wise:
            out = edge_features
        else:
            out = scatter(edge_features, edge_dst, dim=0, dim_size=len(x))

        import pdb; pdb.set_trace()
        
        if self.irrep_in_node == self.irrep_out:
            out = out + self_x

        out = self.linear_out(out)
        return out


class InnerProduct(torch.nn.Module):
    def __init__(self, irrep_in):
        super(InnerProduct, self).__init__()
        self.irrep_in = o3.Irreps(irrep_in).simplify()
        irrep_out = o3.Irreps([(mul, "0e") for mul, _ in self.irrep_in])
        instr = [
            (i, i, i, "uuu", False, 1 / ir.dim)
            for i, (mul, ir) in enumerate(self.irrep_in)
        ]
        self.tp = o3.TensorProduct(
            self.irrep_in,
            self.irrep_in,
            irrep_out,
            instr,
            irrep_normalization="component",
        )
        self.irrep_out = irrep_out.simplify()

    def forward(self, features_1, features_2):
        out = self.tp(features_1, features_2)
        return out


class CueConvNetLayer(torch.nn.Module):
    def __init__(
        self,
        irrep_in_node,
        irrep_hidden,
        irrep_out,
        sh_irrep,
        edge_attr_dim,
        node_attr_dim,
        resnet: bool = True,
        use_norm_gate=True,
        edge_wise=False,
        group=O3_e3nn,
        layer=CUE_LAYER,
    ):
        super(CueConvNetLayer, self).__init__()
        self.nonlinear_scalars = {1: "ssp", -1: "tanh"}
        self.nonlinear_gates = {1: "ssp", -1: "abs"}

        self.irrep_in_node = as_o3_irreps(irrep_in_node)
        self.irrep_hidden = as_o3_irreps(irrep_hidden)
        self.irrep_out = as_o3_irreps(irrep_out)
        self.sh_irrep = as_o3_irreps(sh_irrep)

        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.resnet = resnet and self.irrep_in_node == self.irrep_out

        self.conv = CueConvLayer(
            irrep_in_node=self.irrep_in_node,
            irrep_hidden=self.irrep_hidden,
            sh_irrep=self.sh_irrep,
            irrep_out=self.irrep_out,
            edge_attr_dim=self.edge_attr_dim,
            node_attr_dim=self.node_attr_dim,
            invariant_layers=1,
            invariant_neurons=32,
            avg_num_neighbors=None,
            nonlinear="ssp",
            use_norm_gate=use_norm_gate,
            edge_wise=edge_wise,
            group=group,
            layer=layer,
        )

    def forward(self, data, x):
        old_x = x
        x = self.conv(data, x)
        if self.resnet and self.irrep_out == self.irrep_in_node:
            x = old_x + x
        return x
