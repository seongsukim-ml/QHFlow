import math
import collections
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch_scatter import scatter
import torch_geometric
from torch_geometric.nn import global_mean_pool, global_max_pool

from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Linear, TensorProduct
from e3nn.o3._norm import Norm
from e3nn.math import normalize2mom, perm
from e3nn.util.jit import compile_mode

from .modules import *


_RESCALE = True


def ensure_irreps(irrep):
    """
    Utility function to ensure input is converted to o3.Irreps object.
    
    Args:
        irrep: Either an o3.Irreps object or a string/other representation
               that can be converted to o3.Irreps
    
    Returns:
        o3.Irreps: The input converted to o3.Irreps object
    """
    return irrep if isinstance(irrep, o3.Irreps) else o3.Irreps(irrep)


def get_feasible_irrep(irrep_in1, irrep_in2, cutoff_irrep_out, tp_mode="uvu"):
    irrep_mid = []
    instructions = []

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
    return irrep_mid, instructions

@compile_mode("trace")
class NormGate(torch.nn.Module):
    def __init__(self, irrep):
        super(NormGate, self).__init__()
        self.irrep = irrep
        self.norm = o3.Norm(self.irrep)

        num_mul, num_mul_wo_0 = 0, 0
        for mul, ir in self.irrep:
            num_mul += mul
            if ir.l != 0:
                num_mul_wo_0 += mul

        self.mul = o3.ElementwiseTensorProduct(
            self.irrep[1:], o3.Irreps(f"{num_mul_wo_0}x0e")
        )
        self.fc = nn.Sequential(
            nn.Linear(num_mul, num_mul), nn.SiLU(), nn.Linear(num_mul, num_mul)
        )

        self.num_mul = num_mul
        self.num_mul_wo_0 = num_mul_wo_0

    def forward(self, x):
        norm_x = self.norm(x)[:, self.irrep.slices()[0].stop :]
        f0 = torch.cat([x[:, self.irrep.slices()[0]], norm_x], dim=-1)
        gates = self.fc(f0)
        gated = self.mul(
            x[:, self.irrep.slices()[0].stop :], gates[:, self.irrep.slices()[0].stop :]
        )
        x = torch.cat([gates[:, self.irrep.slices()[0]], gated], dim=-1)
        return x

@compile_mode("trace")
class ConvLayer(torch.nn.Module):
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
    ):
        super(ConvLayer, self).__init__()
        self.avg_num_neighbors = avg_num_neighbors
        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.edge_wise = edge_wise

        self.irrep_in_node = ensure_irreps(irrep_in_node)
        self.irrep_hidden = ensure_irreps(irrep_hidden)
        self.irrep_out = ensure_irreps(irrep_out)
        self.sh_irrep = ensure_irreps(sh_irrep)
        self.nonlinear_layer = get_nonlinear(nonlinear)

        self.irrep_tp_out_node, instruction_node = get_feasible_irrep(
            self.irrep_in_node, self.sh_irrep, self.irrep_hidden, tp_mode="uvu"
        )

        self.tp_node = TensorProduct(
            self.irrep_in_node,
            self.sh_irrep,
            self.irrep_tp_out_node,
            instruction_node,
            shared_weights=False,
            internal_weights=False,
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

        self.linear_out = Linear(
            irreps_in=self.irrep_tp_out_node,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )

        self.use_norm_gate = use_norm_gate
        self.norm_gate = NormGate(self.irrep_in_node)
        self.irrep_linear_out, instruction_node = get_feasible_irrep(
            self.irrep_in_node, o3.Irreps("0e"), self.irrep_in_node
        )
        self.linear_node = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_linear_out,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )
        self.linear_node_pre = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_linear_out,
            internal_weights=True,
            shared_weights=True,
            biases=True,
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

@compile_mode("trace")
class ConvNetLayer(torch.nn.Module):
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
    ):
        super(ConvNetLayer, self).__init__()
        # Use torch.jit.Attribute for TorchScript compatibility
        self.nonlinear_scalars = torch.jit.Attribute({1: "ssp", -1: "tanh"}, Dict[int, str])
        self.nonlinear_gates = torch.jit.Attribute({1: "ssp", -1: "abs"}, Dict[int, str])

        self.irrep_in_node = ensure_irreps(irrep_in_node)
        self.irrep_hidden = ensure_irreps(irrep_hidden)
        self.irrep_out = ensure_irreps(irrep_out)
        self.sh_irrep = ensure_irreps(sh_irrep)

        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.resnet = resnet and self.irrep_in_node == self.irrep_out

        self.conv = ConvLayer(
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
        )

    def forward(self, data, x):
        old_x = x
        x = self.conv(data, x)
        if self.resnet and self.irrep_out == self.irrep_in_node:
            x = old_x + x
        return x

@compile_mode("trace")
class PairNetLayer(torch.nn.Module):
    def __init__(
        self,
        irrep_in_node,
        irrep_bottle_hidden,
        irrep_out,
        sh_irrep,
        edge_attr_dim,
        node_attr_dim,
        resnet: bool = True,
        invariant_layers=1,
        invariant_neurons=8,
        nonlinear="ssp",
    ):
        super(PairNetLayer, self).__init__()
        # Use torch.jit.Attribute for TorchScript compatibility
        self.nonlinear_scalars = torch.jit.Attribute({1: "ssp", -1: "tanh"}, Dict[int, str])
        self.nonlinear_gates = torch.jit.Attribute({1: "ssp", -1: "abs"}, Dict[int, str])
        self.invariant_layers = invariant_layers
        self.invariant_neurons = invariant_neurons
        self.irrep_in_node = ensure_irreps(irrep_in_node)
        self.irrep_bottle_hidden = ensure_irreps(irrep_bottle_hidden)
        self.irrep_out = ensure_irreps(irrep_out)
        self.sh_irrep = ensure_irreps(sh_irrep)

        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.nonlinear_layer = get_nonlinear(nonlinear)

        self.irrep_tp_in_node, _ = get_feasible_irrep(
            self.irrep_in_node, o3.Irreps("0e"), self.irrep_bottle_hidden
        )
        self.irrep_tp_out_node_pair, instruction_node_pair = get_feasible_irrep(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_bottle_hidden,
            tp_mode="uuu",
        )

        self.irrep_tp_out_node_pair_msg, instruction_node_pair_msg = get_feasible_irrep(
            self.irrep_tp_in_node,
            self.sh_irrep,
            self.irrep_bottle_hidden,
            tp_mode="uvu",
        )

        self.linear_node_pair = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_tp_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )

        self.linear_node_pair_n = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )
        self.linear_node_pair_inner = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )

        self.tp_node_pair = TensorProduct(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_tp_out_node_pair,
            instruction_node_pair,
            shared_weights=False,
            internal_weights=False,
        )

        self.irrep_tp_out_node_pair_2, instruction_node_pair_2 = get_feasible_irrep(
            self.irrep_tp_out_node_pair,
            self.irrep_tp_out_node_pair,
            self.irrep_bottle_hidden,
            tp_mode="uuu",
        )

        self.tp_node_pair_2 = TensorProduct(
            self.irrep_tp_out_node_pair,
            self.irrep_tp_out_node_pair,
            self.irrep_tp_out_node_pair_2,
            instruction_node_pair_2,
            shared_weights=True,
            internal_weights=True,
        )

        self.fc_node_pair = FullyConnectedNet(
            [self.edge_attr_dim]
            + invariant_layers * [invariant_neurons]
            + [self.tp_node_pair.weight_numel],
            self.nonlinear_layer,
        )

        self.linear_node_pair_2 = Linear(
            irreps_in=self.irrep_tp_out_node_pair_2,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )

        if self.irrep_in_node == self.irrep_out and resnet:
            self.resnet = True
        else:
            self.resnet = False

        self.linear_node_pair = Linear(
            irreps_in=self.irrep_tp_out_node_pair,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )
        self.norm_gate = NormGate(self.irrep_tp_out_node_pair)
        self.inner_product = InnerProduct(self.irrep_in_node)
        self.norm = o3.Norm(self.irrep_in_node)
        num_mul = 0
        for mul, ir in self.irrep_in_node:
            num_mul = num_mul + mul

        self.norm_gate_pre = NormGate(self.irrep_tp_out_node_pair)
        self.fc = nn.Sequential(
            nn.Linear(self.irrep_in_node[0][0] + num_mul, self.irrep_in_node[0][0]),
            nn.SiLU(),
            nn.Linear(self.irrep_in_node[0][0], self.tp_node_pair.weight_numel),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data, node_attr, node_pair_attr=None):
        dst, src = data.full_edge_index
        node_attr_0 = self.linear_node_pair_inner(node_attr)
        s0 = self.inner_product(node_attr_0[dst], node_attr_0[src])[
            :, self.irrep_in_node.slices()[0].stop :
        ]
        s0 = torch.cat(
            [
                node_attr_0[dst][:, self.irrep_in_node.slices()[0]],
                node_attr_0[src][:, self.irrep_in_node.slices()[0]],
                s0,
            ],
            dim=-1,
        )

        node_attr = self.norm_gate_pre(node_attr)
        node_attr = self.linear_node_pair_n(node_attr)

        node_pair = self.tp_node_pair(
            node_attr[src],
            node_attr[dst],
            self.fc_node_pair(data.full_edge_attr) * self.fc(s0),
        )

        node_pair = self.norm_gate(node_pair)
        node_pair = self.linear_node_pair(node_pair)

        if self.resnet and node_pair_attr is not None:
            node_pair = node_pair + node_pair_attr
        return node_pair

@compile_mode("trace")
class SelfNetLayer(torch.nn.Module):
    def __init__(
        self,
        irrep_in_node,
        irrep_bottle_hidden,
        irrep_out,
        sh_irrep,
        edge_attr_dim,
        node_attr_dim,
        resnet: bool = True,
        nonlinear="ssp",
    ):
        super(SelfNetLayer, self).__init__()
        # Use torch.jit.Attribute for TorchScript compatibility
        self.nonlinear_scalars = torch.jit.Attribute({1: "ssp", -1: "tanh"}, Dict[int, str])
        self.nonlinear_gates = torch.jit.Attribute({1: "ssp", -1: "abs"}, Dict[int, str])
        self.sh_irrep = sh_irrep
        self.irrep_in_node = ensure_irreps(irrep_in_node)
        self.irrep_bottle_hidden = ensure_irreps(irrep_bottle_hidden)
        self.irrep_out = ensure_irreps(irrep_out)

        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.resnet = resnet
        self.nonlinear_layer = get_nonlinear(nonlinear)

        self.irrep_tp_in_node, _ = get_feasible_irrep(
            self.irrep_in_node, o3.Irreps("0e"), self.irrep_bottle_hidden
        )
        self.irrep_tp_out_node, instruction_node = get_feasible_irrep(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_bottle_hidden,
            tp_mode="uuu",
        )

        # - Build modules -
        self.linear_node_1 = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )

        self.linear_node_2 = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )
        self.tp = TensorProduct(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_tp_out_node,
            instruction_node,
            shared_weights=True,
            internal_weights=True,
        )
        self.norm_gate = NormGate(self.irrep_out)
        self.norm_gate_1 = NormGate(self.irrep_in_node)
        self.norm_gate_2 = NormGate(self.irrep_in_node)
        self.linear_node_3 = Linear(
            irreps_in=self.irrep_tp_out_node,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )

    def forward(self, data, x, old_fii):
        old_x = x
        xl = self.norm_gate_1(x)
        xl = self.linear_node_1(xl)
        xr = self.norm_gate_2(x)
        xr = self.linear_node_2(xr)
        x = self.tp(xl, xr)
        if self.resnet:
            x = x + old_x
        x = self.norm_gate(x)
        x = self.linear_node_3(x)
        if self.resnet and old_fii is not None:
            x = old_fii + x
        return x

    @property
    def device(self):
        return next(self.parameters()).device


class Expansion(nn.Module):
    def __init__(self, irrep_in, irrep_out_1, irrep_out_2):
        super(Expansion, self).__init__()
        self.irrep_in = irrep_in
        self.irrep_out_1 = irrep_out_1
        self.irrep_out_2 = irrep_out_2
        self.instructions = self.get_expansion_path(irrep_in, irrep_out_1, irrep_out_2)
        self.num_path_weight = sum(prod(ins[-1]) for ins in self.instructions if ins[3])
        self.num_bias = sum(
            [prod(ins[-1][1:]) for ins in self.instructions if ins[0] == 0]
        )
        if self.num_path_weight > 0:
            self.weights = nn.Parameter(
                torch.rand(self.num_path_weight + self.num_bias)
            )
        self.num_weights = self.num_path_weight + self.num_bias

    def forward(self, x_in, weights=None, bias_weights=None):
        batch_num = x_in.shape[0]
        if len(self.irrep_in) == 1:
            x_in_s = [
                x_in.reshape(batch_num, self.irrep_in[0].mul, self.irrep_in[0].ir.dim)
            ]
        else:
            x_in_s = [
                x_in[:, i].reshape(batch_num, mul_ir.mul, mul_ir.ir.dim)
                for i, mul_ir in zip(self.irrep_in.slices(), self.irrep_in)
            ]

        outputs = {}
        flat_weight_index = 0
        bias_weight_index = 0
        for ins in self.instructions:
            mul_ir_in = self.irrep_in[ins[0]]
            mul_ir_out1 = self.irrep_out_1[ins[1]]
            mul_ir_out2 = self.irrep_out_2[ins[2]]
            x1 = x_in_s[ins[0]]
            x1 = x1.reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim)
            w3j_matrix = (
                o3.wigner_3j(ins[1], ins[2], ins[0]).to(self.device).type(x1.type())
            )
            if ins[3] is True or weights is not None:
                if weights is None:
                    weight = self.weights[
                        flat_weight_index : flat_weight_index + prod(ins[-1])
                    ].reshape(ins[-1])
                    result = (
                        torch.einsum(f"wuv, ijk, bwk-> buivj", weight, w3j_matrix, x1)
                        / mul_ir_in.mul
                    )
                else:
                    weight = weights[
                        :, flat_weight_index : flat_weight_index + prod(ins[-1])
                    ].reshape([-1] + ins[-1])
                    result = torch.einsum(f"bwuv, bwk-> buvk", weight, x1)
                    if ins[0] == 0 and bias_weights is not None:
                        bias_weight = bias_weights[
                            :, bias_weight_index : bias_weight_index + prod(ins[-1][1:])
                        ].reshape([-1] + ins[-1][1:])
                        bias_weight_index += prod(ins[-1][1:])
                        result = result + bias_weight.unsqueeze(-1)
                    result = (
                        torch.einsum(f"ijk, buvk->buivj", w3j_matrix, result)
                        / mul_ir_in.mul
                    )
                flat_weight_index += prod(ins[-1])
            else:
                result = torch.einsum(
                    f"uvw, ijk, bwk-> buivj",
                    torch.ones(ins[-1]).type(x1.type()).to(self.device),
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

    @property
    def device(self):
        return next(self.parameters()).device

    def __repr__(self):
        return (
            f"{self.irrep_in} -> {self.irrep_out_1}x{self.irrep_out_1} and bias {self.num_bias}"
            f"with parameters {self.num_path_weight}"
        )


class OneBody_Reduction(nn.Module):
    r"""The one-body reduction module from the
    `"Informing geometric deep learning with electronic interactionsto accelerate quantum chemistry"
    <https://www.pnas.org/doi/epdf/10.1073/pnas.2205221119>`_ paper

    For each diagonal block of :math:`T`, :math:`T_{AA}` defined for an on-site atom pair :math:`(A, A)` is

    .. math::
        T_{AA}^{\mu\nu} = \langle\Phi_A^{\mu} \vert \hat{H} \vert \Phi_A^{\nu} \rangle

    There exists a set of :math:`T`-independent coefficients :math:`Q_{nlpm}^{\mu\nu}` such that the linear
    transformation :math:`\psi`

    .. math::
        \psi(T_{AA})_{nlpm} = \sum_{\mu\nu}T_{AA}^{\mu\nu}Q_{nlpm^\prime}^{\mu\nu}

    is injective and :math:`h_A=\psi(T_{AA})` satisfies equivariance.
    """

    def __init__(self):
        super(OneBody_Reduction, self).__init__()
        self.norm = Norm("16x0e+8x1o+4x2e")

    def forward(self, data, matrix, keep_block=False):
        if keep_block:
            diagonal_matrix, non_diagonal_matrix = matrix
            node_feats = torch.sum(
                torch.sum(diagonal_matrix.unsqueeze(-1) * data.diagonal_Q, dim=1), dim=1
            )
        else:
            H_size = matrix.size(-1)
            node_feats = torch.sum(
                (
                    matrix.unsqueeze(-1)
                    * data.Q.view(data.num_graphs, H_size, H_size, 60)
                ),
                dim=1,
            ).view(-1, 60)
            node_feats = scatter(node_feats, data.AO_index[0], dim=0)

        norm_irrpes = self.norm(node_feats)
        norm = torch.repeat_interleave(
            norm_irrpes,
            torch.tensor(
                [1 for _ in range(16)] + [3 for _ in range(8)] + [5 for _ in range(4)]
            ).to(norm_irrpes.device),
            dim=-1,
        )
        norm[norm == 0] += 1e-5
        return node_feats / norm


class TensorProductRescale(torch.nn.Module):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
    ):

        super().__init__()

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
        self.rescale = rescale
        self.use_bias = bias

        # e3nn.__version__ == 0.4.4
        # Use `path_normalization` == 'none' to remove normalization factor
        self.tp = o3.TensorProduct(
            irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in2,
            irreps_out=self.irreps_out,
            instructions=instructions,
            normalization=normalization,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            path_normalization="none",
        )

        self.init_rescale_bias()

    def calculate_fan_in(self, ins):
        return {
            "uvw": (self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul),
            "uvu": self.irreps_in2[ins.i_in2].mul,
            "uvv": self.irreps_in1[ins.i_in1].mul,
            "uuw": self.irreps_in1[ins.i_in1].mul,
            "uuu": 1,
            "uvuv": 1,
            "uvu<v": 1,
            "u<vw": self.irreps_in1[ins.i_in1].mul
            * (self.irreps_in2[ins.i_in2].mul - 1)
            // 2,
        }[ins.connection_mode]

    def init_rescale_bias(self) -> None:

        irreps_out = self.irreps_out
        # For each zeroth order output irrep we need a bias
        # Determine the order for each output tensor and their dims
        self.irreps_out_orders = [
            int(irrep_str[-2]) for irrep_str in str(irreps_out).split("+")
        ]
        self.irreps_out_dims = [
            int(irrep_str.split("x")[0]) for irrep_str in str(irreps_out).split("+")
        ]
        self.irreps_out_slices = irreps_out.slices()

        # Store tuples of slices and corresponding biases in a list
        self.bias = None
        self.bias_slices = []
        self.bias_slice_idx = []
        self.irreps_bias = self.irreps_out.simplify()
        self.irreps_bias_orders = [
            int(irrep_str[-2]) for irrep_str in str(self.irreps_bias).split("+")
        ]
        self.irreps_bias_parity = [
            irrep_str[-1] for irrep_str in str(self.irreps_bias).split("+")
        ]
        self.irreps_bias_dims = [
            int(irrep_str.split("x")[0])
            for irrep_str in str(self.irreps_bias).split("+")
        ]
        if self.use_bias:
            self.bias = []
            for slice_idx in range(len(self.irreps_bias_orders)):
                if (
                    self.irreps_bias_orders[slice_idx] == 0
                    and self.irreps_bias_parity[slice_idx] == "e"
                ):
                    out_slice = self.irreps_bias.slices()[slice_idx]
                    out_bias = torch.nn.Parameter(
                        torch.zeros(
                            self.irreps_bias_dims[slice_idx], dtype=self.tp.weight.dtype
                        )
                    )
                    self.bias += [out_bias]
                    self.bias_slices += [out_slice]
                    self.bias_slice_idx += [slice_idx]
        self.bias = torch.nn.ParameterList(self.bias)

        self.slices_sqrt_k = {}
        with torch.no_grad():
            # Determine fan_in for each slice, it could be that each output slice is updated via several instructions
            slices_fan_in = {}  # fan_in per slice
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                fan_in = self.calculate_fan_in(instr)
                slices_fan_in[slice_idx] = (
                    slices_fan_in[slice_idx] + fan_in
                    if slice_idx in slices_fan_in.keys()
                    else fan_in
                )
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                if self.rescale:
                    sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                else:
                    sqrt_k = 1.0
                self.slices_sqrt_k[slice_idx] = (
                    self.irreps_out_slices[slice_idx],
                    sqrt_k,
                )

            # Re-initialize weights in each instruction
            if self.tp.internal_weights:
                for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                    # The tensor product in e3nn already normalizes proportional to 1 / sqrt(fan_in), and the weights are by
                    # default initialized with unif(-1,1). However, we want to be consistent with torch.nn.Linear and
                    # initialize the weights with unif(-sqrt(k),sqrt(k)), with k = 1 / fan_in
                    slice_idx = instr[2]
                    if self.rescale:
                        sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                        weight.data.mul_(sqrt_k)
                    # else:
                    #    sqrt_k = 1.
                    #
                    # if self.rescale:
                    # weight.data.uniform_(-sqrt_k, sqrt_k)
                    #    weight.data.mul_(sqrt_k)
                    # self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

            # Initialize the biases
            # for (out_slice_idx, out_slice, out_bias) in zip(self.bias_slice_idx, self.bias_slices, self.bias):
            #    sqrt_k = 1 / slices_fan_in[out_slice_idx] ** 0.5
            #    out_bias.uniform_(-sqrt_k, sqrt_k)

    def forward_tp_rescale_bias(self, x, y, weight=None):

        out = self.tp(x, y, weight)

        # if self.rescale and self.tp.internal_weights:
        #    for (slice, slice_sqrt_k) in self.slices_sqrt_k.values():
        #        out[:, slice] /= slice_sqrt_k
        if self.use_bias:
            for _, slice, bias in zip(self.bias_slice_idx, self.bias_slices, self.bias):
                # out[:, slice] += bias
                out.narrow(1, slice.start, slice.stop - slice.start).add_(bias)
        return out

    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        return out


class FullyConnectedTensorProductRescale(TensorProductRescale):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
    ):

        instructions = [
            (i_1, i_2, i_out, "uvw", True, 1.0)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]
        super().__init__(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions=instructions,
            bias=bias,
            rescale=rescale,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            normalization=normalization,
        )


class LinearRS(FullyConnectedTensorProductRescale):
    def __init__(self, irreps_in, irreps_out, bias=True, rescale=True):
        super().__init__(
            irreps_in,
            o3.Irreps("1x0e"),
            irreps_out,
            bias=bias,
            rescale=rescale,
            internal_weights=True,
            shared_weights=True,
            normalization=None,
        )

    def forward(self, x):
        y = torch.ones_like(x[:, 0:1])
        out = self.forward_tp_rescale_bias(x, y)
        return out


# From "Geometric and Physical Quantities improve E(3) Equivariant Message Passing"
class EquivariantGraphNorm(nn.Module):
    """Instance normalization for orthonormal representations
    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations `wigner_D` are orthonormal.
    Parameters
    ----------
    irreps : `Irreps`
        representation
    eps : float
        avoid division by zero when we normalize by the variance
    affine : bool
        do we have weight and bias parameters
    reduce : {'mean', 'max'}
        method used to reduce
    """

    def __init__(
        self, irreps, eps=1e-5, affine=True, reduce="mean", normalization="component"
    ):
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        self.mean_shift = nn.Parameter(torch.ones(num_scalar))
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert isinstance(reduce, str), "reduce should be passed as a string value"
        assert reduce in ["mean", "max"], "reduce needs to be 'mean' or 'max'"
        self.reduce = reduce

        assert normalization in [
            "norm",
            "component",
        ], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"

    # @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, node_input, batch, **kwargs):
        """evaluate
        Parameters
        ----------
        node_input : `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        """
        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0
        i_mean_shift = 0

        for (
            mul,
            ir,
        ) in (
            self.irreps
        ):  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            # field = node_input[:, ix: ix + mul * d]  # [batch * sample, mul * repr]
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d

            # [batch * sample, mul, repr]
            field = field.reshape(-1, mul, d)

            # For scalars first compute and subtract the mean
            if ir.l == 0 and ir.p == 1:
                # Compute the mean
                field_mean = global_mean_pool(field, batch).reshape(
                    -1, mul, 1
                )  # [batch, mul, 1]]
                # Subtract the mean
                mean_shift = self.mean_shift[i_mean_shift : (i_mean_shift + mul)]
                mean_shift = mean_shift.reshape(1, mul, 1)
                field = field - field_mean[batch] * mean_shift

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == "component":
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError(
                    "Invalid normalization option {}".format(self.normalization)
                )
            # Reduction method
            if self.reduce == "mean":
                field_norm = global_mean_pool(field_norm, batch)  # [batch, mul]
            elif self.reduce == "max":
                field_norm = global_max_pool(field_norm, batch)  # [batch, mul]
            else:
                raise ValueError("Invalid reduce option {}".format(self.reduce))

            # Then apply the rescaling (divide by the sqrt of the squared_norm, i.e., divide by the norm
            field_norm = (field_norm + self.eps).pow(-0.5)  # [batch, mul]

            if self.affine:
                weight = self.affine_weight[None, iw : iw + mul]  # [batch, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch, mul]

            field = field * field_norm[batch].reshape(
                -1, mul, 1
            )  # [batch * sample, mul, repr]

            if self.affine and d == 1 and ir.p == 1:  # scalars
                bias = self.affine_bias[ib : ib + mul]  # [batch, mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch * sample, mul, repr]

            # Save the result, to be stacked later with the rest
            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        if ix != dim:
            fmt = (
                "`ix` should have reached node_input.size(-1) ({}), but it ended at {}"
            )
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output


# From "Geometric and Physical Quantities improve E(3) Equivariant Message Passing"
class EquivariantInstanceNorm(nn.Module):
    """Instance normalization for orthonormal representations
    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations `wigner_D` are orthonormal.
    Parameters
    ----------
    irreps : `Irreps`
        representation
    eps : float
        avoid division by zero when we normalize by the variance
    affine : bool
        do we have weight and bias parameters
    reduce : {'mean', 'max'}
        method used to reduce
    """

    def __init__(
        self, irreps, eps=1e-5, affine=True, reduce="mean", normalization="component"
    ):
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert isinstance(reduce, str), "reduce should be passed as a string value"
        assert reduce in ["mean", "max"], "reduce needs to be 'mean' or 'max'"
        self.reduce = reduce

        assert normalization in [
            "norm",
            "component",
        ], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"

    # @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, node_input, batch, **kwargs):
        """evaluate
        Parameters
        ----------
        node_input : `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        """
        # batch, *size, dim = node_input.shape  # TODO: deal with batch
        # node_input = node_input.reshape(batch, -1, dim)  # [batch, sample, stacked features]
        # node_input has shape [batch * nodes, dim], but with variable nr of nodes.
        # the node_input batch slices this into separate graphs
        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for (
            mul,
            ir,
        ) in (
            self.irreps
        ):  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            # field = node_input[:, ix: ix + mul * d]  # [batch * sample, mul * repr]
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d

            # [batch * sample, mul, repr]
            field = field.reshape(-1, mul, d)

            # For scalars first compute and subtract the mean
            if ir.l == 0 and ir.p == 1:
                # Compute the mean
                field_mean = global_mean_pool(field, batch).reshape(
                    -1, mul, 1
                )  # [batch, mul, 1]]
                # Subtract the mean
                field = field - field_mean[batch]

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == "component":
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError(
                    "Invalid normalization option {}".format(self.normalization)
                )
            # Reduction method
            if self.reduce == "mean":
                field_norm = global_mean_pool(field_norm, batch)  # [batch, mul]
            elif self.reduce == "max":
                field_norm = global_max_pool(field_norm, batch)  # [batch, mul]
            else:
                raise ValueError("Invalid reduce option {}".format(self.reduce))

            # Then apply the rescaling (divide by the sqrt of the squared_norm, i.e., divide by the norm
            field_norm = (field_norm + self.eps).pow(-0.5)  # [batch, mul]

            if self.affine:
                weight = self.affine_weight[None, iw : iw + mul]  # [batch, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch, mul]

            field = field * field_norm[batch].reshape(
                -1, mul, 1
            )  # [batch * sample, mul, repr]

            if self.affine and d == 1 and ir.p == 1:  # scalars
                bias = self.affine_bias[ib : ib + mul]  # [batch, mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch * sample, mul, repr]

            # Save the result, to be stacked later with the rest
            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        if ix != dim:
            fmt = (
                "`ix` should have reached node_input.size(-1) ({}), but it ended at {}"
            )
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output


class EquivariantLayerNormV2(nn.Module):

    def __init__(self, irreps, eps=1e-5, affine=True, normalization="component"):
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in [
            "norm",
            "component",
        ], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps}, eps={self.eps})"

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, node_input, **kwargs):
        # batch, *size, dim = node_input.shape  # TODO: deal with batch
        # node_input = node_input.reshape(batch, -1, dim)  # [batch, sample, stacked features]
        # node_input has shape [batch * nodes, dim], but with variable nr of nodes.
        # the node_input batch slices this into separate graphs
        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for (
            mul,
            ir,
        ) in (
            self.irreps
        ):  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            # field = node_input[:, ix: ix + mul * d]  # [batch * sample, mul * repr]
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d

            # [batch * sample, mul, repr]
            field = field.reshape(-1, mul, d)

            # For scalars first compute and subtract the mean
            if ir.l == 0 and ir.p == 1:
                # Compute the mean
                field_mean = torch.mean(field, dim=1, keepdim=True)  # [batch, mul, 1]]
                # Subtract the mean
                field = field - field_mean

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == "component":
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError(
                    "Invalid normalization option {}".format(self.normalization)
                )
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)

            # Then apply the rescaling (divide by the sqrt of the squared_norm, i.e., divide by the norm
            field_norm = (field_norm + self.eps).pow(-0.5)  # [batch, mul]

            if self.affine:
                weight = self.affine_weight[None, iw : iw + mul]  # [batch, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch, mul]

            field = field * field_norm.reshape(
                -1, mul, 1
            )  # [batch * sample, mul, repr]

            if self.affine and d == 1 and ir.p == 1:  # scalars
                bias = self.affine_bias[ib : ib + mul]  # [batch, mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch * sample, mul, repr]

            # Save the result, to be stacked later with the rest
            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        if ix != dim:
            fmt = (
                "`ix` should have reached node_input.size(-1) ({}), but it ended at {}"
            )
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output


class EquivariantLayerNormFast(nn.Module):

    def __init__(self, irreps, eps=1e-5, affine=True, normalization="component"):
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in [
            "norm",
            "component",
        ], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"

    def forward(self, node_input, **kwargs):
        """
        Use torch layer norm for scalar features.
        """

        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for (
            mul,
            ir,
        ) in (
            self.irreps
        ):  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d

            if ir.l == 0 and ir.p == 1:
                weight = self.affine_weight[iw : (iw + mul)]
                bias = self.affine_bias[ib : (ib + mul)]
                iw += mul
                ib += mul
                field = F.layer_norm(field, tuple((mul,)), weight, bias, self.eps)
                fields.append(
                    field.reshape(-1, mul * d)
                )  # [batch * sample, mul * repr]
                continue

            # For non-scalar features, use RMS value for std
            field = field.reshape(-1, mul, d)  # [batch * sample, mul, repr]

            if self.normalization == "norm":
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == "component":
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError(
                    "Invalid normalization option {}".format(self.normalization)
                )
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)
            field_norm = 1.0 / ((field_norm + self.eps).sqrt())  # [batch * sample, mul]

            if self.affine:
                weight = self.affine_weight[None, iw : (iw + mul)]  # [1, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch * sample, mul]
            field = field * field_norm.reshape(
                -1, mul, 1
            )  # [batch * sample, mul, repr]

            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        assert ix == dim

        output = torch.cat(fields, dim=-1)
        return output


def get_norm_layer(norm_type):
    if norm_type == "graph":
        return EquivariantGraphNorm
    elif norm_type == "instance":
        return EquivariantInstanceNorm
    elif norm_type == "layer":
        return EquivariantLayerNormV2
    elif norm_type == "fast_layer":
        return EquivariantLayerNormFast
    elif norm_type is None:
        return None
    else:
        raise ValueError("Norm type {} not supported.".format(norm_type))


def get_mul_0(irreps):
    mul_0 = 0
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            mul_0 += mul
    return mul_0


def sort_irreps_even_first(irreps):
    Ret = collections.namedtuple("sort", ["irreps", "p", "inv"])
    out = [(ir.l, -ir.p, i, mul) for i, (mul, ir) in enumerate(irreps)]
    out = sorted(out)
    inv = tuple(i for _, _, i, _ in out)
    p = perm.inverse(inv)
    irreps = o3.Irreps([(mul, (l, -p)) for l, p, _, mul in out])
    return Ret(irreps, p, inv)


class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope

    def forward(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2

    def extra_repr(self):
        return "negative_slope={}".format(self.alpha)


@compile_mode("trace")
class Activation(torch.nn.Module):
    """
    Directly apply activation when irreps is type-0.
    """

    def __init__(self, irreps_in, acts):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)
        assert len(irreps_in) == len(acts), (irreps_in, acts)

        # normalize the second moment
        acts = [normalize2mom(act) if act is not None else None for act in acts]

        from e3nn.util._argtools import _get_device

        irreps_out = []
        for (mul, (l_in, p_in)), act in zip(irreps_in, acts):
            if act is not None:
                if l_in != 0:
                    raise ValueError(
                        "Activation: cannot apply an activation function to a non-scalar input."
                    )

                x = torch.linspace(0, 10, 256, device=_get_device(act))

                a1, a2 = act(x), act(-x)
                if (a1 - a2).abs().max() < 1e-5:
                    p_act = 1
                elif (a1 + a2).abs().max() < 1e-5:
                    p_act = -1
                else:
                    p_act = 0

                p_out = p_act if p_in == -1 else p_in
                irreps_out.append((mul, (0, p_out)))

                if p_out == 0:
                    raise ValueError(
                        "Activation: the parity is violated! The input scalar is odd but the activation is neither even nor odd."
                    )
            else:
                irreps_out.append((mul, (l_in, p_in)))

        self.irreps_in = irreps_in
        self.irreps_out = o3.Irreps(irreps_out)
        self.acts = torch.nn.ModuleList(acts)
        assert len(self.irreps_in) == len(self.acts)

    # def __repr__(self):
    #    acts = "".join(["x" if a is not None else " " for a in self.acts])
    #    return f"{self.__class__.__name__} [{self.acts}] ({self.irreps_in} -> {self.irreps_out})"
    def extra_repr(self):
        output_str = super(Activation, self).extra_repr()
        output_str = output_str + "{} -> {}, ".format(self.irreps_in, self.irreps_out)
        return output_str

    def forward(self, features, dim=-1):
        # directly apply activation without narrow
        if len(self.acts) == 1:
            return self.acts[0](features)

        output = []
        index = 0
        for (mul, ir), act in zip(self.irreps_in, self.acts):
            if act is not None:
                output.append(act(features.narrow(dim, index, mul)))
            else:
                output.append(features.narrow(dim, index, mul * ir.dim))
            index += mul * ir.dim

        if len(output) > 1:
            return torch.cat(output, dim=dim)
        elif len(output) == 1:
            return output[0]
        else:
            return torch.zeros_like(features)


def DepthwiseTensorProduct(
    irreps_node_input,
    irreps_edge_attr,
    irreps_node_output,
    internal_weights=False,
    bias=True,
):
    """
    The irreps of output is pre-determined.
    `irreps_node_output` is used to get certain types of vectors.
    """
    irreps_output = []
    instructions = []

    for i, (mul, ir_in) in enumerate(irreps_node_input):
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_node_output or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_output)
                    irreps_output.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", True))

    irreps_output = o3.Irreps(irreps_output)
    irreps_output, p, _ = sort_irreps_even_first(irreps_output)  # irreps_output.sort()
    instructions = [
        (i_1, i_2, p[i_out], mode, train)
        for i_1, i_2, i_out, mode, train in instructions
    ]
    tp = TensorProductRescale(
        irreps_node_input,
        irreps_edge_attr,
        irreps_output,
        instructions,
        internal_weights=internal_weights,
        shared_weights=internal_weights,
        bias=bias,
        rescale=_RESCALE,
    )
    return tp


class RadialProfile(nn.Module):
    def __init__(self, ch_list, use_layer_norm=True, use_offset=True):
        super().__init__()
        modules = []
        input_channels = ch_list[0]
        for i in range(len(ch_list)):
            if i == 0:
                continue
            if (i == len(ch_list) - 1) and use_offset:
                use_biases = False
            else:
                use_biases = True
            modules.append(nn.Linear(input_channels, ch_list[i], bias=use_biases))
            input_channels = ch_list[i]

            if i == len(ch_list) - 1:
                break

            if use_layer_norm:
                modules.append(nn.LayerNorm(ch_list[i]))
            # modules.append(nn.ReLU())
            # modules.append(Activation(o3.Irreps('{}x0e'.format(ch_list[i])),
            #    acts=[torch.nn.functional.silu]))
            # modules.append(Activation(o3.Irreps('{}x0e'.format(ch_list[i])),
            #    acts=[ShiftedSoftplus()]))
            modules.append(torch.nn.SiLU())

        self.net = nn.Sequential(*modules)

        self.offset = None
        if use_offset:
            self.offset = nn.Parameter(torch.zeros(ch_list[-1]))
            fan_in = ch_list[-2]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.offset, -bound, bound)

    def forward(self, f_in):
        f_out = self.net(f_in)
        if self.offset is not None:
            f_out = f_out + self.offset.reshape(1, -1)
        return f_out


def irreps2gate(irreps):
    irreps_scalars = []
    irreps_gated = []
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            irreps_scalars.append((mul, ir))
        else:
            irreps_gated.append((mul, ir))
    irreps_scalars = o3.Irreps(irreps_scalars).simplify()
    irreps_gated = o3.Irreps(irreps_gated).simplify()
    if irreps_gated.dim > 0:
        ir = "0e"
    else:
        ir = None
    irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()
    return irreps_scalars, irreps_gates, irreps_gated


@compile_mode("script")
class Gate(torch.nn.Module):
    """
    1. Use `narrow` to split tensor.
    2. Use `Activation` in this file.
    """

    def __init__(
        self, irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated
    ):
        super().__init__()
        irreps_scalars = o3.Irreps(irreps_scalars)
        irreps_gates = o3.Irreps(irreps_gates)
        irreps_gated = o3.Irreps(irreps_gated)

        if len(irreps_gates) > 0 and irreps_gates.lmax > 0:
            raise ValueError(
                f"Gate scalars must be scalars, instead got irreps_gates = {irreps_gates}"
            )
        if len(irreps_scalars) > 0 and irreps_scalars.lmax > 0:
            raise ValueError(
                f"Scalars must be scalars, instead got irreps_scalars = {irreps_scalars}"
            )
        if irreps_gates.num_irreps != irreps_gated.num_irreps:
            raise ValueError(
                f"There are {irreps_gated.num_irreps} irreps in irreps_gated, but a different number ({irreps_gates.num_irreps}) of gate scalars in irreps_gates"
            )
        # assert len(irreps_scalars) == 1
        # assert len(irreps_gates) == 1

        self.irreps_scalars = irreps_scalars
        self.irreps_gates = irreps_gates
        self.irreps_gated = irreps_gated
        self._irreps_in = (irreps_scalars + irreps_gates + irreps_gated).simplify()

        self.act_scalars = Activation(irreps_scalars, act_scalars)
        irreps_scalars = self.act_scalars.irreps_out

        self.act_gates = Activation(irreps_gates, act_gates)
        irreps_gates = self.act_gates.irreps_out

        self.mul = o3.ElementwiseTensorProduct(irreps_gated, irreps_gates)
        irreps_gated = self.mul.irreps_out

        self._irreps_out = irreps_scalars + irreps_gated

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps_in} -> {self.irreps_out})"

    def forward(self, features):
        scalars_dim = self.irreps_scalars.dim
        gates_dim = self.irreps_gates.dim
        input_dim = self.irreps_in.dim
        scalars = features.narrow(-1, 0, scalars_dim)
        gates = features.narrow(-1, scalars_dim, gates_dim)
        gated = features.narrow(
            -1, (scalars_dim + gates_dim), (input_dim - scalars_dim - gates_dim)
        )

        scalars = self.act_scalars(scalars)
        if gates.shape[-1]:
            gates = self.act_gates(gates)
            gated = self.mul(gated, gates)
            features = torch.cat([scalars, gated], dim=-1)
        else:
            features = scalars
        return features

    @property
    def irreps_in(self):
        """Input representations."""
        return self._irreps_in

    @property
    def irreps_out(self):
        """Output representations."""
        return self._irreps_out


class SeparableFCTP(torch.nn.Module):
    """
    Use separable FCTP for spatial convolution.
    """

    def __init__(
        self,
        irreps_node_input,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        use_activation=False,
        norm_layer="graph",
        internal_weights=False,
    ):

        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        norm = get_norm_layer(norm_layer)

        self.dtp = DepthwiseTensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            self.irreps_node_output,
            bias=False,
            internal_weights=internal_weights,
        )

        self.dtp_rad = None
        if fc_neurons is not None:
            self.dtp_rad = RadialProfile(fc_neurons + [self.dtp.tp.weight_numel])
            for slice, slice_sqrt_k in self.dtp.slices_sqrt_k.values():
                self.dtp_rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
                self.dtp_rad.offset.data[slice] *= slice_sqrt_k

        irreps_lin_output = self.irreps_node_output
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(
            self.irreps_node_output
        )
        if use_activation:
            irreps_lin_output = irreps_scalars + irreps_gates + irreps_gated
            irreps_lin_output = irreps_lin_output.simplify()
        self.lin = LinearRS(self.dtp.irreps_out.simplify(), irreps_lin_output)

        self.norm = None
        if norm_layer is not None:
            self.norm = norm(self.lin.irreps_out)

        self.gate = None
        if use_activation:
            if irreps_gated.num_irreps == 0:
                gate = Activation(self.irreps_node_output, acts=[torch.nn.SiLU()])
            else:
                gate = Gate(
                    irreps_scalars,
                    [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                    irreps_gates,
                    [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                    irreps_gated,  # gated tensors
                )
            self.gate = gate

    def forward(self, node_input, edge_attr, edge_scalars, batch=None, **kwargs):
        """
        Depthwise TP: `node_input` TP `edge_attr`, with TP parametrized by
        self.dtp_rad(`edge_scalars`).
        """
        weight = None
        if self.dtp_rad is not None and edge_scalars is not None:
            weight = self.dtp_rad(edge_scalars)
        out = self.dtp(node_input, edge_attr, weight)
        out = self.lin(out)
        if self.norm is not None:
            out = self.norm(out, batch=batch)
        if self.gate is not None:
            out = self.gate(out)
        return out


@compile_mode("script")
class Vec2AttnHeads(torch.nn.Module):
    """
    Reshape vectors of shape [N, irreps_mid] to vectors of shape
    [N, num_heads, irreps_head].
    """

    def __init__(self, irreps_head, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = irreps_head
        self.irreps_mid_in = []
        for mul, ir in irreps_head:
            self.irreps_mid_in.append((mul * num_heads, ir))
        self.irreps_mid_in = o3.Irreps(self.irreps_mid_in)
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        N, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.mid_in_indices):
            temp = x.narrow(1, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, self.num_heads, -1)
            out.append(temp)
        out = torch.cat(out, dim=2)
        return out

    def __repr__(self):
        return "{}(irreps_head={}, num_heads={})".format(
            self.__class__.__name__, self.irreps_head, self.num_heads
        )


@compile_mode("script")
class AttnHeads2Vec(torch.nn.Module):
    """
    Convert vectors of shape [N, num_heads, irreps_head] into
    vectors of shape [N, irreps_head * num_heads].
    """

    def __init__(self, irreps_head):
        super().__init__()
        self.irreps_head = irreps_head
        self.head_indices = []
        start_idx = 0
        for mul, ir in self.irreps_head:
            self.head_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        N, _, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.head_indices):
            temp = x.narrow(2, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, -1)
            out.append(temp)
        out = torch.cat(out, dim=1)
        return out

    def __repr__(self):
        return "{}(irreps_head={})".format(self.__class__.__name__, self.irreps_head)


class EquivariantDropout(nn.Module):
    def __init__(self, irreps, drop_prob):
        super(EquivariantDropout, self).__init__()
        self.irreps = irreps
        self.num_irreps = irreps.num_irreps
        self.drop_prob = drop_prob
        self.drop = torch.nn.Dropout(drop_prob, True)
        self.mul = o3.ElementwiseTensorProduct(
            irreps, o3.Irreps("{}x0e".format(self.num_irreps))
        )

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        shape = (x.shape[0], self.num_irreps)
        mask = torch.ones(shape, dtype=x.dtype, device=x.device)
        mask = self.drop(mask)
        out = self.mul(x, mask)
        return out


@compile_mode("script")
class GraphAttention(torch.nn.Module):
    """
    1. Message = Alpha * Value
    2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
    3. 0e -> Activation -> Inner Product -> (Alpha)
    4. (0e+1e+...) -> (Value)
    """
    
    # Class-level type annotations for TorchScript compatibility
    irreps_node_input: o3.Irreps
    irreps_node_attr: o3.Irreps
    irreps_edge_attr: o3.Irreps
    irreps_node_output: o3.Irreps
    irreps_pre_attn: o3.Irreps
    irreps_head: o3.Irreps

    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        irreps_head,
        num_heads,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        alpha_drop=0.1,
        proj_drop=0.1,
    ):

        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = (
            self.irreps_node_input
            if irreps_pre_attn is None
            else o3.Irreps(irreps_pre_attn)
        )
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message

        # Merge src and dst
        self.merge_src = LinearRS(
            self.irreps_node_input, self.irreps_pre_attn, bias=True
        )
        self.merge_dst = LinearRS(
            self.irreps_node_input, self.irreps_pre_attn, bias=False
        )

        irreps_attn_heads = irreps_head * num_heads
        irreps_attn_heads, _, _ = sort_irreps_even_first(
            irreps_attn_heads
        )  # irreps_attn_heads.sort()
        irreps_attn_heads = irreps_attn_heads.simplify()
        mul_alpha = get_mul_0(irreps_attn_heads)
        mul_alpha_head = mul_alpha // num_heads
        irreps_alpha = o3.Irreps("{}x0e".format(mul_alpha))  # for attention score
        irreps_attn_all = (irreps_alpha + irreps_attn_heads).simplify()

        self.sep_act = None
        if self.nonlinear_message:
            # Use an extra separable FCTP and Swish Gate for value
            self.sep_act = SeparableFCTP(
                self.irreps_pre_attn,
                self.irreps_edge_attr,
                self.irreps_pre_attn,
                fc_neurons,
                use_activation=True,
                norm_layer=None,
                internal_weights=False,
            )
            self.sep_alpha = LinearRS(self.sep_act.dtp.irreps_out, irreps_alpha)
            self.sep_value = SeparableFCTP(
                self.irreps_pre_attn,
                self.irreps_edge_attr,
                irreps_attn_heads,
                fc_neurons=None,
                use_activation=False,
                norm_layer=None,
                internal_weights=True,
            )
            self.vec2heads_alpha = Vec2AttnHeads(
                o3.Irreps("{}x0e".format(mul_alpha_head)), num_heads
            )
            self.vec2heads_value = Vec2AttnHeads(self.irreps_head, num_heads)
        else:
            self.sep = SeparableFCTP(
                self.irreps_pre_attn,
                self.irreps_edge_attr,
                irreps_attn_all,
                fc_neurons,
                use_activation=False,
                norm_layer=None,
            )
            self.vec2heads = Vec2AttnHeads(
                (o3.Irreps("{}x0e".format(mul_alpha_head)) + irreps_head).simplify(),
                num_heads,
            )

        self.alpha_act = Activation(
            o3.Irreps("{}x0e".format(mul_alpha_head)), [SmoothLeakyReLU(0.2)]
        )
        self.heads2vec = AttnHeads2Vec(irreps_head)

        self.mul_alpha_head = mul_alpha_head
        self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        torch_geometric.nn.inits.glorot(self.alpha_dot)  # Following GATv2

        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        self.proj = LinearRS(irreps_attn_heads, self.irreps_node_output)
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(
                self.irreps_node_input, drop_prob=proj_drop
            )

    def forward(
        self,
        node_input,
        node_attr,
        edge_src,
        edge_dst,
        edge_attr,
        edge_scalars,
        batch,
        **kwargs,
    ):

        message_src = self.merge_src(node_input)
        message_dst = self.merge_dst(node_input)
        message = message_src[edge_src] + message_dst[edge_dst]

        if self.nonlinear_message:
            weight = self.sep_act.dtp_rad(edge_scalars)
            message = self.sep_act.dtp(message, edge_attr, weight)
            alpha = self.sep_alpha(message)
            alpha = self.vec2heads_alpha(alpha)
            value = self.sep_act.lin(message)
            value = self.sep_act.gate(value)
            value = self.sep_value(
                value, edge_attr=edge_attr, edge_scalars=edge_scalars
            )
            value = self.vec2heads_value(value)
        else:
            message = self.sep(message, edge_attr=edge_attr, edge_scalars=edge_scalars)
            message = self.vec2heads(message)
            head_dim_size = message.shape[-1]
            alpha = message.narrow(2, 0, self.mul_alpha_head)
            value = message.narrow(
                2, self.mul_alpha_head, (head_dim_size - self.mul_alpha_head)
            )

        # inner product
        alpha = self.alpha_act(alpha)
        alpha = torch.einsum("bik, aik -> bi", alpha, self.alpha_dot)
        alpha = torch_geometric.utils.softmax(alpha, edge_dst)
        alpha = alpha.unsqueeze(-1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)
        attn = value * alpha
        attn = scatter(attn, index=edge_dst, dim=0, dim_size=node_input.shape[0])
        attn = self.heads2vec(attn)

        if self.rescale_degree:
            degree = torch_geometric.utils.degree(
                edge_dst, num_nodes=node_input.shape[0], dtype=node_input.dtype
            )
            degree = degree.view(-1, 1)
            attn = attn * degree

        node_output = self.proj(attn)

        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)

        return node_output

    def extra_repr(self):
        output_str = super(GraphAttention, self).extra_repr()
        output_str = output_str + "rescale_degree={}, ".format(self.rescale_degree)
        return output_str


class FullyConnectedTensorProductRescaleSwishGate(FullyConnectedTensorProductRescale):

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
    ):

        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = Gate(
                irreps_scalars,
                [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
        super().__init__(
            irreps_in1,
            irreps_in2,
            gate.irreps_in,
            bias=bias,
            rescale=rescale,
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            normalization=normalization,
        )
        self.gate = gate

    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.gate(out)
        return out


@compile_mode("script")
class FeedForwardNetwork(torch.nn.Module):
    """
    Use two (FCTP + Gate)
    """
    
    # Class-level type annotations for TorchScript compatibility
    irreps_node_input: o3.Irreps
    irreps_node_attr: o3.Irreps
    irreps_mlp_mid: o3.Irreps
    irreps_node_output: o3.Irreps

    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_node_output,
        irreps_mlp_mid=None,
        proj_drop=0.1,
    ):

        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_mlp_mid = (
            o3.Irreps(irreps_mlp_mid)
            if irreps_mlp_mid is not None
            else self.irreps_node_input
        )
        self.irreps_node_output = o3.Irreps(irreps_node_output)

        self.fctp_1 = FullyConnectedTensorProductRescaleSwishGate(
            self.irreps_node_input,
            self.irreps_node_attr,
            self.irreps_mlp_mid,
            bias=True,
            rescale=_RESCALE,
        )
        self.fctp_2 = FullyConnectedTensorProductRescale(
            self.irreps_mlp_mid,
            self.irreps_node_attr,
            self.irreps_node_output,
            bias=True,
            rescale=_RESCALE,
        )

        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(
                self.irreps_node_output, drop_prob=proj_drop
            )

    def forward(self, node_input, node_attr, **kwargs):
        node_output = self.fctp_1(node_input, node_attr)
        node_output = self.fctp_2(node_output, node_attr)
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        return node_output


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class GraphDropPath(nn.Module):
    """
    Consider batch for graph data when dropping paths.
    """

    def __init__(self, drop_prob=None):
        super(GraphDropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x, batch):
        batch_size = batch.max() + 1
        shape = (batch_size,) + (1,) * (
            x.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        ones = torch.ones(shape, dtype=x.dtype, device=x.device)
        drop = drop_path(ones, self.drop_prob, self.training)
        out = x * drop[batch]
        return out

    def extra_repr(self):
        return "drop_prob={}".format(self.drop_prob)


@compile_mode("script")
class TransBlock(torch.nn.Module):
    """
    1. Layer Norm 1 -> GraphAttention -> Layer Norm 2 -> FeedForwardNetwork
    2. Use pre-norm architecture
    """
    
    # Class-level type annotations for TorchScript compatibility
    irreps_node_input: o3.Irreps
    irreps_node_attr: o3.Irreps
    irreps_edge_attr: o3.Irreps
    irreps_node_output: o3.Irreps
    irreps_pre_attn: o3.Irreps
    irreps_head: o3.Irreps
    irreps_mlp_mid: o3.Irreps

    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        irreps_head,
        num_heads,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        alpha_drop=0.1,
        proj_drop=0.1,
        drop_path_rate=0.0,
        irreps_mlp_mid=None,
        norm_layer="layer",
    ):

        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = (
            self.irreps_node_input
            if irreps_pre_attn is None
            else o3.Irreps(irreps_pre_attn)
        )
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = (
            o3.Irreps(irreps_mlp_mid)
            if irreps_mlp_mid is not None
            else self.irreps_node_input
        )

        self.norm_1 = get_norm_layer(norm_layer)(self.irreps_node_input)
        self.ga = GraphAttention(
            irreps_node_input=self.irreps_node_input,
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            irreps_node_output=self.irreps_node_input,
            fc_neurons=fc_neurons,
            irreps_head=self.irreps_head,
            num_heads=self.num_heads,
            irreps_pre_attn=self.irreps_pre_attn,
            rescale_degree=self.rescale_degree,
            nonlinear_message=self.nonlinear_message,
            alpha_drop=alpha_drop,
            proj_drop=proj_drop,
        )

        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        self.norm_2 = get_norm_layer(norm_layer)(self.irreps_node_input)
        # self.concat_norm_output = ConcatIrrepsTensor(self.irreps_node_input,
        #    self.irreps_node_input)
        self.ffn = FeedForwardNetwork(
            irreps_node_input=self.irreps_node_input,  # self.concat_norm_output.irreps_out,
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_node_output,
            irreps_mlp_mid=self.irreps_mlp_mid,
            proj_drop=proj_drop,
        )
        self.ffn_shortcut = None
        if self.irreps_node_input != self.irreps_node_output:
            self.ffn_shortcut = FullyConnectedTensorProductRescale(
                self.irreps_node_input,
                self.irreps_node_attr,
                self.irreps_node_output,
                bias=True,
                rescale=_RESCALE,
            )

    def forward(
        self,
        node_input,
        node_attr,
        edge_src,
        edge_dst,
        edge_attr,
        edge_scalars,
        batch,
        **kwargs,
    ):

        node_output = node_input
        node_features = node_input
        node_features = self.norm_1(node_features, batch=batch)
        # norm_1_output = node_features
        node_features = self.ga(
            node_input=node_features,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_attr,
            edge_scalars=edge_scalars,
            batch=batch,
        )

        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output = node_output + node_features

        node_features = node_output
        node_features = self.norm_2(node_features, batch=batch)
        # node_features = self.concat_norm_output(norm_1_output, node_features)
        node_features = self.ffn(node_features, node_attr)
        if self.ffn_shortcut is not None:
            node_output = self.ffn_shortcut(node_output, node_attr)

        if self.drop_path is not None:
            node_features = self.drop_path(node_features, batch)
        node_output = node_output + node_features

        return node_output

