import torch

# import pytorch_lightning as pl
# from models import get_model

# from src.QHNet_flow.utils import ExponentialMovingAverage, self.post_processing

# from torch_ema import ExponentialMovingAverage
# from transformers import get_polynomial_decay_schedule_with_warmup
from pl_module.base_module import LitModel
from torch_geometric.data import Batch
from utils import AOData, WDs, WDs_batch
from e3nn import o3

import logging
import time
from tqdm import tqdm
from torch_scatter import scatter_sum
from joblib import Parallel, delayed
import concurrent.futures

from pl_module.flow_module_qh9 import LitModel_flow as LitModel_flow_qh9

logger = logging.getLogger(__name__)


class LitModel_flow(LitModel_flow_qh9):
    def __init__(self, conf):
        super().__init__(conf=conf)

    def criterion(
        self, outputs, target, loss_weights, use_t_scale=False, use_mse_and_mae=False
    ):
        if self.qh9:
            return self._criterion_qh9_FT(outputs, target, loss_weights, use_t_scale)
        else:
            return self._criterion(
                outputs, target, loss_weights, use_t_scale, use_mse_and_mae
            )

    def _criterion_qh9_FT(self, outputs, target, loss_weights, use_t_scale=False):
        error_dict = {}
        keys = loss_weights.keys()

        try:
            for key in keys:
                if key == "hamiltonian":
                    row = target.edge_index[0]
                    edge_batch = target.batch[row]
                    diff_diagonal = (
                        outputs[f"{key}_diagonal_blocks"] - target[f"diagonal_{key}"]
                    )
                    mse_diagonal = torch.sum(
                        diff_diagonal**2 * target[f"diagonal_{key}_mask"], dim=[1, 2]
                    )
                    mae_diagonal = torch.sum(
                        torch.abs(diff_diagonal) * target[f"diagonal_{key}_mask"],
                        dim=[1, 2],
                    )
                    count_sum_diagonal = torch.sum(
                        target[f"diagonal_{key}_mask"], dim=[1, 2]
                    )
                    mse_diagonal = scatter_sum(mse_diagonal, target.batch)
                    mae_diagonal = scatter_sum(mae_diagonal, target.batch)
                    count_sum_diagonal = scatter_sum(count_sum_diagonal, target.batch)

                    diff_non_diagonal = (
                        outputs[f"{key}_non_diagonal_blocks"]
                        - target[f"non_diagonal_{key}"]
                    )
                    mse_non_diagonal = torch.sum(
                        diff_non_diagonal**2 * target[f"non_diagonal_{key}_mask"],
                        dim=[1, 2],
                    )
                    mae_non_diagonal = torch.sum(
                        torch.abs(diff_non_diagonal)
                        * target[f"non_diagonal_{key}_mask"],
                        dim=[1, 2],
                    )
                    count_sum_non_diagonal = torch.sum(
                        target[f"non_diagonal_{key}_mask"], dim=[1, 2]
                    )
                    mse_non_diagonal = scatter_sum(mse_non_diagonal, edge_batch)
                    mae_non_diagonal = scatter_sum(mae_non_diagonal, edge_batch)
                    count_sum_non_diagonal = scatter_sum(
                        count_sum_non_diagonal, edge_batch
                    )

                    mae = (
                        (mae_diagonal + mae_non_diagonal)
                        / (count_sum_diagonal + count_sum_non_diagonal)
                    ).mean()
                    mse = (
                        (mse_diagonal + mse_non_diagonal)
                        / (count_sum_diagonal + count_sum_non_diagonal)
                    ).mean()

                    error_dict[key + "_mae"] = mae
                    error_dict[key + "_rmse"] = torch.sqrt(mse)
                    error_dict[key + "_diagonal_mae"] = (
                        mae_diagonal / count_sum_diagonal
                    ).mean()
                    error_dict[key + "_non_diagonal_mae"] = (
                        mae_non_diagonal / count_sum_non_diagonal
                    ).mean()

                    loss = mae + mse
                    if loss.isnan():
                        logger.error(f"loss is nan for {key}")
                        loss = torch.tensor(0.0).to(loss.device)
                        loss.requires_grad = True
                elif key == "waloss":
                    batch_size = target.ptr.shape[0] - 1
                    out_ham = self.build_final_matrix(
                        target,
                        outputs["hamiltonian_diagonal_blocks"],
                        outputs["hamiltonian_non_diagonal_blocks"],
                        transform=True,
                        out_torch=True,
                    )

                    batch_idx = (
                        torch.arange(0, target.ptr.shape[0] - 1)
                        .to(target.atoms.device)
                        .repeat_interleave(target.ptr[1:] - target.ptr[:-1])
                    )
                    num_orbs = scatter_sum(target.atoms.flatten(), batch_idx, dim=0)
                    num_orbs = (num_orbs / 2).long()

                    target_energy = [
                        torch.tensor(target.ef[i][0]).to(self.device).to(self.dtype)
                        for i in range(batch_size)
                    ]
                    target_orb = [
                        torch.tensor(target.cf[i][0]).to(self.device).to(self.dtype)
                        for i in range(batch_size)
                    ]

                    target_energy = torch.nested.as_nested_tensor(target_energy)
                    target_orb = torch.nested.as_nested_tensor(target_orb)
                    out_ham_nested = torch.nested.as_nested_tensor(out_ham)
                    CHC = torch.bmm(target_orb.transpose(-1, -2), out_ham_nested)
                    CHC = torch.bmm(CHC, target_orb)

                    res = []
                    for i in range(len(num_orbs)):
                        diff = CHC[i].diag() - target_energy[i]
                        diff[num_orbs[i] + 1 :] = diff[num_orbs[i] + 1 :] * 0.01
                        res.append(diff)
                    res = torch.cat(res, dim=0)
                    mse = torch.mean(res**2)
                    mae = torch.mean(torch.abs(res))
                    error_dict[key + "_mae"] = mae
                    error_dict[key + "_rmse"] = torch.sqrt(mse)
                    loss = mse + mae
                elif key == "waloss-delta2":
                    batch_size = target.ptr.shape[0] - 1
                    out_ham = self.build_final_matrix(
                        target,
                        outputs["hamiltonian_diagonal_blocks"],
                        outputs["hamiltonian_non_diagonal_blocks"],
                        transform=True,
                        out_torch=True,
                    )

                    batch_idx = (
                        torch.arange(0, target.ptr.shape[0] - 1)
                        .to(target.atoms.device)
                        .repeat_interleave(target.ptr[1:] - target.ptr[:-1])
                    )
                    num_orbs = scatter_sum(target.atoms.flatten(), batch_idx, dim=0)
                    num_orbs = (num_orbs / 2).long()

                    init_orb = [
                        torch.tensor(target.c0[i][0]).to(self.device).to(self.dtype)
                        for i in range(batch_size)
                    ]
                    target_orb = [
                        torch.tensor(target.cf[i][0]).to(self.device).to(self.dtype)
                        for i in range(batch_size)
                    ]
                    target_ham = [
                        torch.tensor(target.h[i]).to(self.device).to(self.dtype)
                        for i in range(batch_size)
                    ]

                    init_orb = torch.nested.as_nested_tensor(init_orb)
                    target_orb = torch.nested.as_nested_tensor(target_orb)
                    out_ham_nested = torch.nested.as_nested_tensor(out_ham)
                    target_ham_nested = torch.nested.as_nested_tensor(target_ham)
                    ham_diff = out_ham_nested - target_ham_nested
                    CHC0 = torch.bmm(init_orb.transpose(-1, -2), ham_diff)
                    CHC0 = torch.bmm(CHC0, init_orb)
                    CHC1 = torch.bmm(target_orb.transpose(-1, -2), ham_diff)
                    CHC1 = torch.bmm(CHC1, target_orb)

                    res0 = []
                    res1 = []
                    for i in range(len(num_orbs)):
                        diff0 = CHC0[i]
                        diff0 = torch.diag(diff0)
                        diff0[num_orbs[i] + 1 :] = diff0[num_orbs[i] + 1 :] * 0.01
                        res0.append(diff0)

                        diff1 = CHC1[i]
                        diff1 = torch.diag(diff1)
                        diff1[num_orbs[i] + 1 :] = diff1[num_orbs[i] + 1 :] * 0.01
                        res1.append(diff1)

                    res0 = torch.cat(res0, dim=0)
                    mse0 = torch.mean(res0**2)
                    mae0 = torch.mean(torch.abs(res0))
                    error_dict[key + "_mae_init"] = mae0
                    error_dict[key + "_rmse_init"] = torch.sqrt(mse0)

                    res1 = torch.cat(res1, dim=0)
                    mse1 = torch.mean(res1**2)
                    mae1 = torch.mean(torch.abs(res1))
                    error_dict[key + "_mae"] = mae1
                    error_dict[key + "_rmse"] = torch.sqrt(mse1)

                    loss = mse1 + mse0 + mae1 + mae0

                elif key == "waloss-delta-cur":
                    batch_size = target.ptr.shape[0] - 1
                    out_ham = self.build_final_matrix(
                        target,
                        outputs["hamiltonian_diagonal_blocks"],
                        outputs["hamiltonian_non_diagonal_blocks"],
                        transform=True,
                        out_torch=True,
                    )

                    batch_idx = (
                        torch.arange(0, target.ptr.shape[0] - 1)
                        .to(target.atoms.device)
                        .repeat_interleave(target.ptr[1:] - target.ptr[:-1])
                    )
                    num_orbs = scatter_sum(target.atoms.flatten(), batch_idx, dim=0)
                    num_orbs = (num_orbs / 2).long()
                    out_energy, out_orb = [], []

                    overlap = self.build_final_matrix(
                        target,
                        target["diagonal_overlap"],
                        target["non_diagonal_overlap"],
                        transform=True,
                        out_torch=True,
                    )

                    # target_ham2 = self.build_final_matrix(
                    #     target,
                    #     target["diagonal_hamiltonian"],
                    #     target["non_diagonal_hamiltonian"],
                    #     transform=True,
                    #     out_torch=True,
                    # )
                    target_ham = [
                        torch.tensor(target.h[i]).to(self.device).to(self.dtype)
                        for i in range(batch_size)
                    ]
                    target_energy = [
                        torch.tensor(target.ef[i][0]).to(self.device).to(self.dtype)
                        for i in range(batch_size)
                    ]
                    target_orb = [
                        torch.tensor(target.cf[i][0]).to(self.device).to(self.dtype)
                        for i in range(batch_size)
                    ]

                    def _process_index(
                        i, overlap, out_ham, cal_func, target_energy, target_orb
                    ):
                        overlap_i = overlap[i].unsqueeze(0)
                        hams_i = out_ham[i].unsqueeze(0)
                        try:
                            energy_i, orb_i = cal_func(overlap_i, hams_i)
                        except Exception as exc:
                            energy_i = target_energy[i].unsqueeze(0)
                            orb_i = target_orb[i].unsqueeze(0)
                        return energy_i[0].detach(), orb_i[0].detach()

                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=8
                    ) as executor:
                        results = list(
                            executor.map(
                                _process_index,
                                range(batch_size),
                                [overlap] * batch_size,
                                [out_ham] * batch_size,
                                [self.cal_orbital_and_energies] * batch_size,
                                [target_energy] * batch_size,
                                [target_orb] * batch_size,
                            )
                        )

                    for energy_i, orb_i in results:
                        out_energy.append(energy_i)
                        out_orb.append(orb_i)
                    ## _CHC = out_orb[0].T @ out_ham[0] @ out_orb[0]
                    ## Has the error of the scale 1e-6. I think it is ok.

                    out_energy = torch.nested.as_nested_tensor(out_energy)
                    out_orb = torch.nested.as_nested_tensor(out_orb)
                    out_ham_nested = torch.nested.as_nested_tensor(out_ham)
                    target_ham_nested = torch.nested.as_nested_tensor(target_ham)
                    ham_diff = out_ham_nested - target_ham_nested

                    CHC = torch.bmm(out_orb.transpose(-1, -2), ham_diff)
                    CHC = torch.bmm(CHC, out_orb)

                    res = []
                    for i in range(len(num_orbs)):
                        diff = CHC[i]
                        diff = torch.diag(diff)
                        diff[num_orbs[i] + 1 :] = diff[num_orbs[i] + 1 :] * 0.01
                        res.append(diff)

                    res = torch.cat(res, dim=0)
                    mse = torch.mean(res**2)
                    mae = torch.mean(torch.abs(res))
                    error_dict[key + "_mae_init"] = mae
                    error_dict[key + "_rmse_init"] = torch.sqrt(mse)

                    loss = mse + mse

                elif key == "waloss-delta-cur2":
                    batch_size = target.ptr.shape[0] - 1
                    out_ham = self.build_final_matrix(
                        target,
                        outputs["hamiltonian_diagonal_blocks"],
                        outputs["hamiltonian_non_diagonal_blocks"],
                        transform=True,
                        out_torch=True,
                    )

                    batch_idx = (
                        torch.arange(0, target.ptr.shape[0] - 1)
                        .to(target.atoms.device)
                        .repeat_interleave(target.ptr[1:] - target.ptr[:-1])
                    )
                    num_orbs = scatter_sum(target.atoms.flatten(), batch_idx, dim=0)
                    num_orbs = (num_orbs / 2).long()
                    out_energy, out_orb = [], []

                    overlap = self.build_final_matrix(
                        target,
                        target["diagonal_overlap"],
                        target["non_diagonal_overlap"],
                        transform=True,
                        out_torch=True,
                    )

                    # target_ham2 = self.build_final_matrix(
                    #     target,
                    #     target["diagonal_hamiltonian"],
                    #     target["non_diagonal_hamiltonian"],
                    #     transform=True,
                    #     out_torch=True,
                    # )
                    target_ham = [
                        torch.tensor(target.h[i]).to(self.device).to(self.dtype)
                        for i in range(batch_size)
                    ]
                    target_energy = [
                        torch.tensor(target.ef[i][0]).to(self.device).to(self.dtype)
                        for i in range(batch_size)
                    ]
                    target_orb = [
                        torch.tensor(target.cf[i][0]).to(self.device).to(self.dtype)
                        for i in range(batch_size)
                    ]

                    def _process_index(
                        i, overlap, out_ham, cal_func, target_energy, target_orb
                    ):
                        overlap_i = overlap[i].unsqueeze(0)
                        hams_i = out_ham[i].unsqueeze(0)
                        try:
                            energy_i, orb_i = cal_func(overlap_i, hams_i)
                        except Exception as exc:
                            energy_i = target_energy[i].unsqueeze(0)
                            orb_i = target_orb[i].unsqueeze(0)
                        return energy_i[0].detach(), orb_i[0].detach()

                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=8
                    ) as executor:
                        results = list(
                            executor.map(
                                _process_index,
                                range(batch_size),
                                [overlap] * batch_size,
                                [out_ham] * batch_size,
                                [self.cal_orbital_and_energies] * batch_size,
                                [target_energy] * batch_size,
                                [target_orb] * batch_size,
                            )
                        )

                    for energy_i, orb_i in results:
                        out_energy.append(energy_i)
                        out_orb.append(orb_i)
                    ## _CHC = out_orb[0].T @ out_ham[0] @ out_orb[0]
                    ## Has the error of the scale 1e-6. I think it is ok.

                    out_energy = torch.nested.as_nested_tensor(out_energy)
                    out_orb = torch.nested.as_nested_tensor(out_orb)
                    out_ham_nested = torch.nested.as_nested_tensor(out_ham)
                    target_ham_nested = torch.nested.as_nested_tensor(target_ham)
                    # ham_diff = out_ham_nested - target_ham_nested

                    CHC = torch.bmm(out_orb.transpose(-1, -2), out_ham_nested)
                    CHC = torch.bmm(CHC, out_orb)

                    res = []
                    for i in range(len(num_orbs)):
                        diff = CHC[i]
                        diff = torch.diag(diff) - target_energy[i]
                        diff[num_orbs[i] + 1 :] = diff[num_orbs[i] + 1 :] * 0.01
                        res.append(diff)

                    res = torch.cat(res, dim=0)
                    mse = torch.mean(res**2)
                    mae = torch.mean(torch.abs(res))
                    error_dict[key + "_mae_init"] = mae
                    error_dict[key + "_rmse_init"] = torch.sqrt(mse)

                    loss = mse + mae

                elif key == "waloss-delta-cur3":
                    batch_size = target.ptr.shape[0] - 1
                    out_ham = self.build_final_matrix(
                        target,
                        outputs["hamiltonian_diagonal_blocks"],
                        outputs["hamiltonian_non_diagonal_blocks"],
                        transform=True,
                        out_torch=True,
                    )

                    batch_idx = (
                        torch.arange(0, target.ptr.shape[0] - 1)
                        .to(target.atoms.device)
                        .repeat_interleave(target.ptr[1:] - target.ptr[:-1])
                    )
                    num_orbs = scatter_sum(target.atoms.flatten(), batch_idx, dim=0)
                    num_orbs = (num_orbs / 2).long()
                    out_energy, out_orb = [], []

                    overlap = self.build_final_matrix(
                        target,
                        target["diagonal_overlap"],
                        target["non_diagonal_overlap"],
                        transform=True,
                        out_torch=True,
                    )

                    # target_ham2 = self.build_final_matrix(
                    #     target,
                    #     target["diagonal_hamiltonian"],
                    #     target["non_diagonal_hamiltonian"],
                    #     transform=True,
                    #     out_torch=True,
                    # )
                    target_ham = [
                        torch.tensor(target.h[i]).to(self.device).to(self.dtype)
                        for i in range(batch_size)
                    ]
                    target_energy = [
                        torch.tensor(target.ef[i][0]).to(self.device).to(self.dtype)
                        for i in range(batch_size)
                    ]
                    target_orb = [
                        torch.tensor(target.cf[i][0]).to(self.device).to(self.dtype)
                        for i in range(batch_size)
                    ]

                    def _process_index(
                        i, overlap, out_ham, cal_func, target_energy, target_orb
                    ):
                        overlap_i = overlap[i].unsqueeze(0)
                        hams_i = out_ham[i].unsqueeze(0)
                        try:
                            energy_i, orb_i = cal_func(overlap_i, hams_i)
                        except Exception as exc:
                            energy_i = target_energy[i].unsqueeze(0)
                            orb_i = target_orb[i].unsqueeze(0)
                        return energy_i[0].detach(), orb_i[0].detach()

                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=8
                    ) as executor:
                        results = list(
                            executor.map(
                                _process_index,
                                range(batch_size),
                                [overlap] * batch_size,
                                [out_ham] * batch_size,
                                [self.cal_orbital_and_energies] * batch_size,
                                [target_energy] * batch_size,
                                [target_orb] * batch_size,
                            )
                        )

                    for energy_i, orb_i in results:
                        out_energy.append(energy_i)
                        out_orb.append(orb_i)
                    ## _CHC = out_orb[0].T @ out_ham[0] @ out_orb[0]
                    ## Has the error of the scale 1e-6. I think it is ok.

                    out_energy = torch.nested.as_nested_tensor(out_energy)
                    out_orb = torch.nested.as_nested_tensor(out_orb)
                    out_ham_nested = torch.nested.as_nested_tensor(out_ham)
                    target_orb = torch.nested.as_nested_tensor(target_orb)
                    target_ham_nested = torch.nested.as_nested_tensor(target_ham)
                    # ham_diff = out_ham_nested - target_ham_nested
                    ham_diff = out_ham_nested

                    CHC = torch.bmm(out_orb.transpose(-1, -2), out_ham_nested)
                    CHC = torch.bmm(CHC, out_orb)
                    CHC1 = torch.bmm(target_orb.transpose(-1, -2), ham_diff)
                    CHC1 = torch.bmm(CHC1, target_orb)

                    res = []
                    res1 = []
                    for i in range(len(num_orbs)):
                        diff = CHC[i]
                        diff = torch.diag(diff) - target_energy[i]
                        diff[num_orbs[i] + 1 :] = diff[num_orbs[i] + 1 :] * 0.01
                        res.append(diff)

                        diff1 = CHC1[i]
                        # diff1 = torch.diag(diff1)
                        diff1 = torch.diag(diff1) - target_energy[i]
                        diff1[num_orbs[i] + 1 :] = diff1[num_orbs[i] + 1 :] * 0.01
                        res1.append(diff1)

                    res = torch.cat(res, dim=0)
                    mse = torch.mean(res**2)
                    mae = torch.mean(torch.abs(res))
                    error_dict[key + "_mae_init"] = mae
                    error_dict[key + "_rmse_init"] = torch.sqrt(mse)

                    res1 = torch.cat(res1, dim=0)
                    mse1 = torch.mean(res1**2)
                    mae1 = torch.mean(torch.abs(res1))
                    error_dict[key + "_mae"] = mae1
                    error_dict[key + "_rmse"] = torch.sqrt(mse1)

                    loss = mse + mae
                    loss = loss + mse1 + mae1

                if use_t_scale:
                    scale = 1 / (1 - torch.min(target.t, torch.tensor(0.9))) ** 2
                    loss = loss * scale

                error_dict[key] = loss
                if "loss" in error_dict.keys():
                    error_dict["loss"] = error_dict["loss"] + loss_weights[key] * loss
                else:
                    error_dict["loss"] = loss_weights[key] * loss
        except Exception as exc:
            raise exc
        return error_dict
