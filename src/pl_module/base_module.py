import torch
import pytorch_lightning as pl
from models import get_model

# from src.QHNet_flow.utils import ExponentialMovingAverage, self.self.post_processing

from torch_ema import ExponentialMovingAverage
from transformers import get_polynomial_decay_schedule_with_warmup
import logging
import time
from tqdm import tqdm
from torch_scatter import scatter_sum
from argparse import Namespace
import numpy as np

logger = logging.getLogger(__name__)

convention_dict = {
    "pyscf_631G": Namespace(
        atom_to_orbitals_map={1: "ss", 6: "ssspp", 7: "ssspp", 8: "ssspp", 9: "ssspp"},
        orbital_idx_map={"s": [0], "p": [2, 0, 1], "d": [0, 1, 2, 3, 4]},
        orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1],
            6: [0, 1, 2, 3, 4],
            7: [0, 1, 2, 3, 4],
            8: [0, 1, 2, 3, 4],
            9: [0, 1, 2, 3, 4],
        },
    ),
    "pyscf_def2svp": Namespace(
        atom_to_orbitals_map={
            1: "ssp",
            6: "sssppd",
            7: "sssppd",
            8: "sssppd",
            9: "sssppd",
        },
        orbital_idx_map={"s": [0], "p": [1, 2, 0], "d": [0, 1, 2, 3, 4]},
        orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2],
            6: [0, 1, 2, 3, 4, 5],
            7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5],
            9: [0, 1, 2, 3, 4, 5],
        },
    ),
    "back2pyscf": Namespace(
        atom_to_orbitals_map={
            1: "ssp",
            6: "sssppd",
            7: "sssppd",
            8: "sssppd",
            9: "sssppd",
        },
        orbital_idx_map={"s": [0], "p": [2, 0, 1], "d": [0, 1, 2, 3, 4]},
        orbital_sign_map={"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1]},
        orbital_order_map={
            1: [0, 1, 2],
            6: [0, 1, 2, 3, 4, 5],
            7: [0, 1, 2, 3, 4, 5],
            8: [0, 1, 2, 3, 4, 5],
            9: [0, 1, 2, 3, 4, 5],
        },
    ),
}


class LitModel(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        # Set up the model on the correct device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.default_type = (
            torch.float64 if conf.data_type == "float64" else torch.float32
        )
        torch.set_default_dtype(self.default_type)

        self.loss_weights = conf.loss_weights
        # self.loss_weights = {"hamiltonian": 1.0}
        self.model = get_model(conf.model)
        self.model.set(device)

        # Optional: set up EMA if enabled
        self.ema = None
        self.use_init_hamiltonian = getattr(conf, "use_init_hamiltonian", False)
        self.use_init_hamiltonian_residue = getattr(
            conf, "use_init_hamiltonian_residue", False
        )
        self.ema_start_epoch = getattr(conf, "ema_start_epoch", -1)
        if self.ema_start_epoch > -1:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.99)

        self.qh9 = getattr(conf, "qh9", False)

        self.save_hyperparameters()
        self._epoch_start_time = None
        self.set(device)
        self.convention_dict = convention_dict
        logger.info(f"use_init_hamiltonian: {self.use_init_hamiltonian}")
        logger.info(
            f"use_init_hamiltonian_residue: {self.use_init_hamiltonian_residue}"
        )
        logger.info(f"ema_start_epoch: {self.ema_start_epoch}")
        logger.info(f"qh9: {self.qh9}")

        self.batch_size = conf.dataset.get("batch_size", 32)
        self.test_batch_size = conf.dataset.get("test_batch_size", 32)

    def configure_optimizers(self):
        torch.set_default_dtype(self.default_type)
        if self.conf.optimizer.lower() == "AdamW".lower():
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=getattr(self.conf, "dataset", {}).get("learning_rate", 5e-4),
                betas=(0.99, 0.999),
                amsgrad=False,
            )
            logger.info(
                f"Optimizer: AdamW with lr: {optimizer.param_groups[0]['lr']}"
            )  #
        else:
            raise NotImplementedError(
                f"Optimizer {self.conf.optimizer} is not implemented."
            )
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=getattr(self.conf, "warmup_step", 1000),
            num_training_steps=getattr(self.conf, "num_training_steps", 200000),
            lr_end=getattr(self.conf, "end_lr", 1e-8),
            power=getattr(self.conf, "scheduler_power", 1.0),
            last_epoch=-1,
        )
        # The scheduler will be updated every training step
        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict]

    def set(self, device):
        self = self.to(device)
        self.orbital_mask = self.get_orbital_mask()
        for key in self.orbital_mask.keys():
            self.orbital_mask[key] = self.orbital_mask[key].to(self.device)

    @staticmethod
    def post_processing(batch, default_type):
        if "hamiltonian" in batch.keys:
            if batch.hamiltonian.dim() == 2:
                batch.hamiltonian = batch.hamiltonian.view(
                    batch.hamiltonian.shape[0] // batch.hamiltonian.shape[1],
                    batch.hamiltonian.shape[1],
                    batch.hamiltonian.shape[1],
                )
        if "overlap" in batch.keys:
            if batch.overlap.dim() == 2:
                batch.overlap = batch.overlap.view(
                    batch.overlap.shape[0] // batch.overlap.shape[1],
                    batch.overlap.shape[1],
                    batch.overlap.shape[1],
                )
        for key in batch.keys:
            if type(batch[key]) == torch.Tensor:
                if torch.is_floating_point(batch[key]):
                    batch[key] = batch[key].type(default_type)
        return batch

    @staticmethod
    def cal_orbital_and_energies(overlap_matrix, full_hamiltonian, EPS=1e-8):
        eigvals, eigvecs = torch.linalg.eigh(overlap_matrix)
        eps = EPS * torch.ones_like(eigvals)
        eigvals = torch.where(eigvals > EPS, eigvals, eps)
        frac_overlap = eigvecs / torch.sqrt(eigvals).unsqueeze(-2)

        Fs = torch.bmm(
            torch.bmm(frac_overlap.transpose(-1, -2), full_hamiltonian), frac_overlap
        )
        orbital_energies, orbital_coefficients = torch.linalg.eigh(Fs)
        orbital_coefficients = torch.bmm(frac_overlap, orbital_coefficients)
        return orbital_energies, orbital_coefficients

    def get_orbital_mask(self):
        idx_1s_2s = torch.tensor([0, 1])
        idx_2p = torch.tensor([3, 4, 5])
        orbital_mask_line1 = torch.cat([idx_1s_2s, idx_2p])
        orbital_mask_line2 = torch.arange(14)
        orbital_mask = {}
        for i in range(1, 11):
            orbital_mask[i] = orbital_mask_line1 if i <= 2 else orbital_mask_line2
        return orbital_mask

    def criterion(self, outputs, target, loss_weights):
        if self.qh9:
            return self._criterion_qh9(outputs, target, loss_weights)
        else:
            return self._criterion(outputs, target, loss_weights)

    @staticmethod
    def _criterion(outputs, target, loss_weights):
        error_dict = {}
        if "waloss" in loss_weights.keys():
            energy, orb = LitModel.cal_orbital_and_energies(
                target.overlap, target.hamiltonian
            )
            target.orbital_energies = torch.diag_embed(energy).to(
                target.hamiltonian.device
            )
            target.orbital_coefficients = orb.to(target.hamiltonian.device)
        if (
            "waloss-delta" in loss_weights.keys()
            or "waloss-delta-2" in loss_weights.keys()
        ):
            energy, orb = LitModel.cal_orbital_and_energies(
                target.overlap, target.hamiltonian
            )
            target.orbital_energies = torch.diag_embed(energy).to(
                target.hamiltonian.device
            )
            target.orbital_coefficients = orb.to(target.hamiltonian.device)

            init_energy, init_orb = LitModel.cal_orbital_and_energies(
                target.overlap, target.init_ham
            )
            target.init_orbital_energies = torch.diag_embed(init_energy).to(
                target.hamiltonian.device
            )
            target.init_orbital_coefficients = init_orb.to(target.hamiltonian.device)

        for key in loss_weights.keys():
            if key == "hamiltonian":
                diff = outputs[key] - target[key]

            elif key == "waloss":
                diff = outputs["hamiltonian"].bmm(target.orbital_coefficients)
                diff = torch.bmm(target.orbital_coefficients.transpose(-1, -2), diff)
                diff = diff - target.orbital_energies

            elif key == "waloss-delta":
                diff1 = outputs["hamiltonian"].bmm(target.orbital_coefficients)
                diff1 = torch.bmm(target.orbital_coefficients.transpose(-1, -2), diff1)
                H0_pred = outputs["hamiltonian"] - (
                    target.hamiltonian - target.init_ham
                )
                diff2 = H0_pred.bmm(target.init_orbital_coefficients)
                diff2 = torch.bmm(
                    target.init_orbital_coefficients.transpose(-1, -2), diff2
                )
                diff = diff1 - diff2
                # temp1 = target.hamiltonian.bmm(target.orbital_coefficients)
                # temp1 = torch.bmm(target.orbital_coefficients.transpose(-1, -2), temp1)
                # temp2 = target.init_ham.bmm(target.init_orbital_coefficients)
                # temp2 = torch.bmm(
                #     target.init_orbital_coefficients.transpose(-1, -2), temp2
                # )
                diff = diff - (target.orbital_energies - target.init_orbital_energies)
            # elif key == "waloss-delta-2":
            #     diff1 = outputs["hamiltonian"].bmm(target.orbital_coefficients)
            #     diff1 = torch.bmm(target.orbital_coefficients.transpose(-1, -2), diff1)
            #     H0_pred = target.hamiltonian - (
            #         outputs["hamiltonian"] - target.init_ham
            #     )
            #     diff2 = H0_pred.bmm(target.init_orbital_coefficients)
            #     diff2 = torch.bmm(
            #         target.init_orbital_coefficients.transpose(-1, -2), diff2
            #     )
            #     diff = diff1 - diff2
            #     diff = torch.diagonal(diff, dim1=-2, dim2=-1, offset=0)
            #     target = torch.diagonal(
            #         target.orbital_energies, dim1=-2, dim2=-1, offset=0
            #     )
            #     diff = diff - target
            elif key == "waloss-delta-2":
                diff1 = outputs["hamiltonian"].bmm(target.orbital_coefficients)
                diff1 = torch.bmm(target.orbital_coefficients.transpose(-1, -2), diff1)
                H0_pred = target.hamiltonian - (
                    outputs["hamiltonian"] - target.init_ham
                )
                diff2 = H0_pred.bmm(target.init_orbital_coefficients)
                diff2 = torch.bmm(
                    target.init_orbital_coefficients.transpose(-1, -2), diff2
                )
                diff = diff1 - diff2
                diff = torch.diagonal(diff, dim1=-2, dim2=-1, offset=0)
                target = torch.diagonal(
                    target.orbital_energies, dim1=-2, dim2=-1, offset=0
                )
                diff = (diff - target) / diff.shape[1]

            mse = torch.mean(diff**2)
            mae = torch.mean(torch.abs(diff))
            error_dict[key + "_mae"] = mae
            error_dict[key + "_rmse"] = torch.sqrt(mse)
            # loss = mse + mae
            loss = mse + mae
            if key == "waloss-delta-2":
                loss = mse
            error_dict[key] = loss
            if "loss" in error_dict:
                error_dict["loss"] += loss_weights[key] * loss
            else:
                error_dict["loss"] = loss_weights[key] * loss

        return error_dict

    @staticmethod
    def _criterion_qh9(outputs, target, loss_weights):
        error_dict = {}
        keys = loss_weights.keys()
        # import pdb

        # pdb.set_trace()
        try:
            for key in keys:
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
                    torch.abs(diff_non_diagonal) * target[f"non_diagonal_{key}_mask"],
                    dim=[1, 2],
                )
                count_sum_non_diagonal = torch.sum(
                    target[f"non_diagonal_{key}_mask"], dim=[1, 2]
                )
                mse_non_diagonal = scatter_sum(mse_non_diagonal, edge_batch)
                mae_non_diagonal = scatter_sum(mae_non_diagonal, edge_batch)
                count_sum_non_diagonal = scatter_sum(count_sum_non_diagonal, edge_batch)

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

                error_dict[key] = loss
                if "loss" in error_dict.keys():
                    error_dict["loss"] = error_dict["loss"] + loss_weights[key] * loss
                else:
                    error_dict["loss"] = loss_weights[key] * loss
        except Exception as exc:
            raise exc
        return error_dict

    def forward(self, batch, H=None):
        keep_blocks = self.qh9
        if self.use_init_hamiltonian:
            output = self.model(batch, batch.init_ham, keep_blocks=keep_blocks)
        else:
            output = self.model(batch, keep_blocks=keep_blocks)
        if self.use_init_hamiltonian_residue:
            if keep_blocks:
                output["hamiltonian_diagonal_blocks"] += batch["diagonal_init_ham"]
                output["hamiltonian_non_diagonal_blocks"] += batch[
                    "non_diagonal_init_ham"
                ]
            else:
                output["hamiltonian"] = output["hamiltonian"] + batch.init_ham
        return output

    def on_train_start(self):
        self._epoch_start_train_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_train_time) / 60.0
        self.log(
            "train/epoch_time_minutes",
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self._epoch_start_train_time = time.time()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update EMA after each training batch if enabled and past the EMA start epoch.
        if self.ema_start_epoch < 0:
            return
        if self.ema is not None and self.current_epoch > self.ema_start_epoch:
            self.ema.update()

    def training_step(self, batch, batch_idx):
        batch = self.post_processing(batch, self.default_type)
        outputs = self(batch)
        errors = self.criterion(outputs, batch, loss_weights=self.loss_weights)

        loss = errors["loss"]
        for key in errors.keys():
            self.log(
                f"train/{key}",
                errors[key],
                on_step=True,
                on_epoch=True,
                prog_bar=True if key == "loss" else False,
                sync_dist=True,
                batch_size=self.batch_size,
            )
        return loss

    def on_validation_start(self):
        self._epoch_start_val_time = time.time()

    def on_validation_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_val_time) / 60.0
        self.log(
            "val/epoch_time_minutes",
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self._epoch_start_val_time = time.time()

    def validation_step(self, batch, batch_idx):
        batch = self.post_processing(batch, self.default_type)
        if self.ema is not None:
            with self.ema.average_parameters():
                ema_outputs = self(batch)
                ema_errors = self.criterion(
                    ema_outputs, batch, loss_weights=self.loss_weights
                )
                ema_loss = ema_errors["loss"]
                for key in ema_errors.keys():
                    self.log(
                        f"val_ema/{key}",
                        ema_errors[key],
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True if key == "loss" else False,
                        sync_dist=True,
                        batch_size=self.batch_size,
                    )
                ema_orb_and_eng_error = self._orb_and_eng_error(ema_outputs, batch)
                for key in ema_orb_and_eng_error.keys():
                    self.log(
                        f"val_ema/{key}",
                        ema_orb_and_eng_error[key],
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True if key == "loss" else False,
                        sync_dist=True,
                        batch_size=self.batch_size,
                    )
        outputs = self(batch)
        errors = self.criterion(outputs, batch, loss_weights=self.loss_weights)
        loss = errors["loss"]
        for key in errors.keys():
            self.log(
                f"val/{key}",
                errors[key],
                on_step=True,
                on_epoch=True,
                prog_bar=True if key == "loss" else False,
                sync_dist=True,
                batch_size=self.batch_size,
            )
        orb_and_eng_error = self._orb_and_eng_error(outputs, batch)
        for key in orb_and_eng_error.keys():
            self.log(
                f"val/{key}",
                orb_and_eng_error[key],
                on_step=True,
                on_epoch=True,
                prog_bar=True if key == "loss" else False,
                sync_dist=True,
                batch_size=self.batch_size,
            )
        return errors

    def on_test_start(self):
        self._epoch_start_test_time = time.time()

    def on_test_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_test_time) / 60.0
        self.log(
            "test/epoch_time_minutes",
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self._epoch_start_test_time = time.time()

    def test_step(self, batch, batch_idx):
        batch = self.post_processing(batch, self.default_type)
        outputs = self(batch)
        errors = self.criterion(outputs, batch, loss_weights=self.loss_weights)
        loss = errors["loss"]
        for key in errors.keys():
            self.log(
                f"test/{key}",
                errors[key],
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=self.test_batch_size,
            )
        if self.qh9:
            assert self.test_batch_size == 1
            error_dicts = self.test_criterion_qh9_fixed(outputs, batch)
            for key in error_dicts.keys():
                self.log(
                    f"test_fix/{key}",
                    error_dicts[key],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                    batch_size=1,
                )
        else:
            orb_and_eng_error = self._orb_and_eng_error(outputs, batch)
            for key in orb_and_eng_error.keys():
                self.log(
                    f"test/{key}",
                    orb_and_eng_error[key],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                    batch_size=self.test_batch_size,
                )
        return errors

    def matrix_transform(self, hamiltonian, atoms, convention="pyscf_def2svp"):
        conv = self.convention_dict[convention]
        orbitals = ""
        orbitals_order = []
        for a in atoms:
            offset = len(orbitals_order)
            orbitals += conv.atom_to_orbitals_map[a.item()]
            orbitals_order += [idx + offset for idx in conv.orbital_order_map[a.item()]]

        transform_indices = []
        transform_signs = []
        for orb in orbitals:
            offset = sum(map(len, transform_indices))
            map_idx = conv.orbital_idx_map[orb]
            map_sign = conv.orbital_sign_map[orb]
            transform_indices.append(np.array(map_idx) + offset)
            transform_signs.append(np.array(map_sign))

        transform_indices = [transform_indices[idx] for idx in orbitals_order]
        transform_signs = [transform_signs[idx] for idx in orbitals_order]
        transform_indices = np.concatenate(transform_indices).astype(np.int32)
        transform_signs = np.concatenate(transform_signs)

        hamiltonian_new = hamiltonian[..., transform_indices, :]
        hamiltonian_new = hamiltonian_new[..., :, transform_indices]
        hamiltonian_new = hamiltonian_new * transform_signs[:, None]
        hamiltonian_new = hamiltonian_new * transform_signs[None, :]

        return hamiltonian_new

    def matrix_transform_torch(self, hamiltonian, atoms, convention="pyscf_def2svp"):
        conv = self.convention_dict[convention]
        orbitals = ""
        orbitals_order = []
        for a in atoms:
            offset = len(orbitals_order)
            orbitals += conv.atom_to_orbitals_map[a.item()]
            orbitals_order += [idx + offset for idx in conv.orbital_order_map[a.item()]]

        transform_indices = []
        transform_signs = []
        for orb in orbitals:
            offset = sum(map(len, transform_indices))
            map_idx = conv.orbital_idx_map[orb]
            map_sign = conv.orbital_sign_map[orb]
            transform_indices.append(np.array(map_idx) + offset)
            transform_signs.append(np.array(map_sign))

        transform_indices = [transform_indices[idx] for idx in orbitals_order]
        transform_signs = [transform_signs[idx] for idx in orbitals_order]
        transform_indices = np.concatenate(transform_indices).astype(np.int32)
        transform_signs = torch.tensor(np.concatenate(transform_signs)).to(
            hamiltonian.device
        )

        hamiltonian_new = hamiltonian[..., transform_indices, :]
        hamiltonian_new = hamiltonian_new[..., :, transform_indices]
        hamiltonian_new = hamiltonian_new * (transform_signs[:, None])
        hamiltonian_new = hamiltonian_new * (transform_signs[None, :])

        return hamiltonian_new

    def build_final_matrix(
        self,
        data,
        diagonal_matrix,
        non_diagonal_matrix,
        transform=True,
        out_torch=False,
        convention="back2pyscf",
    ):
        # concate the blocks together and then select once.
        final_matrix = []
        dst, src = data.full_edge_index
        for graph_idx in range(data.ptr.shape[0] - 1):
            matrix_block_col = []
            for src_idx in range(data.ptr[graph_idx], data.ptr[graph_idx + 1]):
                matrix_col = []
                for dst_idx in range(data.ptr[graph_idx], data.ptr[graph_idx + 1]):
                    if src_idx == dst_idx:
                        matrix_col.append(
                            diagonal_matrix[src_idx]
                            .index_select(
                                -2, self.orbital_mask[data.atoms[dst_idx].item()]
                            )
                            .index_select(
                                -1, self.orbital_mask[data.atoms[src_idx].item()]
                            )
                        )
                    else:
                        mask1 = src == src_idx
                        mask2 = dst == dst_idx
                        index = torch.where(mask1 & mask2)[0].item()

                        matrix_col.append(
                            non_diagonal_matrix[index]
                            .index_select(
                                -2, self.orbital_mask[data.atoms[dst_idx].item()]
                            )
                            .index_select(
                                -1, self.orbital_mask[data.atoms[src_idx].item()]
                            )
                        )
                matrix_block_col.append(torch.cat(matrix_col, dim=-2))
            mat_res = torch.cat(matrix_block_col, dim=-1)
            if transform:
                if out_torch:
                    mat_res = self.matrix_transform_torch(
                        mat_res,
                        data.atoms[data.batch == graph_idx].cpu(),
                        convention,
                    )
                else:
                    mat_res = self.matrix_transform(
                        mat_res.cpu(),
                        data.atoms[data.batch == graph_idx].cpu(),
                        convention,
                    )
            final_matrix.append(mat_res.cuda())
        # final_matrix = torch.stack(final_matrix, dim=0)
        return final_matrix

    def _orb_and_eng_error(self, _outputs, _target):
        loss_weights = {
            "hamiltonian": 1.0,
            "orbital_energies": 1.0,
            "orbital_coefficients": 1.0,
        }
        outputs = _outputs
        target = _target.clone()
        if self.qh9:
            try:
                out_ham = self.build_final_matrix(
                    target,
                    outputs["hamiltonian_diagonal_blocks"],
                    outputs["hamiltonian_non_diagonal_blocks"],
                    transform=True,
                )
                target_ham = self.build_final_matrix(
                    target,
                    target["diagonal_hamiltonian"],
                    target["non_diagonal_hamiltonian"],
                    transform=True,
                )

                # init_ham = self.build_final_matrix(
                #     _target,
                #     target["diagonal_init_ham"],
                #     target["non_diagonal_init_ham"],
                #     transform=True,
                # )
                # import pdb

                # pdb.set_trace()

                # This overlap is used when overlap_dim is not provided.
                # overlap = self.build_final_matrix(
                #     _target,
                #     target["diagonal_overlap"],
                #     target["non_diagonal_overlap"],
                #     transform=True,
                # )

                outputs_orb_energy, outputs_orb_coeff = [], []
                targets_orb_energy, targets_orb_coeff = [], []
                ovlp_start = 0
                ovlp_fin = 0
                for i in range(len(out_ham)):
                    out_ham[i] = out_ham[i].type(self.default_type)
                    target_ham[i] = target_ham[i].type(self.default_type)
                    overlap_dim = target.overlap_dim[i].item()
                    ovlp_start = ovlp_fin
                    ovlp_fin += overlap_dim**2
                    overlap_cur = (
                        target.overlap[ovlp_start:ovlp_fin]
                        .reshape(overlap_dim, overlap_dim)
                        .unsqueeze(0)
                    ).type(self.default_type)
                    energy, orb = self.cal_orbital_and_energies(
                        overlap_cur, out_ham[i].unsqueeze(0)
                    )
                    outputs_orb_energy.append(energy)
                    outputs_orb_coeff.append(orb)

                    target_energy, target_orb = self.cal_orbital_and_energies(
                        overlap_cur, target_ham[i].unsqueeze(0)
                    )
                    targets_orb_energy.append(target_energy)
                    targets_orb_coeff.append(target_orb)

                    # pred_HOMO = outputs['orbital_energies'][:, num_orb-1]
                    # gt_HOMO = batch.orbital_energies[:, num_orb-1]
                    # pred_LUMO = outputs['orbital_energies'][:, num_orb]
                    # gt_LUMO = batch.orbital_energies[:, num_orb]
                error_dict = self.criterion_test_qh9(
                    outputs,
                    out_ham,
                    outputs_orb_energy,
                    outputs_orb_coeff,
                    target,
                    target_ham,
                    targets_orb_energy,
                    targets_orb_coeff,
                )
            except Exception as exc:
                logger.error(f"Error in _orb_and_eng_error: {exc}")
                return {}

        else:

            for key in outputs.keys():
                if isinstance(outputs[key], torch.Tensor):
                    outputs[key] = outputs[key].to("cpu")

            target = target.to("cpu")

            out_ham = outputs["hamiltonian"]
            target_ham = target["hamiltonian"]

            outputs["orbital_energies"], outputs["orbital_coefficients"] = (
                self.cal_orbital_and_energies(target["overlap"], out_ham)
            )

            target.orbital_energies, target.orbital_coefficients = (
                self.cal_orbital_and_energies(target["overlap"], target_ham)
            )

            num_orb = int(target.atoms[target.ptr[0] : target.ptr[1]].sum() / 2)
            (
                outputs["orbital_energies"],
                outputs["orbital_coefficients"],
                target.orbital_energies,
                target.orbital_coefficients,
            ) = (
                outputs["orbital_energies"][:, :num_orb],
                outputs["orbital_coefficients"][:, :, :num_orb],
                target.orbital_energies[:, :num_orb],
                target.orbital_coefficients[:, :, :num_orb],
            )
            error_dict = self._criterion_test(outputs, target, loss_weights)

        return {
            "orbital_energies": error_dict["orbital_energies"],
            "orbital_coefficients": error_dict["orbital_coefficients"],
            "sample_hamiltonian": error_dict["hamiltonian"],
        }

    def _orb_and_eng_error_md17(self, _outputs, _target):
        loss_weights = {
            "hamiltonian": 1.0,
            "orbital_energies": 1.0,
            "orbital_coefficients": 1.0,
            "HOMO": 1.0,
            "LUMO": 1.0,
            "GAP": 1.0,
        }
        outputs = _outputs
        target = _target.clone()
        for key in outputs.keys():
            if isinstance(outputs[key], torch.Tensor):
                outputs[key] = outputs[key].to("cpu")

        target = target.to("cpu")

        out_ham = outputs["hamiltonian"]
        target_ham = target["hamiltonian"]

        outputs["orbital_energies"], outputs["orbital_coefficients"] = (
            self.cal_orbital_and_energies(target["overlap"], out_ham)
        )

        target.orbital_energies, target.orbital_coefficients = (
            self.cal_orbital_and_energies(target["overlap"], target_ham)
        )

        num_orb = int(target.atoms[target.ptr[0] : target.ptr[1]].sum() / 2)
        pred_HOMO = outputs["orbital_energies"][:, num_orb - 1]
        gt_HOMO = target.orbital_energies[:, num_orb - 1]
        pred_LUMO = outputs["orbital_energies"][:, num_orb]
        gt_LUMO = target.orbital_energies[:, num_orb]
        outputs["HOMO"], outputs["LUMO"], outputs["GAP"] = (
            pred_HOMO,
            pred_LUMO,
            pred_LUMO - pred_HOMO,
        )
        target.HOMO, target.LUMO, target.GAP = gt_HOMO, gt_LUMO, gt_LUMO - gt_HOMO

        (
            outputs["orbital_energies"],
            outputs["orbital_coefficients"],
            target.orbital_energies,
            target.orbital_coefficients,
        ) = (
            outputs["orbital_energies"][:, :num_orb],
            outputs["orbital_coefficients"][:, :, :num_orb],
            target.orbital_energies[:, :num_orb],
            target.orbital_coefficients[:, :, :num_orb],
        )
        error_dict = self._criterion_test(outputs, target, loss_weights)

        return error_dict

    def _orb_and_eng_error_md17_helper(self, _outputs, _target):
        loss_weights = {
            "hamiltonian": 1.0,
            "orbital_energies": 1.0,
            "orbital_coefficients": 1.0,
            "HOMO": 1.0,
            "LUMO": 1.0,
            "GAP": 1.0,
        }
        outputs = _outputs
        target = _target.clone()
        for key in outputs.keys():
            if isinstance(outputs[key], torch.Tensor):
                outputs[key] = outputs[key].to("cpu")

        target = target.to("cpu")

        out_ham = outputs["hamiltonian"]
        target_ham = target["hamiltonian"]

        outputs["orbital_energies"], outputs["orbital_coefficients"] = (
            self.cal_orbital_and_energies(target["overlap"], out_ham)
        )

        target.orbital_energies, target.orbital_coefficients = (
            self.cal_orbital_and_energies(target["overlap"], target_ham)
        )

        num_orb = int(target.atoms[target.ptr[0] : target.ptr[1]].sum() / 2)
        pred_HOMO = outputs["orbital_energies"][:, num_orb - 1]
        gt_HOMO = target.orbital_energies[:, num_orb - 1]
        pred_LUMO = outputs["orbital_energies"][:, num_orb]
        gt_LUMO = target.orbital_energies[:, num_orb]
        outputs["HOMO"], outputs["LUMO"], outputs["GAP"] = (
            pred_HOMO,
            pred_LUMO,
            pred_LUMO - pred_HOMO,
        )
        target.HOMO, target.LUMO, target.GAP = gt_HOMO, gt_LUMO, gt_LUMO - gt_HOMO

        (
            outputs["orbital_energies"],
            outputs["orbital_coefficients"],
            target.orbital_energies,
            target.orbital_coefficients,
        ) = (
            outputs["orbital_energies"][:, :num_orb],
            outputs["orbital_coefficients"][:, :, :num_orb],
            target.orbital_energies[:, :num_orb],
            target.orbital_coefficients[:, :, :num_orb],
        )
        error_dict = self._criterion_test(outputs, target, loss_weights)

        return error_dict

    def test_criterion_qh9_fixed(self, _outputs, _target):
        loss_weights = {
            "hamiltonian": 1.0,
            "diagonal_hamiltonian": 1.0,
            "non_diagonal_hamiltonian": 1.0,
            "orbital_energies": 1.0,
            "orbital_coefficients": 1.0,
            "HOMO": 1.0,
            "LUMO": 1.0,
            "GAP": 1.0,
        }
        ## batch size should be 1
        outputs = _outputs
        batch = _target.clone()
        outputs["hamiltonian"] = self.model.build_final_matrix(
            batch,
            outputs["hamiltonian_diagonal_blocks"],
            outputs["hamiltonian_non_diagonal_blocks"],
        ).cpu()
        batch.hamiltonian = self.model.build_final_matrix(
            batch, batch[0].diagonal_hamiltonian, batch[0].non_diagonal_hamiltonian
        ).cpu()
        outputs["hamiltonian"] = outputs["hamiltonian"].type(torch.float64)
        outputs["hamiltonian"] = self.matrix_transform(
            outputs["hamiltonian"],
            batch.atoms.cpu().squeeze().numpy(),
            convention="back2pyscf",
        )

        batch.hamiltonian = batch.hamiltonian.type(torch.float64)
        batch.hamiltonian = self.matrix_transform(
            batch.hamiltonian,
            batch.atoms.cpu().squeeze().numpy(),
            convention="back2pyscf",
        )
        overlap = self.model.build_final_matrix(
            batch, batch[0].diagonal_overlap, batch[0].non_diagonal_overlap
        ).cpu()

        overlap = overlap.type(torch.float64)
        overlap = self.matrix_transform(
            overlap, batch.atoms.cpu().squeeze().numpy(), convention="back2pyscf"
        )

        outputs["orbital_energies"], outputs["orbital_coefficients"] = (
            self.cal_orbital_and_energies(overlap, outputs["hamiltonian"])
        )
        batch.orbital_energies, batch.orbital_coefficients = (
            self.cal_orbital_and_energies(overlap, batch["hamiltonian"])
        )

        num_orb = int(batch.atoms[batch.ptr[0] : batch.ptr[1]].sum() / 2)
        pred_HOMO = outputs["orbital_energies"][:, num_orb - 1]
        gt_HOMO = batch.orbital_energies[:, num_orb - 1]
        pred_LUMO = outputs["orbital_energies"][:, num_orb]
        gt_LUMO = batch.orbital_energies[:, num_orb]
        outputs["HOMO"], outputs["LUMO"], outputs["GAP"] = (
            pred_HOMO,
            pred_LUMO,
            pred_LUMO - pred_HOMO,
        )
        batch.HOMO, batch.LUMO, batch.GAP = gt_HOMO, gt_LUMO, gt_LUMO - gt_HOMO

        (
            outputs["orbital_energies"],
            outputs["orbital_coefficients"],
            batch.orbital_energies,
            batch.orbital_coefficients,
        ) = (
            outputs["orbital_energies"][:, :num_orb],
            outputs["orbital_coefficients"][:, :, :num_orb],
            batch.orbital_energies[:, :num_orb],
            batch.orbital_coefficients[:, :, :num_orb],
        )

        outputs["diagonal_hamiltonian"], outputs["non_diagonal_hamiltonian"] = (
            outputs["hamiltonian_diagonal_blocks"],
            outputs["hamiltonian_non_diagonal_blocks"],
        )
        error_dict = self._criterion_test(outputs, batch, loss_weights)

        return error_dict

    @staticmethod
    def criterion_test_qh9_old(outputs, target, names):
        error_dict = {}
        for key in names:
            if key == "orbital_coefficients":
                error_dict[key] = (
                    torch.cosine_similarity(outputs[key], target[key], dim=1)
                    .abs()
                    .mean()
                )
            else:
                diff = outputs[key] - target[key]
                mae = torch.mean(torch.abs(diff))
                error_dict[key] = mae
        return error_dict

    @staticmethod
    def criterion_test_qh9(
        outputs,
        outputs_ham,
        outputs_energy,
        outputs_coeff,
        target,
        target_ham,
        target_energy,
        target_coeff,
    ):
        # error_dict = {}
        orb_coeff_error = 0
        orb_energy_error = 0
        ham_error = 0
        ham_error_2 = 0
        diag_ham_error = 0
        non_diag_ham_error = 0

        for i in range(len(outputs_ham)):
            ham_error += torch.mean(torch.abs(outputs_ham[i] - target_ham[i]))
            orb_energy_error += torch.mean(
                torch.abs(outputs_energy[i] - target_energy[i])
            )
            orb_coeff_error += (
                torch.cosine_similarity(outputs_coeff[i], target_coeff[i], dim=1)
                .abs()
                .mean()
            )

        row = target.edge_index[0]
        edge_batch = target.batch[row]
        diff_diagonal = (
            outputs[f"hamiltonian_diagonal_blocks"] - target[f"diagonal_hamiltonian"]
        )
        mse_diagonal = torch.sum(
            diff_diagonal**2 * target[f"diagonal_hamiltonian_mask"], dim=[1, 2]
        )
        mae_diagonal = torch.sum(
            torch.abs(diff_diagonal) * target[f"diagonal_hamiltonian_mask"],
            dim=[1, 2],
        )
        count_sum_diagonal = torch.sum(target[f"diagonal_hamiltonian_mask"], dim=[1, 2])
        mse_diagonal = scatter_sum(mse_diagonal, target.batch)
        mae_diagonal = scatter_sum(mae_diagonal, target.batch)
        count_sum_diagonal = scatter_sum(count_sum_diagonal, target.batch)

        diff_non_diagonal = (
            outputs[f"hamiltonian_non_diagonal_blocks"]
            - target[f"non_diagonal_hamiltonian"]
        )
        mse_non_diagonal = torch.sum(
            diff_non_diagonal**2 * target[f"non_diagonal_hamiltonian_mask"],
            dim=[1, 2],
        )
        mae_non_diagonal = torch.sum(
            torch.abs(diff_non_diagonal) * target[f"non_diagonal_hamiltonian_mask"],
            dim=[1, 2],
        )
        count_sum_non_diagonal = torch.sum(
            target[f"non_diagonal_hamiltonian_mask"], dim=[1, 2]
        )
        mse_non_diagonal = scatter_sum(mse_non_diagonal, edge_batch)
        mae_non_diagonal = scatter_sum(mae_non_diagonal, edge_batch)
        count_sum_non_diagonal = scatter_sum(count_sum_non_diagonal, edge_batch)

        mae = (
            (mae_diagonal + mae_non_diagonal)
            / (count_sum_diagonal + count_sum_non_diagonal)
        ).mean()

        ham_error_2 = mae
        diag_ham_error = (mae_diagonal / count_sum_diagonal).mean()
        non_diag_ham_error = (mae_non_diagonal / count_sum_non_diagonal).mean()

        ham_error /= len(outputs_ham)
        orb_energy_error /= len(outputs_ham)
        orb_coeff_error /= len(outputs_ham)
        return {
            "orbital_energies": orb_energy_error,
            "orbital_coefficients": orb_coeff_error,
            "hamiltonian": ham_error,
            "hamiltonian_2": ham_error_2,
            "diag_ham": diag_ham_error,
            "non_diag_ham": non_diag_ham_error,
        }

    @torch.no_grad()
    def test_over_dataset(self, test_data_loader, default_type):
        self.eval()
        total_error_dict = {"total_items": 0}
        loss_weights = {
            "hamiltonian": 1.0,
            "orbital_energies": 1.0,
            "orbital_coefficients": 1.0,
        }
        total_time = 0
        total_graph = 0
        # total_traj = []
        last_traj = []
        logger.info("num of test data: {}".format(len(test_data_loader)))
        for idx, batch in tqdm(enumerate(test_data_loader)):
            batch = self.post_processing(batch, default_type)
            batch = batch.to(self.model.device)
            tic = time.time()
            # ham = batch.hamiltonian.cpu()
            # outputs, traj, _ = self(batch)
            outputs = self(batch, batch.init_ham)

            last_traj.append(outputs["hamiltonian"])

            duration = time.time() - tic
            total_graph = total_graph + batch.ptr.shape[0] - 1
            total_time = duration + total_time
            for key in outputs.keys():
                if isinstance(outputs[key], torch.Tensor):
                    outputs[key] = outputs[key].to("cpu")

            batch = batch.to("cpu")
            outputs["orbital_energies"], outputs["orbital_coefficients"] = (
                self.cal_orbital_and_energies(batch["overlap"], outputs["hamiltonian"])
            )
            batch.orbital_energies, batch.orbital_coefficients = (
                self.cal_orbital_and_energies(batch["overlap"], batch["hamiltonian"])
            )
            num_orb = int(batch.atoms[batch.ptr[0] : batch.ptr[1]].sum() / 2)
            (
                outputs["orbital_energies"],
                outputs["orbital_coefficients"],
                batch.orbital_energies,
                batch.orbital_coefficients,
            ) = (
                outputs["orbital_energies"][:, :num_orb],
                outputs["orbital_coefficients"][:, :, :num_orb],
                batch.orbital_energies[:, :num_orb],
                batch.orbital_coefficients[:, :, :num_orb],
            )
            error_dict = self._criterion_test(outputs, batch, loss_weights)
            secs = duration / batch.hamiltonian.shape[0]
            msg = f"batch {idx} / {secs*100:.2f}(10^-2)s : "
            for key in error_dict.keys():
                if key == "hamiltonian" or key == "orbital_energies":
                    msg += f"{key}: {error_dict[key]*1e6:.3f}(10^-6), "
                elif key == "orbital_coefficients":
                    msg += f"{key}: {error_dict[key]*1e2:.4f}(10^-2)"
                else:
                    msg += f"{key}: {error_dict[key]:.8f}, "

                if key in total_error_dict.keys():
                    total_error_dict[key] += (
                        error_dict[key].item() * batch.hamiltonian.shape[0]
                    )
                else:
                    total_error_dict[key] = (
                        error_dict[key].item() * batch.hamiltonian.shape[0]
                    )
            logger.info(msg)
            total_error_dict["total_items"] += batch.hamiltonian.shape[0]
        for key in total_error_dict.keys():
            if key != "total_items":
                total_error_dict[key] = (
                    total_error_dict[key] / total_error_dict["total_items"]
                )
        last_traj = torch.cat(last_traj, dim=0)
        return total_error_dict, last_traj

    @staticmethod
    def _criterion_test(outputs, target, names):
        error_dict = {}
        for key in names:
            if key == "orbital_coefficients":
                "The shape if [batch, total_orb, num_occ_orb]."
                error_dict[key] = (
                    torch.cosine_similarity(outputs[key], target[key], dim=1)
                    .abs()
                    .mean()
                )
            elif key in ["diagonal_hamiltonian", "non_diagonal_hamiltonian"]:
                diff_blocks = outputs[key] - target[key]
                mae_blocks = torch.sum(
                    torch.abs(diff_blocks) * target[f"{key}_mask"], dim=[1, 2]
                )
                count_sum_blocks = torch.sum(target[f"{key}_mask"], dim=[1, 2])
                if key == "non_diagonal_hamiltonian":
                    row = target.edge_index_full[0]
                    batch = target.batch[row]
                else:
                    batch = target.batch
                mae_blocks = scatter_sum(mae_blocks, batch)
                count_sum_blocks = scatter_sum(count_sum_blocks, batch)
                error_dict[key + "_mae"] = (mae_blocks / count_sum_blocks).mean()
            else:
                diff = outputs[key] - target[key]
                mae = torch.mean(torch.abs(diff))
                error_dict[key] = mae
        return error_dict

    @torch.no_grad()
    def test_over_dataset_qh9(self, test_data_loader, default_type):
        self.eval()
        total_error_dict = {"total_items": 0}
        loss_weights = {
            "hamiltonian": 1.0,
            "diagonal_hamiltonian": 1.0,
            "non_diagonal_hamiltonian": 1.0,
            "orbital_energies": 1.0,
            "orbital_coefficients": 1.0,
            "HOMO": 1.0,
            "LUMO": 1.0,
            "GAP": 1.0,
        }
        total_time = 0
        total_graph = 0
        # total_traj = []
        last_traj = []
        logger.info("num of test data: {}".format(len(test_data_loader)))
        for idx, batch in tqdm(enumerate(test_data_loader)):
            batch = self.post_processing(batch, default_type)
            batch = batch.to(self.model.device)
            tic = time.time()
            # ham = batch.hamiltonian.cpu()
            # outputs, traj, _ = self(batch)
            outputs = self(batch)
            outputs["hamiltonian"] = self.model.build_final_matrix(
                batch,
                outputs["hamiltonian_diagonal_blocks"],
                outputs["hamiltonian_non_diagonal_blocks"],
            ).cpu()
            batch.hamiltonian = self.model.build_final_matrix(
                batch, batch[0].diagonal_hamiltonian, batch[0].non_diagonal_hamiltonian
            ).cpu()
            outputs["hamiltonian"] = outputs["hamiltonian"].type(torch.float64)
            outputs["hamiltonian"] = self.matrix_transform(
                outputs["hamiltonian"],
                batch.atoms.cpu().squeeze().numpy(),
                convention="back2pyscf",
            )

            last_traj.append(outputs["hamiltonian"])

            batch.hamiltonian = batch.hamiltonian.type(torch.float64)
            batch.hamiltonian = self.matrix_transform(
                batch.hamiltonian,
                batch.atoms.cpu().squeeze().numpy(),
                convention="back2pyscf",
            )
            overlap = self.model.build_final_matrix(
                batch, batch[0].diagonal_overlap, batch[0].non_diagonal_overlap
            ).cpu()

            overlap = overlap.type(torch.float64)
            overlap = self.matrix_transform(
                overlap, batch.atoms.cpu().squeeze().numpy(), convention="back2pyscf"
            )

            outputs["orbital_energies"], outputs["orbital_coefficients"] = (
                self.cal_orbital_and_energies(overlap, outputs["hamiltonian"])
            )
            batch.orbital_energies, batch.orbital_coefficients = (
                self.cal_orbital_and_energies(overlap, batch["hamiltonian"])
            )

            num_orb = int(batch.atoms[batch.ptr[0] : batch.ptr[1]].sum() / 2)
            pred_HOMO = outputs["orbital_energies"][:, num_orb - 1]
            gt_HOMO = batch.orbital_energies[:, num_orb - 1]
            pred_LUMO = outputs["orbital_energies"][:, num_orb]
            gt_LUMO = batch.orbital_energies[:, num_orb]
            outputs["HOMO"], outputs["LUMO"], outputs["GAP"] = (
                pred_HOMO,
                pred_LUMO,
                pred_LUMO - pred_HOMO,
            )
            batch.HOMO, batch.LUMO, batch.GAP = gt_HOMO, gt_LUMO, gt_LUMO - gt_HOMO

            (
                outputs["orbital_energies"],
                outputs["orbital_coefficients"],
                batch.orbital_energies,
                batch.orbital_coefficients,
            ) = (
                outputs["orbital_energies"][:, :num_orb],
                outputs["orbital_coefficients"][:, :, :num_orb],
                batch.orbital_energies[:, :num_orb],
                batch.orbital_coefficients[:, :, :num_orb],
            )

            outputs["diagonal_hamiltonian"], outputs["non_diagonal_hamiltonian"] = (
                outputs["hamiltonian_diagonal_blocks"],
                outputs["hamiltonian_non_diagonal_blocks"],
            )
            error_dict = self._criterion_test(outputs, batch, loss_weights)

            duration = time.time() - tic
            total_graph = total_graph + batch.ptr.shape[0] - 1
            total_time = duration + total_time
            for key in outputs.keys():
                if isinstance(outputs[key], torch.Tensor):
                    outputs[key] = outputs[key].to("cpu")

            secs = duration / batch.hamiltonian.shape[0]
            msg = f"batch {idx} / {secs*100:.2f}(10^-2)s : "
            for key in error_dict.keys():
                # if key == "hamiltonian" or key == "orbital_energies":
                if key in [
                    "hamiltonian",
                    "orbital_energies",
                    "non_diagonal_hamiltonian_mae",
                    "diagonal_hamiltonian_mae",
                    "HOMO",
                    "LUMO",
                    "GAP",
                ]:
                    msg += f"{key}: {error_dict[key]*1e6:.3f}(10^-6), "
                elif key == "orbital_coefficients":
                    msg += f"{key}: {error_dict[key]*1e2:.4f}(10^-2)"
                else:
                    msg += f"{key}: {error_dict[key]:.8f}, "

                if key in total_error_dict.keys():
                    total_error_dict[key] += (
                        error_dict[key].item() * batch.hamiltonian.shape[0]
                    )
                else:
                    total_error_dict[key] = (
                        error_dict[key].item() * batch.hamiltonian.shape[0]
                    )
            logger.info(msg)
            total_error_dict["total_items"] += batch.hamiltonian.shape[0]
        for key in total_error_dict.keys():
            if key != "total_items":
                total_error_dict[key] = (
                    total_error_dict[key] / total_error_dict["total_items"]
                )
        last_traj = torch.cat(last_traj, dim=0)
        return total_error_dict, last_traj
