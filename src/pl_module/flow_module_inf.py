import torch

# import pytorch_lightning as pl
# from models import get_model

# from src.QHNet_flow.utils import ExponentialMovingAverage, self.post_processing

# from torch_ema import ExponentialMovingAverage
# from transformers import get_polynomial_decay_schedule_with_warmup
from pl_module.base_module import LitModel
from torch_geometric.data import Batch
from utils import AOData
import logging
import time
from tqdm import tqdm
import pyscf
from pyscf import dft
import numpy as np

logger = logging.getLogger(__name__)
BOHR2ANG = 1 / 1.8897259886  # 0.52917721067


class LitModel_flow(LitModel):
    def __init__(self, conf):
        super().__init__(conf=conf)
        self.batch_mul = conf.flow.get("batch_mul", 1)
        self.use_t_scale = conf.flow.get("use_t_scale", True)
        self.num_ode_steps = conf.flow.get("num_ode_steps", 3)
        self.num_ode_steps_val = conf.flow.get("num_ode_steps_val", 3)
        self.num_ode_steps_inf = conf.flow.get(
            "num_ode_steps_inf", self.num_ode_steps_val
        )
        self.init_gauss = conf.flow.get("init_gauss", False)
        self.error_threshold = conf.flow.get("error_threshold", 1e-5)
        self.use_mse_and_mae = conf.flow.get("use_mse_and_mae", False)
        self.init_gauss_center = conf.flow.get("init_gauss_center", False)
        if self.init_gauss_center == "":
            self.init_gauss_center = False
        self.use_res_target = conf.flow.get("use_res_target", True)
        if self.init_gauss is False:
            self.init_gauss_center = False
        self.use_corrupt_mul = conf.flow.get("use_corrupt_mul", False)
        self.sigma = conf.flow.get("sigma", 0.05)
        self.sample_random = conf.flow.get("sample_random", False)
        self.test_mode = "inference"

        self.save_hyperparameters()

    @staticmethod
    def batch_repeat(batch, mul=1, repeat_style="repeat"):
        assert repeat_style in ["append", "repeat"]
        if mul == 1:
            return batch
        batch_list = []
        for idx in range(batch.num_graphs):
            bb = batch.batch
            pos = batch.pos[bb == idx]
            atoms = batch.atoms[bb == idx]
            forces = batch.force[bb == idx]
            energy = batch.energy[idx]
            overlap = batch.overlap[idx].unsqueeze(0)
            hamiltonian = batch.hamiltonian[idx].unsqueeze(0)
            init_ham = batch.init_ham[idx].unsqueeze(0)
            mask_row = batch.mask_row[bb == idx]

            len_orb = batch.hamiltonian.shape[-1]
            AO_index = batch.AO_index[:, idx * len_orb : (idx + 1) * len_orb]
            AO_index[0] -= batch.ptr[idx]
            AO_index[2] -= idx
            Q = batch.Q[idx * len_orb : (idx + 1) * len_orb]

            data = AOData(
                pos=pos,
                atoms=atoms,
                force=forces,
                energy=energy,
                overlap=overlap,
                hamiltonian=hamiltonian,
                init_ham=init_ham,
                AO_index=AO_index,
                Q=Q,
                mask_row=mask_row,
            )
            if repeat_style == "repeat":
                for _ in range(mul):
                    batch_list.append(data.clone())
            else:
                batch_list.append(data.clone())

        if repeat_style == "append":
            new_batch_list = []
            for _ in range(mul):
                new_batch_list += batch_list
            batch_list = new_batch_list

        return Batch.from_data_list(batch_list)

    def _corrupt(self, batch, batch_t):
        batch.t = batch_t

        random_ham = torch.zeros_like(batch.hamiltonian)
        if self.init_gauss:
            random_ham += torch.randn_like(batch.hamiltonian) * self.sigma
        if self.init_gauss_center:
            random_ham += batch.init_ham

        if self.use_res_target:
            target_ham = batch.hamiltonian - batch.init_ham
        else:
            target_ham = batch.hamiltonian

        batch.random_ham = random_ham
        batch.target_ham = target_ham

        batch_t_reshape = batch_t.reshape(-1, 1, 1)
        batch.init_ham_t = (
            random_ham * (1 - batch_t_reshape) + target_ham * batch_t_reshape
        )
        return batch

    def corrupt(self, batch, mul=1):
        batch = self.batch_repeat(batch, mul)
        # batch = Batch.from_data_list(batch_list)
        batch_t = self.sample_t(batch.hamiltonian.shape[0], batch.hamiltonian.device)
        return self._corrupt(batch, batch_t)

    def corrupt_mul(self, batch):
        batch = self.batch_repeat(batch, 2, repeat_style="append")
        # batch = Batch.from_data_list(batch_list)
        batch_t = self.sample_t(batch.hamiltonian.shape[0], batch.hamiltonian.device)
        batch_t[batch_t.shape[0] // 2 :] = torch.zeros_like(
            batch_t[batch_t.shape[0] // 2 :]
        )
        return self._corrupt(batch, batch_t)

    @staticmethod
    def sample_t(num_batch, device, min_t=0.01):
        t = torch.rand(num_batch, device=device)
        return t * (1 - 2 * min_t) + min_t  # [min_t, 1-min_t]

    @staticmethod
    def criterion(
        outputs, target, loss_weights, use_t_scale=False, use_mse_and_mae=False
    ):
        error_dict = {}
        if "waloss" in loss_weights.keys():
            energy, orb = LitModel.cal_orbital_and_energies(
                target.overlap, target.hamiltonian
            )
            target.orbital_energies = torch.diag_embed(energy).to(
                target.hamiltonian.device
            )
            target.orbital_coefficients = orb.to(target.hamiltonian.device)
        if "waloss-delta" in loss_weights.keys():
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
            scale = 1
            if key == "hamiltonian":
                diff = outputs[key] - target[key]
                if use_t_scale:
                    scale = 1 / (1 - torch.min(target.t, torch.tensor(0.9))) ** 2

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
                diff = diff - (target.orbital_energies - target.init_orbital_energies)

            mse = torch.mean(diff**2)
            mae = torch.mean(torch.abs(diff))
            error_dict[key + "_mae"] = mae
            error_dict[key + "_rmse"] = torch.sqrt(mse)
            # loss = mse + mae
            if key == "hamiltonian":
                if use_mse_and_mae:
                    loss = mse + mae
                else:
                    loss = mse
            elif key == "waloss":
                loss = mse
            elif key == "waloss-delta":
                loss = mse + mae

            loss = loss * scale
            loss = torch.mean(loss)
            error_dict[key] = loss
            if "loss" in error_dict:
                error_dict["loss"] += loss_weights[key] * loss
            else:
                error_dict["loss"] = loss_weights[key] * loss

        for key in loss_weights.keys():
            if key == "waloss" or key == "waloss-delta":
                continue
            for _bin in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
                s, e = _bin
                mask = (target.t >= s) & (target.t < e)
                diff = outputs[key][mask] - target[key][mask]
                mse = torch.mean(diff**2)
                mae = torch.mean(torch.abs(diff))
                error_dict[key + f"_mae@{s:.2f}_{e:.2f}"] = mae
                error_dict[key + f"_rmse@{s:.2f}_{e:.2f}"] = torch.sqrt(mse)

        return error_dict

    def forward(self, batch, H):
        output = self.model(batch, H)
        if self.use_init_hamiltonian_residue:
            output["hamiltonian"] = output["hamiltonian"] + batch.init_ham
            # ouput always target the gt_hamiltonian

        return output

    def training_step(self, batch, batch_idx):
        batch = self.post_processing(batch, self.default_type)
        if self.use_corrupt_mul:
            batch = self.corrupt_mul(batch)
        else:
            batch = self.corrupt(batch, mul=self.batch_mul)
        outputs = self(batch, batch.init_ham_t)
        errors = self.criterion(
            outputs,
            batch,
            loss_weights=self.loss_weights,
            use_t_scale=self.use_t_scale,
            use_mse_and_mae=self.use_mse_and_mae,
        )
        loss = errors["loss"]
        self._log_error(errors, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.post_processing(batch, self.default_type)
        batch_one = batch.clone()
        batch = self.corrupt(batch, mul=self.batch_mul)
        with self.ema.average_parameters():
            ema_outputs = self(batch, batch.init_ham_t)
            ema_errors = self.criterion(
                ema_outputs,
                batch,
                loss_weights=self.loss_weights,
                use_t_scale=self.use_t_scale,
                use_mse_and_mae=self.use_mse_and_mae,
            )
            ema_loss = ema_errors["loss"]
            self._log_error(ema_errors, "val_ema")
            if ema_loss < self.error_threshold:
                self._log_sample_error(
                    batch_one, "val", num_timesteps=self.num_ode_steps_val
                )

        outputs = self(batch, batch.init_ham_t)
        errors = self.criterion(
            outputs,
            batch,
            loss_weights=self.loss_weights,
            use_t_scale=self.use_t_scale,
            use_mse_and_mae=self.use_mse_and_mae,
        )
        loss = errors["loss"]
        self._log_error(errors, "val")
        if loss < self.error_threshold:
            self._log_sample_error(batch_one, "val", num_timesteps=1, post_fix="_1")
            self._log_sample_error(
                batch_one,
                "val",
                num_timesteps=self.num_ode_steps_val,
            )
        return errors

    def test_step(self, batch, batch_idx):
        if self.test_mode == "inference":
            batch = self.post_processing(batch, self.default_type)
            batch_one = batch.clone()
            cycle = getattr(batch_one, "cycle", None)
            init_cycle_time = getattr(batch_one, "init_cycle_time", None)
            ham_calc = getattr(batch_one, "ham_calc", None)
            if self.qh9:
                batch_one.full_edge_index = batch_one.edge_index_full

                batch_ham = self.model.build_final_matrix(
                    batch_one,
                    batch_one[0].diagonal_hamiltonian,
                    batch_one[0].non_diagonal_hamiltonian,
                )
                batch_ham = batch_ham.cpu().numpy()
                batch_one.hamiltonian = (
                    torch.tensor(batch_ham).to(self.device).to(self.default_type)
                )
            else:
                batch_ham = batch_one.hamiltonian.cpu().numpy()

            batch_one.hamiltonian_pyscf = self.matrix_transform(
                batch_one.hamiltonian.cpu().numpy(),
                batch_one.atoms.cpu().squeeze().numpy(),
                convention="back2pyscf",
            )
            batch_one = self.process_target_batch(batch_one)

            if cycle is None:
                init_scf_ret = self.num_scf_steps(batch_one, init_dm_style="1e")
                cycle = init_scf_ret["cycle"]
                init_cycle_time = init_scf_ret["total_time"]
                ham_calc = self.matrix_transform(
                    init_scf_ret["fock"],
                    batch.atoms.cpu().squeeze().numpy(),
                    convention="pyscf_def2svp",
                )
                ham_calc = torch.tensor(ham_calc).unsqueeze(0).to(self.device)
                ham_calc_error = (ham_calc - batch_one.hamiltonian).abs().mean()
                e_tot_calc = init_scf_ret["e_tot"]
                if self.qh9:
                    target_energy = None
                    e_tot_calc_error = None
                else:
                    target_energy = batch_one.energy.cpu()
                    e_tot_calc_error = abs(e_tot_calc - target_energy.numpy()).mean()
            # import pdb

            # pdb.set_trace()
            # fmt: off
            self.log(f"infer/cycle", cycle, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"infer/cycle_time", init_cycle_time, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"infer/ham_calc_mae", ham_calc_error, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"infer/e_tot_calc_error", e_tot_calc_error, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"infer/e_tot_calc", e_tot_calc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            if e_tot_calc_error is not None:
                self.log(f"infer/e_tot_calc_error", e_tot_calc_error, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=1)
            self._log_inference_error(batch_one, "infer_1", num_timesteps=1, post_fix="", init_cycle=cycle, init_cycle_time=init_cycle_time, middle_scf=False, ham_error=ham_calc_error, e_tot_error=e_tot_calc_error, e_tot_gt=target_energy)
            self._log_inference_error(batch_one, "infer_2", num_timesteps=2, post_fix="", init_cycle=cycle, init_cycle_time=init_cycle_time, middle_scf=False, ham_error=ham_calc_error, e_tot_error=e_tot_calc_error, e_tot_gt=target_energy)
            self._log_inference_error(batch_one, "infer_3", num_timesteps=3, post_fix="", init_cycle=cycle, init_cycle_time=init_cycle_time, middle_scf=False, ham_error=ham_calc_error, e_tot_error=e_tot_calc_error, e_tot_gt=target_energy)
            # self._log_inference_error(batch_one, "infer_1", num_timesteps=1, post_fix="_scf", init_cycle=cycle, init_cycle_time=init_cycle_time)
            # self._log_inference_error(batch_one, "infer_2", num_timesteps=2, post_fix="_scf", init_cycle=cycle, init_cycle_time=init_cycle_time)
            # self._log_inference_error(batch_one, "infer_3", num_timesteps=3, post_fix="_scf", init_cycle=cycle,init_cycle_time=init_cycle_time)
            # fmt: on
            return None
        elif self.test_mode == "predict":
            batch = self.post_processing(batch, self.default_type)
            batch_one = batch.clone()
            batch = self.corrupt(batch, mul=self.batch_mul)
            outputs = self(batch, batch.init_ham_t)
            errors = self.criterion(
                outputs,
                batch,
                loss_weights=self.loss_weights,
                use_t_scale=self.use_t_scale,
                use_mse_and_mae=self.use_mse_and_mae,
            )
            loss = errors["loss"]
            self._log_error(errors, "pred_test")
            # fmt: off
            self._log_sample_error(batch_one, "pred", num_timesteps=1, post_fix="_1")
            self._log_sample_error(batch_one, "pred", num_timesteps=2, post_fix="_2")
            self._log_sample_error(batch_one, "pred", num_timesteps=self.num_ode_steps_inf)
            # fmt: on
            return errors
        elif self.test_mode == "predict-mul":
            batch = self.post_processing(batch, self.default_type)
            batch_one = batch.clone()
            batch = self.corrupt(batch, mul=self.batch_mul)
            outputs = self(batch, batch.init_ham_t)
            errors = self.criterion(
                outputs,
                batch,
                loss_weights=self.loss_weights,
                use_t_scale=self.use_t_scale,
                use_mse_and_mae=self.use_mse_and_mae,
            )
            loss = errors["loss"]
            self._log_error(errors, "pred-mul_test")
            # fmt: off
            self._log_sample_error_mul(batch_one, "pred-mul", num_timesteps=self.num_ode_steps_inf, mul=5)
            # fmt: on
            return errors


    def _log_inference_error(
        self,
        batch_one,
        prefix,
        num_timesteps=1,
        post_fix="",
        middle_scf=True,
        init_cycle=None,
        init_cycle_time=None,
        ham_error=None,
        e_tot_error=None,
        e_tot_gt=None,
    ):
        output_scf = init_cycle if init_cycle is not None else False
        sample, traj, pred = self.sample_with_scf(
            batch_one,
            num_timesteps=num_timesteps,
            middle_scf=middle_scf,
            output_scf=output_scf,
        )
        if self.qh9:
            results = self.test_criterion_qh9_fixed(sample, batch_one)
        else:
            results = self._orb_and_eng_error_md17(sample, batch_one)
        # results = self._orb_and_eng_error_md17_helper(sample, batch_one)

        if "sample_time_per_batch" in sample.keys():
            results["sample_time_per_batch"] = sample["sample_time_per_batch"]
        if init_cycle is not None:
            scf_ret = build_matrix(
                sample["mol"],
                dm0=sample["dm_last"],
                error_level=ham_error,
                Hamiltonian_gt=batch_one.hamiltonian_pyscf,
                e_tot_error_level=e_tot_error,
                e_tot_gt=e_tot_gt,
                qh9=self.qh9,
            )
            results["scf_cycle"] = scf_ret["cycle"]
            results["scf_cycle_ratio"] = scf_ret["cycle"] / init_cycle
            results["scf_total_time"] = scf_ret["total_time"]
            results["scf_total_time_ratio"] = scf_ret["total_time"] / init_cycle_time

            achieve_error_cycle = scf_ret["achieve_error_cycle"]
            if achieve_error_cycle is None:
                achieve_error_cycle = scf_ret["cycle"]
            results["scf_achieve_cycle"] = achieve_error_cycle
            results["scf_achieve_cycle_ratio"] = achieve_error_cycle / init_cycle

            e_tot_achive_error_cycle = scf_ret["e_tot_achieve_error_cycle"]
            if e_tot_achive_error_cycle is None:
                e_tot_achive_error_cycle = scf_ret["cycle"]
            results["scf_e_tot_achieve_cycle"] = e_tot_achive_error_cycle
            results["scf_e_tot_achieve_cycle_ratio"] = (
                e_tot_achive_error_cycle / init_cycle
            )

        for key in results.keys():
            self.log(
                f"{prefix}/{key}{post_fix}",
                results[key],
                on_step=True,
                on_epoch=True,
                prog_bar=True if key in ["scf_cycle_ratio"] else False,
                sync_dist=True,
            )

    def process_target_batch(self, target):
        target_ham = target["hamiltonian"]

        target.orbital_energies, target.orbital_coefficients = (
            self.cal_orbital_and_energies(target["overlap"], target_ham)
        )
        num_orb = int(target.atoms[target.ptr[0] : target.ptr[1]].sum() / 2)

        gt_HOMO = target.orbital_energies[:, num_orb - 1]
        gt_LUMO = target.orbital_energies[:, num_orb]

        target.HOMO, target.LUMO, target.GAP = (
            gt_HOMO,
            gt_LUMO,
            gt_LUMO - gt_HOMO,
        )
        target.orbital_energies = target.orbital_energies[:, :num_orb]
        target.orbital_coefficients = target.orbital_coefficients[:, :, :num_orb]
        return target

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
        target = _target.clone().cpu()
        for key in outputs.keys():
            if isinstance(outputs[key], torch.Tensor):
                outputs[key] = outputs[key].to("cpu")

        out_ham = outputs["hamiltonian"]

        outputs["orbital_energies"], outputs["orbital_coefficients"] = (
            self.cal_orbital_and_energies(target["overlap"], out_ham)
        )

        num_orb = int(target.atoms[target.ptr[0] : target.ptr[1]].sum() / 2)
        pred_HOMO = outputs["orbital_energies"][:, num_orb - 1]
        pred_LUMO = outputs["orbital_energies"][:, num_orb]
        outputs["HOMO"], outputs["LUMO"], outputs["GAP"] = (
            pred_HOMO,
            pred_LUMO,
            pred_LUMO - pred_HOMO,
        )
        outputs["orbital_energies"] = outputs["orbital_energies"][:, :num_orb]
        outputs["orbital_coefficients"] = outputs["orbital_energies"][:, :num_orb]

        if getattr(_target, "orbital_energies", None) is None:
            target = _target
            target = target.to("cpu")
            target_ham = target.hamiltonian
            target.orbital_energies, target.orbital_coefficients = (
                self.cal_orbital_and_energies(target["overlap"], target_ham)
            )
            gt_HOMO = target.orbital_energies[:, num_orb - 1]
            gt_LUMO = target.orbital_energies[:, num_orb]

            target.HOMO, target.LUMO, target.GAP = gt_HOMO, gt_LUMO, gt_LUMO - gt_HOMO

            target.orbital_energies = target.orbital_energies[:, :num_orb]
            target.orbital_coefficients = target.orbital_coefficients[:, :, :num_orb]

        error_dict = self._criterion_test(outputs, target, loss_weights)

        return error_dict

    def sample(
        self,
        batch,
        num_timesteps=100,
        min_t=0.01,
        sample_random=True,
    ):
        start_time = time.time()
        device = self.model.device
        lin_t = torch.linspace(min_t, 1.0, num_timesteps + 1).to(device)
        cur_t = lin_t[0]
        batch.init_ham_t = torch.zeros_like(batch.init_ham)
        # batch.init_ham_t_res = batch.init_ham_t
        if sample_random:
            batch.init_ham_t += torch.randn_like(batch.init_ham) * self.sigma
            # batch.init_ham_t_res = batch.init_ham_t
        if self.init_gauss_center:
            batch.init_ham_t += batch.init_ham
            # batch.init_ham_t_res = batch.init_ham_t - batch.init_ham

        hamiltonian_traj = [batch.init_ham_t.cpu()]
        predictions = [None]
        for idx, next_t in enumerate(lin_t[1:]):
            batch.t = cur_t.repeat(batch.init_ham.shape[0])
            outputs = self(batch, batch.init_ham_t)
            dt = next_t - cur_t
            assert dt > 0
            # vector_field = outputs["hamiltonian"] / (1 - cur_t)
            # vector_field = (outputs["hamiltonian"] - batch.init_ham_t) / (1 - cur_t)

            if self.use_res_target:
                target_H = outputs["hamiltonian"] - batch.init_ham
                cur_H = batch.init_ham_t
            else:
                target_H = outputs["hamiltonian"]
                cur_H = batch.init_ham_t

            vector_field = (target_H - cur_H) / (1 - cur_t)

            ham_t = cur_H + vector_field * dt.reshape(-1, 1, 1)
            hamiltonian_traj.append(ham_t.cpu())
            predictions.append(outputs["hamiltonian"].cpu())

            # Update the previous timestep and the current Hamiltonian
            cur_t = next_t
            batch.init_ham_t = ham_t
            # batch.init_ham_t_res = ham_t - batch.init_ham
        if self.use_res_target:
            ham_t = ham_t + batch.init_ham

        res_outputs = {
            "hamiltonian": ham_t,
            "sample_time_all": time.time() - start_time,
            "sample_time_per_batch": (time.time() - start_time)
            / batch.init_ham.shape[0],
        }

        return res_outputs, hamiltonian_traj, predictions

    def sample_qh9(
        self,
        batch,
        num_timesteps=100,
        min_t=0.01,
        sample_random=True,
    ):
        device = self.model.device
        lin_t = torch.linspace(min_t, 1.0, num_timesteps + 1).to(device)
        cur_t = lin_t[0]
        if self.qh9:
            batch.init_ham = batch["diagonal_init_ham"]

        batch.init_ham_t = torch.zeros_like(batch.init_ham)
        # batch.init_ham_t_res = batch.init_ham_t
        if sample_random:
            batch.init_ham_t += torch.randn_like(batch.init_ham) * self.sigma
            # batch.init_ham_t_res = batch.init_ham_t
        if self.init_gauss_center:
            batch.init_ham_t += batch.init_ham
            # batch.init_ham_t_res = batch.init_ham_t - batch.init_ham

        hamiltonian_traj = [batch.init_ham_t.cpu()]
        predictions = [None]
        for idx, next_t in enumerate(lin_t[1:]):
            batch.t = cur_t.repeat(batch.init_ham.shape[0])
            outputs = self(batch, batch.init_ham_t)
            dt = next_t - cur_t
            assert dt > 0
            # vector_field = outputs["hamiltonian"] / (1 - cur_t)
            # vector_field = (outputs["hamiltonian"] - batch.init_ham_t) / (1 - cur_t)
            if self.qh9:
                if self.use_res_target:
                    target_H = (
                        outputs["hamiltonian_diagonal_blocks"]
                        - batch["diagonal_init_ham"]
                    )
                    # target_H = outputs["hamiltonian"] - batch.init_ham
                    cur_H = batch.init_ham_t
                else:
                    target_H = outputs["hamiltonian_diagonal_blocks"]
                    cur_H = batch.init_ham_t
            else:
                if self.use_res_target:
                    target_H = outputs["hamiltonian"] - batch.init_ham
                    cur_H = batch.init_ham_t
                else:
                    target_H = outputs["hamiltonian"]
                    cur_H = batch.init_ham_t

            vector_field = (target_H - cur_H) / (1 - cur_t)

            ham_t = cur_H + vector_field * dt.reshape(-1, 1, 1)
            hamiltonian_traj.append(ham_t.cpu())
            if self.qh9:
                predictions.append(
                    {
                        "hamiltonian_diagonal_blocks": outputs[
                            "hamiltonian_diagonal_blocks"
                        ].cpu(),
                        "hamiltonian_non_diagonal_blocks": outputs[
                            "hamiltonian_non_diagonal_blocks"
                        ].cpu(),
                    }
                )
            else:
                predictions.append(outputs["hamiltonian"].cpu())

            # Update the previous timestep and the current Hamiltonian
            cur_t = next_t
            batch.init_ham_t = ham_t
            # batch.init_ham_t_res = ham_t - batch.init_ham
        if self.use_res_target:
            ham_t = ham_t + batch.init_ham

        if self.qh9:
            res_outputs = {
                "hamiltonian_diagonal_blocks": ham_t,
                "hamiltonian_non_diagonal_blocks": outputs[
                    "hamiltonian_non_diagonal_blocks"
                ],
            }
        else:
            res_outputs = {"hamiltonian": ham_t}

        return res_outputs, hamiltonian_traj, predictions

    def calc_dm0_from_ham(self, batch, overlap_pyscf, cur_ham):
        hamiltonian_t_pyscf = self.matrix_transform(
            cur_ham, batch.atoms.cpu().squeeze().numpy(), convention="back2pyscf"
        )
        orbital_energies, orbital_coefficients = self.cal_orbital_and_energies(
            overlap_pyscf, hamiltonian_t_pyscf
        )

        num_orb = int(batch.atoms[batch.ptr[0] : batch.ptr[1]].sum() / 2)
        orbital_coefficients = orbital_coefficients.squeeze()
        dm0 = (
            orbital_coefficients[:, :num_orb].matmul(
                orbital_coefficients[:, :num_orb].T
            )
            * 2
        )
        dm0 = dm0.cpu().numpy()

        return dm0

    def sample_with_scf(
        self,
        batch,
        num_timesteps=100,
        min_t=0.01,
        sample_random=True,
        middle_scf=False,
        output_scf=True,
    ):
        if self.qh9:
            start_time = time.time()
            res = self.sample_qh9(
                batch,
                num_timesteps=num_timesteps,
                min_t=min_t,
                sample_random=sample_random,
            )
            res["sample_time_all"] = time.time() - start_time
            res["sample_time_per_batch"] = time.time() - start_time
            ham_out = self.model.build_final_matrix(
                batch,
                res["hamiltonian_diagonal_blocks"],
                res["hamiltonian_non_diagonal_blocks"],
            )
            res["hamiltonian"] = ham_out

            mol = pyscf.gto.Mole()
            pos = batch.pos.cpu().numpy()
            if not self.qh9:
                pos = pos * BOHR2ANG
            t = [
                [batch.atoms[atom_idx].cpu().item(), pos[atom_idx]]
                for atom_idx in range(batch.num_nodes)
            ]
            mol.build(verbose=0, atom=t, basis="def2svp", unit="ang")
            overlap_pyscf = torch.from_numpy(mol.intor("int1e_ovlp")).unsqueeze(0)
            res["mol"] = mol
            res["dm_last"] = self.calc_dm0_from_ham(batch, overlap_pyscf, ham_out.cpu())

            return res

        else:
            return self._sample_with_scf(
                batch,
                num_timesteps=num_timesteps,
                min_t=min_t,
                sample_random=sample_random,
                middle_scf=middle_scf,
                output_scf=output_scf,
            )

    def _sample_with_scf(
        self,
        batch,
        num_timesteps=100,
        min_t=0.01,
        sample_random=True,
        middle_scf=False,
        output_scf=True,
    ):
        start_time = time.time()
        device = self.model.device
        lin_t = torch.linspace(min_t, 1.0, num_timesteps + 1).to(device)
        cur_t = lin_t[0]
        batch.init_ham_t = torch.zeros_like(batch.init_ham)
        # batch.init_ham_t_res = batch.init_ham_t
        if sample_random:
            batch.init_ham_t += torch.randn_like(batch.init_ham) * self.sigma
            # batch.init_ham_t_res = batch.init_ham_t
        if self.init_gauss_center:
            batch.init_ham_t += batch.init_ham
            # batch.init_ham_t_res = batch.init_ham_t - batch.init_ham

        hamiltonian_traj = [batch.init_ham_t.cpu()]
        predictions = [None]
        predictions_one_step = [None]
        mol = pyscf.gto.Mole()
        pos = batch.pos.cpu().numpy()
        if not self.qh9:
            pos = pos * BOHR2ANG
        t = [
            [batch.atoms[atom_idx].cpu().item(), pos[atom_idx]]
            for atom_idx in range(batch.num_nodes)
        ]
        mol.build(verbose=0, atom=t, basis="def2svp", unit="ang")
        overlap_pyscf = torch.from_numpy(mol.intor("int1e_ovlp")).unsqueeze(0)

        for idx, next_t in enumerate(lin_t[1:]):
            batch.t = cur_t.repeat(batch.init_ham.shape[0])
            outputs = self(batch, batch.init_ham_t)
            dt = next_t - cur_t
            assert dt > 0
            if not middle_scf:
                pred_H = outputs["hamiltonian"]
            else:
                dm0 = self.calc_dm0_from_ham(
                    batch, overlap_pyscf, outputs["hamiltonian"].cpu()
                )
                ret = build_matrix(mol, dm0=dm0, max_cycle=2)
                hamiltonian_t_pyscf_one_step = ret["fock"]
                hamiltonian_t_pyscf_one_step = self.matrix_transform(
                    hamiltonian_t_pyscf_one_step,
                    batch.atoms.cpu().squeeze().numpy(),
                    convention="pyscf_def2svp",
                )
                predictions_one_step.append(hamiltonian_t_pyscf_one_step)
                hamiltonian_t_pyscf_one_step = (
                    torch.tensor(hamiltonian_t_pyscf_one_step)
                    .unsqueeze(0)
                    .to(self.device)
                )
                pred_H = hamiltonian_t_pyscf_one_step

            if self.use_res_target:
                target_H = pred_H - batch.init_ham
                cur_H = batch.init_ham_t
            else:
                target_H = pred_H
                cur_H = batch.init_ham_t

            vector_field = (target_H - cur_H) / (1 - cur_t)

            ham_t = cur_H + vector_field * dt.reshape(-1, 1, 1)
            hamiltonian_traj.append(ham_t.cpu())
            predictions.append(outputs["hamiltonian"].cpu())

            # Update the previous timestep and the current Hamiltonian
            cur_t = next_t
            batch.init_ham_t = ham_t

        if self.use_res_target:
            ham_t = ham_t + batch.init_ham

        res_outputs = {
            "hamiltonian": ham_t,
            "sample_time_all": time.time() - start_time,
            "sample_time_per_batch": (time.time() - start_time)
            / batch.init_ham.shape[0],
        }

        if output_scf:
            res_outputs["mol"] = mol
            res_outputs["dm_last"] = self.calc_dm0_from_ham(
                batch, overlap_pyscf, ham_t.cpu()
            )

        return res_outputs, hamiltonian_traj, predictions

    def _log_error(self, errors, prefix):
        for key in errors.keys():
            if "@" in key:
                _key, _time_bin = key.split("@")[0], key.split("@")[1]
                self.log(
                    f"{prefix}_{_time_bin}/{_key}_{_time_bin}",
                    errors[key],
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )
            else:
                self.log(
                    f"{prefix}/{key}",
                    errors[key],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True if key == "loss" else False,
                    sync_dist=True,
                )

    def _log_sample_error(self, batch_one, prefix, num_timesteps=1, post_fix=""):
        try:
            sample, traj, pred = self.sample(
                batch_one, num_timesteps=num_timesteps
            )
            results = self._orb_and_eng_error(sample, batch_one)
            if "sample_time_per_batch" in sample.keys():
                results["sample_time_per_batch"] = sample["sample_time_per_batch"]
            for key in results.keys():
                self.log(
                    f"{prefix}/{key}{post_fix}",
                    results[key],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True if key == "loss" else False,
                    sync_dist=True,
                )
        except Exception as e:
            logger.error(f"Error in logging sample error: {e}")

    def _log_sample_error_mul(
        self,
        batch_one,
        prefix,
        num_timesteps=1,
        post_fix="",
        save_pred=False,
        log=True,
        mul=3,
    ):
        try:
            errer_dicts_mul = []
            preds = []
            for i in range(mul):
                sample, traj, pred = self.sample(
                    batch_one, num_timesteps=num_timesteps
                )
                results = self._orb_and_eng_error(sample, batch_one)
                errer_dicts_mul.append(results)
            if log:
                for key in results.keys():
                    val_list = []
                    for i in range(mul):
                        val_list.append(errer_dicts_mul[i][key])
                    val_list = torch.stack(val_list)
                    mean = torch.mean(val_list, dim=0)
                    std = torch.std(val_list, dim=0)

                    self.log(
                        f"{prefix}/{key}_mean_{mul}_{post_fix}",
                        mean,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True if key == "loss" else False,
                        sync_dist=True,
                        batch_size=self.test_batch_size,
                    )
                    self.log(
                        f"{prefix}/{key}_std_{mul}_{post_fix}",
                        std,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True if key == "loss" else False,
                        sync_dist=True,
                        batch_size=self.test_batch_size,
                    )
                    for i in range(mul):
                        self.log(
                            f"{prefix}/{key}_{i}_{post_fix}",
                            errer_dicts_mul[i][key],
                            on_epoch=True,
                            prog_bar=True if key == "loss" else False,
                            sync_dist=True,
                            batch_size=self.test_batch_size,
                        )
            if save_pred:
                return traj, pred
        except Exception as e:
            logger.error(f"Error in logging sample error: {e}")


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
        logger.info(f"num test data: {len(test_data_loader)}")
        logger.info(f"num ode steps: {self.num_ode_steps_inf}")
        for idx, batch in tqdm(enumerate(test_data_loader)):
            batch = self.post_processing(batch, default_type)
            batch = batch.to(self.model.device)
            tic = time.time()
            # ham = batch.hamiltonian.cpu()
            outputs, traj, _ = self.sample(
                batch,
                num_timesteps=self.num_ode_steps_inf,
                sample_random=self.sample_random,
            )
            # outputs = self(batch, batch.init_ham)
            last_traj.append(traj[-1])

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
            error_dict = self.criterion_test(outputs, batch, loss_weights)
            secs = duration / batch.hamiltonian.shape[0]
            msg = f"batch {idx} / [{len(test_data_loader)}] / {secs*100:.2f}(10^-2)s : "
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
        logger.info(f"num ode steps: {self.num_ode_steps_inf}")
        return total_error_dict, last_traj

    def num_scf_steps(self, batch, init_dm_style="minao", dm0=None):
        mol = pyscf.gto.Mole()
        pos = batch.pos.cpu().numpy()
        if not self.qh9:
            pos = pos * BOHR2ANG
        t = [
            [batch.atoms[atom_idx].cpu().item(), pos[atom_idx]]
            for atom_idx in range(batch.num_nodes)
        ]
        mol.build(verbose=0, atom=t, basis="def2svp", unit="ang")
        if dm0 is not None:
            dm0 = dm0
        else:
            if init_dm_style == "minao":
                dm0 = pyscf.scf.hf.init_guess_by_minao(mol)
            elif init_dm_style == "1e":
                dm0 = pyscf.scf.hf.init_guess_by_1e(mol)

        ret = build_matrix(mol, dm0=dm0, qh9=self.qh9)

        return ret


# fmt : off
def get_total_cycles(envs):
    setattr(envs["mf"], "total_cycle", envs["cycle"])
    if envs["mf"].gt is not None:
        print(np.mean(np.abs(envs["fock"] - envs["mf"].gt)), envs["mf"].error_level)
        if (
            np.mean(np.abs(envs["fock"] - envs["mf"].gt)) < envs["mf"].error_level
            and envs["mf"].achieve_error_flag is False
        ):
            setattr(envs["mf"], "achieve_error_flag", True)
            setattr(envs["mf"], "achieve_error_cycle", envs["cycle"])
    if envs["mf"].e_tot_gt is not None:
        print(np.abs(envs["e_tot"] - envs["mf"].e_tot_gt), envs["mf"].e_tot_error_level)
        if (
            np.abs(envs["e_tot"] - envs["mf"].e_tot_gt) < envs["mf"].e_tot_error_level
            and envs["mf"].e_tot_achieve_error_flag is False
        ):
            setattr(envs["mf"], "e_tot_achieve_error_flag", True)
            setattr(envs["mf"], "e_tot_achieve_error_cycle", envs["cycle"])
            print(envs["mf"].e_tot_achieve_error_cycle)
    setattr(envs["mf"], "total_cycle", envs["cycle"])
    setattr(envs["mf"], "_dm_last", envs["dm"])
    print(envs["cycle"], envs["e_tot"])
    info = {
        "fock": envs["fock"],
        "dm": envs["dm"],
        "mo_coeff": envs["mo_coeff"],
        "mo_energy": envs["mo_energy"],
        "e_tot": envs["e_tot"],
    }
    getattr(envs["mf"], "info").append(info)


# fmt : on


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
