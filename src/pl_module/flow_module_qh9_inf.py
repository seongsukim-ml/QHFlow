from re import S
import torch

# import pytorch_lightning as pl
# from models import get_model

# from src.QHNet_flow.utils import ExponentialMovingAverage, self.post_processing

# from torch_ema import ExponentialMovingAverage
# from transformers import get_polynomial_decay_schedule_with_warmup
from pl_module.base_module import LitModel
from torch_geometric.data import Batch
from utils import AOData, get_total_cycles, build_matrix
import logging
import time
from tqdm import tqdm
from torch_scatter import scatter_sum
import pyscf
from pyscf import dft
import numpy as np
from pl_module.flow_module_qh9 import LitModel_flow as LitModel_flow_qh9


logger = logging.getLogger(__name__)
BOHR2ANG = 1 / 1.8897259886  # 0.52917721067


class LitModel_flow(LitModel_flow_qh9):
    def __init__(self, conf):
        super().__init__(conf=conf)
        self.test_mode = None
        self.inf_mul = conf.get("inf_mul", 5)
        self.loss_weights = {
            "hamiltonian": 1.0,
        }
        self.save_pred = conf.get("save_pred", True)

    def _inference(self, batch):
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
        # fmt: off
        self.log(f"infer/cycle", cycle, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"infer/cycle_time", init_cycle_time, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"infer/ham_calc_mae", ham_calc_error, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
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

    def test_step(self, batch, batch_idx):
        if self.test_mode == "inference":
            return self._inference(batch)
        elif self.test_mode == "test":
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
            self._log_error(errors, "test")
            if self.qh9:
                assert self.test_batch_size == 1
                # fmt: off
                self._log_sample_error_test(batch_one, "test_fix", num_timesteps=1, post_fix="_1")
                self._log_sample_error_test(batch_one, "test_fix", num_timesteps=2, post_fix="_2")
                self._log_sample_error_test(batch_one, "test_fix", num_timesteps=self.num_ode_steps_inf)
                # fmt: on
            else:
                # fmt: off
                self._log_sample_error(batch_one, "test", num_timesteps=1, post_fix="_1")
                self._log_sample_error(batch_one, "test", num_timesteps=2, post_fix="_2")
                self._log_sample_error(batch_one, "test", num_timesteps=self.num_ode_steps_inf)
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
            self._log_error(errors, "pred_mul_test")
            if self.qh9:
                assert self.test_batch_size == 1
                # fmt: off
                # self._log_sample_error_test_mul(batch_one, "test_fix", num_timesteps=1, post_fix="_1",mul=self.inf_mul)
                # self._log_sample_error_test_mul(batch_one, "test_fix", num_timesteps=2, post_fix="_2",mul=self.inf_mul)
                self._log_sample_error_test_mul(batch_one, "pred_mul", num_timesteps=self.num_ode_steps_inf,mul=self.inf_mul)
                # fmt: on
            else:
                raise NotImplementedError("test-mul is not implemented for md17")
                # fmt: off
                # self._log_sample_error(batch_one, "test", num_timesteps=1, post_fix="_1")
                # self._log_sample_error(batch_one, "test", num_timesteps=2, post_fix="_2")
                # self._log_sample_error(batch_one, "test", num_timesteps=self.num_ode_steps_inf)
                # fmt: on

            return errors
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
            if self.qh9:
                assert self.test_batch_size == 1
                # fmt: off
                # For debugging
                # self._log_sample_error_test(batch_one, "pred", num_timesteps=1, post_fix="_1")
                # self._log_sample_error_test(batch_one, "pred", num_timesteps=2, post_fix="_2")
                traj, sample =  self._log_sample_error_test(batch_one, "pred", num_timesteps=self.num_ode_steps_inf, save_pred=self.save_pred, log=False)
                if self.save_pred:
                    torch.save(sample, self.output_dir /"sample" / f"pred_{batch_idx}.pt")
                # fmt: on
            else:
                # fmt: off
                self._log_sample_error(batch_one, "pred", num_timesteps=self.num_ode_steps_inf)
                # fmt: on

            # return errors

    def process_target_batch(self, target):
        target_ham = target["hamiltonian"]

        if self.qh9:
            overlap = self.model.build_final_matrix(
                target, target[0].diagonal_overlap, target[0].non_diagonal_overlap
            ).cpu()
            overlap = (
                self.matrix_transform(
                    overlap,
                    target.atoms.cpu().squeeze().numpy(),
                    convention="back2pyscf",
                )
                .to(self.device)
                .to(self.default_type)
            )

        else:
            overlap = target["overlap"]
        target.orbital_energies, target.orbital_coefficients = (
            self.cal_orbital_and_energies(overlap, target_ham)
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
        sample = self.sample_with_scf(
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

    def sample_with_scf(
        self,
        batch,
        num_timesteps=100,
        min_t=0.01,
        sample_random=True,
        middle_scf=False,
        output_scf=False,
    ):
        if self.qh9:
            start_time = time.time()
            res, traj, pred = self.sample_qh9(
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
            return self._sample_with_scf_md17(
                batch,
                num_timesteps=num_timesteps,
                min_t=min_t,
                sample_random=sample_random,
                middle_scf=middle_scf,
                output_scf=output_scf,
            )

    def _sample_with_scf_md17(
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
