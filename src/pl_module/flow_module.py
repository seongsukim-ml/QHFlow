import torch

# import pytorch_lightning as pl
# from models import get_model

# from src.QHNet_flow.utils import ExponentialMovingAverage, self.post_processing

# from torch_ema import ExponentialMovingAverage
# from transformers import get_polynomial_decay_schedule_with_warmup
from pl_module.base_module import LitModel
from torch_geometric.data import Batch
from utils import AOData, WDs, WDs_batch, Expansion
from e3nn import o3
import logging
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LitModel_flow(LitModel):
    def __init__(self, conf):
        super().__init__(conf=conf)
        self.batch_mul = conf.flow.get("batch_mul", 1)
        self.use_t_scale = conf.flow.get("use_t_scale", True)
        self.num_ode_steps = conf.flow.get("num_ode_steps", 10)
        self.num_ode_steps_val = conf.flow.get("num_ode_steps_val", 10)
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
        self.sample_random = conf.flow.get("sample_random", True)
        self.init_p0_type = conf.flow.get("init_p0_type", "gaussian")
        logger.info(f"init_p0_type: {self.init_p0_type}")

        if self.init_p0_type == "expand":
            self.expand = Expansion(
                o3.Irreps("3x0e + 2x1e + 1x2e"),
                o3.Irreps("3x0e + 2x1e + 1x2e"),
                o3.Irreps("3x0e + 2x1e + 1x2e"),
            )

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
            if self.init_p0_type == "gauss" or self.init_p0_type == "so3":
                random_ham += torch.randn_like(batch.hamiltonian) * self.sigma
                i, j = torch.tril_indices(random_ham.shape[-1], random_ham.shape[-1])
                random_ham[:, i, j] = random_ham[:, j, i]
                if self.init_gauss_center:
                    random_ham += batch.init_ham
                if self.init_p0_type == "so3":
                    random_R = o3.rand_matrix(batch.hamiltonian.shape[0])
                    random_WD = WDs_batch(batch, random_R).to(batch.hamiltonian.device)
                    random_ham = torch.bmm(
                        random_WD.transpose(-1, -2),
                        random_ham,
                    ).bmm(random_WD)
            elif self.init_p0_type == "expand":
                random_diag_tensor = o3.Irreps("3x0e + 2x1e + 1x2e").randn(
                    batch.atoms.shape[0], -1, device=batch.hamiltonian.device
                )
                random_non_diag_tensor = o3.Irreps("3x0e + 2x1e + 1x2e").randn(
                    batch.full_edge_index.shape[1], -1, device=batch.hamiltonian.device
                )
                random_diag_ham = self.expand(random_diag_tensor)
                random_non_diag_ham = self.expand(random_non_diag_tensor)
                random_ham = self.build_final_matrix(
                    batch, random_diag_ham, random_non_diag_ham, transform=False
                )
                random_ham = torch.stack(random_ham, dim=0) * self.sigma
            else:
                raise ValueError(f"Unknown init type: {self.init_p0_type}")

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
        self._log_sample_error(batch_one, "val", num_timesteps=1, post_fix="_1")
        self._log_sample_error(batch_one, "val", num_timesteps=self.num_ode_steps_val)

        return errors

    def test_step(self, batch, batch_idx):
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
        self._log_sample_error(batch_one, "test", num_timesteps=1, post_fix="_1")
        self._log_sample_error(batch_one, "test", num_timesteps=2, post_fix="_2")
        self._log_sample_error(batch_one, "test", num_timesteps=self.num_ode_steps_inf)

        return errors

    def sample(
        self,
        batch,
        num_timesteps=100,
        min_t=0.01,
        sample_random=True,
    ):
        device = self.model.device
        lin_t = torch.linspace(min_t, 1.0, num_timesteps + 1).to(device)
        cur_t = lin_t[0]
        batch.init_ham_t = torch.zeros_like(batch.init_ham)
        # batch.init_ham_t_res = batch.init_ham_t
        if sample_random:
            if self.init_p0_type == "gauss" or self.init_p0_type == "so3":
                batch.init_ham_t = torch.randn_like(batch.init_ham) * self.sigma
                i, j = torch.tril_indices(
                    batch.init_ham_t.shape[-1], batch.init_ham_t.shape[-1]
                )
                batch.init_ham_t[:, i, j] = batch.init_ham_t[:, j, i]
                if self.init_gauss_center:
                    batch.init_ham_t += batch.init_ham
                    # batch.init_ham_t_res = batch.init_ham_t - batch.init_ham
                if self.init_p0_type == "so3":
                    random_R = o3.rand_matrix(batch.init_ham.shape[0])
                    batch.random_R = WDs_batch(batch, random_R).to(self.device)
                    batch.init_ham_t = torch.bmm(
                        batch.random_R.transpose(-1, -2),
                        batch.init_ham_t,
                    ).bmm(
                        batch.random_R,
                    )
            elif self.init_p0_type == "expand":
                random_diag_tensor = o3.Irreps("3x0e + 2x1e + 1x2e").randn(
                    batch.atoms.shape[0], -1, device=batch.hamiltonian.device
                )
                random_non_diag_tensor = o3.Irreps("3x0e + 2x1e + 1x2e").randn(
                    batch.full_edge_index.shape[1], -1, device=batch.hamiltonian.device
                )
                random_diag_ham = self.expand(random_diag_tensor)
                random_non_diag_ham = self.expand(random_non_diag_tensor)
                batch.init_ham_t = (
                    torch.stack(
                        self.build_final_matrix(
                            batch, random_diag_ham, random_non_diag_ham, transform=False
                        ),
                        dim=0,
                    )
                    * self.sigma
                )
            else:
                raise ValueError(f"Unknown init type: {self.init_p0_type}")

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

        res_outputs = {"hamiltonian": ham_t}

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
            sample, traj, pred = self.sample(batch_one, num_timesteps=num_timesteps)
            orb_and_eng_error = self._orb_and_eng_error(sample, batch_one)
            for key in orb_and_eng_error.keys():
                self.log(
                    f"{prefix}/{key}{post_fix}",
                    orb_and_eng_error[key],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True if key == "loss" else False,
                    sync_dist=True,
                )
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
