import torch
import pytorch_lightning as pl
from models import get_model

from collections import defaultdict
from typing import List

from torch_ema import ExponentialMovingAverage
from transformers import get_polynomial_decay_schedule_with_warmup
import logging
import time
import numpy as np
from tqdm import tqdm
from torch_scatter import scatter_sum
from argparse import Namespace
from torch_geometric.data import Batch

import os

# PySCF imports for inference functionality
import pyscf
from pyscf import dft
from common.metric import cal_orbital_and_energies, cal_orbital_and_energies_variable_size_grouped

# ==========================================        
# Constants and Configuration
# ==========================================

logger = logging.getLogger(__name__)

# Unit conversion constants
BOHR2ANG = 1 / 1.8897259886  # 0.52917721067 - Bohr to Angstrom conversion (MD17)

# Default optimizer parameters
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_BETAS = (0.99, 0.999)
DEFAULT_WARMUP_STEPS = 1000
DEFAULT_TRAINING_STEPS = 200000
DEFAULT_END_LR = 1e-8
DEFAULT_SCHEDULER_POWER = 1.0

# EMA parameters
DEFAULT_EMA_DECAY = 0.99

# Orbital mask indices
ORBITAL_1S_2S_INDICES = torch.tensor([0, 1])
ORBITAL_2P_INDICES = torch.tensor([3, 4, 5])
ORBITAL_MASK_SIZE_LINE1 = 5  # For atoms 1-2
ORBITAL_MASK_SIZE_LINE2 = 14  # For atoms 3-10

# Tolerance for eigenvalue calculations
EIGENVALUE_TOLERANCE = 1e-8

# SCF calculation defaults
DEFAULT_MAX_SCF_CYCLES = 50
DEFAULT_DFT_GRID_LEVEL = 3

# Convention dictionaries for orbital transformations
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
    """
    PyTorch Lightning module for quantum chemistry Hamiltonian prediction.
    
    This module supports both regular training/testing and inference modes,
    including SCF calculations and orbital energy/coefficient computations.
    """
    
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        
        # Device and data type setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.default_type = (
            torch.float64 if conf.data_type == "float64" else torch.float32
        )
        torch.set_default_dtype(self.default_type)
        
        # Model configuration
        self.loss_weights = conf.loss_weights
        self.model = get_model(conf.model)
        self.model.set(device)
        
        # EMA setup
        self.ema = None
        self.ema_start_epoch = getattr(conf, "ema_start_epoch", -1)
        if self.ema_start_epoch > -1:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=DEFAULT_EMA_DECAY)
        
        # Hamiltonian options
        self.use_init_hamiltonian = getattr(conf, "use_init_hamiltonian", False)
        self.use_init_hamiltonian_residue = getattr(
            conf, "use_init_hamiltonian_residue", False
        )
        
        # Dataset flags
        self.qh9 = getattr(conf, "qh9", False)
        
        # Batch sizes
        self.batch_size = conf.dataset.get("batch_size", 32)
        self.test_batch_size = conf.dataset.get("test_batch_size", 32)
        
        # Test mode for inference (can be "inference", "test", or "predict")
        self.test_mode = getattr(conf, "test_mode", "test")
        
        # Setup
        self.save_hyperparameters()
        self._epoch_start_time = None
        self.set(device)
        self.convention_dict = convention_dict
        
        # Logging
        logger.info(f"use_init_hamiltonian: {self.use_init_hamiltonian}")
        logger.info(f"use_init_hamiltonian_residue: {self.use_init_hamiltonian_residue}")
        logger.info(f"ema_start_epoch: {self.ema_start_epoch}")
        logger.info(f"qh9: {self.qh9}")
        logger.info(f"test_mode: {self.test_mode}")

    # ==========================================
    # Core PyTorch Lightning Methods
    # ==========================================
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        torch.set_default_dtype(self.default_type)
        
        if self.conf.optimizer.lower() == "AdamW".lower():
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=getattr(self.conf, "dataset", {}).get("learning_rate", DEFAULT_LEARNING_RATE),
                betas=DEFAULT_BETAS,
                amsgrad=False,
            )
            logger.info(f"Optimizer: AdamW with lr: {optimizer.param_groups[0]['lr']}")
        else:
            raise NotImplementedError(f"Optimizer {self.conf.optimizer} is not implemented.")
        
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=getattr(self.conf, "warmup_step", DEFAULT_WARMUP_STEPS),
            num_training_steps=getattr(self.conf, "num_training_steps", DEFAULT_TRAINING_STEPS),
            lr_end=getattr(self.conf, "end_lr", DEFAULT_END_LR),
            power=getattr(self.conf, "scheduler_power", DEFAULT_SCHEDULER_POWER),
            last_epoch=-1,
        )
        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict]

    def set(self, device):
        """Set device and initialize orbital masks."""
        self = self.to(device)
        print(f"Setting device to {device}")
        self.orbital_mask = self.get_orbital_mask()
        for key in self.orbital_mask.keys():
            self.orbital_mask[key] = self.orbital_mask[key].to(device)

    def forward(self, batch, H=None):
        """Forward pass through the model."""
        keep_blocks = self.qh9
        if self.use_init_hamiltonian:
            output = self.model(batch, batch.init_ham, keep_blocks=keep_blocks)
        else:
            output = self.model(batch, keep_blocks=keep_blocks)
        
        if self.use_init_hamiltonian_residue:
            if keep_blocks:
                output["hamiltonian_diagonal_blocks"] += batch["diagonal_init_ham"]
                output["hamiltonian_non_diagonal_blocks"] += batch["non_diagonal_init_ham"]
            else:
                output["hamiltonian"] = output["hamiltonian"] + batch["init_ham"]
        return output

    # ==========================================
    # Training Methods
    # ==========================================
    
    def on_train_start(self):
        self._epoch_start_train_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_train_time) / 60.0
        self.log("train/epoch_time_minutes", epoch_time, on_step=False, on_epoch=True, prog_bar=False)
        self._epoch_start_train_time = time.time()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA after each training batch if enabled."""
        if self.ema_start_epoch < 0:
            return
        if self.ema is not None and self.current_epoch > self.ema_start_epoch:
            self.ema.update()

    def training_step(self, batch, batch_idx):
        """Training step."""
        batch = self.post_processing(batch, self.default_type)
        outputs = self(batch)
        errors = self.criterion(outputs, batch, loss_weights=self.loss_weights)
        self.cur_batch_size = len(batch)

        loss = errors["loss"]
        for key in errors.keys():
            self.log(
                f"train/{key}",
                errors[key],
                on_step=True,
                on_epoch=True,
                prog_bar=True if key == "loss" else False,
                sync_dist=True,
                batch_size=self.cur_batch_size,
            )
        return loss

    # ==========================================
    # Validation Methods
    # ==========================================
    
    def on_validation_start(self):
        self._epoch_start_val_time = time.time()

    def on_validation_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_val_time) / 60.0
        self.log("val/epoch_time_minutes", epoch_time, on_step=False, on_epoch=True, prog_bar=False)
        self._epoch_start_val_time = time.time()

    def validation_step(self, batch, batch_idx):
        """Validation step with optional EMA evaluation."""
        batch = self.post_processing(batch, self.default_type)
        self.cur_batch_size = len(batch)
        
        # EMA validation if available
        if self.ema is not None:
            with self.ema.average_parameters():
                ema_outputs = self(batch)
                ema_errors = self.criterion(ema_outputs, batch, loss_weights=self.loss_weights)
                
                for key in ema_errors.keys():
                    self.log(
                        f"val_ema/{key}",
                        ema_errors[key],
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True if key == "loss" else False,
                        sync_dist=True,
                        batch_size=self.cur_batch_size,
                    )
                
                ema_metrics = self.metric(ema_outputs, batch)
                for key in ema_metrics.keys():
                    self.log(
                        f"val_ema/sample_{key}",
                        ema_metrics[key],
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True if key == "loss" else False,
                        sync_dist=True,
                        batch_size=self.cur_batch_size,
                    )
        
        # Regular validation
        outputs = self(batch)
        errors = self.criterion(outputs, batch, loss_weights=self.loss_weights)
        
        for key in errors.keys():
            self.log(
                f"val/{key}",
                errors[key],
                on_step=True,
                on_epoch=True,
                prog_bar=True if key == "loss" else False,
                sync_dist=True,
                batch_size=self.cur_batch_size,
            )
        
        metrics = self.metric(outputs, batch)
        for key in metrics.keys():
            self.log(
                f"val/sample_{key}",
                metrics[key],
                on_step=True,
                on_epoch=True,
                prog_bar=True if key == "loss" else False,
                sync_dist=True,
                batch_size=self.cur_batch_size,
            )
        return errors

    # ==========================================
    # Test Methods
    # ==========================================
    
    def on_test_start(self):
        self._epoch_start_test_time = time.time()

    def on_test_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_test_time) / 60.0
        self.log("test/epoch_time_minutes", epoch_time, on_step=False, on_epoch=True, prog_bar=False)
        self._epoch_start_test_time = time.time()

    def test_step(self, batch, batch_idx):
        """
        Test step with multiple modes:
        - inference: Run SCF calculations and detailed analysis
        - test: Standard model evaluation
        - predict: Save predictions for later analysis
        """
        if self.test_mode == "inference":
            return self._inference_step(batch, batch_idx)
        elif self.test_mode == "test":
            return self._test_step_standard(batch, batch_idx)
        elif self.test_mode == "predict":
            return self._predict_step(batch, batch_idx)
        else:
            return self._test_step_standard(batch, batch_idx)  # Default to standard test

    def _inference_step(self, batch, batch_idx):
        """Inference step with SCF calculations."""
        batch = self.post_processing(batch, self.default_type)
        batch_one = batch.clone()
        
        # Get cached values or initialize
        cycle = getattr(batch_one, "cycle", None)
        init_cycle_time = getattr(batch_one, "init_cycle_time", None)
        ham_calc = getattr(batch_one, "ham_calc", None)
        
        # Build Hamiltonian matrix
        if self.qh9:
            batch_one.full_edge_index = batch_one.edge_index_full
            batch_ham = self.build_final_matrix(
                batch_one,
                batch_one[0].diagonal_hamiltonian,
                batch_one[0].non_diagonal_hamiltonian,
            )
            # batch_ham = batch_ham.cpu().numpy()
            # batch_one.hamiltonian = (
            #     torch.tensor(batch_ham).to(self.device).to(self.default_type)
            # )
            batch_one.hamiltonian = batch_ham.clone()
            batch_ham = batch_ham.cpu()
        else:
            batch_ham = batch_one.hamiltonian.clone().cpu()
        
        # Transform to PySCF format
        batch_one.hamiltonian_pyscf = self.matrix_transform(
            batch_ham,
            batch_one.atoms,
            convention="back2pyscf",
        ).cpu()
        
        batch_one = self.process_target_batch(batch_one)

        # Run SCF if not cached
        if cycle is None:
            init_scf_ret = self.num_scf_steps(batch_one, init_dm_style="1e")
            cycle = init_scf_ret["cycle"]
            init_cycle_time = init_scf_ret["total_time"]
            ham_calc = self.matrix_transform(
                init_scf_ret["fock"],
                batch.atoms,
                convention="pyscf_def2svp",
            ).unsqueeze(0).to(self.device)
            # ham_calc = torch.tensor(ham_calc).unsqueeze(0).to(self.device)
            ham_calc_error = (ham_calc - batch_one.hamiltonian).abs().mean()
            e_tot_calc = init_scf_ret["e_tot"]

            if self.qh9:
                target_energy = None
                e_tot_calc_error = None
            else:
                target_energy = batch_one.energy.cpu()
                e_tot_calc_error = abs(e_tot_calc - target_energy.numpy()).mean()
        
        batch.hamiltonian_pyscf = batch_one.hamiltonian_pyscf

        # Log results
        self.log(f"infer/cycle", cycle, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=1)
        self.log(f"infer/cycle_time", init_cycle_time, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=1)
        self.log(f"infer/ham_calc_mae", ham_calc_error, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=1)
        self.log(f"infer/e_tot_calc", e_tot_calc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if e_tot_calc_error is not None:
            self.log(f"infer/e_tot_calc_error", e_tot_calc_error, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=1)
        
        self._log_inference_error(
            batch, "infer", 
            init_cycle=cycle, 
            init_cycle_time=init_cycle_time, 
            ham_error=ham_calc_error, 
            e_tot_error=e_tot_calc_error, 
            e_tot_gt=target_energy
        )
        return None

    def _test_step_standard(self, batch, batch_idx):
        """Standard test step."""
        batch = self.post_processing(batch, self.default_type)
        outputs = self(batch)
        errors = self.criterion(outputs, batch, loss_weights=self.loss_weights)
        self.cur_batch_size = len(batch)
        
        for key in errors.keys():
            self.log(
                f"test/{key}",
                errors[key],
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=self.cur_batch_size,
            )
        
        # assert self.test_batch_size == 1
        metrics = self.metric(outputs, batch)
        for key in metrics.keys():
            self.log(
                f"test/sample_{key}",
                metrics[key],
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=self.cur_batch_size,
            )
        return errors

    def _batch_has_ground_truth_hamiltonian(self, batch):
        return hasattr(batch, "diagonal_hamiltonian") or hasattr(batch, "hamiltonian")

    def _predict_step(self, batch, batch_idx, log_outputs=True):
        """Prediction step that saves outputs."""
        batch = self.post_processing(batch, self.default_type)
        outputs = self(batch)
        self.cur_batch_size = len(batch)

        if hasattr(self, 'output_dir'):
            if not os.path.exists(self.output_dir / "pred"):
                os.makedirs(self.output_dir / "pred", exist_ok=True)
        # log the error if the batch has the ground truth hamiltonian
        if self._batch_has_ground_truth_hamiltonian(batch):
            if hasattr(self, 'output_dir'):
                if not os.path.exists(self.output_dir / "gt"):
                    os.makedirs(self.output_dir / "gt", exist_ok=True)
                
        if self.qh9:
            # assert self.test_batch_size == 1
            outputs["hamiltonian"] = self.build_final_matrix(
                batch,
                outputs["hamiltonian_diagonal_blocks"],
                outputs["hamiltonian_non_diagonal_blocks"],
                transform=True,
                convention="back2pyscf",
            )
            gt_overlap = self.build_final_matrix(
                batch,
                batch.diagonal_overlap,
                batch.non_diagonal_overlap,
                transform=True,
                convention="back2pyscf",
            )
            
            if self._batch_has_ground_truth_hamiltonian(batch):
                gt_hamiltonian = self.build_final_matrix(
                    batch,
                    batch.diagonal_hamiltonian,
                    batch.non_diagonal_hamiltonian,
                    transform=True,
                    convention="back2pyscf",
                )                
            
            for i in range(self.cur_batch_size):
                pred = {
                    "pred_hamiltonian": outputs["hamiltonian"][i].cpu(),
                    "overlap": gt_overlap[i].cpu(),
                    "pos": batch[i].pos.cpu(),
                    "atoms": batch[i].atoms.cpu(),
                    "format":"pyscf",
                    "length_unit":"angstrom",
                }
                if hasattr(self, 'output_dir'):
                    torch.save(pred, self.output_dir / "pred" / f"pred_{batch_idx}_{i}.pt")
                if self._batch_has_ground_truth_hamiltonian(batch):
                    gt = {
                        "hamiltonian": gt_hamiltonian[i].cpu(),
                        "overlap": gt_overlap[i].cpu(),
                        "pos": batch[i].pos.cpu(),
                        "atoms": batch[i].atoms.cpu(),
                        "format":"pyscf",
                        "length_unit":"angstrom",
                    }
                    if hasattr(self, 'output_dir'):
                        torch.save(gt, self.output_dir / "gt" / f"gt_{batch_idx}_{i}.pt")
  
        else:
            for i in range(self.cur_batch_size):
                overlap = batch[i]["overlap"].squeeze(0).cpu()
                atoms = batch[i].atoms.squeeze(1).cpu()
                pos = batch[i].pos.cpu()
                pred = {
                    "pred_hamiltonian": outputs["hamiltonian"][i].cpu(),
                    "overlap": overlap,                   
                    "pos": pos,
                    "atoms": atoms,
                    "format":"dev2svp", # Need to change to pyscf for dft calculation
                    "length_unit":"bohr", # Need to change to angstrom for dft calculation
                }
                if hasattr(self, 'output_dir'):
                    torch.save(pred, self.output_dir / "pred" / f"pred_{batch_idx}_{i}.pt")
                if self._batch_has_ground_truth_hamiltonian(batch):
                    gt = {
                        "hamiltonian": batch[i].hamiltonian.squeeze(0).cpu(),
                        "overlap": overlap,
                        "pos": pos,
                        "atoms": atoms,
                        "format":"dev2svp", # Need to change to pyscf for dft calculation
                        "length_unit":"bohr", # Need to change to angstrom for dft calculation
                    }
                    if hasattr(batch[i], "init_ham"):
                        gt["init_ham"] = batch[i].init_ham.squeeze(0).cpu()
                    if hasattr(batch[i], "energy"):
                        gt["energy"] = batch[i].energy.cpu()
                    if hasattr(batch[i], "force"):
                        gt["force"] = batch[i].force.cpu()
                    if hasattr(self, 'output_dir'):
                        torch.save(gt, self.output_dir / "gt" / f"gt_{batch_idx}_{i}.pt")


        # only log the metrics if batch has the ground truth hamiltonian
        if self._batch_has_ground_truth_hamiltonian(batch) and log_outputs:
            metrics = self.metric(outputs, batch)
            for key in metrics.keys():
                self.log(
                    f"pred/sample_{key}",
                    metrics[key],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                    batch_size=self.cur_batch_size,
                )
        return None

    # ==========================================
    # Data Processing Methods
    # ==========================================
    
    @staticmethod
    def post_processing(batch, default_type):
        """Post-process batch data to ensure correct tensor shapes and types."""
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
            if torch.is_floating_point(batch[key]):
                batch[key] = batch[key].type(default_type)
        return batch

    def get_orbital_mask(self):
        """Get orbital masks for different atomic numbers."""
        orbital_mask_line1 = torch.cat([ORBITAL_1S_2S_INDICES, ORBITAL_2P_INDICES])
        orbital_mask_line2 = torch.arange(ORBITAL_MASK_SIZE_LINE2)
        orbital_mask = {}
        for i in range(1, 11):
            orbital_mask[i] = orbital_mask_line1 if i <= 2 else orbital_mask_line2
        return orbital_mask

    def process_target_batch(self, target):
        """Process target batch to compute orbital properties."""
        target_ham = target["hamiltonian"]

        if self.qh9:
            overlap = self.build_final_matrix(
                target,
                target.diagonal_overlap,
                target.non_diagonal_overlap,
                transform=True,
                convention="back2pyscf",
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

    # ==========================================
    # Mathematical Methods
    # ==========================================
    
    @staticmethod
    def cal_orbital_and_energies(overlap_matrix, full_hamiltonian, method="eigh", tol=EIGENVALUE_TOLERANCE):
        """Calculate orbital energies and coefficients from overlap and Hamiltonian matrices.
        This function solves the generalized eigenvalue problem HC = SCE, where:
        - H is the Hamiltonian matrix
        - S is the overlap matrix 
        - C are the orbital coefficients
        - E are the orbital energies
        
        Args:
            overlap_matrix (Tensor or List[Tensor]): Batch of overlap matrices [B, N, N] or List of overlap matrices [N, N]
            full_hamiltonian (Tensor or List[Tensor]): Batch of Hamiltonian matrices [B, N, N] or List of Hamiltonian matrices [N, N]
            method (str): Method to use for solving the generalized eigenvalue problem.
                - "eigh": Use eigenvalue decomposition (Fast but unstable)
                - "cholesky": Use Cholesky decomposition (Slow but stable)
            tol (float): Tolerance for small eigenvalues
        Returns:
            Tuple[Tensor, Tensor] or Tuple[List[Tensor], List[Tensor]]: Tuple containing:
                - orbital_energies: Eigenvalues [B, N] or List of Eigenvalues [N]
                - orbital_coefficients: Eigenvectors [B, N, N] or List of Eigenvectors [N, N]
        Notes:
            The orbital energies of the converted matrix are the same as the original matrix of MD17 but not in QH9
            The orbital coefficients of the converted matrix are not the same as the original matrix of both MD17 and QH9
        """
        assert method in ["eigh", "cholesky"], f"Invalid method: {method}"
        if isinstance(overlap_matrix, list) and isinstance(full_hamiltonian, list):
            return cal_orbital_and_energies_variable_size_grouped(
                overlap_matrix,
                full_hamiltonian,
                method,
                tol=tol,
            )
        else:
            return cal_orbital_and_energies(
                overlap_matrix,
                full_hamiltonian,
                method,
                tol=tol,
            )

    def matrix_transform(self, hamiltonian, data, convention="pyscf_def2svp"):
        if isinstance(hamiltonian, list):
            return self.matrix_transform_list(hamiltonian, data, convention)
        else:
            if isinstance(data, Batch):
                assert hamiltonian.shape[0] == data.ptr.shape[0] - 1, f"Hamiltonian shape {hamiltonian.shape} does not match data ptr shape {data.ptr.shape}"
                return self.matrix_transform_single(hamiltonian, data.atoms, convention)
            else:
                assert isinstance(data, list) or isinstance(data, torch.Tensor), f"Data must be a list or a tensor, got {type(data)}"
                return self.matrix_transform_single(hamiltonian, data, convention)

    def matrix_transform_list(self, hamiltonian_list, data, convention="pyscf_def2svp"):
        """Transform matrix between different orbital conventions - CUDA optimized version."""
        assert convention in self.convention_dict, f"Invalid convention: {convention}"
        
        final_matrix_list = []
        
        for graph_idx in range(data.ptr.shape[0] - 1):
            hamiltonian = hamiltonian_list[graph_idx]
            atoms = data.atoms[data.batch == graph_idx]
            mat_res = self.matrix_transform(hamiltonian, atoms, convention)
            final_matrix_list.append(mat_res)
            
        return final_matrix_list

    def matrix_transform_single(self, hamiltonian, atoms, convention="pyscf_def2svp"):
        """Transform matrix between different orbital conventions - CUDA optimized version."""
        assert convention in self.convention_dict, f"Invalid convention: {convention}"
        conv = self.convention_dict[convention]
        
        # Get device from hamiltonian tensor
        device = hamiltonian.device
        dtype = hamiltonian.dtype
        
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
            # Convert to torch tensors directly on the correct device
            transform_indices.append(torch.tensor(map_idx, device=device, dtype=torch.long) + offset)
            transform_signs.append(torch.tensor(map_sign, device=device, dtype=dtype))

        # Reorder according to orbitals_order
        transform_indices = [transform_indices[idx] for idx in orbitals_order]
        transform_signs = [transform_signs[idx] for idx in orbitals_order]
        
        # Concatenate using torch.cat instead of np.concatenate
        transform_indices = torch.cat(transform_indices)
        transform_signs = torch.cat(transform_signs)

        # Apply transformation using torch indexing
        hamiltonian_new = hamiltonian[..., transform_indices, :]
        hamiltonian_new = hamiltonian_new[..., :, transform_indices]
        
        # Apply signs using torch operations
        hamiltonian_new = hamiltonian_new * transform_signs.unsqueeze(-1)
        hamiltonian_new = hamiltonian_new * transform_signs.unsqueeze(-2)

        return hamiltonian_new

    def build_final_matrix(
        self,
        data,
        diagonal_matrix,
        non_diagonal_matrix,
        transform=False,
        convention="back2pyscf",
        dtype=None,
    ):
        """Build final matrix from diagonal and non-diagonal blocks."""
        final_matrix = []
        if hasattr(data, "full_edge_index"):
            dst, src = data.full_edge_index
        else:
            dst, src = data.edge_index_full
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
                mat_res = self.matrix_transform(
                    mat_res,
                    data.atoms[data.batch == graph_idx],
                    convention,
                )
            if dtype is not None:
                mat_res = mat_res.type(dtype)
            final_matrix.append(mat_res)
        return final_matrix

    # ==========================================
    # Loss Functions
    # ==========================================
    
    def criterion(self, outputs, target, loss_weights):
        """Main criterion function that dispatches to appropriate loss calculation."""
        if self.qh9:
            return self._criterion_qh9(outputs, target, loss_weights)
        else:
            return self._criterion(outputs, target, loss_weights)

    @staticmethod
    def _criterion(outputs, target, loss_weights):
        """Standard loss calculation for non-QH9 datasets."""
        error_dict = {}
        
        # Pre-compute orbital energies and coefficients if needed
        if "waloss" in loss_weights.keys():
            energy, orb = cal_orbital_and_energies(target.overlap, target.hamiltonian)
            target.orbital_energies = torch.diag_embed(energy).to(
                target.hamiltonian.device
            )
            target.orbital_coefficients = orb.to(target.hamiltonian.device)

        # Calculate losses for each component
        for key in loss_weights.keys():
            if key == "hamiltonian":
                diff = outputs[key] - target[key]

            elif key == "waloss":
                diff = outputs["hamiltonian"].bmm(target.orbital_coefficients)
                diff = torch.bmm(target.orbital_coefficients.transpose(-1, -2), diff)
                diff = diff - target.orbital_energies

            mse = torch.mean(diff**2)
            mae = torch.mean(torch.abs(diff))
            error_dict[key + "_mae"] = mae
            error_dict[key + "_rmse"] = torch.sqrt(mse)
            
            loss = mse + mae
            error_dict[key] = loss
            
            if "loss" in error_dict:
                error_dict["loss"] += loss_weights[key] * loss
            else:
                error_dict["loss"] = loss_weights[key] * loss

        return error_dict

    @staticmethod
    def _criterion_qh9(outputs, target, loss_weights):
        """Loss calculation for QH9 dataset with block structure."""
        error_dict = {}
        keys = loss_weights.keys()
        
        try:
            for key in keys:
                row = target.edge_index[0]
                edge_batch = target.batch[row]
                
                # Diagonal blocks
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

                # Non-diagonal blocks
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

                # Combined metrics
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
                error_dict[key + "_diagonal_mae"] = (mae_diagonal / count_sum_diagonal).mean()
                error_dict[key + "_non_diagonal_mae"] = (mae_non_diagonal / count_sum_non_diagonal).mean()

                loss = mae + mse
                if loss.isnan():
                    if mae.isnan():
                        logger.error(f"MAE is NaN - mae_diagonal: {mae_diagonal}, mae_non_diagonal: {mae_non_diagonal}")
                    if mse.isnan():
                        logger.error(f"MSE is NaN - mse_diagonal: {mse_diagonal}, mse_non_diagonal: {mse_non_diagonal}")
                    loss = torch.tensor(1e-8).to(loss.device).requires_grad_(True)
                
                error_dict[key + "_loss"] = loss

                if "loss" in error_dict.keys():
                    error_dict["loss"] = error_dict["loss"] + loss_weights[key] * loss
                else:
                    error_dict["loss"] = loss_weights[key] * loss
                    
        except Exception as exc:
            raise exc
        return error_dict

    # ==========================================
    # Evaluation Methods
    # ==========================================
    
    def metric(self, _outputs, _target):
        """Calculate orbital energy and coefficient errors."""
        outputs = _outputs
        target = _target.clone()
        
        if self.qh9:
            return self._metric_qh9(outputs, target)
        else:
            return self._metric_md17(outputs, target)

    @staticmethod
    def _metric_calc(outputs, target, names):
        """Test metric for evaluation."""
        error_dict = {}
        for key in names:
            if key == "orbital_coefficients":
                if isinstance(outputs[key], list):
                    error_list = []
                    for i in range(len(outputs[key])):
                        cosine_similarity = torch.cosine_similarity(outputs[key][i], target[key][i], dim=0)
                        error_list.append(cosine_similarity.abs().mean())
                    error_dict[key] = torch.mean(torch.stack(error_list))
                else:
                    assert outputs[key].dim() == 3, "The shape of outputs must be [batch, total_orbitals, num_occupied_orbitals] or List[Tensor]"
                    assert target[key].dim() == 3,  "The shape of target must be [batch, total_orbitals, num_occupied_orbitals] or List[Tensor]"
                    cosine_similarity = torch.cosine_similarity(outputs[key], target[key], dim=1)
                    error_dict[key] = cosine_similarity.abs().mean()
            elif key in ["diagonal_hamiltonian", "non_diagonal_hamiltonian"]:
                """ Metric of diagonal and non-diagonal blocks needs to be tested (Not reported in the paper) """
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
                if isinstance(outputs[key], list):
                    assert isinstance(target[key], list), "target must be a list"
                    assert len(outputs[key]) == len(target[key]), "outputs and target must have the same length"
                    error_list = []
                    for i in range(len(outputs[key])):
                        diff = outputs[key][i] - target[key][i]
                        mae = torch.mean(torch.abs(diff))
                        error_list.append(mae)
                    error_dict[key] = torch.mean(torch.stack(error_list))
                else:
                    diff = outputs[key] - target[key]
                    mae = torch.mean(torch.abs(diff))
                    error_dict[key] = mae
        return error_dict

    def _metric_md17(self, _outputs, _target):
        """Calculate orbital energy and coefficient errors for MD17 dataset with HOMO/LUMO/GAP."""
        metric_weights = {
            "hamiltonian": 1.0,
            "orbital_energies": 1.0,
            "orbital_coefficients": 1.0,
            "HOMO": 1.0,
            "LUMO": 1.0,
            "GAP": 1.0,
        }
        outputs = _outputs
        target = _target
        
        # for key in outputs.keys():
        #     if isinstance(outputs[key], torch.Tensor):
        #         outputs[key] = outputs[key].to("cpu")

        # target = target.to("cpu")

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
        error_dict = self._metric_calc(outputs, target, metric_weights)

        return error_dict

    def _metric_qh9(self, _outputs, _target):
        """Fixed test criterion for QH9 dataset."""
        metric_weights = {
            "hamiltonian": 1.0,
            "diagonal_hamiltonian": 1.0,
            "non_diagonal_hamiltonian": 1.0,
            "orbital_energies": 1.0,
            "orbital_coefficients": 1.0,
            "HOMO": 1.0,
            "LUMO": 1.0,
            "GAP": 1.0,
        }
        
        outputs = _outputs
        batch = _target
        # Build matrices
        outputs["hamiltonian"] = self.build_final_matrix(
            batch,
            outputs["hamiltonian_diagonal_blocks"],
            outputs["hamiltonian_non_diagonal_blocks"],
            transform=True,
            convention="back2pyscf",
            dtype=torch.float64,
        )
        
        batch.hamiltonian = self.build_final_matrix(
            batch,
            batch.diagonal_hamiltonian,
            batch.non_diagonal_hamiltonian,
            transform=True,
            convention="back2pyscf",
            dtype=torch.float64,
        )
        
        # Overlap matrix is shared across batch and
        overlap = self.build_final_matrix(
            batch,
            batch.diagonal_overlap,
            batch.non_diagonal_overlap,
            transform=True,
            convention="back2pyscf",
            dtype=torch.float64,
        )

        # Calculate orbital properties
        outputs["orbital_energies"], outputs["orbital_coefficients"] = (
            self.cal_orbital_and_energies(overlap, outputs["hamiltonian"])
        )
        batch.orbital_energies, batch.orbital_coefficients = (
            self.cal_orbital_and_energies(overlap, batch["hamiltonian"])
        )
        
        num_orb=[]
        pred_HOMO=[]
        pred_LUMO=[]
        gt_HOMO=[]
        gt_LUMO=[]
        for i in range(len(batch)):
            num_orb.append(int(batch.atoms[batch.ptr[i] : batch.ptr[i+1]].sum() / 2))
            pred_HOMO.append(outputs["orbital_energies"][i][num_orb[i] - 1])
            pred_LUMO.append(outputs["orbital_energies"][i][num_orb[i]])
            gt_HOMO.append(batch.orbital_energies[i][num_orb[i] - 1])
            gt_LUMO.append(batch.orbital_energies[i][num_orb[i]])
        
        pred_HOMO = torch.stack(pred_HOMO)
        pred_LUMO = torch.stack(pred_LUMO)
        gt_HOMO = torch.stack(gt_HOMO)
        gt_LUMO = torch.stack(gt_LUMO)
        
        outputs["HOMO"] = pred_HOMO
        outputs["LUMO"] = pred_LUMO
        outputs["GAP"] = pred_LUMO - pred_HOMO
        
        batch["HOMO"] = gt_HOMO
        batch["LUMO"] = gt_LUMO
        batch["GAP"] = gt_LUMO - gt_HOMO

        for i in range(len(batch)):
            outputs["orbital_energies"][i] = outputs["orbital_energies"][i][:num_orb[i]]
            outputs["orbital_coefficients"][i] = outputs["orbital_coefficients"][i][:, :num_orb[i]] 
            batch.orbital_energies[i] = batch.orbital_energies[i][:num_orb[i]]
            batch.orbital_coefficients[i] = batch.orbital_coefficients[i][:, :num_orb[i]]
        
        outputs["diagonal_hamiltonian"] = outputs["hamiltonian_diagonal_blocks"]
        outputs["non_diagonal_hamiltonian"] = outputs["hamiltonian_non_diagonal_blocks"]

        error_dict = self._metric_calc(outputs, batch, metric_weights)

        return error_dict

    # Deprecated
    @staticmethod
    def metric_test_qh9_old(
        outputs,
        outputs_ham,
        outputs_energy,
        outputs_coeff,
        target,
        target_ham,
        target_energy,
        target_coeff,
    ):
        """Test criterion for QH9 dataset with multiple graphs."""
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


    # ==========================================
    # PySCF Integration Methods
    # ==========================================
    
    def num_scf_steps(self, batch, init_dm_style="minao", dm0=None):
        """Run SCF calculation using PySCF."""
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
        """Calculate density matrix from Hamiltonian."""
        hamiltonian_t_pyscf = self.matrix_transform(
            cur_ham,
            batch.atoms,
            convention="back2pyscf"
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

    def _forward_with_scf(self, batch):
        """Forward pass with SCF setup for inference."""
        start_time = time.time()
        batch = batch.to(self.device)
        outputs = self(batch)
        
        if self.qh9:
            ham_out = self.build_final_matrix(
                batch,
                outputs["hamiltonian_diagonal_blocks"],
                outputs["hamiltonian_non_diagonal_blocks"],
            )
        else:
            ham_out = outputs["hamiltonian"]

        res = {
            "hamiltonian": ham_out,
            "sample_time_all": time.time() - start_time,
            "sample_time_per_batch": (time.time() - start_time) / batch.num_graphs,
        }

        if self.qh9:
            res["hamiltonian_diagonal_blocks"] = outputs["hamiltonian_diagonal_blocks"]
            res["hamiltonian_non_diagonal_blocks"] = outputs["hamiltonian_non_diagonal_blocks"]

        # Setup PySCF molecule
        mol = pyscf.gto.Mole()
        pos = batch.pos.cpu().numpy()
        if not self.qh9:
            pos = pos * BOHR2ANG # Convert Bohr to Angstrom
        
        atom_list = [
            [batch.atoms[atom_idx].cpu().item(), pos[atom_idx]]
            for atom_idx in range(batch.num_nodes)
        ]
        mol.build(verbose=0, atom=atom_list, basis="def2svp", unit="ang")
        overlap_pyscf = torch.from_numpy(mol.intor("int1e_ovlp")).unsqueeze(0)
        res["mol"] = mol
        res["dm_last"] = self.calc_dm0_from_ham(batch, overlap_pyscf, ham_out)

        return res

    def _log_inference_error(
        self,
        batch_one,
        prefix,
        init_cycle=None,
        init_cycle_time=None,
        post_fix="",
        ham_error=None,
        e_tot_error=None,
        e_tot_gt=None,
    ):
        """Log detailed inference errors including SCF results."""
        sample = self._forward_with_scf(batch_one)
        results = self.metric(sample, batch_one)
            
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
                batch_size=1,
            )

    # ==========================================
    # Batch Testing Methods (Deprecated) / TODO: Remove
    # ==========================================
    
    # Deprecated
    @torch.no_grad()
    def test_over_dataset(self, test_data_loader, default_type):
        """Test over entire dataset (non-QH9)."""
        self.eval()
        total_error_dict = {"total_items": 0}
        metric_weights = {
            "hamiltonian": 1.0,
            "orbital_energies": 1.0,
            "orbital_coefficients": 1.0,
        }
        total_time = 0
        total_graph = 0
        last_traj = []
        
        logger.info("num of test data: {}".format(len(test_data_loader)))
        for idx, batch in tqdm(enumerate(test_data_loader)):
            batch = self.post_processing(batch, default_type)
            batch = batch.to(self.model.device)
            tic = time.time()
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
            error_dict = self._metric_calc(outputs, batch, metric_weights)
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

    # Deprecated
    @torch.no_grad()
    def test_over_dataset_qh9(self, test_data_loader, default_type):
        """Test over entire dataset (QH9)."""
        self.eval()
        total_error_dict = {"total_items": 0}
        metric_weights = {
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
        last_traj = []
        
        logger.info("num of test data: {}".format(len(test_data_loader)))
        for idx, batch in tqdm(enumerate(test_data_loader)):
            batch = self.post_processing(batch, default_type)
            batch = batch.to(self.model.device)
            tic = time.time()
            outputs = self(batch)
            
            outputs["hamiltonian"] = self.build_final_matrix(
                batch,
                outputs["hamiltonian_diagonal_blocks"],
                outputs["hamiltonian_non_diagonal_blocks"],
            ).cpu()
            batch.hamiltonian = self.build_final_matrix(
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
            overlap = self.build_final_matrix(
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
            error_dict = self._metric_calc(outputs, batch, metric_weights)

            duration = time.time() - tic
            total_graph = total_graph + batch.ptr.shape[0] - 1
            total_time = duration + total_time
            for key in outputs.keys():
                if isinstance(outputs[key], torch.Tensor):
                    outputs[key] = outputs[key].to("cpu")

            secs = duration / batch.hamiltonian.shape[0]
            msg = f"batch {idx} / {secs*100:.2f}(10^-2)s : "
            for key in error_dict.keys():
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


# ==========================================
# PySCF Helper Functions
# ==========================================

def get_total_cycles(envs):
    """Callback function for SCF cycle tracking."""
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


def build_matrix(
    mol,
    dm0=None,
    error_level=None,
    Hamiltonian_gt=None,
    max_cycle=DEFAULT_MAX_SCF_CYCLES,
    e_tot_error_level=None,
    e_tot_gt=None,
    qh9=False,
):
    """Build matrix using PySCF SCF calculation."""
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
        scf_eng.grids.level = DEFAULT_DFT_GRID_LEVEL

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