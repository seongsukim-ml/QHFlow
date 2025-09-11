"""
Flow Module for Quantum Hamiltonian Learning

This module implements a PyTorch Lightning flow-based model for learning quantum 
Hamiltonians. It supports both MD17 and QH9 datasets with different matrix 
representations (full matrices vs. block-diagonal structures).

The flow model uses normalizing flows to learn the distribution of Hamiltonian 
matrices by training a neural ODE that transforms noise to target Hamiltonians.

Key Features:
- Support for both MD17 (full matrices) and QH9 (block-diagonal) datasets
- Multiple initialization strategies (Gaussian, SO(3), expansion-based)
- Time-scaled loss functions for improved training stability
- Comprehensive evaluation metrics including orbital energies and coefficients
- Flexible sampling with configurable ODE steps
F integration
- Advanced finetuning support with multiple loss variants
- Finetune mode support
"""

import torch
import logging
import time
from tqdm.rich import tqdm
from torch_scatter import scatter_sum
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# PySCF imports for inference functionality
import pyscf
from pyscf import dft

from pl_module.base_module import LitModel, convention_dict, build_fock_matrix
from torch_geometric.data import Batch
from utils import AOData, WDs, WDs_batch, Expansion, get_total_cycles
from e3nn import o3
import os

# ==========================================
# Constants and Configuration
# ==========================================

logger = logging.getLogger(__name__)
BOHR2ANG = 1 / 1.8897259886  # 0.52917721067

# Flow model defaults
DEFAULT_BATCH_MUL = 1  # Number of times to repeat each batch during training
DEFAULT_USE_T_SCALE = False  # Whether to use time-dependent loss scaling
DEFAULT_NUM_ODE_STEPS = 3  # Default ODE integration steps for training
DEFAULT_NUM_ODE_STEPS_VAL = 3  # Default ODE integration steps for validation
DEFAULT_NUM_ODE_STEPS_QH9 = 3  # Default ODE steps for QH9 dataset (typically faster)

# Noise and initialization parameters
DEFAULT_INIT_GAUSS = True  # Whether to use prior distribution
DEFAULT_ERROR_THRESHOLD = 1e-5  # Error threshold for conditional logging
DEFAULT_USE_MSE_AND_MAE = True  # Whether to use both MSE and MAE in loss
DEFAULT_INIT_GAUSS_CENTER = False  # Whether to center Gaussian noise around initial Hamiltonian
DEFAULT_USE_RES_TARGET = True  # Whether to predict residual (H - H_init) or full H
DEFAULT_USE_CORRUPT_MUL = False  # Whether to use corruption multiplication strategy
DEFAULT_SIGMA = 1.00  # Standard deviation for noise initialization
DEFAULT_SAMPLE_RANDOM = True  # Whether to sample random initial states during inference

# Initialization type options: "gauss"(="gaussian"),"so3" (rotate gaussian matrix), "expand", "expand_sym"
DEFAULT_INIT_P0_TYPE_MD17 = "gaussian"  # Default for MD17 dataset
DEFAULT_INIT_P0_TYPE_QH9 = "expand"  # Default for QH9 dataset

# Time sampling parameters
DEFAULT_MIN_T = 0.01  # Minimum time value to avoid singularities
TIME_SCALE_MAX_T = 0.9  # Maximum time value for loss scaling

# Time bins for evaluation - used to analyze performance across different time ranges
# TIME_BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
TIME_BINS = []
# Default loss weights for testing different components
DEFAULT_LOSS_WEIGHTS_MD17 = {
    "hamiltonian": 10.0,
}

DEFAULT_LOSS_WEIGHTS_QH9 = {
    "hamiltonian": 10.0,
}

DEFAULT_METRIC_WEIGHTS_MD17 = {
    "hamiltonian": 1.0,
    "orbital_energies": 1.0,
    "orbital_coefficients": 1.0,
}

DEFAULT_METRIC_WEIGHTS_QH9 = {
    "hamiltonian": 1.0,
    "diagonal_hamiltonian": 1.0,
    "non_diagonal_hamiltonian": 1.0,
    "orbital_energies": 1.0,
    "orbital_coefficients": 1.0,
    "HOMO": 1.0,  # Highest Occupied Molecular Orbital
    "LUMO": 1.0,  # Lowest Unoccupied Molecular Orbital
    "GAP": 1.0,   # HOMO-LUMO gap
}

# Finetune-specific loss weights
DEFAULT_FINETUNE_LOSS_WEIGHTS_MD17 = {
    "hamiltonian": 10.0,
    "waloss": 20.0, 
}

DEFAULT_FINETUNE_LOSS_WEIGHTS_QH9 = {
    "hamiltonian": 10.0,
    "waloss": 20.0,
}

# E3NN irreps configuration for equivariant neural networks
# 3x0e: 3 channels of 0-dimensional irreps (three s orbitals)
# 2x1e: 2 channels of 3-dimensional irreps (two p orbitals)
# 1x2e: 1 channel of 5-dimensional irreps (one d orbital)
DEFAULT_IRREPS = "3x0e + 2x1e + 1x2e"

# Inference and finetuning parameters
DEFAULT_TEST_MUL = 5     # Used for test-mul mode (test with multiple samples for variance measurement)
DEFAULT_SAVE_PRED = True # Whether to save predictions




class LitModel_flow(LitModel):
    """
    Unified Flow-based Lightning Module for Quantum Hamiltonian Learning
    
    This class extends the base LitModel to implement normalizing flows for 
    learning quantum Hamiltonians. It automatically detects whether to use
    MD17 or QH9 mode based on the dataset configuration.
    
    The flow model works by:
    1. Starting with noise (Gaussian or structured)
    2. Using a neural ODE to evolve the noise toward the target Hamiltonian
    3. Training the vector field to minimize the reconstruction error
    """
    
    def __init__(self, conf):
        """
        Initialize the flow model with configuration parameters.
        
        Args:
            conf: Configuration object containing model and dataset parameters
        """
        super().__init__(conf=conf)
        
        # Extract flow-specific configuration with appropriate defaults
        self.batch_mul = conf.flow.get("batch_mul", DEFAULT_BATCH_MUL)
        self.use_t_scale = conf.flow.get("use_t_scale", DEFAULT_USE_T_SCALE)
        
        # Set ODE steps based on dataset type
        if self.qh9:
            default_ode_steps = DEFAULT_NUM_ODE_STEPS_QH9
        else:
            default_ode_steps = DEFAULT_NUM_ODE_STEPS
            
        self.num_ode_steps = conf.flow.get("num_ode_steps", default_ode_steps)
        self.num_ode_steps_val = conf.flow.get("num_ode_steps_val", default_ode_steps)
        self.num_ode_steps_test = conf.flow.get("num_ode_steps_test", self.num_ode_steps_val)
        
        # Noise and initialization parameters
        self.init_gauss = conf.flow.get("init_gauss", DEFAULT_INIT_GAUSS)
        self.error_threshold = conf.flow.get("error_threshold", DEFAULT_ERROR_THRESHOLD)
        self.use_mse_and_mae = conf.flow.get("use_mse_and_mae", DEFAULT_USE_MSE_AND_MAE)
        self.init_gauss_center = conf.flow.get("init_gauss_center", DEFAULT_INIT_GAUSS_CENTER)
        
        # Handle empty string case for init_gauss_center
        if self.init_gauss_center == "" or self.init_gauss_center is None:
            self.init_gauss_center = False
            
        # Consistency check: can't center without Gaussian initialization
        if self.init_gauss is False:
            self.init_gauss_center = False
            
        self.use_res_target = conf.flow.get("use_res_target", DEFAULT_USE_RES_TARGET)
        self.use_corrupt_mul = conf.flow.get("use_corrupt_mul", DEFAULT_USE_CORRUPT_MUL)
        self.sigma = conf.flow.get("sigma", DEFAULT_SIGMA)
        self.sample_random = conf.flow.get("sample_random", DEFAULT_SAMPLE_RANDOM)
        
        # Set default initialization type based on dataset
        if self.qh9:
            default_init_type = DEFAULT_INIT_P0_TYPE_QH9
        else:
            default_init_type = DEFAULT_INIT_P0_TYPE_MD17
            
        self.init_p0_type = conf.flow.get("init_p0_type", default_init_type)
        logger.info(f"init_p0_type: {self.init_p0_type}")
        
        # Initialize expansion module for equivariant initialization if needed
        if self.init_p0_type in ["expand", "expand_sym"]:
            self.expand = Expansion(
                o3.Irreps(DEFAULT_IRREPS),
                o3.Irreps(DEFAULT_IRREPS),
                o3.Irreps(DEFAULT_IRREPS),
            )
        
        # Test mode for inference (can be "inference", "test", "predict", or "predict-mul")
        self.mode = conf.get("mode", "train")
        self.test_mode = conf.get("test_mode", "test")
        
        # Finetune mode configuration
        self.finetune_mode = conf.get("finetune_mode", False)
        if self.finetune_mode:
            if self.mode != "train":
                raise ValueError("Finetune mode is only supported in train mode")            
            # Use finetune-specific loss weights
            if self.qh9:
                self.loss_weights = conf.get("finetune_loss_weights", DEFAULT_FINETUNE_LOSS_WEIGHTS_QH9)
                self.metric_weights = conf.get("metric_weights", DEFAULT_METRIC_WEIGHTS_QH9)
            else:
                self.loss_weights = conf.get("finetune_loss_weights", DEFAULT_FINETUNE_LOSS_WEIGHTS_MD17)
                self.metric_weights = conf.get("metric_weights", DEFAULT_METRIC_WEIGHTS_MD17)
            logger.info(f"Finetune mode enabled with loss weights: {self.loss_weights}")
        else:
            # Use standard loss weights
            if self.qh9:
                self.loss_weights = conf.get("loss_weights", DEFAULT_LOSS_WEIGHTS_QH9)
                self.metric_weights = conf.get("metric_weights", DEFAULT_METRIC_WEIGHTS_QH9)
            else:
                self.loss_weights = conf.get("loss_weights", DEFAULT_LOSS_WEIGHTS_MD17)
                self.metric_weights = conf.get("metric_weights", DEFAULT_METRIC_WEIGHTS_MD17)
        
        # Inference and parameters
        self.test_mul = conf.get("test_mul", DEFAULT_TEST_MUL) # Used for test-mul mode
        self.save_pred = conf.get("save_pred", DEFAULT_SAVE_PRED)
        self.log_n_steps_ODE_test = conf.get("log_n_steps_ODE_test", [1,2])
        self.log_n_steps_ODE_val = conf.get("log_n_steps_ODE_val", [1])
        assert type(self.log_n_steps_ODE_test) in [list, tuple, set, None]
        assert type(self.log_n_steps_ODE_val) in [list, tuple, set, None]
        
        # Setup convention dictionary
        self.convention_dict = convention_dict
        
        self.save_hyperparameters()
        self.batch_size = conf.dataset.get("batch_size", 32)


    # ==========================================
    # Batch Processing Methods
    # ==========================================

    def batch_repeat(self, batch, mul=1, repeat_style="repeat"):
        """
        Repeat batch data for data augmentation during training.
        
        Args:
            batch: Input batch data
            mul: Number of times to repeat the batch
            repeat_style: "repeat" or "append" - how to handle repetition
            
        Returns:
            Repeated batch data or original batch if mul=1
        """
        if mul == 1:
            return batch

        # QH9 dataset doesn't support multiplication > 1 due to memory constraints
        if self.qh9:
            if mul > 1:
                raise ValueError("mul > 1 is not supported for qh9")
            return batch
        else:
            return self._batch_repeat_md17(batch, mul, repeat_style)

    @staticmethod
    def _batch_repeat_md17(batch, mul=1, repeat_style="repeat"):
        """
        Repeat batch data for MD17 dataset (supports full matrix representation).
        
        Args:
            batch: Input batch data
            mul: Number of times to repeat
            repeat_style: "repeat" (duplicate each sample) or "append" (duplicate all samples)
            
        Returns:
            New batch with repeated data
        """
        assert repeat_style in ["append", "repeat"]
        if mul == 1:
            return batch
            
        batch_list = []
        
        # Extract individual graphs from the batch
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

            # Extract orbital-related data
            len_orb = batch.hamiltonian.shape[-1]
            AO_index = batch.AO_index[:, idx * len_orb : (idx + 1) * len_orb]
            AO_index[0] -= batch.ptr[idx]
            AO_index[2] -= idx
            Q = batch.Q[idx * len_orb : (idx + 1) * len_orb]

            # Create AOData object for this graph
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
            
            # Add to batch list based on repeat style
            if repeat_style == "repeat":
                for _ in range(mul):
                    batch_list.append(data.clone())
            else:
                batch_list.append(data.clone())

        # For append style, repeat the entire list
        if repeat_style == "append":
            new_batch_list = []
            for _ in range(mul):
                new_batch_list += batch_list
            batch_list = new_batch_list

        return Batch.from_data_list(batch_list)

    # ==========================================
    # Flow Corruption Methods
    # ==========================================

    def _corrupt_md17(self, batch, batch_t):
        """
        Add noise corruption to MD17 batch data (full matrix representation).
        
        This method implements the forward process of the flow model by interpolating
        between noise and target Hamiltonians based on time parameter t.
        
        Args:
            batch: Input batch data
            batch_t: Time parameters for each sample in batch
            
        Returns:
            Corrupted batch with interpolated Hamiltonians
        """
        batch.t = batch_t
        random_ham = torch.zeros_like(batch.hamiltonian)
        
        if self.init_gauss:
            if self.init_p0_type in ["gauss", "gaussian", "so3"]:
                # Standard Gaussian noise
                random_ham += torch.randn_like(batch.hamiltonian) * self.sigma
                
                # Ensure Hermitian symmetry by copying lower triangular to upper
                i, j = torch.tril_indices(random_ham.shape[-1], random_ham.shape[-1])
                random_ham[:, i, j] = random_ham[:, j, i]
                
                # Optionally center around initial Hamiltonian
                if self.init_gauss_center:
                    random_ham += batch.init_ham
                    
                # Apply SO(3) rotation for equivariant initialization
                if self.init_p0_type == "so3":
                    random_R = o3.rand_matrix(batch.hamiltonian.shape[0])
                    random_WD = WDs_batch(batch, random_R).to(batch.hamiltonian.device)
                    random_ham = torch.bmm(
                        random_WD.transpose(-1, -2),
                        random_ham,
                    ).bmm(random_WD)
                    
            elif self.init_p0_type == "expand":
                # Use equivariant expansion for structured initialization
                random_diag_tensor = o3.Irreps(DEFAULT_IRREPS).randn(
                    batch.atoms.shape[0], -1, device=batch.hamiltonian.device
                )
                random_non_diag_tensor = o3.Irreps(DEFAULT_IRREPS).randn(
                    batch.full_edge_index.shape[1], -1, device=batch.hamiltonian.device
                )
                random_diag_ham = self.expand(random_diag_tensor)
                random_non_diag_ham = self.expand(random_non_diag_tensor)
            
                random_ham = self.build_final_matrix(
                    batch, random_diag_ham, random_non_diag_ham, transform=False
                )
                random_ham = torch.stack(random_ham, dim=0) * self.sigma
                
            elif self.init_p0_type == "expand_sym":
                # Symmetric expansion initialization
                random_diag_tensor = o3.Irreps(DEFAULT_IRREPS).randn(
                    batch.atoms.shape[0], -1, device=batch.hamiltonian.device
                )
                random_non_diag_tensor = o3.Irreps(DEFAULT_IRREPS).randn(
                    batch.full_edge_index.shape[1], -1, device=batch.hamiltonian.device
                )
                random_diag_ham = self.expand(random_diag_tensor)
                random_non_diag_ham = self.expand(random_non_diag_tensor)
                random_ham = self.build_final_matrix(
                    batch, random_diag_ham, random_non_diag_ham, transform=False
                )
                random_ham = torch.stack(random_ham, dim=0) * self.sigma
                
                # Enforce symmetry
                i, j = torch.tril_indices(random_ham.shape[-1], random_ham.shape[-1])
                random_ham[:, i, j] = random_ham[:, j, i]
            else:
                raise ValueError(f"Unknown init type: {self.init_p0_type}")

        # Determine target: residual (H - H_init) or full Hamiltonian
        if self.use_res_target:
            target_ham = batch.hamiltonian - batch.init_ham
        else:
            target_ham = batch.hamiltonian

        # Store for debugging/analysis
        batch.random_ham = random_ham
        batch.target_ham = target_ham

        # Linear interpolation: H(t) = (1-t) * noise + t * target
        batch_t_reshape = batch_t.reshape(-1, 1, 1)
        batch.init_ham_t = (
            random_ham * (1 - batch_t_reshape) + target_ham * batch_t_reshape
        )
        return batch

    def _corrupt_qh9(self, batch, batch_t):
        """
        Add noise corruption to QH9 batch data (block-diagonal representation).
        
        QH9 uses a block-diagonal structure where only diagonal blocks are corrupted
        while non-diagonal blocks are handled separately.
        
        Args:
            batch: Input batch data with block structure
            batch_t: Time parameters for each sample in batch
            
        Returns:
            Corrupted batch with interpolated diagonal Hamiltonians
        """
        batch.t = batch_t
        random_ham = torch.zeros_like(batch["diagonal_hamiltonian"])
        
        if self.init_gauss:
            if self.init_p0_type in ["gauss", "gaussian", "so3"]:
                # Gaussian noise for diagonal blocks
                random_ham += (
                    torch.randn_like(batch["diagonal_hamiltonian"]) * self.sigma
                )
                
                # Note: Symmetry is handled differently for blocks
                # Individual blocks maintain their own symmetry
                i, j = torch.tril_indices(random_ham.shape[-1], random_ham.shape[-1])
                random_ham[:, i, j] = random_ham[:, j, i]
                
                if self.init_gauss_center:
                    random_ham += batch["diagonal_init_ham"]
                    
                if self.init_p0_type == "so3":
                    random_R = o3.rand_matrix(batch["diagonal_hamiltonian"].shape[0])
                    random_WD = WDs_batch(batch, random_R).to(self.device)
                    random_ham = torch.bmm(
                        random_WD.transpose(-1, -2),
                        random_ham,
                    ).bmm(random_WD)
                    
            elif self.init_p0_type == "expand":
                # Equivariant initialization for QH9
                random_diag_tensor = o3.Irreps(DEFAULT_IRREPS).randn(
                    batch.atoms.shape[0], -1, device=self.device
                )
                random_diag_ham = self.expand(random_diag_tensor)
                random_ham = random_diag_ham * self.sigma
            else:
                raise ValueError(f"Unknown init type: {self.init_p0_type}")

        # Target determination for QH9
        if self.use_res_target:
            target_ham = batch["diagonal_hamiltonian"] - batch["diagonal_init_ham"]
        else:
            target_ham = batch["diagonal_hamiltonian"]

        batch.random_ham = random_ham
        batch.target_ham = target_ham

        # Linear interpolation for diagonal blocks
        batch_t_reshape = batch_t.reshape(-1, 1, 1)
        batch.init_ham_t = (
            random_ham * (1 - batch_t_reshape) + target_ham * batch_t_reshape
        )
        return batch

    def corrupt(self, batch, mul=1):
        """
        Apply corruption (noise addition) to batch data.
        
        This is the main interface for adding noise during training.
        It automatically selects the appropriate corruption method based on dataset type.
        
        Args:
            batch: Input batch data
            mul: Batch multiplication factor
            
        Returns:
            Corrupted batch with noise added
        """
        # First repeat the batch if needed
        batch = self.batch_repeat(batch, mul)
        
        # Sample random time parameters
        if self.qh9:
            num_ham = batch["diagonal_hamiltonian"].shape[0]
        else:
            num_ham = batch.hamiltonian.shape[0]
        batch_t = self.sample_t(num_ham, batch.atoms.device)

        # Apply appropriate corruption method
        if self.qh9:
            return self._corrupt_qh9(batch, batch_t)
        else:
            return self._corrupt_md17(batch, batch_t)

    def corrupt_mul(self, batch):
        """
        Special corruption method that creates a batch with mixed time parameters.
        
        This method creates a batch where half the samples have t=0 (pure noise)
        and half have random t values. Used for certain training strategies.
        
        Args:
            batch: Input batch data
            
        Returns:
            Corrupted batch with mixed time parameters
        """
        batch = self.batch_repeat(batch, 2, repeat_style="append")
        
        if self.qh9:
            num_ham = batch["diagonal_hamiltonian"].shape[0]
        else:
            num_ham = batch.hamiltonian.shape[0]

        batch_t = self.sample_t(num_ham, batch.atoms.device)
        # Set second half to t=0 (pure noise)
        batch_t[batch_t.shape[0] // 2 :] = torch.zeros_like(
            batch_t[batch_t.shape[0] // 2 :]
        )
        
        if self.qh9:
            return self._corrupt_qh9(batch, batch_t)
        else:
            return self._corrupt_md17(batch, batch_t)

    @staticmethod
    def sample_t(num_batch, device, min_t=DEFAULT_MIN_T):
        """
        Sample random time parameters for the flow model.
        
        Time parameters are sampled uniformly from [min_t, 1-min_t] to avoid
        singularities at t=0 and t=1.
        
        Args:
            num_batch: Number of time parameters to sample
            device: Device to place tensors on
            min_t: Minimum time value to avoid singularities
            
        Returns:
            Tensor of sampled time parameters
        """
        t = torch.rand(num_batch, device=device)
        return t * (1 - 2 * min_t) + min_t  # [min_t, 1-min_t]

    # ==========================================
    # Loss Functions and Criteria
    # ==========================================

    def criterion(self, outputs, target, loss_weights, use_t_scale=False, use_mse_and_mae=False):
        """
        Unified criterion function that dispatches to appropriate loss calculation.
        
        Args:
            outputs: Model predictions
            target: Ground truth data
            loss_weights: Dictionary of loss component weights
            use_t_scale: Whether to apply time-dependent scaling
            use_mse_and_mae: Whether to use both MSE and MAE
            
        Returns:
            Dictionary of computed losses
        """
        if self.finetune_mode:
            # Use finetune-specific criterion
            if self.qh9:
                return self._criterion_qh9_finetune(outputs, target, loss_weights, use_t_scale)
            else:
                return self._criterion_md17_finetune(outputs, target, loss_weights, use_t_scale, use_mse_and_mae)
        else:
            # Use standard criterion
            if self.qh9:
                return self._criterion_qh9(outputs, target, loss_weights, use_t_scale)
            else:
                return self._criterion_md17(outputs, target, loss_weights, use_t_scale, use_mse_and_mae)

    @staticmethod
    def _criterion_qh9(outputs, target, loss_weights, use_t_scale=False):
        """
        Loss calculation for QH9 dataset with block-diagonal structure.
        
        Computes losses separately for diagonal and non-diagonal blocks,
        then combines them using scatter operations for batching.
        
        Args:
            outputs: Model predictions with block structure
            target: Ground truth with block structure
            loss_weights: Dictionary of loss component weights
            use_t_scale: Whether to apply time-dependent scaling
            
        Returns:
            Dictionary of computed losses
        """
        error_dict = {}
        keys = loss_weights.keys()
        
        try:
            for key in keys:
                row = target.edge_index[0]
                edge_batch = target.batch[row]
                
                # Diagonal blocks loss computation
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
                
                # Aggregate over batch using scatter operations
                mse_diagonal = scatter_sum(mse_diagonal, target.batch)
                mae_diagonal = scatter_sum(mae_diagonal, target.batch)
                count_sum_diagonal = scatter_sum(count_sum_diagonal, target.batch)

                # Non-diagonal blocks loss computation
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
                
                # Aggregate non-diagonal blocks
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

                # Store individual component errors
                error_dict[key + "_mae"] = mae
                error_dict[key + "_rmse"] = torch.sqrt(mse)
                error_dict[key + "_diagonal_mae"] = (
                    mae_diagonal / count_sum_diagonal
                ).mean()
                error_dict[key + "_non_diagonal_mae"] = (
                    mae_non_diagonal / count_sum_non_diagonal
                ).mean()

                # Combined loss
                loss = mae + mse
                
                # Handle NaN values gracefully
                if loss.isnan():
                    logger.error(f"loss is nan for {key}")
                    loss = torch.tensor(0.0).to(loss.device)
                    loss.requires_grad = True

                # Apply time-dependent scaling if requested
                if use_t_scale:
                    scale = 1 / (1 - torch.min(target.t, torch.tensor(TIME_SCALE_MAX_T))) ** 2
                    loss = loss * scale

                error_dict[key] = loss
                
                # Accumulate total loss
                if "loss" in error_dict.keys():
                    error_dict["loss"] = error_dict["loss"] + loss_weights[key] * loss
                else:
                    error_dict["loss"] = loss_weights[key] * loss
                    
        except Exception as exc:
            logger.error(f"Error in QH9 criterion computation: {exc}")
            raise exc
            
        return error_dict

    @staticmethod
    def _criterion_md17(outputs, target, loss_weights, use_t_scale=False, use_mse_and_mae=False):
        """
        Loss calculation for MD17 dataset with full matrix representation.
        
        Supports multiple loss types including direct Hamiltonian loss,
        Walsh-Bowers (waloss) for orbital-based loss, and delta variants.
        
        Args:
            outputs: Model predictions
            target: Ground truth data
            loss_weights: Dictionary of loss component weights
            use_t_scale: Whether to apply time-dependent scaling
            use_mse_and_mae: Whether to use both MSE and MAE
            
        Returns:
            Dictionary of computed losses
        """
        error_dict = {}
        
        # Pre-compute orbital energies and coefficients for Walsh-Bowers loss
        if "waloss" in loss_weights.keys():
            energy, orb = LitModel.cal_orbital_and_energies(
                target.overlap, target.hamiltonian
            )
            target.orbital_energies = torch.diag_embed(energy).to(target.atoms.device)
            target.orbital_coefficients = orb.to(target.atoms.device)

        # Compute losses for each component
        for key in loss_weights.keys():
            scale = 1
            
            if key == "hamiltonian":
                # Direct Hamiltonian reconstruction loss
                diff = outputs[key] - target[key]
                if use_t_scale:
                    scale = 1 / (1 - torch.min(target.t, torch.tensor(TIME_SCALE_MAX_T))) ** 2

            elif key == "waloss":
                # Walsh-Bowers loss: projects Hamiltonian onto orbital subspace
                diff = outputs["hamiltonian"].bmm(target.orbital_coefficients)
                diff = torch.bmm(target.orbital_coefficients.transpose(-1, -2), diff)
                diff = diff - target.orbital_energies

            # Compute error metrics
            mse = torch.mean(diff**2)
            mae = torch.mean(torch.abs(diff))
            error_dict[key + "_mae"] = mae
            error_dict[key + "_rmse"] = torch.sqrt(mse)
            
            # Determine loss combination based on loss type
            if key == "hamiltonian":
                if use_mse_and_mae:
                    loss = mse + mae
                else:
                    loss = mse
            elif key == "waloss":
                loss = mse
            # Apply scaling and averaging
            loss = loss * scale
            loss = torch.mean(loss)
            error_dict[key] = loss
            
            # Accumulate total loss
            if "loss" in error_dict:
                error_dict["loss"] += loss_weights[key] * loss
            else:
                error_dict["loss"] = loss_weights[key] * loss

        # Compute time-binned losses for analysis
        for key in loss_weights.keys():
            if key in ["waloss"]:
                continue
            for _bin in TIME_BINS:
                s, e = _bin
                mask = (target.t >= s) & (target.t < e)
                if mask.sum() > 0:  # Only compute if there are samples in this bin
                    diff = outputs[key][mask] - target[key][mask]
                    mse = torch.mean(diff**2)
                    mae = torch.mean(torch.abs(diff))
                    error_dict[key + f"_mae@{s:.2f}_{e:.2f}"] = mae
                    error_dict[key + f"_rmse@{s:.2f}_{e:.2f}"] = torch.sqrt(mse)

        return error_dict

    # Deprecated
    def criterion_finetune(self, outputs, target, loss_weights, use_t_scale=False, use_mse_and_mae=False):
        """
        Enhanced criterion function for finetuning with advanced loss variants.
        
        Args:
            outputs: Model predictions
            target: Ground truth data
            loss_weights: Dictionary of loss component weights
            use_t_scale: Whether to apply time-dependent scaling
            use_mse_and_mae: Whether to use both MSE and MAE
            
        Returns:
            Dictionary of computed losses
        """
        if self.qh9:
            return self._criterion_qh9_finetune(outputs, target, loss_weights, use_t_scale)
        else:
            return self._criterion_md17_finetune(outputs, target, loss_weights, use_t_scale, use_mse_and_mae)

    def _criterion_qh9_finetune(self, outputs, target, loss_weights, use_t_scale=False):
        """
        Enhanced loss calculation for QH9 dataset with advanced finetuning variants.
        
        Args:
            outputs: Model predictions
            target: Ground truth data
            loss_weights: Dictionary of loss component weights
            use_t_scale: Whether to apply time-dependent scaling
            
        Returns:
            Dictionary of computed losses
        """
        error_dict = {}
        keys = loss_weights.keys()

        try:
            for key in keys:
                if key == "hamiltonian":
                    # Standard Hamiltonian loss
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
                    # Walsh-Bowers loss for orbital-based optimization
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

    @staticmethod
    def _criterion_md17_finetune(outputs, target, loss_weights, use_t_scale=False, use_mse_and_mae=False):
        """
        Enhanced loss calculation for MD17 dataset with advanced finetuning variants.
        
        Args:
            outputs: Model predictions
            target: Ground truth data
            loss_weights: Dictionary of loss component weights
            use_t_scale: Whether to apply time-dependent scaling
            use_mse_and_mae: Whether to use both MSE and MAE
            
        Returns:
            Dictionary of computed losses
        """
        error_dict = {}
        
        # Pre-compute orbital energies and coefficients for Walsh-Bowers loss
        if "waloss" in loss_weights.keys():
            energy, orb = LitModel.cal_orbital_and_energies(
                target.overlap, target.hamiltonian
            )
            target.orbital_energies = torch.diag_embed(energy).to(target.atoms.device)
            target.orbital_coefficients = orb.to(target.atoms.device)

        # Compute losses for each component
        for key in loss_weights.keys():
            scale = 1
            
            if key == "hamiltonian":
                # Direct Hamiltonian reconstruction loss
                diff = outputs[key] - target[key]
                if use_t_scale:
                    scale = 1 / (1 - torch.min(target.t, torch.tensor(TIME_SCALE_MAX_T))) ** 2

            elif key == "waloss":
                # Walsh-Bowers loss: projects Hamiltonian onto orbital subspace
                diff = outputs["hamiltonian"].bmm(target.orbital_coefficients)
                diff = torch.bmm(target.orbital_coefficients.transpose(-1, -2), diff)
                diff = diff - target.orbital_energies

            elif key == "orbital_energies":
                # Orbital energy loss
                pred_energy, _ = LitModel.cal_orbital_and_energies(
                    target.overlap, outputs["hamiltonian"]
                )
                gt_energy, _ = LitModel.cal_orbital_and_energies(
                    target.overlap, target.hamiltonian
                )
                diff = pred_energy - gt_energy

            elif key == "orbital_coefficients":
                # Orbital coefficient loss
                _, pred_coeff = LitModel.cal_orbital_and_energies(
                    target.overlap, outputs["hamiltonian"]
                )
                _, gt_coeff = LitModel.cal_orbital_and_energies(
                    target.overlap, target.hamiltonian
                )
                diff = pred_coeff - gt_coeff

            # Compute error metrics
            mse = torch.mean(diff**2)
            mae = torch.mean(torch.abs(diff))
            error_dict[key + "_mae"] = mae
            error_dict[key + "_rmse"] = torch.sqrt(mse)
            
            # Determine loss combination based on loss type
            if key == "hamiltonian":
                if use_mse_and_mae:
                    loss = mse + mae
                else:
                    loss = mse
            elif key == "waloss":
                loss = mse
            elif key in ["orbital_energies", "orbital_coefficients"]:
                loss = mse + mae
            else:
                loss = mse
                
            # Apply scaling and averaging
            loss = loss * scale
            loss = torch.mean(loss)
            error_dict[key] = loss
            
            # Accumulate total loss
            if "loss" in error_dict:
                error_dict["loss"] += loss_weights[key] * loss
            else:
                error_dict["loss"] = loss_weights[key] * loss

        # Compute time-binned losses for analysis
        for key in loss_weights.keys():
            if key in ["waloss", "orbital_energies", "orbital_coefficients"]:
                continue
            for _bin in TIME_BINS:
                s, e = _bin
                mask = (target.t >= s) & (target.t < e)
                if mask.sum() > 0:  # Only compute if there are samples in this bin
                    diff = outputs[key][mask] - target[key][mask]
                    mse = torch.mean(diff**2)
                    mae = torch.mean(torch.abs(diff))
                    error_dict[key + f"_mae@{s:.2f}_{e:.2f}"] = mae
                    error_dict[key + f"_rmse@{s:.2f}_{e:.2f}"] = torch.sqrt(mse)

        return error_dict

    # ==========================================
    # Forward Pass and Model Interface
    # ==========================================

    def forward(self, batch, H):
        """
        Forward pass through the model.
        
        Args:
            batch: Input batch data
            H: Input Hamiltonian (can be noise or intermediate state)
            
        Returns:
            Model outputs (predicted Hamiltonian or vector field)
        """
        keep_blocks = self.qh9
        output = self.model(batch, H, keep_blocks=keep_blocks)
        
        # Add residual connection if using residual targets
        if self.use_init_hamiltonian_residue:
            if keep_blocks:
                output["hamiltonian_diagonal_blocks"] += batch["diagonal_init_ham"]
                output["hamiltonian_non_diagonal_blocks"] += batch["non_diagonal_init_ham"]
            else:
                output["hamiltonian"] += batch["init_ham"]

        return output

    # ==========================================
    # Training, Validation, and Testing Steps
    # ==========================================

    def training_step(self, batch, batch_idx):
        """
        Single training step for the flow model.
        
        Args:
            batch: Training batch data
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        batch = self.post_processing(batch, self.default_type)
        # Apply corruption strategy
        if self.use_corrupt_mul:
            batch = self.corrupt_mul(batch)
        else:
            batch = self.corrupt(batch, mul=self.batch_mul)
            
        # Forward pass
        outputs = self(batch, batch.init_ham_t)
        self.cur_batch_size = len(batch)
        
        # Compute losses based on mode (finetune or not)
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
        """
        Single validation step for the flow model.
        
        Includes both EMA evaluation (if available) and regular model evaluation,
        plus conditional sampling-based evaluation when loss is below threshold.
        
        Args:
            batch: Validation batch data
            batch_idx: Batch index
            
        Returns:
            Validation errors
        """
        batch = self.post_processing(batch, self.default_type)
        batch_one = batch.clone()
        batch = self.corrupt(batch, mul=self.batch_mul)
        self.cur_batch_size = len(batch)

        # EMA evaluation if available
        if self.ema is not None:
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
                
                # Conditional sampling evaluation for EMA
                if self.error_threshold is None or ema_loss < self.error_threshold:
                    self._log_sample_metric(
                        batch_one, "val", num_timesteps=self.num_ode_steps_val
                    )

        # Regular model evaluation
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
        
        # Conditional sampling evaluation
        if self.error_threshold is None or loss < self.error_threshold:
            for n_steps in self.log_n_steps_ODE_val:
                self._log_sample_metric(batch_one, "val", num_timesteps=n_steps, post_fix=f"_{n_steps}")
            self._log_sample_metric(batch_one, "val", num_timesteps=self.num_ode_steps_val)
            
        return errors

    def test_step(self, batch, batch_idx):
        """
        Single test step for the flow model with multiple modes.
        
        Supports different test modes:
        - inference: Run SCF calculations and detailed analysis
        - test: Standard model evaluation
        - predict: Save predictions for later analysis
        - predict-mul: Multiple prediction evaluation
        
        Args:
            batch: Test batch data
            batch_idx: Batch index
            
        Returns:
            Test errors
        """
        if self.test_mode == "inference":
            return self._inference_step(batch, batch_idx)
        elif self.test_mode == "test":
            return self._test_step_standard(batch, batch_idx)
        elif self.test_mode == "predict":
            return self._predict_step(batch, batch_idx)
        elif self.test_mode == "test-mul":
            return self._test_mul_step(batch, batch_idx)
        else:
            return self._test_step_standard(batch, batch_idx)  # Default to standard test

    def _inference_step(self, batch, batch_idx):
        """Inference step with SCF calculations."""
        assert self.qh9, "Inference is only supported for QH9"
        assert self.test_batch_size == 1, "QH9 test batch size must be 1"
        batch = self.post_processing(batch, self.default_type)
        batch_one = batch.clone()
        self.cur_batch_size = len(batch)
        # Get cached values or initialize
        cycle = getattr(batch_one, "cycle", None)
        init_cycle_time = getattr(batch_one, "init_cycle_time", None)
        ham_calc = getattr(batch_one, "ham_calc", None)
        
        # Build Hamiltonian matrix
        if self.qh9:
            batch_one.full_edge_index = batch_one.edge_index_full
            batch_ham = self.build_final_matrix(
                batch_one,
                batch_one.diagonal_hamiltonian,
                batch_one.non_diagonal_hamiltonian,
            )
            batch_one.hamiltonian = batch_ham.clone()
            batch_ham = batch_ham.cpu().numpy()
        else:
            batch_ham = batch_one.hamiltonian.cpu().numpy()
        
        # Transform to PySCF format
        batch_one.hamiltonian_pyscf = self.matrix_transform(
            batch_ham,
            batch_one,
            convention="e3nn_to_pyscf_def2svp",
        )
        
        batch_one = self.process_target_batch(batch_one)

        # Run SCF if not cached
        if cycle is None:
            init_scf_ret = self.num_scf_steps(batch_one, init_dm_style="1e")
            cycle = init_scf_ret["cycle"]
            init_cycle_time = init_scf_ret["total_time"]
            ham_calc = self.matrix_transform(
                init_scf_ret["fock"],
                batch_one,
                convention="e3nn_to_pyscf_def2svp",
            ).unsqueeze(0)
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
        batch_one = batch.clone()
        batch = self.corrupt(batch, mul=self.batch_mul)
        self.cur_batch_size = len(batch)
        
        # Standard loss computation
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
        
        # Comprehensive sampling evaluation
        if self.qh9:
            # assert self.test_batch_size == 1, "QH9 test batch size must be 1"
            # Use QH9-specific test evaluation
            for n_steps in self.log_n_steps_ODE_test:
                self._log_sample_metric(batch_one, "test", num_timesteps=n_steps, post_fix=f"_{n_steps}")
            self._log_sample_metric(batch_one, "test", num_timesteps=self.num_ode_steps_test)
        else:
            # Standard MD17 evaluation
            for n_steps in self.log_n_steps_ODE_test:
                self._log_sample_metric(batch_one, "test", num_timesteps=n_steps, post_fix=f"_{n_steps}")
            self._log_sample_metric(batch_one, "test", num_timesteps=self.num_ode_steps_test)
            
        return errors



    def _predict_step(self, batch, batch_idx, log_outputs=True):
        """Prediction step that saves outputs."""
        batch = self.post_processing(batch, self.default_type)
        batch_one = batch.clone()
        self.cur_batch_size = len(batch)

        # outputs = self(batch, batch.init_ham_t)
        if hasattr(self, 'output_dir'):
            if not os.path.exists(self.output_dir / "pred"):
                os.makedirs(self.output_dir / "pred", exist_ok=True)
        # log the error if the batch has the ground truth hamiltonian
        if self._batch_has_ground_truth_hamiltonian(batch):
            if hasattr(self, 'output_dir'):
                if not os.path.exists(self.output_dir / "gt"):
                    os.makedirs(self.output_dir / "gt", exist_ok=True)   
        
        if self.qh9:
            # assert self.test_batch_size == 1, "QH9 test batch size must be 1"
            # Save predictions
            sample, traj, _pred = self.sample(batch_one, num_timesteps=self.num_ode_steps_test)
            sample["hamiltonian"] = self.build_final_matrix(
                batch_one,
                sample["hamiltonian_diagonal_blocks"],
                sample["hamiltonian_non_diagonal_blocks"],
                transform=True,
                convention="e3nn_to_pyscf_def2svp",
            )
            gt_overlap = self.build_final_matrix(
                batch_one,
                batch_one.diagonal_overlap,
                batch_one.non_diagonal_overlap,
                transform=True,
                convention="e3nn_to_pyscf_def2svp",
            )
            if self._batch_has_ground_truth_hamiltonian(batch_one):
                gt_hamiltonian = self.build_final_matrix(
                    batch_one,
                    batch_one.diagonal_hamiltonian,
                    batch_one.non_diagonal_hamiltonian,
                    transform=True,
                    convention="e3nn_to_pyscf_def2svp",
                )

            for i in range(self.cur_batch_size):
                pred = {
                    "pred_hamiltonian": sample["hamiltonian"][i].cpu(),
                    "overlap": gt_overlap[i].cpu(),
                    "pos": batch_one[i].pos.cpu(),
                    "atoms": batch_one[i].atoms.squeeze(1).cpu(),
                    "format":"pyscf_def2svp",
                    "length_unit":"angstrom",
                }
                if hasattr(self, 'output_dir'):
                    torch.save(pred, self.output_dir / "pred" / f"pred_{batch_idx}_{i}.pt")
                if self._batch_has_ground_truth_hamiltonian(batch):
                    gt = {
                        "hamiltonian": gt_hamiltonian[i].cpu(),
                        "overlap": gt_overlap[i].cpu(),
                        "pos": batch_one[i].pos.cpu(),
                        "atoms": batch_one[i].atoms.squeeze(1).cpu(),
                        "format":"pyscf_def2svp",
                        "length_unit":"angstrom",
                    }
                    if hasattr(self, 'output_dir'):
                        torch.save(gt, self.output_dir / "gt" / f"gt_{batch_idx}_{i}.pt")
        else:
            sample, traj, _pred = self.sample(batch_one, num_timesteps=self.num_ode_steps_test)
            for i in range(self.cur_batch_size):
                overlap = batch_one[i]["overlap"].squeeze(0).cpu()
                atoms = batch_one[i].atoms.squeeze(1).cpu()
                pos = batch_one[i].pos.cpu()
                pred = {
                    "pred_hamiltonian": sample["hamiltonian"][i].cpu(),
                    "overlap": overlap,                   
                    "pos": pos,
                    "atoms": atoms,
                    "format":"e3nn", # Need to change to pyscf for dft calculation
                    "length_unit":"bohr", # Need to change to angstrom for dft calculation
                }
                if hasattr(self, 'output_dir'):
                    torch.save(pred, self.output_dir / "pred" / f"pred_{batch_idx}_{i}.pt")
                if self._batch_has_ground_truth_hamiltonian(batch_one):
                    gt = {
                        "hamiltonian": batch_one[i].hamiltonian.squeeze(0).cpu(),
                        "overlap": overlap,
                        "pos": pos,
                        "atoms": atoms,
                        "format":"e3nn", # Need to change to pyscf for dft calculation
                        "length_unit":"bohr" # Need to change to angstrom for dft calculation
                    }
                    if hasattr(batch_one[i], "init_ham"):
                        gt["init_ham"] = batch_one[i].init_ham.squeeze(0).cpu()
                    if hasattr(batch_one[i], "energy"):
                        gt["energy"] = batch_one[i].energy.cpu()
                    if hasattr(batch_one[i], "force"):
                        gt["force"] = batch_one[i].force.cpu()
                    if hasattr(self, 'output_dir'):
                        torch.save(gt, self.output_dir / "gt" / f"gt_{batch_idx}_{i}.pt")

        if self._batch_has_ground_truth_hamiltonian(batch_one) and log_outputs:
            metrics = self.metric(sample, batch_one)
            for key in metrics.keys():
                self.log(
                    f"pred/sample_{key}",
                    metrics[key],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True if key == "loss" else False,
                    sync_dist=True,
                    batch_size=self.cur_batch_size,
                )            
        return None

    def _test_mul_step(self, batch, batch_idx):
        """Multiple prediction step for ensemble evaluation."""
        batch = self.post_processing(batch, self.default_type)
        batch_one = batch.clone()
        batch = self.corrupt(batch, mul=self.batch_mul)
        self.cur_batch_size = len(batch)
        
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
            assert self.test_batch_size == 1, "QH9 test batch size must be 1"
            self._log_sample_metric_qh9_mul(
                batch_one, "pred_mul", num_timesteps=self.num_ode_steps_test, mul=self.test_mul
            )
        else:
            raise NotImplementedError("predict-mul is not implemented for md17")
            
        return errors

    # ==========================================
    # Sampling and Generation Methods
    # ==========================================

    def sample(
        self,
        batch,
        num_timesteps=100,
        min_t=DEFAULT_MIN_T,
        sample_random=True,
    ):
        """
        Sample from the flow model using neural ODE integration.
        
        This is the main interface for generation. It starts from noise and
        evolves toward the target distribution using the learned vector field.
        
        Args:
            batch: Input batch data (provides molecular structure)
            num_timesteps: Number of ODE integration steps
            min_t: Minimum time value to start integration
            sample_random: Whether to start from random noise
            
        Returns:
            Tuple of (final_outputs, trajectory, predictions)
        """
        if self.qh9:
            return self.sample_qh9(
                batch,
                num_timesteps=num_timesteps,
                min_t=min_t,
                sample_random=sample_random,
            )
        else:
            return self.sample_md17(
                batch,
                num_timesteps=num_timesteps,
                min_t=min_t,
                sample_random=sample_random,
            )

    def sample_md17(
        self,
        batch,
        num_timesteps=100,
        min_t=DEFAULT_MIN_T,
        sample_random=True,
    ):
        """
        Sample from flow model for MD17 dataset (full matrices).
        
        Uses Euler integration to solve the neural ODE:
        dH/dt = f(H, t) where f is the learned vector field
        
        Args:
            batch: Input batch data
            num_timesteps: Number of integration steps
            min_t: Starting time value
            sample_random: Whether to use random initialization
            
        Returns:
            Tuple of (final_outputs, hamiltonian_trajectory, prediction_trajectory)
        """
        device = self.model.device
        lin_t = torch.linspace(min_t, 1.0, num_timesteps + 1).to(device)
        cur_t = lin_t[0]
        
        # Initialize Hamiltonian trajectory
        batch.init_ham_t = torch.zeros_like(batch.init_ham)
        
        if sample_random:
            # Initialize with structured noise based on init_p0_type
            if self.init_p0_type in ["gauss", "gaussian", "so3"]:
                batch.init_ham_t = torch.randn_like(batch.init_ham) * self.sigma
                
                # Ensure Hermitian symmetry
                i, j = torch.tril_indices(
                    batch.init_ham_t.shape[-1], batch.init_ham_t.shape[-1]
                )
                batch.init_ham_t[:, i, j] = batch.init_ham_t[:, j, i]
                
                if self.init_gauss_center:
                    batch.init_ham_t += batch.init_ham
                    
                if self.init_p0_type == "so3":
                    # Apply random SO(3) rotation
                    random_R = o3.rand_matrix(batch.init_ham.shape[0])
                    batch.random_R = WDs_batch(batch, random_R).to(self.device)
                    batch.init_ham_t = torch.bmm(
                        batch.random_R.transpose(-1, -2),
                        batch.init_ham_t,
                    ).bmm(batch.random_R)
                    
            elif self.init_p0_type == "expand":
                # Equivariant expansion initialization
                random_diag_tensor = o3.Irreps(DEFAULT_IRREPS).randn(
                    batch.atoms.shape[0], -1, device=batch.hamiltonian.device
                )
                random_non_diag_tensor = o3.Irreps(DEFAULT_IRREPS).randn(
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

        # Store trajectories for analysis
        hamiltonian_traj = [batch.init_ham_t.cpu()]
        predictions = [None]  # First prediction is None (initial state)
        
        # Neural ODE integration using Euler method
        for idx, next_t in enumerate(lin_t[1:]):
            batch.t = cur_t.repeat(batch.init_ham.shape[0])
            
            # Get vector field prediction
            outputs = self(batch, batch.init_ham_t)
            dt = next_t - cur_t
            assert dt > 0, "Time step must be positive"
            
            # Compute vector field based on target type
            if self.use_res_target:
                target_H = outputs["hamiltonian"] - batch.init_ham
                cur_H = batch.init_ham_t
            else:
                target_H = outputs["hamiltonian"]
                cur_H = batch.init_ham_t

            # Vector field: dH/dt = (target - current) / (1 - t)
            vector_field = (target_H - cur_H) / (1 - cur_t)

            # Euler step: H_{t+dt} = H_t + vector_field * dt
            ham_t = cur_H + vector_field * dt.reshape(-1, 1, 1)
            
            # Store trajectory
            hamiltonian_traj.append(ham_t.cpu())
            predictions.append(outputs["hamiltonian"].cpu())

            # Update for next iteration
            cur_t = next_t
            batch.init_ham_t = ham_t
            
        # Add residual if using residual targets
        if self.use_res_target:
            ham_t = ham_t + batch.init_ham

        res_outputs = {"hamiltonian": ham_t}
        return res_outputs, hamiltonian_traj, predictions

    def sample_qh9(
        self,
        batch,
        num_timesteps=100,
        min_t=DEFAULT_MIN_T,
        sample_random=True,
    ):
        """
        Sample from flow model for QH9 dataset (block-diagonal structure).
        
        Similar to MD17 sampling but works with block-diagonal representation
        where diagonal and non-diagonal blocks are handled separately.
        
        Args:
            batch: Input batch data with block structure
            num_timesteps: Number of integration steps
            min_t: Starting time value
            sample_random: Whether to use random initialization
            
        Returns:
            Tuple of (final_outputs, hamiltonian_trajectory, prediction_trajectory)
        """
        device = self.model.device
        lin_t = torch.linspace(min_t, 1.0, num_timesteps + 1).to(device)
        cur_t = lin_t[0]
        
        if self.qh9:
            batch.init_ham = batch["diagonal_init_ham"]

        batch.init_ham_t = torch.zeros_like(batch.init_ham)
        random_ham = torch.zeros_like(batch.init_ham_t)
        
        if self.init_gauss:
            if self.init_p0_type in ["gauss", "gaussian", "so3"]:
                random_ham += (
                    torch.randn_like(batch["diagonal_hamiltonian"]) * self.sigma
                )
                
                if self.init_gauss_center:
                    random_ham += batch["diagonal_init_ham"]
                    
                if self.init_p0_type == "so3":
                    random_R = o3.rand_matrix(batch["diagonal_hamiltonian"].shape[0])
                    random_WD = WDs_batch(batch, random_R).to(self.device)
                    random_ham = torch.bmm(
                        random_WD.transpose(-1, -2),
                        random_ham,
                    ).bmm(random_WD)
                    
            elif self.init_p0_type == "expand":
                # Equivariant initialization for QH9
                random_diag_tensor = o3.Irreps(DEFAULT_IRREPS).randn(
                    batch.atoms.shape[0], -1, device=self.device
                )
                random_diag_ham = self.expand(random_diag_tensor)
                random_ham = random_diag_ham * self.sigma
            else:
                raise ValueError(f"Unknown init type: {self.init_p0_type}")
                
            batch.init_ham_t = random_ham
        else:
            # Non-Gaussian initialization options
            if self.init_p0_type == "gauss":
                pass  # Already zeros
            elif self.init_p0_type == "so3":
                identity_R = torch.eye(batch.init_ham.shape[-1]).to(self.device)
                identity_R = identity_R.repeat(batch.init_ham.shape[0], 1, 1)
                batch.init_ham_t = (
                    WDs_batch(batch, identity_R).to(self.device) * self.sigma
                )
            else:
                raise ValueError(f"Unknown init type: {self.init_p0_type}")

        # Store trajectories
        hamiltonian_traj = [batch.init_ham_t.cpu()]
        predictions = [None]  # len(predictions) = num_timesteps + 1
        
        # Neural ODE integration
        for idx, next_t in enumerate(lin_t[1:]):
            batch.t = cur_t.repeat(batch.init_ham.shape[0])
            outputs = self(batch, batch.init_ham_t)
            dt = next_t - cur_t
            assert dt > 0
            
            # Compute vector field for QH9
            if self.qh9:
                if self.use_res_target:
                    target_H = outputs["hamiltonian_diagonal_blocks"]- batch["diagonal_init_ham"]
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
            
            # Store trajectories
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

            # Update state
            cur_t = next_t
            batch.init_ham_t = ham_t
            
        # Add residual if needed
        if self.use_res_target:
            ham_t = ham_t + batch.init_ham

        # Prepare outputs
        if self.qh9:
            res_outputs = {
                "hamiltonian_diagonal_blocks": ham_t,
                "hamiltonian_non_diagonal_blocks": outputs["hamiltonian_non_diagonal_blocks"],
            }
        else:
            res_outputs = {"hamiltonian": ham_t}

        return res_outputs, hamiltonian_traj, predictions

    # ==========================================
    # Logging and Evaluation Methods
    # ==========================================

    def _log_error(self, errors, prefix):
        """
        Log error metrics to the logger.
        
        Handles both regular errors and time-binned errors with appropriate
        organization in the logging hierarchy.
        
        Args:
            errors: Dictionary of error metrics
            prefix: Logging prefix (e.g., "train", "val", "test")
        """
        for key in errors.keys():
            if "@" in key:
                # Time-binned errors get special treatment
                _key, _time_bin = key.split("@")[0], key.split("@")[1]
                self.log(
                    f"{prefix}_{_time_bin}/{_key}_{_time_bin}",
                    errors[key],
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=self.cur_batch_size,
                )
            else:
                # Regular errors
                self.log(
                    f"{prefix}/{key}",
                    errors[key],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True if key == "loss" else False,
                    sync_dist=True,
                    batch_size=self.cur_batch_size,
                )

    def _log_sample_metric(self, batch_one, prefix, num_timesteps=1, post_fix="", save_pred=False, log=True):
        """
        Log sampling-based errors by running full generation and evaluating results.
        
        This is more expensive than direct loss computation but provides better
        insight into actual generation quality.
        
        Args:
            batch_one: Single batch for evaluation
            prefix: Logging prefix
            num_timesteps: Number of ODE steps for sampling
            post_fix: Additional suffix for metric names
            save_pred: Whether to save predictions
            log: Whether to log results
            
        Returns:
            Tuple of (trajectory, sample) if save_pred=True
        """
        try:
            sample, traj, pred = self.sample(batch_one, num_timesteps=num_timesteps)
            if log:
                metrics = self.metric(sample, batch_one)
                for key in metrics.keys():
                    self.log(
                        f"{prefix}/sample_{key}{post_fix}",
                        metrics[key],
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True if key == "loss" else False,
                        sync_dist=True,
                        batch_size=self.cur_batch_size,
                    )
            if save_pred:
                return traj, sample
        except Exception as e:
            logger.error(f"Error in logging sample error: {e}")

    def _log_sample_metric_qh9_mul(self, batch_one, prefix, num_timesteps=1, post_fix="", mul=5):
        """
        Multiple sampling evaluation for ensemble analysis.
        
        Args:
            batch_one: Single batch for evaluation
            prefix: Logging prefix
            num_timesteps: Number of ODE steps
            post_fix: Additional suffix for metric names
            mul: Number of multiple samples to generate
        """
        try:
            all_samples = []
            all_errors = []
            
            for i in range(mul):
                sample, traj, pred = self.sample(batch_one, num_timesteps=num_timesteps)
                error_dicts = self._metric_qh9(sample, batch_one)
                all_samples.append(sample)
                all_errors.append(error_dicts)
            
            # Compute ensemble statistics
            ensemble_errors = {}
            for key in all_errors[0].keys():
                values = [errors[key].item() for errors in all_errors]
                ensemble_errors[f"{key}_mean"] = np.mean(values)
                ensemble_errors[f"{key}_std"] = np.std(values)
                ensemble_errors[f"{key}_min"] = np.min(values)
                ensemble_errors[f"{key}_max"] = np.max(values)
            
            # Log ensemble statistics
            for key in ensemble_errors.keys():
                self.log(
                    f"{prefix}/{key}{post_fix}",
                    ensemble_errors[key],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True if "mean" in key else False,
                    sync_dist=True,
                    batch_size=self.cur_batch_size,
                )
            return all_samples, all_errors
        except Exception as e:
            logger.error(f"Error in logging multiple sample error: {e}")

    def _log_inference_error(
        self,
        batch_one,
        prefix,
        num_timesteps=1,
        init_cycle=None,
        init_cycle_time=None,
        post_fix="",
        ham_error=None,
        e_tot_error=None,
        e_tot_gt=None,
        min_t=DEFAULT_MIN_T,    
    ):
        """Log detailed inference errors including SCF results."""
        sample = self._sample_with_scf(
            batch_one,
            num_timesteps=num_timesteps,
            min_t=min_t,
        )
        results = self.metric(sample, batch_one)
            
        if "sample_time_per_batch" in sample.keys():
            results["sample_time_per_batch"] = sample["sample_time_per_batch"]
            
        if init_cycle is not None:
            scf_ret = build_fock_matrix(
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
            
    def _sample_with_scf(
        self,
        batch,
        num_timesteps=DEFAULT_NUM_ODE_STEPS,
        min_t=DEFAULT_MIN_T,
        sample_random=True,
    ):
        if self.qh9:
            return self._sample_with_scf_qh9(
                batch,
                num_timesteps=num_timesteps,
                min_t=min_t,
                sample_random=sample_random,
            )
        else:
            raise ValueError("Only QH9 is supported for now")

    def _sample_with_scf_qh9(
        self,
        batch,
        num_timesteps=DEFAULT_NUM_ODE_STEPS,
        min_t=DEFAULT_MIN_T,
        sample_random=True,
    ):
        start_time = time.time()
        res, traj, pred = self.sample_qh9(
            batch,
            num_timesteps=num_timesteps,
            min_t=min_t,
            sample_random=sample_random,
        )
        res["sample_time_all"] = time.time() - start_time
        res["sample_time_per_batch"] = time.time() - start_time
        ham_out = self.build_final_matrix(
            batch,
            res["hamiltonian_diagonal_blocks"],
            res["hamiltonian_non_diagonal_blocks"],
        )
        res["hamiltonian"] = ham_out

        mol = pyscf.gto.Mole()
        pos = batch.pos.cpu().numpy()
        atom_list = [
            [batch.atoms[atom_idx].cpu().item(), pos[atom_idx]]
            for atom_idx in range(batch.num_nodes)
        ]
        mol.build(verbose=0, atom=atom_list, basis="def2svp", unit="ang")
        overlap_pyscf = torch.from_numpy(mol.intor("int1e_ovlp")).unsqueeze(0)
        res["mol"] = mol
        res["dm_last"] = self.calc_dm0_from_ham(batch, overlap_pyscf, ham_out.cpu())

        return res

    # ==========================================
    # Dataset Testing Methods (Deprecated)
    # ==========================================

    @torch.no_grad()
    def test_over_dataset(self, test_data_loader, default_type):
        """
        Comprehensive evaluation over entire test dataset.
        
        Automatically detects dataset type and uses appropriate evaluation method.
        
        Args:
            test_data_loader: DataLoader for test data
            default_type: Default tensor type for computation
            
        Returns:
            Tuple of (aggregated_errors, trajectory_samples)
        """
        if self.qh9:
            return self.test_over_dataset_qh9(test_data_loader, default_type)
        else:
            return self.test_over_dataset_md17(test_data_loader, default_type)

    @torch.no_grad()
    def test_over_dataset_md17(self, test_data_loader, default_type):
        """
        Test evaluation for MD17 dataset with full matrix representation.
        
        Args:
            test_data_loader: DataLoader for test data
            default_type: Default tensor type
            
        Returns:
            Tuple of (total_error_dict, last_trajectory)
        """
        self.eval()
        total_error_dict = {"total_items": 0}
        loss_weights = DEFAULT_LOSS_WEIGHTS_MD17.copy()
        total_time = 0
        total_graph = 0
        last_traj = []
        
        logger.info(f"num test data: {len(test_data_loader)}")
        logger.info(f"num ode steps: {self.num_ode_steps_test}")
        
        for idx, batch in tqdm(enumerate(test_data_loader)):
            batch = self.post_processing(batch, default_type)
            batch = batch.to(self.model.device)
            tic = time.time()
            
            # Generate samples using flow model
            outputs, traj, _ = self.sample(
                batch,
                num_timesteps=self.num_ode_steps_test,
                sample_random=self.sample_random,
            )
            last_traj.append(traj[-1])

            duration = time.time() - tic
            total_graph = total_graph + batch.ptr.shape[0] - 1
            total_time = duration + total_time
            
            # Move to CPU for evaluation
            for key in outputs.keys():
                if isinstance(outputs[key], torch.Tensor):
                    outputs[key] = outputs[key].to("cpu")

            batch = batch.to("cpu")
            
            # Compute orbital properties
            outputs["orbital_energies"], outputs["orbital_coefficients"] = (
                self.cal_orbital_and_energies(batch["overlap"], outputs["hamiltonian"])
            )
            batch.orbital_energies, batch.orbital_coefficients = (
                self.cal_orbital_and_energies(batch["overlap"], batch["hamiltonian"])
            )
            
            # Truncate to occupied orbitals only
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
            
            # Compute test errors
            error_dict = self._criterion_test(outputs, batch, loss_weights)
            secs = duration / batch.hamiltonian.shape[0]
            
            # Log progress
            msg = f"batch {idx} / [{len(test_data_loader)}] / {secs*100:.2f}(10^-2)s : "
            for key in error_dict.keys():
                if key == "hamiltonian" or key == "orbital_energies":
                    msg += f"{key}: {error_dict[key]*1e6:.3f}(10^-6), "
                elif key == "orbital_coefficients":
                    msg += f"{key}: {error_dict[key]*1e2:.4f}(10^-2)"
                else:
                    msg += f"{key}: {error_dict[key]:.8f}, "

                # Accumulate errors
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
            
        # Average errors
        for key in total_error_dict.keys():
            if key != "total_items":
                total_error_dict[key] = (
                    total_error_dict[key] / total_error_dict["total_items"]
                )
        last_traj = torch.cat(last_traj, dim=0)
        logger.info(f"num ode steps: {self.num_ode_steps_test}")
        return total_error_dict, last_traj

    @torch.no_grad()
    def test_over_dataset_qh9(self, test_data_loader, default_type):
        """
        Test evaluation for QH9 dataset with block-diagonal structure.
        
        Args:
            test_data_loader: DataLoader for test data
            default_type: Default tensor type
            
        Returns:
            Tuple of (total_error_dict, last_trajectory)
        """
        self.eval()
        total_error_dict = {"total_items": 0}
        loss_weights = DEFAULT_LOSS_WEIGHTS_QH9.copy()
        total_time = 0
        total_graph = 0
        last_traj = []
        
        logger.info("num of test data: {}".format(len(test_data_loader)))
        
        for idx, batch in tqdm(enumerate(test_data_loader)):
            batch = self.post_processing(batch, default_type)
            batch = batch.to(self.model.device)
            tic = time.time()
            
            # Generate samples
            outputs, traj, _ = self.sample(
                batch,
                num_timesteps=self.num_ode_steps_test,
                sample_random=self.sample_random,
            )

            # Also get direct model output for comparison
            outputs = self(batch)
            outputs["hamiltonian"] = self.model.build_final_matrix(
                batch,
                outputs["hamiltonian_diagonal_blocks"],
                outputs["hamiltonian_non_diagonal_blocks"],
            ).cpu()
            batch.hamiltonian = self.model.build_final_matrix(
                batch, batch[0].diagonal_hamiltonian, batch[0].non_diagonal_hamiltonian
            ).cpu()
            
            # Transform to appropriate precision and format
            outputs["hamiltonian"] = outputs["hamiltonian"].type(torch.float64)
            outputs["hamiltonian"] = self.matrix_transform(
                outputs["hamiltonian"],
                batch.atoms.cpu().squeeze().numpy(),
                convention="e3nn_to_pyscf_def2svp",
            )

            last_traj.append(outputs["hamiltonian"])

            batch.hamiltonian = batch.hamiltonian.type(torch.float64)
            batch.hamiltonian = self.matrix_transform(
                batch.hamiltonian,
                batch.atoms.cpu().squeeze().numpy(),
                convention="e3nn_to_pyscf_def2svp",
            )
            
            # Build overlap matrix
            overlap = self.model.build_final_matrix(
                batch, batch[0].diagonal_overlap, batch[0].non_diagonal_overlap
            ).cpu()

            overlap = overlap.type(torch.float64)
            overlap = self.matrix_transform(
                overlap, batch.atoms.cpu().squeeze().numpy(), convention="e3nn_to_pyscf_def2svp"
            )

            # Compute orbital properties
            outputs["orbital_energies"], outputs["orbital_coefficients"] = (
                self.cal_orbital_and_energies(overlap, outputs["hamiltonian"])
            )
            batch.orbital_energies, batch.orbital_coefficients = (
                self.cal_orbital_and_energies(overlap, batch["hamiltonian"])
            )

            # Compute HOMO/LUMO/GAP
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

            # Truncate to occupied orbitals
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

            # Prepare block outputs for evaluation
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
