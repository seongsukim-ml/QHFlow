#!/usr/bin/env python3
"""
Common setup utilities for experiments.
"""
import os
import sys
import torch
import logging
import shutil
import subprocess
import pytorch_lightning as pl
from pathlib import Path
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Mode mapping dictionary
MODE_DICT = {
    "train": "train",
    "test": "test",
    "eval": "eval",
    "evaluation": "eval",
    "inference": "inference",
    "infer": "inference",
    "predict": "predict",
    "pred": "predict",
}


def setup_paths():
    """Setup Python path and get root directory."""
    # Get the absolute path to the parent directory
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Insert the parent directory at the beginning of sys.path
    sys.path.insert(0, parent_dir)
    return parent_dir


def setup_auxiliary_basis(output_dir: Path):
    """Copy auxiliary basis file and set basis."""
    shutil.copy(os.path.join(os.path.dirname(__file__), "..", "experiment", "auxiliary.gbs"), output_dir)
    logger.info("Copied auxiliary basis file")
    cmd = subprocess.Popen("set basis AUXILIARY", shell=True)
    cmd.wait()
    logger.info("Set auxiliary basis")


def setup_tensor_type_and_seed(conf: DictConfig):
    """Setup tensor type and random seed."""
    # Set the default tensor type
    default_type = torch.float64 if conf.data_type == "float64" else torch.float32
    torch.set_default_dtype(default_type)
    
    if default_type == torch.float32:
        # Enable TensorFloat32 for better performance on Ampere GPUs
        logger.info("Setting float32 matmul precision to high")
        torch.set_float32_matmul_precision('high')

    # Set random seed
    seed = conf.get("seed", 0)
    logger.info(f"Seed: {seed}")
    pl.seed_everything(seed)


def get_root_path():
    """Get the root path for dataset loading."""
    root_path = os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-4]))
    logger.info(f"Root path: {root_path}")
    return root_path


def get_mode(conf: DictConfig):
    """Get the current mode from configuration."""
    mode = conf.get("mode", "train")
    return MODE_DICT.get(mode, mode)


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
