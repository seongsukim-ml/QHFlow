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

# Ex: /mnt/QHFlow/src
DEFAULT_SRC_PATH  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def setup_paths(src_path: str=None):
    """Setup Python path and get src directory."""
    # Get the absolute path to the parent directory
    src_path = DEFAULT_SRC_PATH if src_path is None else src_path  
    # Insert the parent directory at the beginning of sys.path
    sys.path.insert(0, src_path)
    logger.info(f"Add src_path to sys.path: {src_path}")
    return src_path


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


def get_root_path(conf: DictConfig=None):
    """Get the root path for dataset loading."""
    root_path = DEFAULT_ROOT_PATH if conf is None else conf.get("root_path", DEFAULT_ROOT_PATH)
    logger.info(f"Root path: {root_path}")
    if not os.path.exists(root_path):
        logger.error(f"Root path {root_path} does not exist")
        raise FileNotFoundError(f"Root path {root_path} does not exist")
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
