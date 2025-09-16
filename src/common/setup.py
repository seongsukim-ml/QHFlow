#!/usr/bin/env python3
"""
Common setup utilities for experiments.
"""
import os
import sys
import torch
import shutil
import subprocess
import pytorch_lightning as pl
from pathlib import Path
from omegaconf import DictConfig
from common.custom_logger import get_logger

logger = get_logger(__file__)

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

# Ex: /mnt/QHFlow
DEFAULT_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))
# Ex: /mnt/QHFlow/src
DEFAULT_SRC_PATH  = os.path.join(DEFAULT_ROOT_PATH, "src")

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


def get_dataset_path(conf: DictConfig=None):
    """Get the root path for dataset loading."""
    data_path = DEFAULT_ROOT_PATH if conf is None else conf.get("dataset_path", DEFAULT_DATASET_PATH)
    logger.info(f"Dataset path: {data_path}")
    if not os.path.exists(data_path):
        logger.error(f"Dataset path {data_path} does not exist")
        raise FileNotFoundError(f"Dataset path {data_path} does not exist")
    return data_path


""" Deprecated / Used in old code """
def get_root_path(conf: DictConfig=None):
    return get_dataset_path(conf)


def get_mode(conf: DictConfig):
    """Get the current mode from configuration."""
    mode = conf.get("mode", "train")
    return MODE_DICT.get(mode, mode)