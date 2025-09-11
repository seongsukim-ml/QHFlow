#!/usr/bin/env python3
"""
Common QH9 fine-tuning utilities for experiments.
"""
import os
from omegaconf import DictConfig
from common.custom_logger import get_logger

logger = get_logger(__file__)

def load_qh9_finetune_dataset(conf: DictConfig, root_path: str):
    """Load QH9 dataset for fine-tuning with orbital and energies calculation."""
    from dataset_module.qh9_datasets_split import QH9Stable, QH9Dynamic
    
    dataset_name = conf.dataset.dataset_name
    logger.info(f"Loading {dataset_name} dataset for fine-tuning...")
    
    if dataset_name == "QH9Stable":
        dataset = QH9Stable(
            os.path.join(root_path, "dataset"),
            split=conf.dataset.split,
            cal_orbital_and_energies=True,
        )
    elif dataset_name == "QH9Dynamic":
        dataset = QH9Dynamic(
            os.path.join(root_path, "dataset"),
            split=conf.dataset.split,
            version=conf.dataset.version,
            cal_orbital_and_energies=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def setup_qh9_finetune_model(conf: DictConfig, pl_model_cls, output_dir):
    """Setup QH9 fine-tuning model with pretrained weights."""
    from common.checkpoint_utils import _find_best_checkpoint
    
    # Get original checkpoint
    original_ckpt = conf.get("original_ckpt")
    if original_ckpt is None or original_ckpt == "None":
        original_ckpt = None
        logger.info("No original checkpoint provided")
    logger.info(f"Original checkpoint: {original_ckpt}")

    # Find best checkpoint for resuming
    ckpt_path = _find_best_checkpoint(conf, output_dir)
    
    # If no checkpoint found, use original checkpoint
    if ckpt_path is None or ckpt_path == "":
        ckpt_path = original_ckpt
    
    logger.info(f"Checkpoint path: {ckpt_path}")
    
    # Load model based on checkpoint type
    if ckpt_path is not None:
        if ckpt_path == original_ckpt:
            print("Using pretrained model as weights")
            lit_model_pretrained = pl_model_cls.load_from_checkpoint(
                ckpt_path, conf=conf, strict=False
            )
            lit_model = pl_model_cls(conf)
            lit_model.model = lit_model_pretrained.model
            load_model_ckpt = None
        else:
            lit_model = pl_model_cls.load_from_checkpoint(
                ckpt_path, conf=conf, strict=False
            )
            load_model_ckpt = ckpt_path
    else:
        lit_model = pl_model_cls(conf)
        load_model_ckpt = None
    
    return lit_model, ckpt_path, load_model_ckpt


def validate_qh9_finetune_config(conf: DictConfig):
    """Validate QH9 fine-tuning configuration."""
    if conf.get("num_training_steps") is None:
        raise ValueError("num_training_steps must be specified for fine-tuning")
    
    # Check if dataset supports fine-tuning
    dataset_name = conf.dataset.dataset_name
    if dataset_name not in ["QH9Stable", "QH9Dynamic"]:
        raise ValueError(f"Dataset {dataset_name} does not support fine-tuning")
