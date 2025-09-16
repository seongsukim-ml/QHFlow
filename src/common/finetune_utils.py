"""
Common fine-tuning utilities for experiments.
"""
from pathlib import Path
from omegaconf import DictConfig
from common.custom_logger import get_logger

logger = get_logger(__file__)

def setup_finetune_model(conf: DictConfig, pl_model_cls, output_dir: Path):
    """Setup fine-tuning model with pretrained weights."""
    from common.checkpoint_utils import _find_best_checkpoint
    
    # Get original checkpoint
    original_ckpt = conf.get("original_ckpt")
    if original_ckpt is None:
        raise ValueError("original_ckpt is None")
    logger.info(f"Original checkpoint: {original_ckpt}")

    # Find best checkpoint for resuming
    ckpt_path = _find_best_checkpoint(conf, output_dir)
    
    # If no checkpoint found, use original checkpoint
    if ckpt_path is None:
        ckpt_path = original_ckpt
    
    logger.info(f"Checkpoint path: {ckpt_path}")
    
    # Load model based on checkpoint type
    if ckpt_path == original_ckpt or ckpt_path == "":
        # Load pretrained model and transfer weights
        lit_model_pretrained = pl_model_cls.load_from_checkpoint(
            ckpt_path, conf=conf, strict=False
        )
        lit_model = pl_model_cls(conf)
        lit_model.model = lit_model_pretrained.model
    else:
        # Load fine-tuned model
        lit_model = pl_model_cls.load_from_checkpoint(
            ckpt_path, conf=conf, strict=False
        )
    
    return lit_model, ckpt_path


def validate_finetune_config(conf: DictConfig):
    """Validate fine-tuning configuration."""
    if conf.get("original_ckpt") is None:
        raise ValueError("original_ckpt must be specified for fine-tuning")
    
    if conf.get("num_training_steps") is None:
        raise ValueError("num_training_steps must be specified for fine-tuning")
