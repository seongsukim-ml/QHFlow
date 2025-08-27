#!/usr/bin/env python3
"""
Common checkpoint utilities for experiments.
"""
import os
import logging
from pathlib import Path
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def get_checkpoint_path(conf: DictConfig, output_dir: Path):
    """Get checkpoint path from configuration or find the best checkpoint."""
    ckpt_path = conf.get("ckpt", "")

    if ckpt_path == "":
        ckpt_path = None
    
    logger.info(f"ckpt_path: {ckpt_path if ckpt_path is not None else 'None'}")
    
    if ckpt_path is None:
        ckpt_path = _find_best_checkpoint(conf, output_dir)
    
    return ckpt_path


def _find_best_checkpoint(conf: DictConfig, output_dir: Path):
    """Find the best checkpoint from wandb directory."""
    run_id = None
    
    # Check for latest run
    if (output_dir / "wandb" / "latest-run").exists():
        run_id = [
            file.name
            for file in (output_dir / "wandb" / "latest-run").iterdir()
            if "wandb" in file.name
        ][0][4:12]
    elif conf.wandb.run_id is not None and conf.wandb.run_id != "":
        run_id = conf.wandb.run_id

    if run_id is not None:
        ckpt_path = output_dir / conf.wandb.project / run_id / "checkpoints"
        ckpt_path_list = list(ckpt_path.glob("*.ckpt"))
        ckpt_path_list = [
            path for path in ckpt_path_list if "best" in path.stem
        ]
        logger.info(f"Found {len(ckpt_path_list)} checkpoints")
        
        if len(ckpt_path_list) > 0:
            ckpt_path_list = sorted(
                ckpt_path_list, key=lambda x: int(x.stem.split("-")[1].split("#")[1])
            )
            return ckpt_path_list[-1]
    
    return None


def setup_wandb_logger(conf: DictConfig, output_dir: Path, run_id=None):
    """Setup Weights & Biases logger."""
    from pytorch_lightning.loggers import WandbLogger
    import omegaconf
    
    # Create wandb directory
    os.makedirs(output_dir / "wandb", exist_ok=True)
    
    wandb_logger = WandbLogger(
        project=conf.wandb.project,
        name=conf.wandb.run_name,
        save_dir=output_dir,
        mode=getattr(conf.wandb, "mode", "online"),
        id=run_id,
        tags=getattr(conf.wandb, "tags", None),
        resume="allow",
    )

    # Log hyperparameters
    wandb_config = omegaconf.OmegaConf.to_container(
        conf, resolve=True, throw_on_missing=True
    )
    wandb_logger.log_hyperparams(wandb_config)
    
    return wandb_logger
