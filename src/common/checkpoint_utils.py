"""
Common checkpoint utilities for experiments.
"""
import os
from pathlib import Path
from omegaconf import DictConfig
from common.custom_logger import set_logger

logger = set_logger()


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
    # Check for latest run
    ckpt_path = output_dir / "checkpoints"
    ckpt_path_list = list(ckpt_path.glob("*.ckpt"))
    if conf.get("use_last_ckpt", False):
        ckpt_path_last = list(ckpt_path.glob("last.ckpt"))
        if len(ckpt_path_last) == 0:
            logger.info("No last checkpoint found, using None")
            return None
        else:
            logger.info(f"Using last checkpoint: {ckpt_path_last[0]}")
            return ckpt_path_last[0]
    else:
        ckpt_path_list = [
            path for path in ckpt_path_list if "weights" in path.stem
        ]
        logger.info(f"Found {len(ckpt_path_list)} checkpoints")
        if len(ckpt_path_list) > 0:
            try:
                ckpt_path_list_sorted = sorted(
                    ckpt_path_list,
                    key=lambda x: int(x.stem.split("-")[1].split("=")[1]),
                    reverse=True,
                )
                return ckpt_path_list_sorted[0]
            except:
                logger.info("Error in finding the best checkpoint")
                return None
    logger.info("No checkpoints found, using None")
    return None


def setup_wandb_logger(conf: DictConfig, output_dir: Path):
    """Setup Weights & Biases logger."""
    from pytorch_lightning.loggers import WandbLogger
    import omegaconf
    
    # Create wandb directory
    os.makedirs(output_dir / "wandb", exist_ok=True)

    resume = conf.wandb.get("resume", "allow")
    if resume == "allow" or resume == "must":
        if (output_dir / "wandb" / "latest-run").exists():
            run_id = [
                file.name
                for file in (output_dir / "wandb" / "latest-run").iterdir()
                if "wandb" in file.name
            ][0][4:12]
        else:
            run_id = conf.wandb.get("run_id", None)
            run_id = None if run_id == "Null" or run_id == "null" else run_id
        logger.info(f"run_id: {run_id}")

    wandb_logger = WandbLogger(
        project=conf.wandb.project,
        name=conf.wandb.run_name,
        entity=conf.wandb.get("entity", None),
        save_dir=output_dir,
        mode=getattr(conf.wandb, "mode", "online"),
        id=run_id,
        tags=getattr(conf.wandb, "tags", None),
        resume=resume,
    )

    # Log hyperparameters
    wandb_config = omegaconf.OmegaConf.to_container(
        conf, resolve=True, throw_on_missing=True
    )
    wandb_logger.log_hyperparams(wandb_config)
    
    return wandb_logger
