#!/usr/bin/env python3
"""
Common training utilities for experiments.
"""
import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import DictConfig
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_callbacks(conf: DictConfig, output_dir: Path):
    """Setup training callbacks."""
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="weights-{epoch:02d}-{val_loss:.2f}", # {epoch:02d} is the number of epoch
        monitor=conf.get("monitor", "val/loss"),
        mode=conf.get("monitor_mode", "min"),
        save_top_k=conf.get("save_top_k", 1),
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_JOIN_CHAR = "#"
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    return callbacks


def setup_logger(conf: DictConfig, output_dir: Path):
    """Setup logging configuration."""
    loggers = []
    
    # Weights & Biases logger
    if conf.get("wandb", {}).get("enabled", False):
        wandb_logger = WandbLogger(
            project=conf.wandb.get("project", "QHFlow"),
            name=conf.wandb.get("name", None),
            log_model=conf.wandb.get("log_model", False),
            save_dir=output_dir,
        )
        loggers.append(wandb_logger)
    
    return loggers


def setup_trainer(conf: DictConfig, callbacks, loggers, output_dir: Path):
    """Setup PyTorch Lightning trainer."""
    
    trainer_kwargs = {
        "accelerator": conf.get("accelerator", "gpu"), # deprecated in v1.5.0
        "devices": conf.get("devices", 1),
        "precision": conf.get("precision", 32),
        "callbacks": callbacks,
        "logger": loggers,
        "enable_progress_bar": conf.get("enable_progress_bar", True),
        "enable_checkpointing": conf.get("enable_checkpointing", True),
        "log_every_n_steps": conf.get("log_every_n_steps", 50),
        "num_nodes": conf.get("num_nodes", 1),
        "num_sanity_val_steps": conf.get("num_sanity_val_steps", 2),
        "profiler": "simple" if conf.get("profiler", "simple") == "simple" else None,
    }
    
    trainer_kwargs["max_steps"] = conf.get("num_training_steps", -1)
    
    if trainer_kwargs["max_steps"] == -1:
        trainer_kwargs["max_epochs"] = conf.get("max_epochs", 1000)   
    # Add gradient clipping if specified
    if conf.get("gradient_clip_val"):
        trainer_kwargs["gradient_clip_val"] = conf.gradient_clip_val
    
    # Add strategy if specified
    if conf.get("strategy"):
        trainer_kwargs["strategy"] = conf.strategy
        
    if conf.get("data_type") == "float64":
        trainer_kwargs["precision"] = 64
    
    trainer = pl.Trainer(**trainer_kwargs)
    return trainer


def log_training_config(conf: DictConfig):
    """Log training configuration."""
    logger.info("Training Configuration:")
    logger.info(f"  Max steps: {conf.get('num_training_steps', -1)}")
    if conf.get("num_training_steps", -1) == -1:
        logger.info(f"  Max epochs: {conf.get('max_epochs', 1000)}")
    # logger.info(f"  Accelerator: {conf.get('accelerator', 'auto')}")
    logger.info(f"  Strategy: {conf.get('strategy', "None")}")
    logger.info(f"  Devices: {conf.get('devices', 1)}")
    logger.info(f"  Data type: {conf.get('data_type', 'float32')}")
