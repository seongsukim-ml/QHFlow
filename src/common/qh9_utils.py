#!/usr/bin/env python3
"""
Common QH9 utilities for experiments.
"""
import logging
import os
from torch_geometric.loader import DataLoader
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def load_qh9_dataset(conf: DictConfig, root_path: str):
    """Load QH9 dataset based on configuration."""
    from dataset_module.qh9_datasets_split import QH9Stable, QH9Dynamic
    
    dataset_name = conf.dataset.dataset_name
    logger.info(f"Loading {dataset_name} dataset...")
    
    if dataset_name == "QH9Stable":
        dataset = QH9Stable(
            os.path.join(root_path, "dataset"),
            split=conf.dataset.split,
        )
    elif dataset_name == "QH9Dynamic":
        dataset = QH9Dynamic(
            os.path.join(root_path, "dataset"),
            split=conf.dataset.split,
            version=conf.dataset.version,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def create_qh9_data_loaders(dataset, conf: DictConfig):
    """Create train, validation, and test data loaders for QH9."""
    train_dataset = dataset[dataset.train_mask]
    valid_dataset = dataset[dataset.val_mask]
    test_dataset = dataset[dataset.test_mask]
    
    # Handle partial validation if specified
    if getattr(conf, "partial_val", None) is not None:
        assert conf.partial_val > 0 and conf.partial_val <= 1
        valid_dataset = valid_dataset[: int(len(valid_dataset) * conf.partial_val)]
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=conf.dataset.train_batch_size,
        shuffle=True,
        num_workers=conf.dataset.num_workers,
        pin_memory=conf.dataset.pin_memory,
    )
    
    val_loader = DataLoader(
        valid_dataset,
        batch_size=conf.dataset.train_batch_size,
        shuffle=False,
        num_workers=conf.dataset.num_workers,
        pin_memory=conf.dataset.pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=conf.dataset.test_batch_size,
        shuffle=False,
        num_workers=conf.dataset.num_workers,
        pin_memory=conf.dataset.pin_memory,
    )
    
    return train_loader, val_loader, test_loader, train_dataset, valid_dataset, test_dataset


def setup_warmup_training(conf: DictConfig, lit_model, train_dataset, wandb_logger, callbacks):
    """Setup warmup training for Real_QHNet if needed."""
    import pytorch_lightning as pl
    from torch_geometric.loader import DataLoader
    
    # Check if warmup is needed for Real_QHNet
    if (
        conf.model.version.lower() == "Real_QHNet".lower()
        and conf.get("warmup_step") is not None
        and conf.get("mode", "train") != "test"
    ):
        logger.info("Warmup training for Real_QHNet")
        
        # Store original learning rate
        real_lr = conf.dataset.learning_rate
        warmup_lr = 1e-3
        conf.dataset.learning_rate = warmup_lr
        
        # Create warmup trainer
        warmup_trainer = pl.Trainer(
            max_steps=conf.warmup_step,
            logger=wandb_logger,
            callbacks=callbacks,
            precision=64 if conf.data_type == "float64" else 32,
            log_every_n_steps=conf.dataset.train_batch_interval,
            accelerator="auto",
            devices=1,
            enable_progress_bar=True,
            gradient_clip_val=5.0,
        )
        
        # Create warmup data loader with smaller batch size
        train_loader_warmup = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=conf.dataset.num_workers,
            pin_memory=conf.dataset.pin_memory,
        )
        
        # Run warmup training
        warmup_trainer.fit(
            lit_model,
            train_dataloaders=train_loader_warmup,
        )
        
        # Restore original learning rate
        conf.dataset.learning_rate = real_lr
        
        return True
    
    return False


def create_inference_loader(mode: str, test_dataset, conf: DictConfig):
    """Create inference data loader based on mode."""
    if mode == "inference":
        inf_loader = DataLoader(
            test_dataset[:300],
            batch_size=1,
            shuffle=False,
            num_workers=conf.dataset.num_workers,
            pin_memory=conf.dataset.pin_memory,
        )
    elif mode == "predict-mul":
        inf_loader = DataLoader(
            test_dataset,
            batch_size=conf.dataset.test_batch_size,
            shuffle=False,
            num_workers=conf.dataset.num_workers,
            pin_memory=conf.dataset.pin_memory,
        )
    elif mode == "predict":
        inf_loader = DataLoader(
            test_dataset,
            batch_size=conf.dataset.test_batch_size,
            shuffle=False,
            num_workers=conf.dataset.num_workers,
            pin_memory=conf.dataset.pin_memory,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return inf_loader
