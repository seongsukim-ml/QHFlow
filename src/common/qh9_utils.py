#!/usr/bin/env python3
"""
Common QH9 utilities for experiments.
"""
import logging
import os
from torch_geometric.loader import DataLoader
from omegaconf import DictConfig
from torch.utils.data import DistributedSampler
import torch.distributed as dist

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

def _create_qh9_dataset(dataset, conf: DictConfig):
    train_dataset = dataset[dataset.train_mask]
    valid_dataset = dataset[dataset.val_mask]
    test_dataset = dataset[dataset.test_mask]
    
    return train_dataset, valid_dataset, test_dataset

def create_qh9_data_loaders(dataset, conf: DictConfig, batch_size=[None, None, None]):
    """Create train, validation, and test data loaders for QH9."""
    train_dataset, valid_dataset, test_dataset = _create_qh9_dataset(dataset, conf)
    return _create_qh9_data_loaders(train_dataset, valid_dataset, test_dataset, conf, batch_size)

def _create_qh9_data_loaders(train_dataset, valid_dataset, test_dataset, conf: DictConfig, batch_size=[None, None, None]):
    # Handle partial validation if specified
    if getattr(conf, "partial_val", None) is not None:
        assert conf.partial_val > 0 and conf.partial_val <= 1
        original_valid_size = len(valid_dataset)
        valid_dataset = valid_dataset[: int(len(valid_dataset) * conf.partial_val)]
        print(f"Using partial validation: {conf.partial_val} ({original_valid_size} -> {len(valid_dataset)}) (for speed up)")
    
    train_batch_size = conf.dataset.train_batch_size
    valid_batch_size = conf.dataset.valid_batch_size
    test_batch_size = conf.dataset.test_batch_size
    
    if batch_size is not None:
        if isinstance(batch_size, int):
            batch_size = [batch_size, batch_size, batch_size]
        if len(batch_size) == 1:
            batch_size = batch_size * 3
        if len(batch_size) == 2:
            batch_size = batch_size + [batch_size[-1]]
        if len(batch_size) != 3:
            raise ValueError(f"Batch size must be a int or a list of 1, 2, 3 elements: {batch_size}")
        if batch_size[0] is not None: 
            train_batch_size = batch_size[0]
            print(f"Using custom train batch size: {train_batch_size} instead of config batch size {conf.dataset.train_batch_size}")
        if batch_size[1] is not None:
            valid_batch_size = batch_size[1]
            print(f"Using custom valid batch size: {valid_batch_size} instead of config batch size {conf.dataset.valid_batch_size}")
        if batch_size[2] is not None:
            test_batch_size = batch_size[2]
            print(f"Using custom test batch size: {test_batch_size} instead of config batch size {conf.dataset.test_batch_size}")

    use_ddp = conf.get("strategy", "None") == "ddp"
    if not use_ddp:
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=conf.dataset.num_workers,
            pin_memory=conf.dataset.pin_memory,
        )
        
        val_loader = DataLoader(
            valid_dataset,
            batch_size=valid_batch_size,
            shuffle=False,
            num_workers=conf.dataset.num_workers,
            pin_memory=conf.dataset.pin_memory,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=conf.dataset.num_workers,
            pin_memory=conf.dataset.pin_memory,
        )
    else:
        # Potential issue with automatic sampler replacement
        # By default, Lightning replaces shuffle=True with DistributedSampler in DDP mode.
        # However, this replacement may not work reliably when using PyG's torch_geometric.loader.DataLoader (subclass)
        # combined with list/mask indexing for subsets. This can cause each rank to process all data
        # or have different batch counts, leading to hangs at all-reduce synchronization points.

        # Batch size imbalance
        # When drop_last=False, the last batch size can vary between ranks.
        # This can cause step count mismatches when combined with Lightning's collect/sync logic, leading to hangs.

        # Worker explosion/pipeline bottleneck 
        # If num_workers × number of GPUs is too high, a setup that works fine with 2 GPUs
        # may hang with 4 GPUs due to IPC/file lock/CPU saturation.
        print("Using DDP")
        print("Num workers: ", conf.dataset.num_workers)
        # sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=conf.dataset.pin_memory,
            persistent_workers=False,
            drop_last=True,
            # sampler=sampler,
        )
        val_loader = DataLoader(
            valid_dataset,
            batch_size=valid_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=conf.dataset.pin_memory,
            persistent_workers=False,
            drop_last=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=conf.dataset.pin_memory,
        )
    
    return train_loader, val_loader, test_loader

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
            num_sanity_val_steps=8,
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
