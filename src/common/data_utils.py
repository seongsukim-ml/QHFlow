#!/usr/bin/env python3
"""
Common data utilities for experiments.
"""
import os
import logging
from torch_geometric.loader import DataLoader
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def create_data_loaders(train_dataset, valid_dataset, test_dataset, conf: DictConfig):
    """Create train, validation, and test data loaders."""
    
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
    
    return train_loader, val_loader, test_loader


def get_dataset_path(root_path: str, dataset_name: str):
    """Get the dataset path."""
    return os.path.join(root_path, "dataset")


def log_dataset_info(dataset, train_dataset, valid_dataset, test_dataset):
    """Log dataset information."""
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Validation size: {len(valid_dataset)}")
    logger.info(f"Test size: {len(test_dataset)}")
