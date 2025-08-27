#!/usr/bin/env python3
"""
Common data utilities for experiments.
"""
import os
import logging
from torch_geometric.loader import DataLoader
from omegaconf import DictConfig
from dataset_module.ori_dataset import MD17_DFT, random_split, get_mask

logger = logging.getLogger(__name__)

def load_md17_dataset(conf: DictConfig, root_path: str):
    return MD17_DFT(
        os.path.join(root_path, "dataset"),
        name=conf.dataset.dataset_name,
        transform=get_mask,
    )

def create_md17_data_loaders(dataset, conf: DictConfig, batch_size=[None, None, None]):
    """Create train, validation, and test data loaders."""
    train_dataset, valid_dataset, test_dataset = _create_md17_dataset(dataset, conf)
    return _create_md17_data_loaders(train_dataset, valid_dataset, test_dataset, conf, batch_size)

def _create_md17_dataset(dataset, conf: DictConfig):
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset,
        [
            conf.dataset.num_train,
            conf.dataset.num_valid,
            len(dataset) - (conf.dataset.num_train + conf.dataset.num_valid),
        ],
        seed=conf.split_seed,
    )

    return train_dataset, valid_dataset, test_dataset

def _create_md17_data_loaders(train_dataset, valid_dataset, test_dataset, conf: DictConfig, batch_size=[None, None, None]):
    """Create train, validation, and test data loaders."""
    
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
    
    return train_loader, val_loader, test_loader


def get_dataset_path(root_path: str, dataset_name: str):
    """Get the dataset path."""
    return os.path.join(root_path, "dataset")


def log_dataset_info(dataset, train_dataset, valid_dataset, test_dataset):
    """Log dataset information."""
    logger.info(f"Dataset size:      {len(dataset):>8}")
    logger.info(f"Train size:        {len(train_dataset):>8} ({len(train_dataset) / len(dataset):.2%})")
    logger.info(f"Validation size:   {len(valid_dataset):>8} ({len(valid_dataset) / len(dataset):.2%})")    
    logger.info(f"Test size:         {len(test_dataset):>8} ({len(test_dataset) / len(dataset):.2%})")
