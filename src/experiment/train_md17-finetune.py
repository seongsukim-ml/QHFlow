#!/usr/bin/env python3
import hydra
import os
from pathlib import Path
from dataset_module.ori_dataset import MD17_DFT, random_split, get_mask

# Import common utilities
from common.setup import (
    setup_paths, setup_auxiliary_basis, setup_tensor_type_and_seed,
    get_root_path, get_mode
)
from common.data_utils import create_md17_data_loaders, log_dataset_info, load_md17_dataset
from common.training_utils import setup_callbacks, setup_logger, setup_trainer, log_training_config
from common.finetune_utils import setup_finetune_model, validate_finetune_config

# Setup paths and import models
setup_paths()

import warnings
warnings.filterwarnings("ignore")

from pl_module import get_pl_model

import torch

from common.custom_logger import get_logger
logger = get_logger(__file__)

NUM_THREADS = 16
logger.info(f"Using torch.set_num_threads({NUM_THREADS})")
torch.set_num_threads(NUM_THREADS)

@hydra.main(config_path="../config_md17", config_name="config_flow-lw10-wa")
def main(conf):
    # Validate fine-tuning configuration
    validate_finetune_config(conf)
    
    # Setup output directory and auxiliary basis
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    setup_auxiliary_basis(output_dir)

    # Setup tensor type and seed
    setup_tensor_type_and_seed(conf)

    # Load the dataset
    root_path = get_root_path()
    logger.info(f"Loading {conf.dataset.dataset_name} dataset...")
    
    dataset = load_md17_dataset(conf, root_path)
    
    train_loader, val_loader, test_loader = create_md17_data_loaders(dataset, conf)

    # Log dataset information
    log_dataset_info(dataset, train_loader.dataset, val_loader.dataset, test_loader.dataset)

    # Initialize the LightningModule
    pl_model_cls = get_pl_model(conf)
    logger.info(f"Using model: {pl_model_cls}")

    # Get and validate mode
    mode = get_mode(conf)
    assert mode in ["train", "test", "eval", "inference", "predict"]

    if mode in ["train", "test", "inference", "predict"]:
        # Setup fine-tuning model
        lit_model, ckpt_path = setup_finetune_model(conf, pl_model_cls, output_dir)
        setattr(lit_model, "test_mode", "test")
        
        # Import checkpoint utilities
        from common.checkpoint_utils import setup_wandb_logger
        
        # Setup wandb logger
        wandb_logger = setup_wandb_logger(conf, output_dir)
        wandb_logger.watch(model=lit_model, log_freq=500)

        # Setup callbacks and trainer
        callbacks = setup_callbacks(conf, output_dir)
        trainer = setup_trainer(conf, callbacks, [wandb_logger], output_dir)
        log_training_config(conf)
        
        # Start training/testing
        _run_finetune_training_or_testing(mode, trainer, lit_model, train_loader, val_loader, test_loader, ckpt_path, conf)

    elif mode == "eval":
        _run_evaluation(conf, pl_model_cls, test_loader, output_dir)


def _run_finetune_training_or_testing(mode, trainer, lit_model, train_loader, val_loader, test_loader, ckpt_path, conf):
    """Run fine-tuning training or testing based on mode."""
    
    if mode == "train":
        trainer.fit(
            lit_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        logger.info("Testing...")
        trainer.test(lit_model, test_loader, ckpt_path="best")
    elif mode == "test":
        # Test the model with the test metric (smaller metrics)
        trainer.test(lit_model, test_loader, ckpt_path=ckpt_path)
    elif mode in ["inference", "predict"]:
        # predict the model with many metric (larger metrics)
        from torch_geometric.loader import DataLoader
        inf_loader = DataLoader(
            test_loader.dataset[:300],
            batch_size=1,
            shuffle=False,
            num_workers=conf.dataset.num_workers,
            pin_memory=conf.dataset.pin_memory,
        )
        setattr(lit_model, "test_mode", "inference")
        logger.info(f"{lit_model.test_mode}...")
        trainer.test(lit_model, inf_loader, ckpt_path=ckpt_path)


def _run_evaluation(conf, pl_model_cls, test_loader, output_dir):
    """Run evaluation mode."""    
    model_ckpt = conf.model_ckpt
    lit_model = pl_model_cls.load_from_checkpoint(model_ckpt, conf=conf)
    logger.info("Model loaded")
    
    logger.info("Testing...")
    default_type = torch.float64 if conf.data_type == "float64" else torch.float32
    errors, h_output = lit_model.test_over_dataset(test_loader, default_type)
    
    msg = f"dataset {conf.dataset.dataset_name}: {errors.get('total_items')} :"
    for key in errors.keys():
        if key == "hamiltonian" or key == "orbital_energies":
            msg += f"{key}: {errors[key]*1e6:.3f}(10^-6), "
        elif key == "orbital_coefficients":
            msg += f"{key}: {errors[key]*1e2:.4f}(10^-2)"
        elif key == "total_items":
            msg += f"{key}: {errors[key]:d}, "
        else:
            msg += f"{key}: {errors[key]:.8f}, "
    
    logger.info(msg)
    output_dir_name = "output"
    os.makedirs(output_dir / output_dir_name, exist_ok=True)
    with open(output_dir / output_dir_name / "results.txt", "w") as f:
        f.write(msg)
    torch.save(h_output, output_dir / output_dir_name / "h_output.pt")


if __name__ == "__main__":
    main()
