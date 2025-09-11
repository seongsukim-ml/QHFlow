#!/usr/bin/env python3
import hydra
import os
import torch.multiprocessing
from pathlib import Path

# Import common utilities
from common.setup import (
    setup_paths, setup_auxiliary_basis, setup_tensor_type_and_seed,
    get_root_path, get_mode
)
from common.qh9_utils import (
    create_qh9_data_loaders, setup_warmup_training,
    create_inference_loader, dataset_abbr, dataset_full_name
)
from common.qh9_finetune_utils import (
    load_qh9_finetune_dataset, setup_qh9_finetune_model, validate_qh9_finetune_config
)
from common.data_utils import log_dataset_info
from common.checkpoint_utils import setup_wandb_logger
from common.training_utils import setup_callbacks, setup_logger, setup_trainer, log_training_config

# Setup multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

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


@hydra.main(config_path="../config_qh9", config_name="config_flow-cont-ft-wa")
def main(conf):
    # Validate QH9 fine-tuning configuration
    validate_qh9_finetune_config(conf)
    
    # Setup output directory and auxiliary basis
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    setup_auxiliary_basis(output_dir)

    # Setup tensor type and seed
    setup_tensor_type_and_seed(conf)

    # Load the dataset
    root_path = get_root_path()
    dataset = load_qh9_finetune_dataset(conf, root_path)
    
    # Create data loaders
    train_loader, val_loader, test_loader= create_qh9_data_loaders(dataset, conf)
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset
    
    
    # Log dataset information
    log_dataset_info(dataset, train_loader.dataset, val_loader.dataset, test_loader.dataset)

    # Initialize the LightningModule
    pl_model_cls = get_pl_model(conf)
    logger.info(f"Using model: {pl_model_cls}")

    # Get and validate mode
    mode = get_mode(conf)
    assert mode in ["train", "test", "eval", "inference", "predict"]
    if mode in ["train", "test", "inference", "predict"]:
        # Setup QH9 fine-tuning model
        lit_model, ckpt_path, load_model_ckpt = setup_qh9_finetune_model(conf, pl_model_cls, output_dir)
        
        # Import checkpoint utilities
        conf.wandb.tags += ["qh9", dataset_abbr(conf.dataset.dataset_full_name)]
        
        # Setup wandb logger
        wandb_logger = setup_wandb_logger(conf, output_dir)
        wandb_logger.watch(model=lit_model, log_freq=500)

        # Setup callbacks and trainer
        callbacks = setup_callbacks(conf, output_dir)
        trainer = setup_trainer(conf, callbacks, [wandb_logger], output_dir)
        log_training_config(conf)
        
        # Setup warmup training if needed
        setup_warmup_training(conf, lit_model, train_dataset, wandb_logger, callbacks)
        # Start training/testing
        _run_qh9_finetune_training_or_testing(mode, trainer, lit_model, train_loader, val_loader, test_loader, ckpt_path, load_model_ckpt, conf, test_dataset)

    elif mode == "eval":
        _run_qh9_finetune_evaluation(conf, pl_model_cls, test_dataset, output_dir)


def _run_qh9_finetune_training_or_testing(mode, trainer, lit_model, train_loader, val_loader, test_loader, ckpt_path, load_model_ckpt, conf, test_dataset):
    """Run QH9 fine-tuning training or testing based on mode."""
    
    if mode == "train":
        trainer.fit(
            lit_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=load_model_ckpt,
        )
        logger.info("Testing...")
        trainer.test(lit_model, test_loader, ckpt_path="best")
    elif mode == "test":
        # Test the model
        trainer.test(lit_model, test_loader, ckpt_path=ckpt_path)
    elif mode in ["inference", "predict"]:
        # Create inference loader
        inf_loader = create_inference_loader(mode, test_dataset, conf)
        setattr(lit_model, "test_mode", "inference")
        logger.info(f"{lit_model.test_mode}...")
        trainer.test(lit_model, inf_loader, ckpt_path=ckpt_path)


def _run_qh9_finetune_evaluation(conf, pl_model_cls, test_dataset, output_dir):
    """Run QH9 fine-tuning evaluation mode."""
    import torch
    from torch_geometric.loader import DataLoader
    
    # Find best checkpoint
    ckpt_path = output_dir / conf.wandb.project
    ckpt_path_list = list(ckpt_path.glob("**/*.ckpt"))
    ckpt_path_list = [path for path in ckpt_path_list if "best" in path.stem]
    ckpt_path_list = sorted(ckpt_path_list, key=lambda x: int(x.stem.split("=")[1]))
    
    if len(ckpt_path_list) == 0:
        ckpt_path = None
    else:
        ckpt_path = ckpt_path_list[-1]
    
    logger.info(f"Checkpoint path: {ckpt_path}")

    # Create evaluation data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=conf.dataset.num_workers,
        pin_memory=conf.dataset.pin_memory,
    )

    lit_model = pl_model_cls.load_from_checkpoint(ckpt_path, conf=conf)
    logger.info("Model loaded")
    
    logger.info("Testing...")
    default_type = torch.float64 if conf.data_type == "float64" else torch.float32
    errors, h_output = lit_model.test_over_dataset_qh9(test_loader, default_type)
    
    msg = f"dataset {conf.dataset.dataset_name}: {errors.get('total_items')} :"
    for key in errors.keys():
        if key in [
            "hamiltonian",
            "orbital_energies",
            "non_diagonal_hamiltonian",
            "diagonal_hamiltonian",
        ]:
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
