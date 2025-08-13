#!/usr/bin/env python3
import os
import sys
import torch
import hydra
import logging
import shutil
import subprocess
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch_geometric.loader import DataLoader
from dataset_module.ori_dataset import MD17_DFT, random_split, get_mask

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Insert the parent directory at the beginning of sys.path
sys.path.insert(0, parent_dir)

from models import get_model, get_pl_model
import math


# import copy
from pathlib import Path
import warnings
import omegaconf

logger = logging.getLogger(__name__)

mode_dict = {
    "train": "train",
    "test": "test",
    "eval": "eval",
    "evaluation": "eval",
    "inference": "inference",
    "infer": "inference",
    math.inf: "inference",
    "predict": "predict",
    "pred": "predict",
}


@hydra.main(config_path="../config_md17", config_name="config_flow-lw10-wa")
def main(conf):
    # Copy the auxiliary basis file to the output directory.
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    shutil.copy(os.path.join(os.path.dirname(__file__), "auxiliary.gbs"), output_dir)
    logger.info("Copied auxiliary basis file")
    cmd = subprocess.Popen("set basis AUXILIARY", shell=True)
    cmd.wait()
    logger.info("Set auxiliary basis")

    # Set the default tensor type and seed.
    default_type = torch.float64 if conf.data_type == "float64" else torch.float32
    torch.set_default_dtype(default_type)

    pl.seed_everything(0)

    # Load the dataset.
    root_path = os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-4]))

    logger.info(f"Root path: {root_path}")
    logger.info(f"Loading {conf.dataset.dataset_name} dataset...")
    dataset = MD17_DFT(
        os.path.join(root_path, "dataset"),
        name=conf.dataset.dataset_name,
        transform=get_mask,
    )
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset,
        [
            conf.dataset.num_train,
            conf.dataset.num_valid,
            len(dataset) - (conf.dataset.num_train + conf.dataset.num_valid),
        ],
        seed=conf.split_seed,
    )

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

    # Initialize the wandb logger.

    # Initialize the LightningModule.
    pl_model_cls = get_pl_model(conf)
    logger.info(f"Using model: {pl_model_cls}")
    lit_model = pl_model_cls(conf)

    mode = getattr(conf, "mode", "train").lower()
    mode = mode_dict.get(mode, mode)
    assert mode in ["train", "test", "eval", "inference", "predict"]
    setattr(lit_model, "test_mode", "test")

    if mode in ["train", "test", "inference", "predict"]:
        os.makedirs(output_dir / "wandb", exist_ok=True)
        run_id = None
        original_ckpt = conf.original_ckpt
        if original_ckpt is None:
            raise ValueError("original_ckpt is None")
        logger.info(f"Original checkpoint: {original_ckpt}")

        ckpt_path = None
        resume = "allow"

        if (output_dir / "wandb" / "latest-run").exists():
            run_id = [
                file.name
                for file in (output_dir / "wandb" / "latest-run").iterdir()
                if "wandb" in file.name
            ][0][4:12]
            # resume = "must"
        elif conf.wandb.run_id is not None and conf.wandb.run_id != "":
            run_id = conf.wandb.run_id

        if run_id is not None:
            ckpt_path = output_dir / conf.wandb.project / run_id / "checkpoints"
            ckpt_path_list = list(ckpt_path.glob("*.ckpt"))
            ckpt_path_list = [path for path in ckpt_path_list if "best" in path.stem]
            logger.info(f"Found {len(ckpt_path_list)} checkpoints")
            ckpt_path_list = sorted(
                ckpt_path_list, key=lambda x: int(x.stem.split("=")[1])
            )
            if len(ckpt_path_list) == 0:
                ckpt_path = None
            else:
                ckpt_path = ckpt_path_list[-1]

        if ckpt_path is None:
            ckpt_path = original_ckpt
        logger.info(f"Checkpoint path: {ckpt_path}")
        if ckpt_path == original_ckpt or ckpt_path == "":
            lit_model_pretrained = pl_model_cls.load_from_checkpoint(
                ckpt_path, conf=conf, strict=False
            )
            lit_model.model = lit_model_pretrained.model
        else:
            lit_model = pl_model_cls.load_from_checkpoint(
                ckpt_path,
                conf=conf,
                strict=False,
            )

        wandb_logger = WandbLogger(
            project=conf.wandb.project,
            name=conf.wandb.run_name,
            save_dir=output_dir,
            mode=getattr(conf.wandb, "mode", "online"),
            id=run_id,
            tags=getattr(conf.wandb, "tags", None),
            resume=resume,
        )

        wandb_config = omegaconf.OmegaConf.to_container(
            conf, resolve=True, throw_on_missing=True
        )
        wandb_logger.log_hyperparams(wandb_config)

        # Set up model checkpointing.
        callbacks = []
        callbacks.append(
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                save_last=True,
                filename="best-{epoch:02d}",
            )
        )
        callbacks.append(LearningRateMonitor(logging_interval="step"))

        wandb_logger.watch(
            model=lit_model,
            log_freq=500,
        )

        # Create the PyTorch Lightning Trainer.
        trainer = pl.Trainer(
            max_steps=conf.num_training_steps,
            logger=wandb_logger,
            callbacks=callbacks,
            precision=64 if conf.data_type == "float64" else 32,
            log_every_n_steps=conf.dataset.train_batch_interval,
            accelerator="auto",
            devices=1,
            # val_check_interval=conf.dataset.validation_interval,
            check_val_every_n_epoch=conf.check_val_every_n_epoch,
            enable_progress_bar=True,
            gradient_clip_val=(
                conf.dataset.clip_norm if conf.dataset.use_gradient_clipping else None
            ),
        )
        warnings.filterwarnings("ignore")
        # Start training
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
        elif mode == "inference" or "predict":
            # predict the model with many metric (larger metrics)
            inf_loader = DataLoader(
                test_dataset[:300],
                batch_size=1,
                shuffle=False,
                num_workers=conf.dataset.num_workers,
                pin_memory=conf.dataset.pin_memory,
            )
            setattr(lit_model, "test_mode", "inference")
            logger.info(f"{lit_model.test_mode}...")
            trainer.test(lit_model, inf_loader, ckpt_path=ckpt_path)

    elif mode == "eval":
        model_ckpt = conf.model_ckpt
        lit_model = pl_model_cls.load_from_checkpoint(model_ckpt, conf=conf)
        logger.info("Model loaded")
        # Create the PyTorch Lightning Trainer.
        warnings.filterwarnings("ignore")
        logger.info("Testing...")
        errors, h_output = lit_model.test_over_dataset(test_loader, default_type)
        msg = f"dataset {conf.dataset.dataset_name}: {errors.get('total_items')} :"

        for key in errors.keys():
            if key == "hamiltonian" or key == "orbital_energies":
                msg += f"{key}: {errors[key]*1e6:.3f}(10^-6), "
            elif key == "orbital_coefficients":
                msg += f"{key}: {errors[key]*1e2:.4f}(10^-2)"
            elif key == "total_items":
                # int value
                msg += f"{key}: {errors[key]:d}, "
            else:
                msg += f"{key}: {errors[key]:.8f}, "
        logger.info(msg)
        output_dir_name = f"output"
        os.makedirs(output_dir / output_dir_name, exist_ok=True)
        with open(output_dir / output_dir_name / "results.txt", "w") as f:
            f.write(msg)
        torch.save(h_output, output_dir / output_dir_name / "h_output.pt")


if __name__ == "__main__":
    main()
