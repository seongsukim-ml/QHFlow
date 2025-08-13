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
from pytorch_lightning.plugins import environments
from torch_geometric.loader import DataLoader

# from ori_dataset_traj import MD17_DFT_trajectory, random_split, get_mask
from dataset_module.qh9_datasets_split import QH9Stable, QH9Dynamic

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Insert the parent directory at the beginning of sys.path
sys.path.insert(0, parent_dir)

from models import get_model, get_pl_model

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
# import copy
from pathlib import Path
import warnings
import omegaconf

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

global output_dir


@hydra.main(config_path="../config_qh9", config_name="config_flow-cont-ft-wa")
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

    seed = conf.get("seed", 0)
    logger.info(f"Seed: {seed}")
    pl.seed_everything(seed)

    # Load the dataset.
    root_path = os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-4]))

    logger.info(f"Root path: {root_path}")
    logger.info(f"Loading {conf.dataset.dataset_name} dataset...")

    if conf.dataset.dataset_name == "QH9Stable":
        dataset = QH9Stable(
            os.path.join(root_path, "dataset"),
            split=conf.dataset.split,
            cal_orbital_and_energies=True,
        )
    elif conf.dataset.dataset_name == "QH9Dynamic":
        dataset = QH9Dynamic(
            os.path.join(root_path, "dataset"),
            split=conf.dataset.split,
            version=conf.dataset.version,
            cal_orbital_and_energies=True,
        )
    train_dataset = dataset[dataset.train_mask]
    valid_dataset = dataset[dataset.val_mask]
    if getattr(conf, "partial_val", None) is not None:
        assert conf.partial_val > 0 and conf.partial_val <= 1
        valid_dataset = valid_dataset[: int(len(valid_dataset) * conf.partial_val)]
    test_dataset = dataset[dataset.test_mask]

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

    pl_model_cls = get_pl_model(conf)
    logger.info(f"Using model: {pl_model_cls}")
    lit_model = pl_model_cls(conf)

    mode = getattr(conf, "mode", "train")

    assert mode in ["train", "test", "eval", "inference", "predict"]
    if mode in ["train", "test", "inference", "predict"]:
        # Initialize the wandb logger.
        os.makedirs(output_dir / "wandb", exist_ok=True)

        run_id = getattr(conf.wandb, "run_id", None)
        original_ckpt = conf.original_ckpt
        if original_ckpt is None or original_ckpt == "None":
            original_ckpt = None
            logger.info("No original checkpoint provided")
        logger.info(f"Original checkpoint: {original_ckpt}")

        ckpt_path = None
        resume = "allow"
        logger.info(f"ckpt_path: {ckpt_path}")
        if ckpt_path is None or ckpt_path == "":
            if (output_dir / "wandb" / "latest-run").exists():
                run_id = [
                    file.name
                    for file in (output_dir / "wandb" / "latest-run").iterdir()
                    if "wandb" in file.name
                ][0][4:12]
                resume = "must"
            elif conf.wandb.run_id is not None and conf.wandb.run_id != "":
                run_id = conf.wandb.run_id

            if run_id is not None:
                ckpt_path = output_dir / conf.wandb.project / run_id / "checkpoints"
                ckpt_path_list = list(ckpt_path.glob("*.ckpt"))
                ckpt_path_list = [
                    path for path in ckpt_path_list if "best" in path.stem
                ]
                ckpt_path_list = sorted(
                    ckpt_path_list, key=lambda x: int(x.stem.split("=")[1])
                )
                if len(ckpt_path_list) == 0:
                    ckpt_path = None
                else:
                    ckpt_path = ckpt_path_list[-1]

        if ckpt_path is None or ckpt_path == "":
            ckpt_path = original_ckpt
        logger.info(f"Checkpoint path: {ckpt_path}")
        if ckpt_path is not None:
            if (ckpt_path == original_ckpt):
                print("Using pretrained model as weights")
                lit_model_pretrained = pl_model_cls.load_from_checkpoint(
                    ckpt_path, conf=conf, strict=False
                )
                lit_model.model = lit_model_pretrained.model
                load_model_ckpt = None
            else:
                lit_model = pl_model_cls.load_from_checkpoint(
                    ckpt_path,
                    conf=conf,
                    strict=False,
                )
                load_model_ckpt = ckpt_path
        else:
            lit_model = pl_model_cls(conf)
            load_model_ckpt = None

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
                save_top_k=-1,
                save_last=True,
                filename="best-{epoch:02d}",
            )
        )
        callbacks.append(LearningRateMonitor(logging_interval="step"))

        # Initialize the LightningModule.

        wandb_logger.watch(
            model=lit_model,
            log_freq=500,
        )

        warnings.filterwarnings("ignore")

        # Warmup training for Real_QHNet which is unstable
        if (
            conf.model.version.lower() == "Real_QHNet".lower()
            and run_id is None
            and mode != "test"
        ):
            logger.info("Warmup training for Real_QHNet")
            real_lr = conf.dataset.learning_rate
            warmup_lr = 1e-3
            conf.dataset.learning_rate = warmup_lr
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

            train_loader_warmup = DataLoader(
                train_dataset,
                batch_size=4,
                shuffle=True,
                num_workers=conf.dataset.num_workers,
                pin_memory=conf.dataset.pin_memory,
            )
            warmup_trainer.fit(
                lit_model,
                train_dataloaders=train_loader_warmup,
            )
            conf.dataset.learning_rate = real_lr

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
            gradient_clip_val=5.0,
        )
        # Start training.
        if mode == "train":
            trainer.fit(
                lit_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=load_model_ckpt,
                # ckpt_path=ckpt_path,
            )
            logger.info("Testing...")
            trainer.test(lit_model, test_loader, ckpt_path="best")
        elif mode == "test":
            # Test the model.
            trainer.test(lit_model, test_loader, ckpt_path=ckpt_path)
        elif mode == "inference" or "predict":
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
        ckpt_path = output_dir / conf.wandb.project
        ckpt_path_list = list(ckpt_path.glob("**/*.ckpt"))
        ckpt_path_list = [path for path in ckpt_path_list if "best" in path.stem]
        ckpt_path_list = sorted(ckpt_path_list, key=lambda x: int(x.stem.split("=")[1]))
        if len(ckpt_path_list) == 0:
            ckpt_path = None
        else:
            ckpt_path = ckpt_path_list[-1]
        logger.info(f"Checkpoint path: {ckpt_path}")

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=conf.dataset.num_workers,
            pin_memory=conf.dataset.pin_memory,
        )

        lit_model = pl_model_cls.load_from_checkpoint(ckpt_path, conf=conf)
        logger.info("Model loaded")
        # Create the PyTorch Lightning Trainer.
        warnings.filterwarnings("ignore")
        logger.info("Testing...")
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
