# High-order Equivariant Flow Matching for Density Functional Theory Hamiltonian Prediction (QHFlow)

By Seongsu Kim, Aug, 2025 [[arxiv]](https://arxiv.org/abs/2505.18817) [[PDF]](https://arxiv.org/pdf/2505.18817)

🌟 This repository contains an implementation of the QHFlow for DFT Hamiltonian prediction. This repository is still updating.

## Packages and Requirements

All codes are run with python 3.12 and CUDA 12.1. A similar environment should also work, as this project does not rely on some rapidly changing packages.

```bash
# Example CUDA 12.1 with torch 2.4.1
conda create -n qhflow python=3.12 psi4 -y
conda activate qhflow

pip install pyscf
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

pip install -r requirements.txt
```

## Directory and Files
```
.
├── src/                       # Source code
│   ├── experiment/            # Training/finetune/inference entrypoints
│   ├── config_md17/           # MD17 configs (dataset/model)
│   ├── config_qh9/            # QH9 configs (dataset/model)
│   ├── dataset_module/        # Dataset loaders and split utilities
│   ├── models/                # QHNet/Real_QHNet & Flow variants
│   ├── pl_module/             # PyTorch Lightning modules
│   ├── utils.py
│   └── auxiliary.gbs
├── dataset/                   # Data root (auto or manual download)
├── _my_scripts/               # Helper scripts for dataset processing 
├── requirements.txt
├── ckpts                      # Pretrained/finetuned checkpoints files
├── README.md
└── auxiliary.gbs
```

## Dataset
MD17 is downloaded automatically, but the QH9 dataset requires manual download due to gdown instability.

To download QH9, use the commands below:

```bash
mkdir -p ./dataset/QH9Stable/raw/
gdown https://drive.google.com/uc?id=1LcEJGhB8VUGkuyb0oQ_9ANJdSkky9xMS -O ./dataset/QH9Stable/raw/QH9Stable.db

mkdir -p ./dataset/QH9Dynamic_300k/raw/
gdown https://drive.google.com/uc?id=1sbf-sFhh3ZmhXgTcN2ke_la39MaG0Yho -O ./dataset/QH9Dynamic_300k/raw/QH9Dynamic_300k.db
```

Processing from raw files to torch datasets runs automatically on the first training run.
Or, you can process manually with the split process:
```bash
python -m dataset_module.qh9_datasets_split \
    --name=${NAME}  \
    --num_chunks=30 --chunk_idx=${DB_IDX} \
    --split=${SPLIT}
```
where NAME is the dataset name (`QH9Stable` / `QH9Dynamic`). Use the following SPLIT options:
- `QH9Stable`: `random`, `size_ood`
- `QH9Dynamic`: `geometry`, `mol`

Data is assembled automatically when the final chunk is processed.

**Note:** We plan to provide pre-processed datasets for all datasets to facilitate easier setup and usage.

## Saved Checkpoints

We plan to provide pre-trained model checkpoints for all datasets. Currently, we can provide checkpoints upon request. The checkpoint files are organized as follows:

**MD17 Dataset:**
```bash
ckpts/md17/${DATASET}/checkpoints/weights.ckpt
# continune_ckpt=../ckpts/md17/water/checkpoints/weights.ckpt           # Example
```

**QH9 Dataset:**
```bash
ckpts/${DATASET}/${SPLIT}/checkpoints/weights.ckpt       # Pretrained
ckpts/${DATASET}/${SPLIT}-FT/checkpoints/weights.ckpt    # Finetuned

# continune_ckpt=${ROOT}$/ckpts/QH9Stable/random/checkpoints/weights.ckpt     # Example (Pretrained)
# continune_ckpt=${ROOT}$/ckpts/QH9Stable/random-FT/checkpoints/weights.ckpt  # Example (Finetuned)
```

Where `${DATASET}` and `${SPLIT}` should be replaced with the specific dataset and split names:
- **MD17 DATASET**: `ethanol`, `malondialdehyde`, `uracil`, `water`
- **QH9 DATASET**: `QH9Stable`, `QH9Dynamic`
  - **QH9Stable SPLIT**: `random`, `size_ood`
  - **QH9Dynamic SPLIT**: `geometry`, `mol`

To use these checkpoints, specify the path in the `continune_ckpt` parameter when running inference or prediction commands. `${ROOT}` is the path of this repository or the parent path of the checkpoints directory.

## Usage

### Prerequisites
All commands should be run from the `QHFlow/src` directory.

### Available Datasets:
- **MD17 DATASET**: `ethanol`, `malondialdehyde`, `uracil`, `water`
- **QH9 DATASET**: `QH9Stable`, `QH9Dynamic`
  - **QH9Stable SPLIT (dataset.split)**: `random`, `size_ood`
  - **QH9Dynamic SPLIT (dataset.split)**: `geometry`, `mol`

### Tips

**Training Tips:**
- You can enable Weights & Biases logging with `wandb.mode=online`
- Training automatically resumes when interrupted
- Use `CUDA_VISIBLE_DEVICES` to specify GPU devices: `CUDA_VISIBLE_DEVICES=0,1 python -m experiment.train_md17 dataset=water`

**Performance Tips:**
- For faster training, use multiple GPUs with `CUDA_VISIBLE_DEVICES=0,1,2,3`
- Monitor GPU memory usage and adjust batch size if needed
- Use mixed precision training for memory efficiency (enabled by default)

**Debugging Tips:**
- Check logs in the `logs/` directory for detailed training information
- Use `mode=debug` for additional debugging information
- Monitor validation metrics to ensure proper training progress

### Train

```bash
python -m experiment.train_md17 dataset=${DATASET}
python -m experiment.train_qh9  dataset=${DATASET} dataset.split=${SPLIT}
```

**Examples:**

```bash
# Train MD17 model
python -m experiment.train_md17 dataset=water

# Train QH9 model
python -m experiment.train_qh9 dataset=QH9Stable dataset.split=random
```

### Finetune
```bash
python -m experiment.train_qh9-finetune dataset=${DATASET} dataset.split=${SPLIT} +original_ckpt=${PRETRAINED_CKPT}
```

**Example:**
```bash
python -m experiment.train_qh9-finetune dataset=QH9Stable dataset.split=random +original_ckpt=../ckpts/QH9Stable/random/checkpoints/weights.ckpt
```

### Inference (SCF acceleration measure)
```bash
python -m experiment.train_md17 mode=inference dataset=${DATASET} continune_ckpt=${CKPT}
python -m experiment.train_qh9 mode=inference dataset=${DATASET} dataset.split=${SPLIT} continune_ckpt=${CKPT}
```

**Examples:**
```bash
# MD17 inference
python -m experiment.train_md17 mode=inference dataset=water continune_ckpt=${ROOT}/ckpts/md17/water/checkpoints/weights.ckpt

# QH9 inference
python -m experiment.train_qh9 mode=inference dataset=QH9Stable dataset.split=random continune_ckpt=${ROOT}/ckpts/QH9Stable/random/checkpoints/weights.ckpt
```

### Prediction (Saving the Output)

This mode is used to predict test files and save individual Hamiltonian matrices for each sample. The predictions are saved to disk for further analysis.

**Output Format:**
- Hamiltonian matrices are saved as individual files
- Each prediction corresponds to a test sample
- Files are organized by dataset and model configuration

```bash
python -m experiment.train_md17 mode=predict dataset=${DATASET} continune_ckpt=${CKPT}
python -m experiment.train_qh9 mode=predict dataset=${DATASET} dataset.split=${SPLIT} continune_ckpt=${CKPT}
```

**Examples:**
```bash
# MD17 prediction
python -m experiment.train_md17 mode=predict dataset=water continune_ckpt=${ROOT}/ckpts/md17/water/checkpoints/weights.ckpt

# QH9 prediction
python -m experiment.train_qh9 mode=predict dataset=QH9Stable dataset.split=random continune_ckpt=${ROOT}/ckpts/QH9Stable/random/checkpoints/weights.ckpt
```

**Output Location:**
- Predictions are typically saved in the `outputs/` directory
- Each run creates timestamped subdirectories for organization

### Note about Metrics

The validation metrics of physical properties (e.g., orbital energies, Hamiltonian MAE) on QH9 can be unstable since the metric code is designed for batch size 1. Test and inference metrics have no issue since the batch size is fixed to 1. Multi-batch metric implementation is possible, but we use batch size 1 to ensure the bug-free behavior we tested.

**Note:** Although the physical metric implementation is unstable on multi-batch, the loss is not affected by these metrics, so training and tracking are perfectly fine.

## Citation
```
@article{kim2025high,
  title={High-order Equivariant Flow Matching for Density Functional Theory Hamiltonian Prediction},
  author={Kim, Seongsu and Kim, Nayoung and Kim, Dongwoo and Ahn, Sungsoo},
  journal={arXiv preprint arXiv:2505.18817},
  year={2025}
}
```

## Acknowledgements
This project is based on the repo [AIRS](https://github.com/divelab/AIRS.git) (QHNet).

**MD17 Dataset**: [Revised MD17 dataset (rMD17)](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038)

**QH9 Dataset**: [QHBench/QH9](https://github.com/divelab/AIRS/tree/main/OpenDFT/QHBench/QH9)