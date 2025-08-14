# High-order Equivariant Flow Matching for Density Functional Theory Hamiltonian Prediction (QHFlow)

By Seongsu Kim, Aug, 2025 [[arxiv]](https://arxiv.org/abs/2505.18817) [[PDF]](https://arxiv.org/pdf/2505.18817)

🌟 This repository contains an implementation of the QHFlow for DFT Hamiltonian prediction.

## Packages and Requirements

All codes are run with python 3.12 and CUDA 11.8. A similar environment should also work, as this project does not rely on some rapidly changing packages.

```bash
# Example CUDA 11.8 with torch 2.4.1
conda create -n qhflow python=3.12 psi4 -y
conda activate qhflow

pip install pyscf
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index https://download.pytorch.org/whl/cu118
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

pip install pytorch-lightning==1.8.5 hydra-core
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

## Usage

### Train
```bash
python -m experiment.train_md17 dataset=${DATASET}
python -m experiment.train_qh9  dataset=${DATASET} dataset.split=${SPLIT}
```

- **MD17 DATASET**: `ethanol`, `malondialdehyde`, `uracil`, `water`
- **QH9 DATASET**: `QH9Stable`, `QH9Dynamic`
  - **QH9Stable SPLIT (dataset.split)**: `random`, `size_ood`
  - **QH9Dynamic SPLIT (dataset.split)**: `geometry`, `mol`

### Finetune
```bash
python -m experiment.train_qh9-finetune  dataset=${DATASET} dataset.split=${SPLIT} +original_ckpt=${PRETRAINED_CKPT}
```

### SCF acceleration measure
```bash
python -m experiment.train_md17 mode=inference dataset=${DATASET} continune_ckpt=${CKPT}
python -m experiment.train_qh9  mode=inference dataset=${DATASET} dataset.split=${SPLIT} continune_ckpt=${CKPT}
```

### Prediction (saving the output)
```bash
python -m experiment.train_md17 mode=predict dataset=${DATASET} continune_ckpt=${CKPT}
python -m experiment.train_qh9  mode=predict dataset=${DATASET} dataset.split=${SPLIT} continune_ckpt=${CKPT}
```

Tips: You can enable Weights & Biases logging with `wandb.mode=online`. Training automatically resumes when interrupted.

The validation metrics of physical properties (e.g., orbital energies, Hamiltonian MAE) on QH9 can be unstable since the metric code is designed for batch size 1. Test and inference metrics have no issue since the batch size is fixed to 1. Multi-batch metric implementation is possible, but we use batch size 1 to ensure the bug-free behavior we tested.

(Although the physical metric implementation is unstable on multi-batch, the loss is not affected by these metrics, so training and tracking are perfectly fine.)

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
This project is based on the repo [AIRS](https://github.com/divelab/AIRS.git).