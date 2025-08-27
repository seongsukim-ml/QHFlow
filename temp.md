conda activate qhflow
cd ~/25DFT/QHFlow/src

CUDA_VISIBLE_DEVICES=4 python -m experiment.train_md17 model=QHFlow wandb.mode="online"
CUDA_VISIBLE_DEVICES=4 python -m experiment.train_md17 dataset=ethanol model=QHFlow wandb.mode="online"
CUDA_VISIBLE_DEVICES=5 python -m experiment.train_md17 dataset=malondialdehyde model=QHFlow wandb.mode="online"
CUDA_VISIBLE_DEVICES=6 python -m experiment.train_md17 dataset=uracil model=QHFlow wandb.mode="online"



1. Using uv to make the folder
2. Upload the pre-trained weights
3. Upload the sample dataset for test

```bash
conda create -n qhflow python=3.12 psi4
conda activate qhflow
pip install pyscf
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index https://download.pytorch.org/whl/cu118
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
pip install ase
pip install -r requirements.txt

```

```bash
# CPU-only
conda create -n qhflow python=3.12 psi4
conda activate qhflow
# conda install -c conda-forge libstdcxx-ng
# conda install gcc
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install pyscf
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install ase
pip install -r requirements.txt

```


```bash

uv init
uv venv
source .venv/bin/activate

```

uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index https://download.pytorch.org/whl/cu118
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

uv pip install pyscf pytorch3d
uv pip install pytorch-lightning



conda activate qhflow
cd ~/25DFT/QHFlow/src


python -m dataset_module.qh9_datasets_split --name=QH9Stable --num_chunks=30 --chunk_idx=0 


python -m dataset_module.qh9_datasets_split --name=QH9Dynamic --num_chunks=30 --chunk_idx=0 --split=mol

for i in {0..15}; do
    python -m dataset_module.qh9_datasets_split --name=QH9Dynamic --num_chunks=30 --chunk_idx=$i --split=geo
    if [ $? -eq 0 ]; then
        echo "Successfully completed chunk $i"
    else
        echo "Failed on chunk $i"
        exit 1
    fi
done

for i in {16..29}; do
    python -m dataset_module.qh9_datasets_split --name=QH9Dynamic --num_chunks=30 --chunk_idx=$i --split=mol
    if [ $? -eq 0 ]; then
        echo "Successfully completed chunk $i"
    else
        echo "Failed on chunk $i"
        exit 1
    fi
done