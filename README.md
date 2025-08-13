## Install
``` bash
conda create -n p4_QH4 python=3.9 pytorch==2.1.2 pytorch-cuda=12.1  psi4 pyscf=2.2.1 pytorch3d pytorch-lightning==1.8.5 -c pytorch -c nvidia -c pyscf -c pytorch3d 
conda activate p4_QH4
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html --no-cache-dir
pip install pytorch-lightning==1.8.5
# pip install hydra-core
# pip install ase==3.22.1
# pip install torch_ema tqdm wandb PyYAML
# pip install e3nn gdown transformers tensorboard lmdb
pip install -r requirements.txt
pip install scipy==1.10
pip install pydantic==1.10.21
```



## Dataset
MD17 dataset will be downloaded automatically, but the QH9 dataset requires the manual downloads due to unstability of gdown.

To download the QH9, you can download it with below command:

```bash
mkdir -p ./dataset/QH9Stable/raw/
gdown https://drive.google.com/uc?id=1LcEJGhB8VUGkuyb0oQ_9ANJdSkky9xMS -O ./dataset/QH9Stable/raw/QH9Stable.db

mkdir -p ./dataset/QH9Dynamic_300k/raw/
gdown https://drive.google.com/uc?id=1sbf-sFhh3ZmhXgTcN2ke_la39MaG0Yho -O ./dataset/QH9Dynamic_300k/raw/QH9Dynamic_300k.db
```

And then we need process raw dataset to the torch dataset, and it can be done automatically when excute the train.
Or, it can be done manually with the splitted process:
```bash
python -m dataset_module.qh9_datasets_split \
    --name=${NAME}  \
    --num_chunks=30 --chunk_idx=${DB_IDX} \
    --split=${SPLIT}
```
where Name is the name of the dataset (QH9Stable / QH9Dynamic), split is the name of the split (iid,ood,geo,mol), and num_chunks is the number of the total idx, and DB_IDX is the current index of the splitted process.
Assembling of the data is automatically done when final chunk is made.


## Excution
### Train
```bash
python -m experiment.train_md17 dataset=${DATASET}
python -m experiment.train_qh9  dataset=${DATASET} dataset.split=${SPLIT}
```

### Finetune
```bash
# pytyon -m experiment.train_md17-finetune dataset=${DATASET} +original_ckpt=${PRETRAINED_CKPT}
python -m experiment.train_qh9-finetune  dataset=${DATASET} dataset.split=${SPLIT} +original_ckpt=${PRETRAINED_CKPT}
```

### SCF acceleartion measure
```bash
pytyon -m experiment.train_md17 mode=inference dataset=${DATASET} mode=predict continune_ckpt=${CKPT}
python -m experiment.train_qh9  mode=inference dataset=${DATASET} mode=predict dataset.split=${SPLIT} continune_ckpt=${CKPT}
```

### Prediction (saving the output)
```bash
pytyon -m experiment.train_md17 mode=predict dataset=${DATASET} continune_ckpt=${CKPT}
python -m experiment.train_qh9  mode=predict dataset=${DATASET} dataset.split=${SPLIT} continune_ckpt=${CKPT}
```

Tips: You can turn on the wandb for detailed logs and outputs with `wandb.mode=online`. The train commands automatically resume when train is distruptted in the middle of the train.

The validation metrics of physical properties (ex. orbital energies, Ham mae) on QH9 is somewhat instable since our metric code is designed for the 1-batch. The test metric and inference metric have no issue since the batch size is fixed to 1. Multi-batch metric implementation is possible, but we use 1-batch code to sure about the bug-free that we tested.

(Although the physical metric implementation is instable on multi-batch, the loss is not affects by these metric so that the tracking the training and train itself are perfectly fine.)

## Acknowledgements
This project is based on the repo [AIRS](https://github.com/divelab/AIRS.git).