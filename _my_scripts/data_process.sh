
python -m dataset_module.qh9_datasets --pdb

# sample
conda activate qhflow && cd ~/25DFT/QHFlow/src
python -m dataset_module.qh9_datasets_shard --shard_num=10 --shard_idx=0-1 --prefix="_sample"
python -m dataset_module.qh9_datasets_shard --shard_num=10 --shard_idx=2-3 --prefix="_sample"
python -m dataset_module.qh9_datasets_shard --shard_num=10 --shard_idx=4-5 --prefix="_sample"
python -m dataset_module.qh9_datasets_shard --shard_num=10 --shard_idx=6-7 --prefix="_sample"
python -m dataset_module.qh9_datasets_shard --shard_num=10 --shard_idx=8-9 --prefix="_sample"


# sample
conda activate qhflow && cd ~/25DFT/QHFlow/src
python -m dataset_module.qh9_datasets_shard --shard_num=10 --shard_idx=0-1 --name=QH9Dynamic
python -m dataset_module.qh9_datasets_shard --shard_num=10 --shard_idx=2-3 --name=QH9Dynamic
python -m dataset_module.qh9_datasets_shard --shard_num=10 --shard_idx=4-5 --name=QH9Dynamic
python -m dataset_module.qh9_datasets_shard --shard_num=10 --shard_idx=6-7 --name=QH9Dynamic
python -m dataset_module.qh9_datasets_shard --shard_num=10 --shard_idx=8-9 --name=QH9Dynamic

# shard
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=0 --prefix="_shard"   
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=1 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=2 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=3 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=4 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=5 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=6 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=7 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=8 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=9 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=10 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=11 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=12 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=13 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=14 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=15 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=16 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=17 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=18 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=19 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=20 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=21 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=22 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=23 --prefix="_shard"       
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=24 --prefix="_shard"   
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=25 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=26 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=27 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=28 --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=29 --prefix="_shard"


conda activate qhflow && cd ~/25DFT/QHFlow/src

python -m dataset_module.qh9_datasets_shard --shard_num=30 --shard_idx=-1 --name=QH9Dynamic --prefix="_shard"

python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=0-4 --name=QH9Dynamic   --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=5-9 --name=QH9Dynamic   --prefix="_shard"

python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=10-14 --name=QH9Dynamic --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=15-19 --name=QH9Dynamic --prefix="_shard"

python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=20-24 --name=QH9Dynamic --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=25-29 --name=QH9Dynamic --prefix="_shard"

python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=30-34 --name=QH9Dynamic --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=35-39 --name=QH9Dynamic --prefix="_shard"

python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=40-44 --name=QH9Dynamic --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=45-49 --name=QH9Dynamic --prefix="_shard"

python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=50-54 --name=QH9Dynamic --prefix="_shard"
python -m dataset_module.qh9_datasets_shard --shard_num=60 --shard_idx=55-59 --name=QH9Dynamic --prefix="_shard"


# QM9Rowan in storage server
conda activate qhflow && cd ~/hw_data1/QHFlow/src
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python -m dataset_module.qh9_datasets_rowan --shard_num=90 --shard_idx=0-5 --name=QM9Rowan
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python -m dataset_module.qh9_datasets_rowan --shard_num=90 --shard_idx=5-10 --name=QM9Rowan
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python -m dataset_module.qh9_datasets_rowan --shard_num=90 --shard_idx=10-15 --name=QM9Rowan
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python -m dataset_module.qh9_datasets_rowan --shard_num=90 --shard_idx=15-20 --name=QM9Rowan




python -m dataset_module.qh9_datasets_rowan --shard_num=90 --shard_idx=-1 --name=QM9Rowan
