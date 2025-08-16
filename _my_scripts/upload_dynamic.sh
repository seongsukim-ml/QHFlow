cd $HOME/25DFT/QHFlow/dataset   

cd $HOME/25DFT/QHFlow/dataset/QH9Dynamic_300k/processed

cd $HOME/25DFT/QHFlow/dataset && \
$HOME/gdrive files upload --recursive --chunk-size 256 \
    --print-chunk-errors --print-chunk-info \
    --parent 19-Hi2VpcI2YcZBweMeiRgU9iBLlwkK8a \
    QH9Dynamic_300k/processed/QH9Dynamic_0_completed.lmdb

cd $HOME/shared/QHFlow/dataset && \
$HOME/_my_initial/gdrive files upload --recursive --chunk-size 256 \
    --print-chunk-errors --print-chunk-info \
    --parent 19-Hi2VpcI2YcZBweMeiRgU9iBLlwkK8a \
    QH9Dynamic_300k/processed/QH9Dynamic_1_completed.lmdb

cd $HOME/25DFT/QHFlow/dataset && \
$HOME/gdrive files upload --recursive --chunk-size 256 \
    --print-chunk-errors --print-chunk-info \
    --parent 1sPZ81P_wv6DRXnops3Lj-QYW9f4_NiNr \
    QH9Dynamic_300k/processed/QH9Dynamic.lmdb

cd $HOME/25DFT/QHFlow/dataset && \
$HOME/gdrive files upload --recursive --chunk-size 256 \
    --print-chunk-errors --print-chunk-info \
    --parent 1sPZ81P_wv6DRXnops3Lj-QYW9f4_NiNr \
    QH9Dynamic_300k/processed/processed_QH9Dynamic_mol.pt

    

cd $HOME/25DFT/QHFlow/dataset && \
$HOME/gdrive files upload --recursive --chunk-size 256 \
    --print-chunk-errors --print-chunk-info \
    --parent 1cW7LngAWTGEdOj_SB9da2_R3uPCVuf6j \
    QH9Stable/processed/processed_QH9Stable_size_ood.pt