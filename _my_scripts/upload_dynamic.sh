cd $HOME/25DFT/QHFlow/dataset

cd $HOME/25DFT/QHFlow/dataset/QH9Dynamic_300k/processed

$HOME/gdrive files upload --recursive --chunk-size 256 \
    --print-chunk-errors --print-chunk-info \
    --parent 19-Hi2VpcI2YcZBweMeiRgU9iBLlwkK8a \
    QH9Dynamic_300k/processed/QH9Dynamic_0_completed.lmdb

$HOME/_my_initial/gdrive files upload --recursive --chunk-size 256 \
    --print-chunk-errors --print-chunk-info \
    --parent 19-Hi2VpcI2YcZBweMeiRgU9iBLlwkK8a \
    QH9Dynamic_300k/processed/QH9Dynamic_0_completed.lmdb
