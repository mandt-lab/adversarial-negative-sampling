#!/bin/bash

Binarize the labels, take first prediction for every data point, remove
labels that have no training point, and split heldout data set into
validation set and test set.

python ./binarize-labels.py ../../dat/Wikipedia-500K
python ./binarize-labels.py ../../dat/Amazon-670K


# Binarize the features and split holdout data set in the same way.

# Wikipedia-500K data set

python ./binarize-features.py ../../dat/Wikipedia-500K/trn_ft_mat_dense.txt \
    -o ../../dat/Wikipedia-500K/train-features-full.np

python ./binarize-features.py ../../dat/Wikipedia-500K/tst_ft_mat_dense.txt \
    -f ../../dat/Wikipedia-500K/valid-indices.np \
    -o ../../dat/Wikipedia-500K/valid-features-full.np

python ./binarize-features.py ../../dat/Wikipedia-500K/tst_ft_mat_dense.txt \
    -f ../../dat/Wikipedia-500K/test-indices.np \
    -o ../../dat/Wikipedia-500K/test-features-full.np


# Amazon-670K data set

python ./binarize-features.py ../../dat/Amazon-670K/trn_ft_mat_dense.txt \
    -o ../../dat/Amazon-670K/train-features-full.np

python ./binarize-features.py ../../dat/Amazon-670K/tst_ft_mat_dense.txt \
    -f ../../dat/Amazon-670K/valid-indices.np \
    -o ../../dat/Amazon-670K/valid-features-full.np

python ./binarize-features.py ../../dat/Amazon-670K/tst_ft_mat_dense.txt \
    -f ../../dat/Amazon-670K/test-indices.np \
    -o ../../dat/Amazon-670K/test-features-full.np

echo 'Done. Next, run the jupyter notebook "pca.ipynb".'
