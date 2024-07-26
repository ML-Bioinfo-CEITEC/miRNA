#!/bin/bash

# Function to check if gdown is installed
is_gdown_installed() {
    python -c "import gdown" &> /dev/null
    return $?
}

# Check if gdown is installed
if is_gdown_installed; then
    echo "Package 'gdown' is already installed."
else
    echo "Package 'gdown' is not installed. Installing..."
    pip install gdown
fi

# Create folder structure
mkdir -p "data/preprocessed_for_ml"

# https://drive.google.com/file/d/18hP00PC8fRRs-woyXaSIrXfYu8V1JZFn/view?usp=sharing
gdown "18hP00PC8fRRs-woyXaSIrXfYu8V1JZFn&confirm=t" --output "data/preprocessed_for_ml/3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.train.pkl"
# https://drive.google.com/file/d/1A3wdtRshVhfvGjJu5HsNp7t5VukL6mXc/view?usp=sharing
gdown "1A3wdtRshVhfvGjJu5HsNp7t5VukL6mXc&confirm=t" --output "data/preprocessed_for_ml/3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.test.pkl"