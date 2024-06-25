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

# https://drive.google.com/file/d/1m6c3be4OVqVv72vAD5UnR_Ar-8IUoeex/view?usp=sharing
gdown "1m6c3be4OVqVv72vAD5UnR_Ar-8IUoeex&confirm=t" --output "data/preprocessed_for_ml/3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.train.pkl"
# https://drive.google.com/file/d/15dZZQAEXAqsbBpkjVfWvew9oM4Eu-8PL/view?usp=sharing
gdown "15dZZQAEXAqsbBpkjVfWvew9oM4Eu-8PL&confirm=t" --output "data/preprocessed_for_ml/3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.test.pkl"