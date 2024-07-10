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


# UCSC id map
# https://drive.google.com/file/d/1geMB7JMUkkDwtwulCw01ruSLqMBEdXiT/view?usp=sharing
gdown "1geMB7JMUkkDwtwulCw01ruSLqMBEdXiT&confirm=t" --output "GRCh37.p13 hg19/UCSC/id_map"

# utr_sequences_hg19
# https://drive.google.com/file/d/11PFjBLQJ-rUF8nioh7pd1MzDs1fsSCPG/view?usp=sharing
gdown "11PFjBLQJ-rUF8nioh7pd1MzDs1fsSCPG&confirm=t" --output "GRCh37.p13 hg19/utr_sequences_hg19.txt"
