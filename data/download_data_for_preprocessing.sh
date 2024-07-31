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
mkdir -p "data/GRCh37.p13 hg19/UCSC"
mkdir -p "data/target_scan_8"

# UCSC id map
# https://drive.google.com/file/d/1geMB7JMUkkDwtwulCw01ruSLqMBEdXiT/view?usp=sharing
gdown "1geMB7JMUkkDwtwulCw01ruSLqMBEdXiT&confirm=t" --output "data/GRCh37.p13 hg19/UCSC/id_map"

# utr_sequences_hg19
# https://drive.google.com/file/d/11PFjBLQJ-rUF8nioh7pd1MzDs1fsSCPG/view?usp=sharing
gdown "11PFjBLQJ-rUF8nioh7pd1MzDs1fsSCPG&confirm=t" --output "data/GRCh37.p13 hg19/utr_sequences_hg19.txt"

wget -P /data/target_scan_8/Conserved_Site_Context_Scores.txt.zip https://www.targetscan.org/vert_80/vert_80_data_download/Conserved_Site_Context_Scores.txt.zip
wget -P /data/target_scan_8/Nonconserved_Site_Context_Scores.txt.zip https://www.targetscan.org/vert_80/vert_80_data_download/Nonconserved_Site_Context_Scores.txt.zip
wget -P /data/target_scan_8/Predicted_Targets_Context_Scores.txt.zip https://www.targetscan.org/vert_80/vert_80_data_download/Predicted_Targets_Context_Scores.default_predictions.txt.zip

