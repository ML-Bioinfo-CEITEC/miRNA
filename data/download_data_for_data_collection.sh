#!/bin/bash

# Ensure the script is being run from the "data" directory
if [ "${PWD##*/}" != "data" ]; then
    echo "Please run this script from the 'data' directory."
    exit 1
fi

# Create folder structure
echo "Creating folder structure..."
mkdir -p "GRCh37.p13 hg19/UCSC"
mkdir -p "fold_change"
mkdir -p "target_scan_8"

# Function to download file from Google Drive using wget
download_from_gdrive() {
    local file_id=$1
    local dest_path=$2

    echo "Downloading $dest_path..."
    wget --no-check-certificate "https://drive.google.com/uc?export=download&id=$file_id" -O "$dest_path"
}

# Download files from Google Drive (replace with actual Google Drive file IDs)
IN_UTR_SEQ_TXT_PATH="GRCh37.p13 hg19/UCSC/3utr_sequences.txt"
MIRNA_FCS="fold_change/mirna_fcs.csv"
TS_GENE_INFO="target_scan_8/Gene_info.txt"
ID_MAP="GRCh37.p13 hg19/UCSC/id_map"

# Replace with actual Google Drive file IDs
IN_UTR_SEQ_FILE_ID="12HjygyA2Q8mGw2j3Lw5gt9PC1Un_C0W9"
MIRNA_FCS_FILE_ID="1HnDH40DArkBgsbiAvdoiCuu39QYFAqn4"
TS_GENE_INFO_FILE_ID="1ZCKXpvVP66tTV5wIN1I006dGyCYMQfQk"
ID_MAP_FILE_ID="1ptdZQL6Sn6a0zDZyz9KR8O-gcXn6m8Mh"

# Download the files
download_from_gdrive "$IN_UTR_SEQ_FILE_ID" "$IN_UTR_SEQ_TXT_PATH"
download_from_gdrive "$MIRNA_FCS_FILE_ID" "$MIRNA_FCS"
download_from_gdrive "$TS_GENE_INFO_FILE_ID" "$TS_GENE_INFO"
download_from_gdrive "$ID_MAP_FILE_ID" "$ID_MAP"

echo "All files downloaded successfully."
