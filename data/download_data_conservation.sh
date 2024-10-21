#!/bin/bash

# Ensure the script is being run from the "data" directory
if [ "${PWD##*/}" != "data" ]; then
    echo "Please run this script from the 'data' directory."
    exit 1
fi

# Create the necessary folder structure
echo "Creating folder structure..."
mkdir -p "conservation"

# Define the download URL and destination path
# source: http://hgdownload.cse.ucsc.edu/goldenpath/hg19/phyloP100way/
CONSERVATION_URL="http://hgdownload.cse.ucsc.edu/goldenpath/hg19/phyloP100way/hg19.100way.phyloP100way.bw"
CONSERVATION_FILE_PATH="conservation/hg19.100way.phyloP100way.bw"

# Download the conservation file
echo "Downloading conservation file..."
wget "$CONSERVATION_URL" -O "$CONSERVATION_FILE_PATH"

echo "Conservation file downloaded successfully to $CONSERVATION_FILE_PATH."
