#!/bin/bash

# Define file paths
CONSERVATION_PATH='../../data/conservation/hg19.100way.phyloP100way.bw'
UTR_DF_MIRNA_FC_CHR_PATH='../../data/processed/GRCh37.p13 hg19/UCSC/3utr.sequences.refseq_id.mirna_fc.pkl'
UTR_DF_MIRNA_FC_CHR_CONSERVATION_PATH='../../data/conservation/hg19/UCSC/3utr.sequences.refseq_id.mirna_fc.chr.conservation.pkl'

# Run the Python script with the specified arguments
python process_conservation.py \
    --conservation_path "$CONSERVATION_PATH" \
    --utr_df_mirna_fc_chr_path "$UTR_DF_MIRNA_FC_CHR_PATH" \
    --utr_df_mirna_fc_chr_conservation_path "$UTR_DF_MIRNA_FC_CHR_CONSERVATION_PATH"
