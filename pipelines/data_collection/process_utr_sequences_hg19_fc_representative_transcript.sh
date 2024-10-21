#!/bin/bash

# Paths to your input files
IN_UTR_SEQ_TXT_PATH='../../data/GRCh37.p13 hg19/UCSC/3utr_sequences.txt'
MIRNA_FCS='../../data/fold_change/mirna_fcs.csv'
TS_GENE_INFO='../../data/target_scan_8/Gene_info.txt'
ID_MAP='../../data/GRCh37.p13 hg19/UCSC/id_map'
OUT_UTR_DF_MIRNA_FC_PATH='../../data/processed/GRCh37.p13 hg19/3utr.sequences.refseq_id.mirna_fc.pkl'

# Run the Python script
python process_utrs.py "$IN_UTR_SEQ_TXT_PATH" "$MIRNA_FCS" "$TS_GENE_INFO" "$ID_MAP" "$OUT_UTR_DF_MIRNA_FC_PATH"
