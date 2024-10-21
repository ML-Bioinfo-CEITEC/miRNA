#!/bin/bash

# Define variables
MIRNA_NAME='hsa-let-7c-5p'
DATASET_NAME='mirna_fcs'
OUTPUT_FOLDER='../../data/preprocessed_for_ml/GRCh37.p13 hg19/UCSC/'
FOLD_CHANGE_FILE_PATH='../../data/fold_change/mirna_fcs.csv'
MIRNA_FC_EXPERIMENT_DATA_URL='https://raw.githubusercontent.com/ML-Bioinfo-CEITEC/miRNA/main/modules/evaluation/mirna_fcs.csv'
EXPLAINABILITY_SCORES_DATASET="../../data/scanned/GRCh37.p13 hg19/UCSC/3utr.sequences.refseq_id.mirna_fc.explainability_scores_${MIRNA_NAME}"
UTR_CONSERVATION_PATH='../../data/conservation/hg19/UCSC/3utr.sequences.refseq_id.mirna_fc.chr.conservation'

# Run the Python script
python process_mirna_data.py \
    --output_folder "${OUTPUT_FOLDER}" \
    --mirna_name "${MIRNA_NAME}" \
    --dataset_name "${DATASET_NAME}" \
    --fold_change_file_path "${FOLD_CHANGE_FILE_PATH}" \
    --mirna_fc_experiment_data_url "${MIRNA_FC_EXPERIMENT_DATA_URL}" \
    --explainability_scores_dataset "${EXPLAINABILITY_SCORES_DATASET}" \
    --utr_conservation_path "${UTR_CONSERVATION_PATH}" \
    --random_state "${RANDOM_STATE}"
