#!/bin/bash

# Set variables for the file paths and parameters
MIRNA_NAME='hsa-miR-16-5p'
PREDICTION_THRESHOLD=0
ID_COLUMN='RefSeq ID'
MODEL_PATH='models/miRBind.h5'
SEQUENCE_SOURCE_PATH='../../data/processed/GRCh37.p13 hg19/UCSC/3utr.sequences.refseq_id.mirna_fc.pkl'
EXPLAINABILITY_BACKGROUND_DATA_PATH='evaluation_set_1_1_CLASH2013_paper.tsv'
SAVE_EXPLAINABILITY_SCORES_PATH='../../data/scanned/GRCh37.p13 hg19/UCSC/3utr.sequences.refseq_id.mirna_fc.explainability_scores.json'
SAVE_SCANNING_ERRORS_PATH='../../data/scanned/GRCh37.p13 hg19/UCSC/3utr.sequences.refseq_id.mirna_fc.explainability_scores.scanning_errors.txt'

# Run the Python script with the specified arguments
python3 scan_sequences.py \
    --dataset_name $DATASET_NAME \
    --sequence_source_path "$SEQUENCE_SOURCE_PATH" \
    --model_path "$MODEL_PATH" \
    --explainability_background_data_path "$EXPLAINABILITY_BACKGROUND_DATA_PATH" \
    --save_explainability_scores_path "$SAVE_EXPLAINABILITY_SCORES_PATH" \
    --save_scanning_errors_path "$SAVE_SCANNING_ERRORS_PATH" \
    --prediction_threshold $PREDICTION_THRESHOLD \
    --id_column "$ID_COLUMN" \
    --mirna_name $MIRNA_NAME
    
