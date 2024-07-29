#!/bin/bash

data_splits=("train" "test")

for split in "${data_splits[@]}"
do
  python create_seeds_prediction_file.py \
  --data_file_path "../data/preprocessed_for_ml/3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.${split}.pkl" \
  --out_folder_path "../data/predictions/seed_counts/" \
  --output_prefix "seed_counts.class_preds"
done