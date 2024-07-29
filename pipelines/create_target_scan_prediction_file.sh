#!/bin/bash

data_splits=("train" "test")

for split in "${data_splits[@]}"
do
  python create_target_scan_prediction_file.py \
  --data_file_path "../data/preprocessed_for_ml/3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.${split}.pkl" \
  --out_folder_path "../data/predictions/target_scan/" \
  --output_prefix "target_scan.conserved_nonconserved.class_preds" \
  --data_file_prediction_column "weighted context++ score percentile" \
  --result_prediction_column "weighted context++ score percentile (filled NaNs)"
done
