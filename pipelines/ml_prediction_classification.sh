#!/bin/bash


python ml_prediction_classification.py \
--train_file_path "../data/preprocessed_for_ml/3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.train.pkl" \
--test_file_path "../data/preprocessed_for_ml/3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.test.pkl" \
--out_folder_path "../data/predictions/" \
--run_name "seeds.signal.local_features" 
# --run_name 'seeds.signal.local_features.model_optimisation'
# --run_name 'seeds.signal'
# --run_name 'seeds'