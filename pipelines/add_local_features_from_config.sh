#!/bin/bash

# test dataset
python add_local_features_from_config.py \
--folder_path "../data/preprocessed_for_ml/" \
--data_file "3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.test.pkl"

# train dataset
python add_local_features_from_config.py \
--folder_path "../data/preprocessed_for_ml/" \
--data_file "3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.train.pkl"