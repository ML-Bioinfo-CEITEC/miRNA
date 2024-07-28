#!/bin/bash

# test dataset
python add_local_features.py \
--folder_path "../data/preprocessed_for_ml/" \
--data_file "3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.test.pkl"

# train dataset
python add_local_features.py \
--folder_path "../data/preprocessed_for_ml/" \
--data_file "3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.train.pkl"