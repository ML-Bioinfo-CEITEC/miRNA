#!/bin/bash

# evaluate_new_feature.sh
# --run_output_sufix # Name suffix for your resulting train/test datasets
# --config_path # Path to USER_CONFIG containing your extraction functions for local features
# --data_folder_path # Folder containg train/test dataset 
# --train_dataset # Train dataset name 
# --test_dataset # Test dataset name
# --run_name # Folder where you will find your predictions 
# --comparison_name # Folder where you will find your plots
# --prediction_folders # Folders cointaining prediction you wish to compare with
# --methods_to_plot # Methods you wish to compare with


# Run on 1.8.
./evaluate_new_feature.sh \
--run_output_sufix ".user_local_features" \
--config_path "user_config.py" \
--data_folder_path "../data/preprocessed_for_ml/" \
--train_dataset "3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.train.pkl" \
--test_dataset "3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.test.pkl" \
--run_name "seeds.signal.local_features" \
--comparison_name "random_forest-seeds.signal.local_features.target_scan" \
--prediction_folders "seeds.signal.local_features/,seeds.signal/,seeds/,target_scan/,seed_counts/" \
--methods_to_plot "weighted context++ score percentile (filled NaNs),random_forest.seeds,random_forest.seeds.signal,random_forest.seeds.signal.local_features,random_forest.seeds.signal.local_features"


# Run on 6.8.
./extract_local_features.sh \
--run_output_sufix ".user_local_features" \
--config_path "user_config.py" \
--data_folder_path "../data/preprocessed_for_ml/" \
--train_dataset "3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.train.pkl" \
--test_dataset "3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.test.pkl" \


# Run on 7.8.
./train_and_evaluate_models.sh \
--train_dataset "../data/preprocessed_for_ml/3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.train.pkl" \
--test_dataset "../data/preprocessed_for_ml/3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.test.pkl" \
--run_output_sufix ".user_local_features" \
--run_name "seeds.signal.local_features" \
--comparison_name "random_forest-seeds.signal.local_features.target_scan" \
--prediction_folders_to_plot "seeds.signal.local_features/,seeds.signal/,seeds/,target_scan/,seed_counts/" \
--methods_to_plot "weighted context++ score percentile (filled NaNs),random_forest.seeds,random_forest.seeds.signal,random_forest.seeds.signal.local_features,random_forest.seeds.signal.local_features"