#!/bin/bash


# Path to the config file
CONFIG_PATH="user_config.py"
run_output_sufix=".user_local_features"

data_folder_path="../data/preprocessed_for_ml/"
train_dataset="3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.train.pkl"
test_dataset="3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.test.pkl"


# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config_path) CONFIG_PATH="$2"; shift ;;
        --run_output_sufix) run_output_sufix="$2"; shift ;;
        --data_folder_path) data_folder_path="$2"; shift ;;
        --train_dataset) train_dataset="$2"; shift ;;
        --test_dataset) test_dataset="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

#Extract local features
# train dataset
echo "Running add_local_features_from_config.py for $train_dataset"
echo $(date '+%Y-%m-%d %H:%M:%S')
python add_local_features_from_config.py \
--folder_path "$data_folder_path" \
--data_file "$train_dataset" \
--config_path "$CONFIG_PATH" \
--output_sufix "$run_output_sufix" 

# test dataset
echo "Running add_local_features_from_config.py for $test_dataset"
echo $(date '+%Y-%m-%d %H:%M:%S')
python add_local_features_from_config.py \
--folder_path "$data_folder_path" \
--data_file "$test_dataset" \
--config "$CONFIG_PATH"
--output_sufix "$run_output_sufix" 