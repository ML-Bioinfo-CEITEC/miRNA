#!/bin/bash


# Path to the config file
### TODO fix seeds path
run_output_sufix=".user_local_features"

train_dataset="../data/preprocessed_for_ml/3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.train.pkl"
test_dataset="../data/preprocessed_for_ml/3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.test.pkl"

prediction_folders_to_plot=("seeds.signal.local_features.model_optimisation/" "seeds.signal.local_features/" "seeds.signal/" "seeds/" "target_scan/" "seed_counts/")
methods_to_plot=("weighted context++ score percentile (filled NaNs)" "random_forest.seeds" "random_forest.seeds.signal" "random_forest.seeds.signal.local_features" "random_forest.seeds.signal.local_features")

run_name="seeds.signal.local_features"
comparison_name="random_forest-seeds.signal.local_features.target_scan"


seeds_folder_path="../data/predictions/seed_counts/"
predictions_folder_path="../data/predictions/"
plots_folder_path="../plots/"

seeds_output_prefix="seeds_counts.class_preds.${run_output_sufix}"


target_scan_predictions_folder="../data/predictions/target_scan/"
target_scan_output_prefix="target_scan.conserved_nonconserved.class_preds"
target_scan_input_prediction_column="weighted context++ score percentile"
target_scan_resulting_prediction_column="weighted context++ score percentile (filled NaNs)"


# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --train_dataset) train_dataset="$2"; shift ;;
        --test_dataset) test_dataset="$2"; shift ;;
        --run_output_sufix) run_output_sufix="$2"; shift ;;
        --run_name) run_name="$2"; shift ;;
        --comparison_name) comparison_name="$2"; shift ;;
        --seeds_folder_path) seeds_folder_path="$2"; shift ;;
        --predictions_folder_path) predictions_folder_path="$2"; shift ;;
        --plots_folder_path) plots_folder_path="$2"; shift ;;
        --prediction_folders_to_plot) IFS=',' read -r -a prediction_folders_to_plot <<< "$2"; shift ;;
        --methods_to_plot) IFS=',' read -r -a methods_to_plot <<< "$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


#Extract targetscan predictions 
echo "Running create_target_scan_prediction_file.py for $train_dataset"
echo $(date '+%Y-%m-%d %H:%M:%S')
python create_target_scan_prediction_file.py \
--data_file_path "$train_dataset" \
--out_folder_path "$target_scan_predictions_folder" \
--output_prefix "$target_scan_output_prefix" \
--data_file_prediction_column "$target_scan_input_prediction_column" \
--result_prediction_column "$target_scan_resulting_prediction_column"

echo "Running create_target_scan_prediction_file.py for $test_dataset"
echo $(date '+%Y-%m-%d %H:%M:%S')
python create_target_scan_prediction_file.py \
--data_file_path "$test_dataset" \
--out_folder_path "$target_scan_predictions_folder" \
--output_prefix "$target_scan_output_prefix" \
--data_file_prediction_column "$target_scan_input_prediction_column" \
--result_prediction_column "$target_scan_resulting_prediction_column"


#Extract seeds predictions
echo "Running create_seeds_prediction_file.py for $train_dataset"
echo $(date '+%Y-%m-%d %H:%M:%S')
python create_seeds_prediction_file.py \
--data_file_path "$train_dataset" \
--out_folder_path "$seeds_folder_path" \
--output_prefix "$seeds_output_prefix"

echo "Running create_seeds_prediction_file.py for $test_dataset"
echo $(date '+%Y-%m-%d %H:%M:%S')
python create_seeds_prediction_file.py \
--data_file_path "$test_dataset" \
--out_folder_path "$seeds_folder_path" \
--output_prefix "$seeds_output_prefix"


#Run ML classification
echo "Running ml_prediction_classification.py for train: $train_dataset and test: $test_dataset"
echo $(date '+%Y-%m-%d %H:%M:%S')
python ml_prediction_classification.py \
--train_file_path "$train_dataset" \
--test_file_path "$test_dataset" \
--out_folder_path "$predictions_folder_path" \
--run_name "$run_name" 
# --run_name 'seeds.signal.local_features.model_optimisation'
# --run_name 'seeds.signal'
# --run_name 'seeds'


#Run ML evaluation and plotting
echo "Running ml_plot_classification.py for $predictions_folder_pathh"
echo $(date '+%Y-%m-%d %H:%M:%S')
python ml_plot_classification.py \
    --predictions_folder_path "$predictions_folder_path" \
    --prediction_folders_to_plot "${prediction_folders_to_plot[@]}" \
    --methods_to_plot "${methods_to_plot[@]}" \
    --comparison_name "$comparison_name" \
    --out_folder_path "$plots_folder_path"
