#!/bin/bash

# Default values for the arguments
predictions_folder_path="../data/predictions/"
prediction_folders=("seeds.signal.local_features.model_optimisation/" "seeds.signal.local_features/" "seeds.signal/" "seeds/" "target_scan/" "seed_counts/")
methods_to_plot=("weighted context++ score percentile (filled NaNs)" "random_forest.seeds" "random_forest.seeds.signal" "random_forest.seeds.signal.local_features")
comparison_name="random_forest-seeds.signal.local_features.target_scan"
out_folder_path="../plots/"

# Calling the Python script with the default arguments
python ml_plot_classification.py \
    --predictions_folder_path "$predictions_folder_path" \
    --prediction_folders "${prediction_folders[@]}" \
    --methods_to_plot "${methods_to_plot[@]}" \
    --comparison_name "$comparison_name" \
    --out_folder_path "$out_folder_path"
