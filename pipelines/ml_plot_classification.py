import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from os.path import isfile, join

from funmirtar.models.seeds import SEED_TYPE_TO_EXTRACTION_FUNCTION
from funmirtar.models.constants import CLASSIFICATION_COLUMNS
from funmirtar.utils.plots import plot_prc_with_seeds
from funmirtar.utils.file import make_dir_with_parents, load_and_merge_pickle_dataframes


def main():
    parser = argparse.ArgumentParser(description='Compare and plot the evaluation of given predictions.')
    parser.add_argument(
        '--predictions_folder_path', 
        type=str, 
        default="../data/predictions/",
        help='Path to the folder containing the prediction folders.'
    )
    parser.add_argument(
        '--prediction_folders', 
        type=str, 
        nargs='+',
        default=[
            "seeds.signal.local_features.model_optimisation/", 
            "seeds.signal.local_features/", 
            "seeds.signal/", 
            "seeds/", 
            "target_scan/",
            "seed_counts/", # Necessery part for now because we force plotting with seeds
        ],
        # TODO make the seeds in plotting optional
        help="Paths to the folders containing the prediction files. 'seed_counts/' are necessery for now because we force plotting with seeds"
    )
    parser.add_argument(
        '--methods_to_plot', 
        type=str, 
        nargs='+',
        default=[
            'weighted context++ score percentile (filled NaNs)',
            'random_forest.seeds',
            'random_forest.seeds.signal',
            'random_forest.seeds.signal.local_features'
        ],
        help='Names of the columns containing the predictions to compare.'
    )
    parser.add_argument(
        '--comparison_name',
        type=str, 
        default= "random_forest-seeds.signal.local_features.target_scan",
        # "model_comparison-seeds.signal.local_features.target_scan"
        help='Name of the comparison. The resulting folder with plots and metrics will be named this way.'
    )
    parser.add_argument(
        '--out_folder_path', 
        type=str, 
        default="../plots/",
        help='Path to the folder that will containe the folder with the results. The final location will be out_folder_path + comparison_name.'
    )
    args = parser.parse_args()

    COMPARISON_NAME = args.comparison_name
    METHODS_TO_PLOT = args.methods_to_plot
    PREDICTIONS_DIR_PATH = args.predictions_folder_path
    PREDICTIONS_FOLDERS = args.prediction_folders
    DIR_PATHS = []
    
    for prediction_path in PREDICTIONS_FOLDERS:
        DIR_PATHS.append(f"{PREDICTIONS_DIR_PATH}{prediction_path}")
    
    out_folder_path = args.out_folder_path
    out_folder_path = f"{out_folder_path}{COMPARISON_NAME}/"

    paths_train = []
    paths_test = []
    
    for path in DIR_PATHS:
        files = [f for f in listdir(path) if isfile(join(path, f))]
        train_files = [file for file in files if 'train.pkl' in file]
        test_files = [file for file in files if 'test.pkl' in file]
        if train_files:
            paths_train.append(path + train_files[0])
        if test_files:
             paths_test.append(path + test_files[0])

    data_train = load_and_merge_pickle_dataframes(paths_train, CLASSIFICATION_COLUMNS)
    data_test = load_and_merge_pickle_dataframes(paths_test, CLASSIFICATION_COLUMNS)

    make_dir_with_parents(out_folder_path)

    _, ax = plot_prc_with_seeds(
        data_test,
        SEED_TYPE_TO_EXTRACTION_FUNCTION,
        METHODS_TO_PLOT,
        title='PR-curve on test dataset',
        path=out_folder_path + '/test'
    )

    _, ax = plot_prc_with_seeds(
        data_train,
        SEED_TYPE_TO_EXTRACTION_FUNCTION,
        METHODS_TO_PLOT,
        title='PR-curve on train dataset',
        path=out_folder_path + '/train'
    )


if __name__ == "__main__":
    main()




