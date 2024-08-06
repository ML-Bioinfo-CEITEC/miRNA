import argparse
import pandas as pd
from pathlib import Path

from funmirtar.models.constants import CLASSIFICATION_COLUMNS
from funmirtar.utils.file import make_dir_with_parents, get_file_path_ending


def main():
    parser = argparse.ArgumentParser(description='Extract targetscan predictions into a dataframe file out of a given dataframe.')
    parser.add_argument(
        '--data_file_path',
        type=str, 
        default= '../data/preprocessed_for_ml/3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.train.pkl',
        help='Path to the dataset file.'
    )
    parser.add_argument(
        '--out_folder_path', 
        type=str, 
        default="../data/predictions/target_scan/",
        help='Path to the folder containing the dataset files.'
    )
    parser.add_argument(
        '--output_prefix',
        type=str, 
        default= 'target_scan.conserved_nonconserved.class_preds',
        help='The naming prefix of the dataset file that comes before the suffixes: \'train/test\' and file format.'
    )
    parser.add_argument(
        '--data_file_prediction_column',
        type=str, 
        default= 'weighted context++ score percentile',
        help='A column of the dataframe found at the data_file_path that will be used for classification prediction.'
    )
    parser.add_argument(
        '--result_prediction_column',
        type=str, 
        default= 'weighted context++ score percentile (filled NaNs)',
        help='A new column name that will be used to store classification prediction.'
    )
    
    args = parser.parse_args()

    IN_PATH = args.data_file_path
    
    OUT_FOLDER_PATH = args.out_folder_path
    OUT_FILE_PATH = args.output_prefix
    OUT_FULL_PATH = OUT_FOLDER_PATH + OUT_FILE_PATH + get_file_path_ending(IN_PATH) 
    
    PREDICTION_COLUMN = args.data_file_prediction_column
    EXTRACTED_PREDICTION_COLUMN = args.result_prediction_column

    data_df = pd.read_pickle(IN_PATH)
    data_df[EXTRACTED_PREDICTION_COLUMN] = data_df[PREDICTION_COLUMN].fillna(0,inplace=False)
    
    out_columns=[]
    out_columns.extend(CLASSIFICATION_COLUMNS)
    out_columns.extend([EXTRACTED_PREDICTION_COLUMN])    

    make_dir_with_parents(OUT_FOLDER_PATH)
    
    print(f"TargetScan resulting file is at {OUT_FULL_PATH}")
    data_df[out_columns].to_pickle(OUT_FULL_PATH)
    
    
if __name__ == "__main__":
    main()


