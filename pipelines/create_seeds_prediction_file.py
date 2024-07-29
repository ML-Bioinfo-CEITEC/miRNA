import argparse
import pandas as pd
from pathlib import Path

from funmirtar.models.constants import SEED_COUNT_COLUMNS, CLASSIFICATION_COLUMNS
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
        default="../data/predictions/seed_counts/",
        help='Path to the folder containing the dataset files.'
    )
    parser.add_argument(
        '--output_prefix',
        type=str, 
        default= 'seed_counts.class_preds',
        help='The naming prefix of the dataset file that comes before the suffixes: \'train/test\' and file format.'
    )
    args = parser.parse_args()
    
    IN_PATH = args.data_file_path
    
    OUT_FOLDER_PATH = args.out_folder_path
    OUT_FILE_PATH = args.output_prefix
    OUT_FULL_PATH = OUT_FOLDER_PATH + OUT_FILE_PATH + get_file_path_ending(IN_PATH) 
    
    OUT_COLUMNS=[]
    OUT_COLUMNS.extend(CLASSIFICATION_COLUMNS)
    OUT_COLUMNS.extend(SEED_COUNT_COLUMNS)

    data_df = pd.read_pickle(IN_PATH)

    make_dir_with_parents(OUT_FOLDER_PATH)

    data_df[OUT_COLUMNS].to_pickle(OUT_FULL_PATH)
    
    
if __name__ == "__main__":
    main()
