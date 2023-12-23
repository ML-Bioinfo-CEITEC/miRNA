import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from utils import MIRNAS_NAME_SEQUENCE_DICT, save_dataframe_to_csv, load_data_from_pickle

DEFAULT_CLS_NO_CHANGE_UPPER_THRESHOLD = -0.1
DEFAULT_CLS_CHANGE_LOWER_THRESHOLD = -0.2
DEFAULT_SEQUENCE_LENGTH_THRESHOLD = 16300

SEED = 42

REGRESSION_LABEL_COLUMN = 'log2fold'
MRNA_SEQ_COLUMN = 'utr3'
MIRNA_SEQ_COLUMN = 'miRNA'
COLUMNS_TO_KEEP_FROM_INPUT_DATA = [MIRNA_SEQ_COLUMN, MRNA_SEQ_COLUMN, REGRESSION_LABEL_COLUMN]
REGRESSION_LABEL_COLUMN_CLAMPED_TO_ZERO = 'log2fold_clamped'
CLASSIFICATION_LABEL_COLUMN = 'cls_label'
SEQUENCE_PAIRS_WITH_LABEL_EXT = '.nlp'
DATA_WITH_POSITIVE_LABEL_CLAMPED_TO_ZERO_EXT = '.clamped_to_zero'
CLASSIFICATION_DATA_EXT = '.cls'
SEQUENCE_LENGTH_BELOW_EXT = '.seq_len_below'
BALANCED_DOWNSAMPLES_EXT = '.balanced_down'

def transform_to_sequence_pairs_with_label(data, mirna_sequences, columns):
    seq_array = []
    for miRNA in mirna_sequences.keys():
        for i in data.index.values:
            if not np.isnan(data[miRNA][i]):
                seq_array.append([
                    mirna_sequences[miRNA],
                    str(data['sequence'][i]),
                    data[miRNA][i]
                ])
    return pd.DataFrame(seq_array, columns = columns)

def clamp_positive_to_zero(data_df, original_column_name, new_column_name):
    
    data_df[new_column_name] = data_df[original_column_name].map(lambda x: 0 if x > 0 else x)
    return data_df

def convert_to_cls_task(data_df, 
                        reg_label_column, 
                        cls_label_column, 
                        no_change_upper_threshold, 
                        change_lower_threshold):
    
    data_df[cls_label_column] = data_df[reg_label_column].apply(lambda x: 0 if x > no_change_upper_threshold else 2 if x > change_lower_threshold else 1)
    data_df = data_df[data_df[cls_label_column] != 2].reset_index(drop=True)
    
    return data_df

def remove_sequence_above_len_threshold(data_df, seq_column, len_threshold):
    data_df.drop(data_df[data_df[seq_column].str.len() > len_threshold].index, inplace=True)
    return data_df

def balance_by_downsampling(df, label_column, seed=42):
    """Downsample the dominant class to balance the dataset."""
    # Identify the class with the minimum number of samples
    min_class = df[label_column].value_counts().idxmin()
    
    # Separate data into classes
    minority_class = df[df[label_column] == min_class]
    majority_class = df[df[label_column] != min_class]
    
    # Downsample the majority class
    majority_downsampled = majority_class.sample(len(minority_class), replace=False, random_state=seed)
    # majority_downsampled = resample(majority_class,
    #                                replace=False,  # Set to True if you want to allow duplicates
    #                                n_samples=len(minority_class),
    #                                random_state=seed)
    
    # Combine the minority class with the downsampled majority class
    balanced_df = pd.concat([minority_class, majority_downsampled])
    # Shuffle samples to mix labels
    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
    
    return balanced_df

def add_extension(path, extension):
    return path.with_suffix(path.suffix + extension)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filepath', type=str,
                        help='Path to input file with data')
    parser.add_argument('output_dir_path', type=str,
                        help='Path to output directory where processed data will be saved.')
    parser.add_argument('--cls_no_change_upper_threshold', type=float,
                        help='Upper threshold for log2fold with no change.'
                             'Used for converting regression task to classification task.')
    parser.add_argument('--cls_change_lower_threshold', type=float,
                        help='Lower threshold for log2fold with some change.'
                             'Used for converting regression task to classification task.')
    args = parser.parse_args()
    
    if args.cls_no_change_upper_threshold is None:
        args.cls_no_change_upper_threshold = DEFAULT_CLS_NO_CHANGE_UPPER_THRESHOLD
        
    if args.cls_change_lower_threshold is None:
        args.cls_change_lower_threshold = DEFAULT_CLS_CHANGE_LOWER_THRESHOLD
        
    output_dir = Path(args.output_dir_path)
    if not output_dir.exists():
        print(f"Creating output directory {output_dir}")
        output_dir.mkdir(parents=True)
    
    print("Loading input data.")
    input_file = Path(args.input_filepath)
    data = load_data_from_pickle(input_file)
    
    print("Converting data to table with triplets - miRNA sequence, mRNA sequence and label")
    sequence_pairs_with_label_df \
        = transform_to_sequence_pairs_with_label(data, 
                                                 MIRNAS_NAME_SEQUENCE_DICT,
                                                 COLUMNS_TO_KEEP_FROM_INPUT_DATA)
    del data # release data from memory
    sequence_pairs_with_label_path = output_dir / add_extension(Path(input_file.name), 
                                                                SEQUENCE_PAIRS_WITH_LABEL_EXT)
    save_dataframe_to_csv(sequence_pairs_with_label_df, sequence_pairs_with_label_path)
    
    print("Adding column with regression label values where positives values are clamped to zero.")
    data_with_positive_label_clamped_to_zero_df \
        = clamp_positive_to_zero(sequence_pairs_with_label_df, 
                                 REGRESSION_LABEL_COLUMN, 
                                 REGRESSION_LABEL_COLUMN_CLAMPED_TO_ZERO)
    del sequence_pairs_with_label_df # release sequence_pairs_with_label_df from memory
    data_with_positive_label_clamped_to_zero_path \
        = add_extension(sequence_pairs_with_label_path, 
                        DATA_WITH_POSITIVE_LABEL_CLAMPED_TO_ZERO_EXT)
    save_dataframe_to_csv(data_with_positive_label_clamped_to_zero_df,
                          data_with_positive_label_clamped_to_zero_path)
    
    print(f"Convert to classification task. Regression label from range (0, {args.cls_no_change_upper_threshold}) is classification label 0 and regression labels from range ({args.cls_change_lower_threshold}, max) is classification label 1. Middle area is not used.")
    cls_data = convert_to_cls_task(data_with_positive_label_clamped_to_zero_df, 
                                   REGRESSION_LABEL_COLUMN_CLAMPED_TO_ZERO, 
                                   CLASSIFICATION_LABEL_COLUMN, 
                                   args.cls_no_change_upper_threshold, 
                                   args.cls_change_lower_threshold)
    del data_with_positive_label_clamped_to_zero_df
    cls_data_path = add_extension(data_with_positive_label_clamped_to_zero_path, 
                                  CLASSIFICATION_DATA_EXT)
    save_dataframe_to_csv(cls_data, cls_data_path)
    
    print(f"Removing samples where {MRNA_SEQ_COLUMN} columns contains sequence longer than {DEFAULT_SEQUENCE_LENGTH_THRESHOLD}.")
    shorter_seq_data = remove_sequence_above_len_threshold(cls_data, MRNA_SEQ_COLUMN, DEFAULT_SEQUENCE_LENGTH_THRESHOLD)
    del cls_data
    shorter_seq_data_path = add_extension(cls_data_path,
                                          SEQUENCE_LENGTH_BELOW_EXT)
    save_dataframe_to_csv(shorter_seq_data, shorter_seq_data_path)
    
    print("Downsampling dominant class to create a balanced dataset.")
    balanced_data = balance_by_downsampling(shorter_seq_data, CLASSIFICATION_LABEL_COLUMN, seed=SEED)
    del shorter_seq_data
    balanced_path = add_extension(shorter_seq_data_path,
                                  BALANCED_DOWNSAMPLES_EXT)
    save_dataframe_to_csv(balanced_data, balanced_path)
    