import os
import re
import pandas as pd
from pathlib import Path
from functools import reduce


def make_dir_with_parents(path):
    path_obj = Path(os.path.dirname(path))
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def extend_path_by_suffix_before_filetype(
    path,
    suffix,
):
    path = os.path.normpath(path)
    path = path.split(os.sep)
    file = path[-1].split('.')
    return os.path.join(*path[:-1], '.'.join(file[:-1]) + suffix + '.' + file[-1])


def insert_text_before_the_end_of_path(file_path, insert_text):
    file_format_suffix = str(file_path).split('.')[-1]
    if file_path.endswith(f'.test.{file_format_suffix}'):
        base = file_path[:-9]  # Remove '.test.pkl'
        new_path = f"{base}{insert_text}.test.{file_format_suffix}"
    elif file_path.endswith(f'.train.{file_format_suffix}'):
        base = file_path[:-10]  # Remove '.train.pkl'
        new_path = f"{base}{insert_text}.train.{file_format_suffix}"
    elif file_path.endswith(f'.{file_format_suffix}'):
        base = file_path[:-4]  # Remove '.pkl'
        new_path = f"{base}{insert_text}.{file_format_suffix}"
    else:
        raise ValueError("The file path does not end with a recognized pattern.")
    return new_path


def get_file_path_ending(file_path, endings_list = ['.test','.train','']):
    file_format_suffix = str(file_path).split('.')[-1]
    
    for ending in endings_list:
        file_ending_with_format = f'{ending}.{file_format_suffix}'
        if file_path.endswith(file_ending_with_format):
            return file_ending_with_format

    raise ValueError("The file path does not end with a recognized pattern.")
    
    
def extract_file_name_from_path(original):
    # Remove the prefix path up to and including the last "/"
    prefix_removed = re.sub(r'^.*/', '', original)
    return prefix_removed


def extract_file_ending(original):
    # Remove the suffix "train", "test", and always ".pkl"
    suffix_removed = re.sub(r'(.train|.test)?\.pkl$', '', original)
    return suffix_removed


def merge_dataframes(dataframes, merge_on_columns):
    return reduce(
        lambda x, y: pd.merge(x, y, on = merge_on_columns), dataframes
    )


def load_and_merge_pickle_dataframes(pickle_paths, merge_on_columns):
    dataframes = []
    for path in pickle_paths:
        df = pd.read_pickle(path)
        dataframes.append(df)
    merged_df = merge_dataframes(dataframes, merge_on_columns)
    return merged_df