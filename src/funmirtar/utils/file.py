import os
from pathlib import Path


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