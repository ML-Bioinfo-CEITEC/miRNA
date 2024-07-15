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
