import argparse
import pandas as pd
from pathlib import Path

from funmirtar.models.local_features import get_binding_site_features_from_config
from funmirtar.models.global_features import get_only_positive_conservation
from funmirtar.models.local_features_config import load_user_config
from funmirtar.utils.file import insert_text_before_the_end_of_path


# def load_user_config(config_path):
#     spec = importlib.util.spec_from_file_location("config", config_path)
#     config = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(config)
#     return config.USER_CONFIG


def main():
    parser = argparse.ArgumentParser(description='Add local features based on the given local-features config and dataset paths.')
    parser.add_argument(
        '--folder_path', 
        type=str, 
        default="../data/preprocessed_for_ml/",
        help='Path to the folder containing the dataset files.'
    )
    parser.add_argument(
        '--data_file',
        type=str, 
        default= '3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.train.pkl',
        help='Name of the dataset file.'
    )
    parser.add_argument(
        '--output_sufix',
        type=str, 
        default= '.user_local_features',
        help='The naming suffix that will be added to the end of the dataset file.'
    )
    parser.add_argument(
        '--config_path', 
        type=str, 
        default="",
        help='Path to the file containing the dictionary config for local feature extraction.'
    )

    args = parser.parse_args()

    FOLDER_PATH = args.folder_path
    OUTPUT_SUFIX = args.output_sufix
    INPUT_PATH = FOLDER_PATH + args.data_file
    
    OUTPUT_PATH = insert_text_before_the_end_of_path(INPUT_PATH, OUTPUT_SUFIX)
    
    CONFIG_PATH = args.config_path
    
    config = load_user_config(CONFIG_PATH)

    data_df = pd.read_pickle(INPUT_PATH)

    # data_df['conservation_phylo'] = data_df.conservation_phylo.map(lambda cons: get_only_positive_conservation(cons))
    features = get_binding_site_features_from_config(data_df, config)
    data_df = pd.concat([data_df, pd.DataFrame(features)], axis=1)

    print(f"Extracted local features resulting file is at {OUTPUT_PATH}")
    data_df.to_pickle(OUTPUT_PATH)
    
    
if __name__ == "__main__":
    main()


