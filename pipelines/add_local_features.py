import argparse
import pandas as pd
from pathlib import Path

from funmirtar.models.local_features import get_binding_site_features
from funmirtar.models.global_features import get_only_positive_conservation
from funmirtar.utils.file import insert_text_before_the_end_of_path


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
        default= '.local_features',
        help='The naming suffix that will be added to the end of the dataset file.'
    )

    args = parser.parse_args()

    FOLDER_PATH = args.folder_path
    OUTPUT_SUFIX = args.output_sufix
    INPUT_PATH = FOLDER_PATH + args.data_file
    
    OUTPUT_PATH = insert_text_before_the_end_of_path(INPUT_PATH, OUTPUT_SUFIX)

    data_df = pd.read_pickle(INPUT_PATH)

    data_df['conservation_phylo'] = data_df.conservation_phylo.map(lambda cons: get_only_positive_conservation(cons))
    features = get_binding_site_features(data_df)
    data_df = pd.concat([data_df, pd.DataFrame(features)], axis=1)

    data_df.to_pickle(OUTPUT_PATH)
    
    
if __name__ == "__main__":
    main()


