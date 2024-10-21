import argparse
from funmirtar.models.constants import TARGETSCAN_COLUMN_TO_SEQUENCE
from funmirtar.utils.data_collection import (
    download_data_from_github_if_not_existing,
    split_train_and_test_set,
    possitive_fold_change_to_zero,
    convert_signals_json_to_pickle,
    merge_fold_change_dataset_with_signals,
    merge_conservation,
    normalize_column_based_on_train_set_NO_LOG,
    pad_empty_signal_based_on_conservation_length,
)


def main():
    parser = argparse.ArgumentParser(description='Process data for miRNA analysis.')    
    parser.add_argument('--output_folder', type=str, default='../../data/preprocessed_for_ml/GRCh37.p13 hg19/UCSC/',
                        help='Path to the output folder.')
    parser.add_argument('--mirna_name', type=str, default='hsa-let-7c-5p',
                        help='Name of the miRNA.')
    parser.add_argument('--dataset_name', type=str, default='mirna_fcs',
                        help='Name of the dataset.')
    parser.add_argument('--fold_change_file_path', type=str, default='../../data/fold_change/mirna_fcs.csv',
                        help='Path to the fold change file.')
    parser.add_argument('--mirna_fc_experiment_data_url', type=str, default='https://raw.githubusercontent.com/ML-Bioinfo-CEITEC/miRNA/main/modules/evaluation/mirna_fcs.csv',
                        help='URL to download the miRNA fold change experimental data.')
    parser.add_argument('--explainability_scores_dataset', type=str, default='',
                        help='Path to the explainability scores dataset.')
    parser.add_argument('--utr_conservation_path', type=str, default='../../data/conservation/hg19/UCSC/3utr.sequences.refseq_id.mirna_fc.chr.conservation',
                        help='Path to the UTR dataframe with conservation data.')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility.')


    args = parser.parse_args()

    OUTPUT_FOLDER = args.output_folder
    MIRNA_NAME = args.mirna_name
    DATASET_NAME = args.dataset_name
    FOLD_CHANGE_FILE_PATH = args.fold_change_file_path
    MIRNA_FC_EXPERIMENT_DATA_URL = args.mirna_fc_experiment_data_url
    EXPLAINABILITY_SCORES_DATASET = args.explainability_scores_dataset
    UTR_CONSERVATION_PATH = args.utr_conservation_path
    RANDOM_STATE = args.random_state

    EXPLAINABILITY_COLUMN_NAMES = ['mirna', 'RefSeq ID', 'signal']
    MERGE_ON = 'RefSeq ID'

    MIRNA_SEQ = TARGETSCAN_COLUMN_TO_SEQUENCE[MIRNA_NAME]

    download_data_from_github_if_not_existing(
        url=MIRNA_FC_EXPERIMENT_DATA_URL,
        output_file_path=FOLD_CHANGE_FILE_PATH,
    )

    split_train_and_test_set(
        input_dataset_file_path=FOLD_CHANGE_FILE_PATH,
        column_to_split=MIRNA_NAME,
        columns_to_keep=['gene_symbol', MIRNA_NAME],
        test_fraction=0.25,
        random_state=RANDOM_STATE,
        splits_distributions_similarity_pvalue_threshold=0.05,
        output_file_path_train=f'{OUTPUT_FOLDER}{DATASET_NAME}.{MIRNA_NAME}.train.pkl',
        output_file_path_test=f'{OUTPUT_FOLDER}{DATASET_NAME}.{MIRNA_NAME}.test.pkl',
    )

    possitive_fold_change_to_zero(
        input_dataset_file_path=f'{OUTPUT_FOLDER}{DATASET_NAME}.{MIRNA_NAME}.train.pkl',
        column_name=MIRNA_NAME,
        output_path_suffix='.positive_fc_to_zero',
    )
    possitive_fold_change_to_zero(
        input_dataset_file_path=f'{OUTPUT_FOLDER}{DATASET_NAME}.{MIRNA_NAME}.test.pkl',
        column_name=MIRNA_NAME,
        output_path_suffix='.positive_fc_to_zero',
    )

    convert_signals_json_to_pickle(
        signals_json_file_path=f"{EXPLAINABILITY_SCORES_DATASET}.json",
        output_file_path=f"{EXPLAINABILITY_SCORES_DATASET}.pkl",
        mirna_name=MIRNA_NAME,
        mirna_seq=MIRNA_SEQ,
        column_names=EXPLAINABILITY_COLUMN_NAMES,
    )

    merge_fold_change_dataset_with_signals(
        signals_df_path=f'{EXPLAINABILITY_SCORES_DATASET}.pkl',
        fold_change_df_path=f'{OUTPUT_FOLDER}{DATASET_NAME}.{MIRNA_NAME}.train.positive_fc_to_zero.pkl',
        merge_on=MERGE_ON,
        mirna_name=MIRNA_NAME,
        output_file_path=f'{OUTPUT_FOLDER}{DATASET_NAME}.{MIRNA_NAME}.train.positive_fc_to_zero.signals.pkl',
    )
    merge_fold_change_dataset_with_signals(
        signals_df_path=f'{EXPLAINABILITY_SCORES_DATASET}.pkl',
        fold_change_df_path=f'{OUTPUT_FOLDER}{DATASET_NAME}.{MIRNA_NAME}.test.positive_fc_to_zero.pkl',
        merge_on=MERGE_ON,
        mirna_name=MIRNA_NAME,
        output_file_path=f'{OUTPUT_FOLDER}{DATASET_NAME}.{MIRNA_NAME}.test.positive_fc_to_zero.signals.pkl',
    )

    merge_conservation(
        conservation_df_path=f'{UTR_CONSERVATION_PATH}.pkl',
        fold_change_df_path=f'{OUTPUT_FOLDER}{DATASET_NAME}.{MIRNA_NAME}.train.positive_fc_to_zero.signals.pkl',
        merge_on=MERGE_ON,
        mirna_name=MIRNA_NAME,
        output_path_suffix='.conservation',
    )
    merge_conservation(
        conservation_df_path=f'{UTR_CONSERVATION_PATH}.pkl',
        fold_change_df_path=f'{OUTPUT_FOLDER}{DATASET_NAME}.{MIRNA_NAME}.test.positive_fc_to_zero.signals.pkl',
        merge_on=MERGE_ON,
        mirna_name=MIRNA_NAME,
        output_path_suffix='.conservation',
    )

    normalize_column_based_on_train_set_NO_LOG(
        input_file_path_train=f'{OUTPUT_FOLDER}{DATASET_NAME}.{MIRNA_NAME}.train.positive_fc_to_zero.signals.conservation.pkl',
        input_file_path_test=f'{OUTPUT_FOLDER}{DATASET_NAME}.{MIRNA_NAME}.test.positive_fc_to_zero.signals.conservation.pkl',
        column='signal',
        output_path_suffix='.normalize',
    )

    pad_empty_signal_based_on_conservation_length(
        input_file_path=f'{OUTPUT_FOLDER}{DATASET_NAME}.{MIRNA_NAME}.train.positive_fc_to_zero.signals.conservation.normalize.pkl',
        to_pad_column_name='signal',
        pad_based_on_column_name='conservation_phylo',
        output_path_suffix='.pad_empty_signal',
    )

    pad_empty_signal_based_on_conservation_length(
        input_file_path=f'{OUTPUT_FOLDER}{DATASET_NAME}.{MIRNA_NAME}.test.positive_fc_to_zero.signals.conservation.normalize.pkl',
        to_pad_column_name='signal',
        pad_based_on_column_name='conservation_phylo',
        output_path_suffix='.pad_empty_signal',
    )

if __name__ == '__main__':
    main()
