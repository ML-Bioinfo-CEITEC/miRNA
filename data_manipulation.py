from requests import get
from os import path
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import json
from collections.abc import Sequence
import numpy as np

from utils import make_dir_with_parents, extend_path_by_suffix_before_filetype, compress_zero_subsequences, check_empty_or_not_array


def download_data_from_github_if_not_existing(url, output_file_path):
    if path.isfile(output_file_path):
        return output_file_path
    else:
        response = get(url)
        if not response.ok:
            raise RuntimeError("Could not download the requested file. HTTP respose not OK, i.e. HTTP respose not 200")

        make_dir_with_parents(output_file_path)
        with open(output_file_path,'w') as file:
            file.write(response.text)

        return output_file_path
    
    
def split_train_and_test_set(
    input_dataset_file_path, 
    column_to_split, 
    columns_to_keep,
    test_fraction, 
    random_state, 
    splits_distributions_similarity_pvalue_threshold,
    output_file_path_train,
    output_file_path_test,
):
    input_dataset = pd.read_csv(input_dataset_file_path,index_col=0, header=0, sep=',')    
    input_dataset = input_dataset.rename(columns={"Gene symbol": "gene_symbol"})
    
    train, test = train_test_split(
        input_dataset[column_to_split].dropna(), 
        test_size = test_fraction, 
        random_state = random_state,
        shuffle=True
    )
    
    ks_test_results = ks_2samp(train, test)
    if ks_test_results.pvalue < splits_distributions_similarity_pvalue_threshold:
        raise ValueError('The train test distributions made by the split are significantly different. Consider a better way to split the dataset by selecting different parameters. P-value: {}'.format(ks_test_results.pvalue))
    """
    Kolmogorov–Smirnov test is a nonparametric test of the equality of continuous one-dimensional probability distributions 
    that can be used to compare compare two samples (two-sample K–S test).
    "How likely is it that we would see two sets of samples like this if they were drawn from the same (but unknown) probability distribution?"
    Under the null hypothesis the two distributions are identical. 
    If the K-S statistic is small or the p-value is high (greater than the significance level, say 5%), 
    then we cannot reject the hypothesis that the distributions of the two samples are the same. 
    Conversely, we can reject the null hypothesis if the p-value is low.
    """
    
    DF_MERGED_TRAIN = pd.merge(train, input_dataset, on='RefSeq ID', how='inner')    
    DF_MERGED_TEST = pd.merge(test, input_dataset, on='RefSeq ID', how='inner')   
    DF_MERGED_TRAIN = DF_MERGED_TRAIN.rename(columns={column_to_split + '_x': column_to_split})
    DF_MERGED_TEST = DF_MERGED_TEST.rename(columns={column_to_split + '_x': column_to_split})
    
    make_dir_with_parents(output_file_path_train)
    make_dir_with_parents(output_file_path_test)
    DF_MERGED_TRAIN[columns_to_keep].to_csv(output_file_path_train)
    DF_MERGED_TEST[columns_to_keep].to_csv(output_file_path_test)
    
    return output_file_path_train, output_file_path_test


def possitive_fold_change_to_zero(
    input_dataset_file_path, 
    column_name,
    output_path_suffix
):
    input_dataset = pd.read_csv(input_dataset_file_path,index_col=0, header=0, sep=',')

    input_dataset.loc[input_dataset[column_name] > 0, column_name] = 0
    
    output_file_path = extend_path_by_suffix_before_filetype(
        path = input_dataset_file_path,
        suffix = output_path_suffix,
    )
    
    make_dir_with_parents(output_file_path)
    input_dataset.to_csv(output_file_path)
    
    return output_file_path


def convert_signals_json_to_pickle(signals_json_file_path, output_file_path, mirna_name, mirna_seq):    
    with open(signals_json_file_path, 'r') as file:    
        data = json.load(file)

    df_signals = pd.DataFrame(data={
        'mirna': [mirna_name for x in range(len(data[mirna_seq]))],
        'gene_symbol': [x[0] for x in data[mirna_seq]], 
        'signal': [x[1] for x in data[mirna_seq]],
    })
    
    make_dir_with_parents(output_file_path)
    df_signals.to_pickle(output_file_path)
    
    return output_file_path


def merge_fold_change_dataset_with_signals(
    signals_df_path,
    fold_change_df_path, 
    merge_on,
    mirna_name,
    output_file_path,
):
    signals_df = pd.read_pickle(signals_df_path)
    fold_change_df = pd.read_csv(fold_change_df_path, index_col=0, header=0, sep=',')    
    
    df_merged = pd.merge(fold_change_df, signals_df, on=merge_on, how='left')    
    
    df_merged = df_merged.rename(columns={mirna_name: "fold_change"})

    df_merged.loc[df_merged['mirna'].isna(), 'mirna'] = mirna_name

    # output_file_path = extend_path_by_suffix_before_filetype(
    #     path = fold_change_df_path,
    #     suffix = output_path_suffix,
    # )
    make_dir_with_parents(output_file_path)
    # df_merged.to_csv(output_file_path)
    df_merged.to_pickle(output_file_path)
    
    return output_file_path


def compress_signal_zeros_in_dataset(
    input_dataset_file_path,
    output_path_suffix,
):
    # df = pd.read_csv(input_dataset_file_path, index_col=0, header=0, sep=',')
    df = pd.read_pickle(input_dataset_file_path)
    df['signal'] = df['signal'].apply(func=compress_zero_subsequences)    
        
    output_file_path = extend_path_by_suffix_before_filetype(
        path = input_dataset_file_path,
        suffix = output_path_suffix,
    )
    
    make_dir_with_parents(output_file_path)
    df.to_pickle(output_file_path)
    
    return output_file_path


def remove_genes_with_no_binding_sites(
    input_dataset_file_path_train,
    output_path_suffix,
):
    df = pd.read_pickle(input_dataset_file_path_train)

    df = df.dropna()
    df = df.where(df.signal.str.len() > 0).dropna()

    output_file_path = extend_path_by_suffix_before_filetype(
        path = input_dataset_file_path_train,
        suffix = output_path_suffix,
    )

    make_dir_with_parents(output_file_path)
    df.to_pickle(output_file_path)
    
    return output_file_path


def predict_genes_without_signal_as_zeros(
    input_file_path,
    prediction_column_name,
    output_path_suffix,
):
    df = pd.read_pickle(input_file_path)    
    df.loc[df.signal.isnull(), prediction_column_name] = 0
    df.loc[df.signal.str.len() == 0, prediction_column_name] = 0
    
    output_file_path = extend_path_by_suffix_before_filetype(
        path = input_file_path,
        suffix = output_path_suffix,
    )

    make_dir_with_parents(output_file_path)
    df.to_pickle(output_file_path)
    
    return output_file_path


def zeros_to_negative(input_array):
    if check_empty_or_not_array(input_array):
        return input_array
    negative_array = [-0.1 if (x == 0) else x for x in input_array]
    return negative_array

def fill_empty_signal(input_array):
    if check_empty_or_not_array(input_array):
        return [0] # this is just an empty signal filling/padding
    return input_array

def apply_map_function_to_column(
    input_file_path,
    column_name,
    map_function,
    output_path_suffix,
):
    df = pd.read_pickle(input_file_path)    
    df[column_name] = df[column_name].apply(map_function)
    
    output_file_path = extend_path_by_suffix_before_filetype(
        path = input_file_path,
        suffix = output_path_suffix,
    )

    make_dir_with_parents(output_file_path)
    df.to_pickle(output_file_path)
    
    return output_file_path


def get_normalize_function(min_val, max_val):
    def normalize(input_array):
        if check_empty_or_not_array(input_array):
            return input_array
        return [((x - min_val) / (max_val - min_val)) for x in input_array]
    return normalize

def normalize_column_based_on_train_set(
    input_file_path_train,
    input_file_path_test,
    column,
    output_path_suffix,
):
    df_train = pd.read_pickle(input_file_path_train)
    df_test = pd.read_pickle(input_file_path_test)
    
    signal_max = df_train[column].explode().max()
    signal_min = df_train[column].explode().min()
    
    df_train[column] = df_train[column].apply(
        func=get_normalize_function(signal_min, signal_max)
    )
    df_test[column] = df_test[column].apply(
        func=get_normalize_function(signal_min, signal_max)
    ) 
    
    output_file_path_train = extend_path_by_suffix_before_filetype(
        path = input_file_path_train,
        suffix = output_path_suffix,
    )
    output_file_path_test = extend_path_by_suffix_before_filetype(
        path = input_file_path_test,
        suffix = output_path_suffix,
    )

    make_dir_with_parents(output_file_path_train)
    df_train.to_pickle(output_file_path_train)
    make_dir_with_parents(output_file_path_test)
    df_test.to_pickle(output_file_path_test)
    
    return output_file_path_train, output_file_path_test


def add_predictions_to_test_dataframe(
    df_test,
    gene_to_predictions,
    output_file_path,
):
    df_predictions = pd.DataFrame(gene_to_predictions.items(), columns=['gene_symbol' ,'prediction'])
    merged = pd.merge(df_test, df_predictions, on='gene_symbol', how='left')
    merged.loc[merged.prediction_x == 0, 'prediction_y'] = 0
    merged = merged.rename(columns={'prediction_y': 'prediction'})
    merged = merged.drop('prediction_x', axis='columns')
    
    make_dir_with_parents(output_file_path)
    df_test.to_pickle(output_file_path)
    
    return merged, output_file_path