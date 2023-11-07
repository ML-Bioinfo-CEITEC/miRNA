import pandas as pd 
import numpy as np
import torch
import torchmetrics
import os
import warnings
import datetime
from pathlib import Path
from math import ceil
from collections.abc import Sequence


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


def compress_zero_subsequences(arr):
    if not isinstance(arr, (Sequence, np.ndarray)):
    # np.isnan(arr)
        return np.nan
    if len(arr) == 0:
        return []
    
    new_arr = []
    zero_count = 0
    
    for num in arr:
        if num == 0:
            zero_count += 1
        else:
            if zero_count > 0:
                new_arr.extend([0] * (ceil((zero_count / 100))))
                zero_count = 0
            new_arr.append(num)
    
    if zero_count > 0:
        new_arr.extend([0] * (ceil((zero_count / 100))))
    
    return new_arr


def check_empty_or_not_array(input_array):
    return (not isinstance(input_array, (Sequence, np.ndarray))) or (len(input_array) == 0)


def get_experiment_time_id(sufix=''):
    return datetime.datetime.now().strftime("%d-%m-%y_%H-%M") + '.' + sufix





def rna_to_dna(rna_sample):
    rna_dic = {
        'A':'A',
        'C':'C',
        'U':'T',
        'G':'G',
    }
    converted = [rna_dic[x.upper()] for x in rna_sample]
    new = ""
    # traverse in the string
    for x in converted:
        new += x
    # return string
    return new


def get_our_miRNAs(as_DNA_string = False):
    mirna_seqs = ['UAGCAGCACGUAAAUAUUGGCG', 'UAAAGUGCUGACAGUGCAGAU', 'UAACACUGUCUGGUAACGAUGU', 'UAAUACUGCCUGGUAAUGAUGA', 'AUGACCUAUGAAUUGACAGAC', 'UGAGGUAGUAGGUUGUAUGGUU', 'AGCAGCAUUGUACAGGGCUAUGA']
    mirna_names = ['hsa-miR-16-5p', 'hsa-miR-106b-5p', 'hsa-miR-200a-3p', 'hsa-miR-200b-3p', 'hsa-miR-215-5p', 'hsa-let-7c-5p', 'hsa-miR-103a-3p']
    if as_DNA_string:
        mirna_seqs = [rna_to_dna(x) for x in mirna_seqs]
    return pd.DataFrame(
            {'mirna_name': mirna_names, 'mirna_sequence':mirna_seqs},
        )


def get_bratel_gene_fc(mirna_name):
    mirna_FCs = pd.read_csv('mirna_fcs.csv',index_col=0, header=0, sep=',')
    # mirna_FCs[['Gene symbol',my_miRNA_name]]
    
    mirna_FCs_dict = mirna_FCs.set_index('Gene symbol')[mirna_name].to_dict()

    return mirna_FCs_dict


def get_labels(mirna_name, input_data, input_data_genes):
    mirna_FCs = pd.read_csv('mirna_fcs.csv',index_col=0, header=0, sep=',')
    # mirna_FCs[['Gene symbol',my_miRNA_name]]
    
    mirna_FCs_dict = get_bratel_gene_fc(mirna_name)
    # df_dict

    input_labels_unfiltered = []
    not_found_genes = []
    gene_indices_to_remove = []
    nan_genes = []

    for i, gene in enumerate(input_data_genes):
        try:
            fc = mirna_FCs_dict[gene]
            if np.isnan(fc):
                nan_genes.append(gene)
                input_labels_unfiltered.append(0)
                gene_indices_to_remove.append(i)            
            else:
                input_labels_unfiltered.append(fc)
        except KeyError as e:
            not_found_genes.append(gene)
            input_labels_unfiltered.append(0)
            gene_indices_to_remove.append(i)
            
    print("We have predicted ", len(not_found_genes), "genes for which we do not have fold change because they are not in the Bartel table, out of total", len(input_labels_unfiltered), "and", len(nan_genes), "nan valued genes in FC table")
    
    for i in gene_indices_to_remove:
        input_labels_unfiltered[i] = "remove"

    input_labels =  [i for i in input_labels_unfiltered if i != "remove"]
    result_data_tensor =  [input_data[i] for i in range(len(input_labels_unfiltered)) if input_labels_unfiltered[i] != "remove"]
    input_data_genes_filtered = [input_data_genes[i] for i in range(len(input_labels_unfiltered)) if input_labels_unfiltered[i] != "remove"]
    
    return input_labels, result_data_tensor, input_data_genes_filtered


# params: 
# dictionary: {} where items are of type 'tensor'
def unpack_tensor_items(dictionary):
    return {k: v.item() for k, v in dictionary.items()}


def get_metrics(predictions, labels):
    metrics = {}

    mae = torchmetrics.MeanAbsoluteError()(predictions, labels)
    mse = torchmetrics.MeanSquaredError()(predictions, labels)
    r2 = torchmetrics.R2Score()(predictions, labels)
    rmse = torch.sqrt(mse)
    
    metrics['mse'] = mse
    metrics['mae'] = mae
    metrics['r2'] = r2
    metrics['rmse'] = rmse
    return metrics
    

def get_baseline_metrics(baseline_prediction, y_test):
    # Baseline metrics
    baseline_mae = torchmetrics.MeanAbsoluteError()(torch.tensor(baseline_prediction), torch.tensor(np.array(y_test)))
    metrics = unpack_tensor_items(
        get_metrics(
            torch.tensor(baseline_prediction), 
            torch.tensor(np.array(y_test))
        )
    )
    # np.corrcoef() can raise a warning if there is division by zero
    # like in the case of Corr(constant_random_variable, other_random_variable)
    # in that case the correlation is undefined
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        baseline_corr = np.corrcoef(baseline_prediction, y_test)[0][1]
    metrics['corr'] = None if np.isnan(baseline_corr) else baseline_corr
    return metrics