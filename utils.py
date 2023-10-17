import pandas as pd 
import numpy as np
import torch
import torchmetrics
import warnings


def rna_to_dna(rna_sample):
    rna_dic = {
        'A':'A',
        'C':'C',
        'U':'T',
        'G':'G',
    }
    return [rna_dic[x.upper()] for x in rna_sample]

def get_bratel_gene_fc(mirna_name):
    mirna_FCs = pd.read_csv('mirna_fcs.csv',index_col=0, header=0, sep=',')
    # mirna_FCs[['Gene symbol',my_miRNA_name]]
    
    mirna_FCs_dict = mirna_FCs.set_index('Gene symbol')[mirna_name].to_dict()

    return mirna_FCs_dict


def get_labels(mirna_name, padded_data_tensor, input_data_genes):
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
            
    print("There is ", len(not_found_genes), "genes for which we do not have fold change because they are not in the Bartel table, out of total", len(input_labels_unfiltered), "and", len(nan_genes), "nan valued genes in FC table")
    
    for i in gene_indices_to_remove:
        input_labels_unfiltered[i] = "remove"

    input_labels =  [i for i in input_labels_unfiltered if i != "remove"]
    result_data_tensor =  [padded_data_tensor[i] for i in range(len(input_labels_unfiltered)) if input_labels_unfiltered[i] != "remove"]
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
