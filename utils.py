import pandas as pd 
import numpy as np

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
