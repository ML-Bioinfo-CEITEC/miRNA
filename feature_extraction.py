from modules.collect_scores.bymaxvalue import bymaxvaluefunc
from modules.collect_scores.bymean import bymeanfunc

import json


FEATURES = [0, 1, 2, 3]
FEATURE_NAMES = ['max', 'mean', 'bs_start', 'transcript_len']

    
# analyze binding sites in genes for the given miRNA. 
# For each gene's binding site, it computes statistics (stat_one and stat_two), the start of the binding site, and the length of gene
# outputs:
#        input_data list 
#        input_data_genes - corresponding gene names
#        transcripts_with_no_bs - genes that have no binding sites
def count_statistics(binding_sites, load_scores_path, miRNA_seq):
    with open(load_scores_path, 'r') as file:
        miRNA_to_gene_score_loaded = json.load(file)
    
    input_data = []
    input_data_genes = []
    transcripts_with_no_bs = []

    scores_per_gene = miRNA_to_gene_score_loaded[miRNA_seq]

    statistical_funcs = [bymaxvaluefunc, bymeanfunc]

    for i in range(len(scores_per_gene)):
        scores = scores_per_gene[i][1]
        gene_name = scores_per_gene[i][0]
        seq_binding = []
        starts, ends, lengths = binding_sites[i]

        # for each binding site: count features
        for k in range(len(starts)):

            if starts[k].any():
            # if start.size > 0:
                # this dictates the number of features
                stat_one = statistical_funcs[0](scores[starts[k]:ends[k]])
                stat_two = statistical_funcs[1](scores[starts[k]:ends[k]])
                # TODO think about the modeling of input data because now the " len(scores)" is the same number in every binding site (quartet) of single transcript
                seq_binding.append([stat_one, stat_two, starts[k], len(scores)])
            else:
                seq_binding.append([0, 0, 0, 0])

        if len(seq_binding) > 0:
            input_data.append(seq_binding)
            input_data_genes.append(gene_name)
        else:
            transcripts_with_no_bs.append(gene_name)
            
    return input_data, input_data_genes, transcripts_with_no_bs


#  Normalize input data
def normalize_statistics(input_data):
    feature_maxmins = {}

    for feature in FEATURES:
        x_min = 999999
        x_max = -1
        for sample in input_data:
            for binding_site in sample:
                item = binding_site[feature]
                if x_min > item:
                    x_min = item
                if x_max < item:
                    x_max = item

        feature_maxmins['feature' + str(feature)] = (x_min, x_max)
    
    input_data_normalized = []

    for sample in input_data:
        new_sample = []
        for binding_site in sample:
            new_binding_site = []
            for feature in FEATURES:
                x_min = list(feature_maxmins.items())[feature][1][0]
                x_max = list(feature_maxmins.items())[feature][1][1]
                item = binding_site[feature]
                item = (item - x_min) / (x_max - x_min)
                new_binding_site.append(item)
            new_sample.append(new_binding_site)
        input_data_normalized.append(new_sample)    
        
    return input_data_normalized