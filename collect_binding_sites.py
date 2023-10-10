import json
import numpy as np 
  
    
# helper functions
def get_islands(arr):
    # Find start and end+1 of each island
    starts = np.array([], int)
    ends = np.array([], int)
    for i in range(len(arr)-1):        
        if arr[i] == 0 and arr[i+1] != 0 or i==0 and arr[i] != 0:
            # starts.append(i+1)
            starts = np.append(starts, i+1)
        if arr[i] != 0 and arr[i+1] == 0:
            # or arr[i] != 0 and len(arr) == i
            # ends.append(i)
            ends = np.append(ends, i+1)

    # Get lengths of each island
    lengths = ends - starts

    # Return as pairs of (start, length)
    # return list(zip(starts, ends, lengths))
    return (starts, ends, lengths)


# main function
def collect_binding_sites(load_scores_path, miRNA_seq):
    binding_sites = []
    #### Load the explainability scoring
    with open(load_scores_path, 'r') as file:
        miRNA_to_gene_score_loaded = json.load(file)

        # Group binding sites == skip the 0s inbetween scores
        for scores in miRNA_to_gene_score_loaded[miRNA_seq]:
            binding_sites.append(get_islands(scores[1]))
    
    return binding_sites
    
        

            
        