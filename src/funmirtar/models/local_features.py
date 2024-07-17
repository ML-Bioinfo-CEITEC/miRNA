import numpy as np
import heapq

def filter_highest_islands(signal, binding_sites_count=10, min_length=6, max_length=22, neighborhood=50, max_overlap=0.5):
    """
    Filters the input signal to isolate the top X islands of miRNA binding sites based on AUC with limited overlap.
    
    Parameters:
    signal (np.array): Input signal array.
    binding_sites_count (int): Number of top islands to retain based on AUC.
    min_length (int): Minimum length of the core of the island.
    max_length (int): Maximum length of the core of the island.
    neighborhood (int): Length of the neighborhood to retain around the islands.
    max_overlap (float): Maximum allowed overlap between islands as a fraction of the island length.
    
    Returns:
    tuple: (Filtered signal with the top X islands and their neighborhoods retained, List of top islands)
    """
    # Initialize the filtered signal with zeros
    filtered_signal = np.zeros_like(signal)
    
    # Calculate the AUC for all possible subarrays within the length range
    island_aucs = []
    for length in range(min_length, max_length + 1):
        for start in range(len(signal) - length + 1):
            end = start + length
            core_signal = signal[start:end]
            auc = np.sum(core_signal)
            island_aucs.append((auc, start, end))
            # print(f"AUC for subarray from {start} to {end} is {auc}")
    
    # Find the top X islands based on AUC with limited overlap
    top_islands = []
    while len(top_islands) < binding_sites_count and island_aucs:
        auc, start, end = heapq.nlargest(1, island_aucs, key=lambda x: x[0])[0]
        island_aucs.remove((auc, start, end))
        
        # Check for overlap with already selected islands
        overlap_allowed = True
        for _, s, e in top_islands:
            overlap = max(0, min(e, end) - max(s, start))
            if overlap / (end - start) > max_overlap:
                overlap_allowed = False
                break
        
        if overlap_allowed:
            top_islands.append((auc, start, end))
    
    # print(f"Top islands based on AUC with limited overlap: {top_islands}")
    
    # Process each top island and apply the neighborhood filter
    for _, start, end in top_islands:
        neighborhood_start = max(start - neighborhood, 0)
        neighborhood_end = min(end + neighborhood, len(signal))
        filtered_signal[neighborhood_start:neighborhood_end] = signal[neighborhood_start:neighborhood_end]
    
    return filtered_signal, top_islands


def analyze_binding_sites(signal, sequence, conservation, binding_sites, neighborhood=50):
    """
    Analyzes the top binding sites of miRNA binding sites and extracts various features.

    Parameters:
    signal (np.array): Input signal array.
    sequence (np.array): Sequence array corresponding to the signal.
    /// conservation (np.array): Conservation array corresponding to the signal. ///
    binding_sites (list): List of top binding sites with (AUC, start, end) tuples.
    neighborhood (int): Length of the neighborhood to analyze around the binding sites.

    Returns:
    dict: Dictionary of features for each binding site.
    """
    features_dict = {}

    for i, (_, start, end) in enumerate(binding_sites):
        # Binding site features
        site_distance_from_start = start
        site_distance_from_end = len(signal) - end
        
        # Determine the closest binding sites on the left and right
        left_sites = [(s, e) for _, s, e in binding_sites if s < start]
        right_sites = [(s, e) for _, s, e in binding_sites if s > end]

        if left_sites:
            closest_left_site_end = max(left_sites, key=lambda x: x[1])[1]
            relative_position_left = start - closest_left_site_end
        else:
            relative_position_left = start

        if right_sites:
            closest_right_site_start = min(right_sites, key=lambda x: x[0])[0]
            relative_position_right = closest_right_site_start - end
        else:
            relative_position_right = len(signal) - end
        
        # conservation_peak = np.max(conservation[start:end])
        # conservation_auc = np.sum(conservation[start:end])
        
        # Signal statistics
        signal_peak = np.max(signal[start:end])
        signal_auc = np.sum(signal[start:end])
        signal_mean = np.mean(signal[start:end])

        features_dict[f'site_distance_from_start_{i}'] = site_distance_from_start
        features_dict[f'site_distance_from_end_{i}'] = site_distance_from_end
        features_dict[f'relative_position_left_{i}'] = relative_position_left
        features_dict[f'relative_position_right_{i}'] = relative_position_right
        # features_dict[f'conservation_peak_{i}'] = conservation_peak
        # features_dict[f'conservation_auc_{i}'] = conservation_auc
        features_dict[f'signal_peak_{i}'] = signal_peak
        features_dict[f'signal_auc_{i}'] = signal_auc
        features_dict[f'signal_mean_{i}'] = signal_mean

        # Neighborhood features
        neighborhood_start = max(start - neighborhood, 0)
        neighborhood_end = min(end + neighborhood, len(signal))

        # Before-neighborhood features
        before_neighborhood_seq = sequence[neighborhood_start:start]
        # before_neighborhood_cons = conservation[neighborhood_start:start]
        before_au_content = (before_neighborhood_seq.count('A') + before_neighborhood_seq.count('T')) / len(before_neighborhood_seq) if len(before_neighborhood_seq) > 0 else 0
        before_cg_content = (before_neighborhood_seq.count('C') + before_neighborhood_seq.count('G')) / len(before_neighborhood_seq) if len(before_neighborhood_seq) > 0 else 0
        # before_cons_peak = np.max(before_neighborhood_cons) if len(before_neighborhood_cons) > 0 else 0
        # before_cons_auc = np.sum(before_neighborhood_cons) if len(before_neighborhood_cons) > 0 else 0

        features_dict[f'before_au_content_{i}'] = before_au_content
        features_dict[f'before_cg_content_{i}'] = before_cg_content
        # features_dict[f'before_cons_peak_{i}'] = before_cons_peak
        # features_dict[f'before_cons_auc_{i}'] = before_cons_auc

        # After-neighborhood features
        after_neighborhood_seq = sequence[end:neighborhood_end]
        # after_neighborhood_cons = conservation[end:neighborhood_end]
        after_au_content = (after_neighborhood_seq.count('A') + after_neighborhood_seq.count('T')) / len(after_neighborhood_seq) if len(after_neighborhood_seq) > 0 else 0
        after_cg_content = (after_neighborhood_seq.count('C') + after_neighborhood_seq.count('G')) / len(after_neighborhood_seq) if len(after_neighborhood_seq) > 0 else 0
        # after_cons_peak = np.max(after_neighborhood_cons) if len(after_neighborhood_cons) > 0 else 0
        # after_cons_auc = np.sum(after_neighborhood_cons) if len(after_neighborhood_cons) > 0 else 0

        features_dict[f'after_au_content_{i}'] = after_au_content
        features_dict[f'after_cg_content_{i}'] = after_cg_content
        # features_dict[f'after_cons_peak_{i}'] = after_cons_peak
        # features_dict[f'after_cons_auc_{i}'] = after_cons_auc

    return features_dict


def get_binding_site_features(df):

    def get_local_features(row):
        signal = row['signal']
        sequence = row['sequence']
        conservation = row['conservation_phylo']
        
        if len(signal) == 0 or len(conservation) == 0:
            return {}
        
        _, top_islands_list = filter_highest_islands(
            signal, 
            binding_sites_count=10, 
            min_length=14,
            max_length=14,
            neighborhood=50, 
            max_overlap=0.2,
        )
        features = analyze_binding_sites(signal, sequence, conservation, top_islands_list)
        return features

    features_dicts = [get_local_features(row) for idx,row in df.iterrows()]

    return features_dicts
