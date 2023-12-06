import pickle
import argparse
import pandas as pd
import numpy as np

mirna_sequences = {'hsa-miR-16-5p': 'UAGCAGCACGUAAAUAUUGGCG', 
                   'hsa-miR-106b-5p': 'UAAAGUGCUGACAGUGCAGAU', 
                   'hsa-miR-200a-3p': 'UAACACUGUCUGGUAACGAUGU', 
                   'hsa-miR-200b-3p': 'UAAUACUGCCUGGUAAUGAUGA', 
                   'hsa-miR-215-5p': 'AUGACCUAUGAAUUGACAGAC', 
                   'hsa-let-7c-5p': 'UGAGGUAGUAGGUUGUAUGGUU', 
                   'hsa-miR-103a-3p': 'AGCAGCAUUGUACAGGGCUAUGA'}

def load_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def transform_data(data, mirna_sequences, columns):
    seq_array = []
    for miRNA in mirna_sequences.keys():
        for i in data.index.values:
            if not np.isnan(data[miRNA][i]):
                seq_array.append([
                    mirna_sequences[miRNA],
                    str(data['sequence'][i]),
                    data[miRNA][i]
                ])
    return pd.DataFrame(seq_array, columns = columns)
    
    
def save_result(df, filepath):
    df.to_csv(filepath, index=False)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filepath', type=str,
                        help='Path to input file with data')
    parser.add_argument('--output_filepath', type=str,
                        help='Optional path to output file.'
                        'Default - the same as input, only adding \'nlp.csv\' to the filename.')
    args = parser.parse_args()
    
    if args.output_filepath is None:
        args.output_filepath = args.input_filepath + ".nlp.csv"

    data = load_data(args.input_filepath)
    seq_df = transform_data(data, mirna_sequences, ['miRNA', 'utr3', 'log2fold'])
    save_result(seq_df, args.output_filepath)