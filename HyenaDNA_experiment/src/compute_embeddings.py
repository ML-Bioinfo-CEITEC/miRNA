#from transformers import PreTrainedModel
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import DataLoader

from utils import save_dataframe_to_csv, load_data_from_csv
from utils import MRNA_SEQ_COLUMN, MIRNA_SEQ_COLUMN, CLASSIFICATION_LABEL_COLUMN
from hyenadna import HyenaDNAEncoder, Pooler, POOL_OPTIONS
from hyenadna_helpers import miRNA_Dataset

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filepath', type=str,
                        help='Path to input file with data')
    parser.add_argument('--output_filepath', type=str,
                        help='Optional path to output file.'
                        'Default - the same as input, only adding \'embedd\' to the filename.')
    args = parser.parse_args()
    
    if args.output_filepath is None:
        args.output_filepath = args.input_filepath + ".embedd"
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)
    
    encoder = HyenaDNAEncoder()
    encoder.eval()
    encoder = encoder.to(device)
    
    data = load_data_from_csv(args.input_filepath)
    dset = miRNA_Dataset(
        data[MIRNA_SEQ_COLUMN].reset_index(drop=True),
        data[MRNA_SEQ_COLUMN].reset_index(drop=True),
        data[CLASSIFICATION_LABEL_COLUMN].reset_index(drop=True),
        max_length = encoder.tokenizer.model_max_length,
        use_padding = True,
        tokenizer=encoder.tokenizer,
        add_eos=False,
    )
    loader = DataLoader(dset, batch_size=16, shuffle=False)
    
    torch.set_grad_enabled(False)
    
    hiddens = []
    print("Running samples through HyenaDNA encoder")
    for idx, (sample, label) in enumerate(tqdm(loader)):
        sample = sample.to(device)
        hidden = encoder(sample)
        hiddens.extend(hidden.detach().cpu())
    
    for mode_pooler in POOL_OPTIONS:
        print("Using pool option", mode_pooler)
        
        embeddings = []
        
        pooler = Pooler(mode = mode_pooler, l_output = 0) 
    
        # put model in train mode
 
        pooler.eval()
        pooler = pooler.to(device)
        
        for hidden in tqdm(hiddens):
            hidden = hidden.to(device)
            embedd = pooler(hidden)

            embedd = ','.join(str(i) for i in list(embedd.detach().cpu().numpy()))
            embeddings.append(embedd)
    
        data['embedd_' + mode_pooler] = embeddings
    
    save_dataframe_to_csv(data, args.output_filepath)