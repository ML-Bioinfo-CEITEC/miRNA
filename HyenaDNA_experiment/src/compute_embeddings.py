from transformers import PreTrainedModel
import argparse
from tqdm import tqdm
import pandas as pd
import re
import os
import json
import torch

from utils import save_dataframe_to_csv
from hyena_helpers import load_model, get_tokenizer, load_data_from_csv

def tokenize_sample(tokenizer, miRNA_seq, mRNA_seq):
    
    return tokenizer(miRNA_seq + "NNNNN" + mRNA_seq)["input_ids"]

def embedd_sample(model, tok_seq, device):
    
    tok_seq = torch.LongTensor(tok_seq).unsqueeze(0)
    tok_seq = tok_seq.to(device)

    # prep model and forward
    model.to(device)
    model.eval()
    with torch.inference_mode():
        embeddings = model(tok_seq)
        
    # Convert the tensor to float if it's not already
    sequence_embedding = embeddings.float()

    # Mean pooling along the sequence length dimension (axis=1)
    mean_pooled_embedding = torch.mean(sequence_embedding, dim=1)
        
    # extracting embeddings as a simple list of values
    return list(mean_pooled_embedding.detach().cpu().numpy()[0])

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
    
    model, max_length = load_model()
    tokenizer = get_tokenizer(max_length)
    data = load_data_from_csv(args.input_filepath)
    
    tqdm.pandas()
    data['tokenized'] = data.progress_apply(lambda x: tokenize_sample(tokenizer, x['miRNA'], x['utr3']), axis=1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    
    # TODO: maybe try batches?
    data['embedded'] = data.progress_apply(lambda x: embedd_sample(model, x['tokenized'], device), axis=1)
    
    # converting list to string on purpose, so we can control saving and then loading from the file
    data["tokenized"] = data["tokenized"].apply(lambda x: ','.join(str(i) for i in x))
    data["embedded"] = data["embedded"].apply(lambda x: ','.join(str(i) for i in x))
    
    save_dataframe_to_csv(data, args.output_filepath)