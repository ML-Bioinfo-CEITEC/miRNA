import re
import numpy as np
import pandas as pd
from Bio import SeqIO
from collections.abc import Sequence


def get_ids_without_version(id_x):
    split = id_x.split('.')
    if len(split) > 1:
        return split[0]
    else:
        return id_x

def parse_biomart_fasta(fasta_path):
    """
     Reads a FASTA file got from the Ensembl BioMart and parses its content into a list of dictionaries. Each dictionary represents a sequence record with attributes extracted from the FASTA's ID string of the sequence. 
    
    Parameters:
    - fasta_path: path to a specific FASTA file with the coresponding structure of the sequence's ID string
    """
    sequences = []
    i = 0
    # Open the file and parse it
    with open(fasta_path, 'r') as fasta_file:
        for record in SeqIO.parse(fasta_file, 'fasta'):
            id_string = record.id
            parts = id_string.split("|")
            sequences.append(
                {
                    # "utr3_start": [int(pos) for pos in parts[1].split(';')],
                    # "utr3_end": [int(pos) for pos in parts[2].split(';')],
                    "ensembl_gene_id": parts[0],
                    "ensembl_gene_id_version": parts[1],
                    "3_utr_start": parts[2],
                    "3_utr_end": parts[3],
                    "chromosome_name": parts[4],
                    "external_gene_name": parts[5],
                    "start_position": parts[6],
                    "end_position": parts[7],
                    "strand": parts[8],
                    "transcript_start": parts[9],
                    "transcript_end": parts[10],
                    "transcript_length": parts[11],
                    "ensembl_transcript_id": parts[12],
                    "ensembl_transcript_id_version": parts[13],

                    "sequence": record.seq,
                }
            )
    return sequences


def get_train_and_test_set(data_folder_path, dataset_name, mirna_names, file_extension_train, file_extension_test):
    input_dataset_file_path_train = '{}{}.{}{}'.format(
        data_folder_path,
        dataset_name, 
        mirna_names[0],
        file_extension_train
    )
    df_train = pd.read_pickle(input_dataset_file_path_train)

    input_dataset_file_path_test = '{}{}.{}{}'.format(
        data_folder_path,
        dataset_name, 
        mirna_names[0],
        file_extension_test
    )
    df_test = pd.read_pickle(input_dataset_file_path_test)

    if len(mirna_names) > 1:
        for name in mirna_names[1:]:
            input_dataset_file_path_train = '{}{}.{}{}'.format(
                data_folder_path,
                dataset_name, 
                name,
                file_extension_train
            )
            df_train = pd.concat(
                [df_train, pd.read_pickle(input_dataset_file_path_train)], 
                axis='index',
                ignore_index=True,
            )

            input_dataset_file_path_test = '{}{}.{}{}'.format(
                data_folder_path,
                dataset_name, 
                name,
                file_extension_test
            )
            df_test = pd.concat(
                [df_test, pd.read_pickle(input_dataset_file_path_test)],
                axis='index',
                ignore_index=True,
            )

    return df_train, df_test


def compress_zero_subsequences(signal):
    if not isinstance(signal, (Sequence, np.ndarray)):
    # np.isnan(signal)
        return np.nan
    if len(signal) == 0:
        return []
    
    new_signal = []
    zero_count = 0
    
    for num in signal:
        if num == 0:
            zero_count += 1
        else:
            if zero_count > 0:
                new_signal.extend([0] * (ceil((zero_count / 100))))
                zero_count = 0
            new_signal.append(num)
    
    if zero_count > 0:
        new_signal.extend([0] * (ceil((zero_count / 100))))
    
    return new_signal


def compress_zero_subsequences_with_conservation(signal, conservation):
    # WHAT IS THE DIFFERENCE HERE?
    # I GUESS WE HAVE [] IF there is no signal prediction
    if not isinstance(signal, (Sequence, np.ndarray)):
    # np.isnan(signal)
        return np.nan, np.nan
    if len(signal) == 0:
        return [], []
    
    new_signal = []
    new_conservation = []
    zero_count = 0
    
    for num, cons in zip(signal, conservation):
        if num == 0:
            zero_count += 1
        else:
            if zero_count > 0:
                new_signal.extend([0] * (ceil((zero_count / 100))))
                new_conservation.extend([0] * (ceil((zero_count / 100))))
                zero_count = 0
            new_signal.append(num)
            new_conservation.append(cons)
    
    if zero_count > 0:
        new_signal.extend([0] * (ceil((zero_count / 100))))
        new_conservation.extend([0] * (ceil((zero_count / 100))))
    
    return new_signal, new_conservation


def check_empty_or_not_array(input_array):
    return (not isinstance(input_array, (Sequence, np.ndarray))) or (len(input_array) == 0)


def parse_UCSC_utr_txt(fasta_path):
    # List to hold all parsed records
    records = []

    # Open the file and parse it
    with open(fasta_path, 'r') as fasta_file:
        for record in SeqIO.parse(fasta_file, 'fasta'):
            # Extract necessary information
            record_id = record.id
            record_name = record.name
            record_description = record.description
            record_sequence = str(record.seq)

            # Extract the latter part of the ID by removing the prefix "hg19_ncbiRefSeq_"
            extracted_id = record_id.replace("hg19_ncbiRefSeq_", "")

            # Use regular expressions to extract chromosome, start, end, and strand
            match = re.search(r'range=(chr[\w]+):(\d+)-(\d+).*strand=([\+\-])', record_description)
            
            if match:
                chromosome = match.group(1)
                start = int(match.group(2))
                end = int(match.group(3))
                strand = match.group(4)
            else:
                chromosome = None
                start = None
                end = None
                strand = None

            # Create a dictionary for the record
            record_dict = {
                "ID": extracted_id,  # Use the extracted ID
                "Name": record_name,
                "Description": record_description,
                "Sequence": record_sequence,
                "Chromosome": chromosome,
                "Start": start,
                "End": end,
                "Strand": strand
            }

            # Append the dictionary to the list
            records.append(record_dict)
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(records)
    
    return df