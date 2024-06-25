from Bio import SeqIO
import pandas as pd


def get_ids_without_version(id_x):
    split = id_x.split('.')
    if len(split) > 1:
        return split[0]
    else:
        return id_x

def parse_biomart_fasta(fasta_path):
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
            # i+=1
            # if i > 10:
                # break
    return sequences