import pandas as pd
import pyBigWig
import numpy as np
import argparse

def get_conservation(bw_file, chrom, ensembl_start, ensembl_end):
    # Adjust Ensembl coordinates to UCSC format (0-based, half-open at the end)
    ucsc_start = ensembl_start - 1
    ucsc_end = ensembl_end  # The end coordinate remains the same

    if not chrom.startswith('chr'):
        chrom = 'chr' + chrom

    conservation_scores = bw_file.values(chrom, ucsc_start, ucsc_end)

    return conservation_scores

def main():
    parser = argparse.ArgumentParser(description='Process conservation scores for UTRs.')
    parser.add_argument('--conservation_path', required=True, help='Path to the conservation BigWig file')
    parser.add_argument('--utr_df_mirna_fc_chr_path', required=True, help='Path to the input UTR dataframe pickle')
    parser.add_argument('--utr_df_mirna_fc_chr_conservation_path', required=True, help='Path to the output UTR dataframe pickle with conservation data')

    args = parser.parse_args()

    # Assign arguments to variables
    CONSERVATION_PATH = args.conservation_path
    UTR_DF_MIRNA_FC_CHR_PATH = args.utr_df_mirna_fc_chr_path
    UTR_DF_MIRNA_FC_CHR_CONSERVATION_PATH = args.utr_df_mirna_fc_chr_conservation_path

    # Load the UTR dataframe
    df = pd.read_pickle(UTR_DF_MIRNA_FC_CHR_PATH)

    # Rename columns for consistency
    df = df.rename({'Start': 'utr3_start', 'End': 'utr3_end', 'Chromosome': 'chromosome', 'Strand': 'strand'}, axis=1)

    # Open the BigWig file
    bw_file = pyBigWig.open(CONSERVATION_PATH)

    df_conservation = []
    errors = []
    error_loci = []

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        chrom = row.chromosome
        ensembl_start = row.utr3_start  # Ensembl coordinates (1-based, inclusive)
        ensembl_end = row.utr3_end      # Ensembl coordinates (1-based, inclusive)
        strand = row.strand

        if pd.isnull(ensembl_start):
            df_conservation.append([])
            continue

        if isinstance(ensembl_start, (float, int, np.integer)):
            ensembl_start = [int(ensembl_start)]
            ensembl_end = [int(ensembl_end)]
        else:
            ensembl_start = [int(x) for x in str(ensembl_start).split(';')]
            ensembl_end = [int(x) for x in str(ensembl_end).split(';')]

        if len(ensembl_start) > 1:
            ensembl_start.sort()
            ensembl_end.sort()

        row_conservation = []
        for start, end in zip(ensembl_start, ensembl_end):
            try:
                exon_conservation = get_conservation(bw_file, chrom, start, end)
                row_conservation.extend(exon_conservation)
            except RuntimeError as er:
                errors.append(er)
                error_loci.append((start, end, chrom))

        df_conservation.append(row_conservation)

    # Close the BigWig file
    bw_file.close()

    # Add conservation data to the dataframe
    df['conservation_phylo'] = df_conservation

    # Save the updated dataframe
    df.to_pickle(UTR_DF_MIRNA_FC_CHR_CONSERVATION_PATH)

if __name__ == '__main__':
    main()
