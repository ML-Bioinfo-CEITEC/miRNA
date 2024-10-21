import pandas as pd
import numpy as np
from Bio.Seq import Seq
import argparse
from funmirtar.utils.data_processing import parse_UCSC_utr_txt

def main():
    parser = argparse.ArgumentParser(description='Process UTR sequences with miRNA fold changes.')

    parser.add_argument('--in_utr_seq_txt_path', default='../../data/GRCh37.p13 hg19/UCSC/3utr_sequences.txt', help='Input UTR sequence txt path')
    parser.add_argument('--mirna_fcs', default='../../data/fold_change/mirna_fcs.csv', help='miRNA fold change CSV path')
    parser.add_argument('--ts_gene_info', default='../../data/target_scan_8/Gene_info.txt', help='TargetScan gene info TXT path')
    parser.add_argument('--id_map', default='../../data/GRCh37.p13 hg19/UCSC/id_map', help='ID map path')
    parser.add_argument('--out_utr_df_mirna_fc_path', default='../../data/processed/GRCh37.p13 hg19/3utr.sequences.refseq_id.mirna_fc.pkl', help='Output path for processed data')

    args = parser.parse_args()

    IN_UTR_SEQ_TXT_PATH = args.in_utr_seq_txt_path
    MIRNA_FCS = args.mirna_fcs
    TS_GENE_INFO = args.ts_gene_info
    ID_MAP = args.id_map
    OUT_UTR_DF_MIRNA_FC_PATH = args.out_utr_df_mirna_fc_path

    SEQUENCE_UNAVAILABLE = Seq("Sequenceunavailable")

    utrs_df = parse_UCSC_utr_txt(IN_UTR_SEQ_TXT_PATH)

    utrs_df = utrs_df.rename({'Sequence': 'sequence', 'ID': 'ID_versioned'}, axis='columns')
    utrs_df['RefSeq ID'] = utrs_df.ID_versioned.map(lambda ID: ID.split('.')[0])

    utrs_with_seq = utrs_df[utrs_df.sequence != SEQUENCE_UNAVAILABLE]

    mirna_fc = pd.read_csv(MIRNA_FCS)

    ucsc_id_map = pd.read_csv(
        ID_MAP,
        comment='#',
        header=None,
        names=[
            "knownGene.name",
            "knownGene.chrom",
            "kgAlias.kgID",
            "kgAlias.alias",
            "kgXref.kgID",
            "kgXref.mRNA",
            "kgXref.geneSymbol",
            "kgXref.refseq",
            "knownToEnsembl.name",
            "knownToEnsembl.value",
            "knownToRefSeq.name",
            "knownToRefSeq.value"
        ],
        delimiter='\t'
    )

    ucsc_id_map['RefSeq ID'] = ucsc_id_map['knownToRefSeq.value'].fillna(ucsc_id_map['kgXref.refseq'])
    ucsc_id_map.drop_duplicates(inplace=True)

    ts_gene_info = pd.read_csv(TS_GENE_INFO, sep='\t')
    ts_gene_info = ts_gene_info[['Transcript ID', 'Representative transcript?']]

    mirna_fc_ucsc_id = pd.merge(
        mirna_fc,
        ucsc_id_map,
        left_on='RefSeq ID',
        right_on='RefSeq ID',
        how='left',
    ).drop_duplicates()

    # Join UTR sequences with miRNA FC and IDs
    mirna_fc_ucsc_id_seq = pd.merge(
        mirna_fc_ucsc_id,
        utrs_with_seq,
        left_on='RefSeq ID',
        right_on='RefSeq ID',
        how='left',
    )

    ts_gene_info['ensembl_id_no_version'] = ts_gene_info['Transcript ID'].map(lambda ID: ID.split('.')[0])

    # Merge with TargetScan gene info
    mirna_fc_ucsc_id_seq_repre = pd.merge(
        mirna_fc_ucsc_id_seq,
        ts_gene_info,
        left_on='knownToEnsembl.value',
        right_on='ensembl_id_no_version',
        how='left',
    )

    ts_gene_info = pd.merge(
        ts_gene_info,
        ucsc_id_map,
        left_on='ensembl_id_no_version',
        right_on='knownToEnsembl.value',
        how='left',
    )

    utrs_with_seq_repre = pd.merge(
        utrs_with_seq,
        ts_gene_info,
        left_on='RefSeq ID',
        right_on='RefSeq ID',
        how='left',
    )

    # Keep representative/longest UTR per RefSeq ID
    mirna_fc_ucsc_id_seq_repre["sequence_origin"] = None
    mirna_fc_ucsc_id_seq_repre['utr3_length'] = mirna_fc_ucsc_id_seq_repre.apply(
        lambda a: len(a['sequence']) if not pd.isna(a['sequence']) else 0, axis=1
    )

    def get_seq_len(a):
        if a is None or a == '' or pd.isna(a):
            return 0
        else:
            return len(a)

    def has_sequence(index, df):
        return pd.notnull(index) and get_seq_len(df.iloc[index].sequence) > 0

    utrs_longest_seq = []
    refseq_id_not_resolved = []

    groupby_refseq = mirna_fc_ucsc_id_seq_repre.groupby("RefSeq ID")
    groupby_gene = utrs_with_seq_repre.groupby('kgXref.geneSymbol')
    groupby_ensembl_id = utrs_with_seq_repre.groupby('Transcript ID')

    for group_name, group in groupby_refseq:
        repre_index_per_gene = None
        longest_utr_index_per_gene = None
        repre_index_per_ensembl_id = None
        longest_utr_index_per_ensembl_id = None

        repre_transcript_index = group['Representative transcript?'].idxmax()
        longest_utr_index = group.sequence.map(lambda a: get_seq_len(a)).idxmax()
        longest_transcript_index = pd.to_numeric(group['utr3_length']).idxmax()

        genes_of_refseq_id = list(set(group['Gene symbol']))
        ensembl_ids_of_refseq_id = list(set(group['Transcript ID']))

        for gene in genes_of_refseq_id:
            if pd.notnull(gene) and gene in groupby_gene.groups.keys():
                group_of_gene_symbol = groupby_gene.get_group(gene)
                repre_index_per_gene = group_of_gene_symbol['Representative transcript?'].idxmax()
                longest_utr_index_per_gene = group_of_gene_symbol.sequence.map(lambda a: get_seq_len(a)).idxmax()

        for ensembl_id in ensembl_ids_of_refseq_id:
            if pd.notnull(ensembl_id) and ensembl_id in groupby_ensembl_id.groups.keys():
                group_of_ensembl_id = groupby_ensembl_id.get_group(ensembl_id)
                repre_index_per_ensembl_id = group_of_ensembl_id['Representative transcript?'].idxmax()
                longest_utr_index_per_ensembl_id = group_of_ensembl_id.sequence.map(lambda a: get_seq_len(a)).idxmax()

        indexes = {
            'repre_transcript': {
                'index': repre_transcript_index,
                'is_original_sequence': True,
                'dataframe': mirna_fc_ucsc_id_seq_repre,
            },
            'longest_utr': {
                'index': longest_utr_index,
                'is_original_sequence': True,
                'dataframe': mirna_fc_ucsc_id_seq_repre,
            },
            'repre_per_ensembl_id': {
                'index': repre_index_per_ensembl_id,
                'is_original_sequence': False,
                'dataframe': utrs_with_seq_repre
            },
            'longest_utr_per_ensembl_id': {
                'index': longest_utr_index_per_ensembl_id,
                'is_original_sequence': False,
                'dataframe': utrs_with_seq_repre
            },
            'repre_per_gene': {
                'index': repre_index_per_gene,
                'is_original_sequence': False,
                'dataframe': utrs_with_seq_repre
            },
            'longest_utr_per_gene': {
                'index': longest_utr_index_per_gene,
                'is_original_sequence': False,
                'dataframe': utrs_with_seq_repre
            },
        }

        for name, index in indexes.items():
            if has_sequence(index['index'], index['dataframe']):
                selected_sequence_row = index['dataframe'].iloc[index['index']]
                if index['is_original_sequence']:
                    selected_sequence_row["sequence_origin"] = 'RefSeq ID'
                    utrs_longest_seq.append(selected_sequence_row)
                elif pd.isnull(longest_transcript_index):
                    if group.shape[0] == 1:
                        original_row = group.iloc[0].copy()
                        original_row['sequence'] = selected_sequence_row['sequence']
                        if 'gene' in name:
                            original_row["sequence_origin"] = 'Gene symbol'
                        else:
                            original_row["sequence_origin"] = 'Transcript ID'
                        utrs_longest_seq.append(original_row)
                else:
                    original_row = mirna_fc_ucsc_id_seq_repre.iloc[longest_transcript_index]
                    original_row['sequence'] = selected_sequence_row['sequence']
                    utrs_longest_seq.append(original_row)
                break
        else:
            refseq_id_not_resolved.append(group_name)

    utrs_selected_seq_df = pd.DataFrame(utrs_longest_seq)

    missing_sequences = utrs_selected_seq_df[utrs_selected_seq_df.sequence.isna()]['RefSeq ID'].values
    utrs_selected_seq_df[utrs_selected_seq_df['RefSeq ID'].isin(missing_sequences)].to_csv('missing_sequences.csv')

    utrs_selected_seq_df.to_pickle(OUT_UTR_DF_MIRNA_FC_PATH)

if __name__ == '__main__':
    main()
