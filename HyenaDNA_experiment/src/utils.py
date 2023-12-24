import pickle
import pandas as pd

MIRNAS_NAME_SEQUENCE_DICT = {'hsa-miR-16-5p': 'UAGCAGCACGUAAAUAUUGGCG', 
                             'hsa-miR-106b-5p': 'UAAAGUGCUGACAGUGCAGAU', 
                             'hsa-miR-200a-3p': 'UAACACUGUCUGGUAACGAUGU', 
                             'hsa-miR-200b-3p': 'UAAUACUGCCUGGUAAUGAUGA', 
                             'hsa-miR-215-5p': 'AUGACCUAUGAAUUGACAGAC', 
                             'hsa-let-7c-5p': 'UGAGGUAGUAGGUUGUAUGGUU', 
                             'hsa-miR-103a-3p': 'AGCAGCAUUGUACAGGGCUAUGA'}

REGRESSION_LABEL_COLUMN = 'log2fold'
MRNA_SEQ_COLUMN = 'utr3'
MIRNA_SEQ_COLUMN = 'miRNA'
COLUMNS_TO_KEEP_FROM_INPUT_DATA = [MIRNA_SEQ_COLUMN, MRNA_SEQ_COLUMN, REGRESSION_LABEL_COLUMN]
REGRESSION_LABEL_COLUMN_CLAMPED_TO_ZERO = 'log2fold_clamped'
CLASSIFICATION_LABEL_COLUMN = 'cls_label'

def load_data_from_pickle(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def load_data_from_csv(filepath):
    return pd.read_csv(filepath)

def save_dataframe_to_csv(df, filepath):
    df.to_csv(filepath, index=False)