## Data processing from scratch
### 1) Sequence and LFC merging
```process_utr_sequences_hg19.fc.representative_transcript.ipynb```

First step of the pipeline. Takes 3'UTRs, LFCs, mapping of Ensembl and RefSeq IDs, and information from TS about what transcript is representative. It returns a dataset with labels, sequences, IDs and other transcript info. This sequences are later used for scanning.
- Gets hg19 3’UTR sequences by matches of 
    - RefSeq ID (through an ID map from UCSC), if that fails then:
    - Ensembl ID, if that fails then:
    - Gene Symbol
### 2) Scanning - producing explainability signal

Will be added later
### 3) Merge explainability signal with sequences and conservation, split to train/test

Will be added later. This produces the final train/test datasets for the 7miRs HCT116 transfection



## Running the local feature extraction
```extract_local_features.sh```

The following files are expected to run the local feature extraction for a new local feature:

```CONFIG_PATH```
Given by the CONFIG_PATH parameter. In this config you include the functions the script will use to extract the local features 
="user_config.py"


data_folder_path="../data/preprocessed_for_ml/"
train_dataset="3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.train.pkl"
test_dataset="3utr.sequences.refseq_id.mirna_fc.seed_cls.sequence.signal.conservation.seed_cls.ts_preds.local_features.test.pkl"



run_output_sufix=".user_local_features"


## Running the model training

```train_and_evaluate_models.sh```
