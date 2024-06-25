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