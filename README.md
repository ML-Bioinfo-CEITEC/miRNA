## Scanning the transcriptome for ago2:miRNA to mRNA binding sites and predicting fold change

### Set up the environment
Create a conda env from environment/spec-file.txt by running ```conda create --name YOUR_ENVIRONMENT_NAME --file spec-file.txt```

### Download the data
Run the ```download_data.ipynb```

### Example ML notebooks
After you download the datasets, you can play with example notebooks ```ML_training_and_evaluation.regression.ipynb``` and ```ML_training_and_evaluation.classification.ipynb```

### Data processing
#### 1) Sequence and LFC merging
```process_utr_sequences_hg19.fc.representative_transcript.ipynb```

First step of the pipeline. Takes 3'UTRs, LFCs, mapping of Ensembl and RefSeq IDs, and information from TS about what transcript is representative. It returns a dataset with labels, sequences, IDs and other transcript info. This sequences are later used for scanning.
- Gets hg19 3’UTR sequences by matches of 
    - RefSeq ID (through an ID map from UCSC), if that fails then:
    - Ensembl ID, if that fails then:
    - Gene Symbol
#### 2) Scanning - producing explainability signal

Will be added later
#### 3) Merge explainability signal with sequences and conservation, split to train/test

Will be added later. This produces the final train/test datasets for the 7miRs HCT116 transfection

### Data links
#### 7miRs HCT116 transfection - Train set
[https://drive.google.com/drive/folders/15pX8qrauPJbKYo6r07jRtBLZaRTU6LdO?usp=sharing](https://drive.google.com/file/d/1m6c3be4OVqVv72vAD5UnR_Ar-8IUoeex/view?usp=drive_link)
#### 7miRs HCT116 transfection - Test set
[https://drive.google.com/file/d/15dZZQAEXAqsbBpkjVfWvew9oM4Eu-8PL/view?usp=drive_link](https://drive.google.com/file/d/15dZZQAEXAqsbBpkjVfWvew9oM4Eu-8PL/view?usp=drive_link)
