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
The script takes a config containing extraction functions and the path of the train and test dataset. It extracts the features and adds them to the datasets. It saves the updated datasets as new files.

The following arguments are expected to run the local feature extraction for a new local feature:

```CONFIG_PATH```
Given by the CONFIG_PATH parameter. In this config, you include the functions the script will use to extract the local features. See the example in ```user_config.py```.

```data_folder_path``` is the folder where ```train_dataset```, ```test_dataset``` lie. The following are the names of the specific files. 

```run_output_sufix``` is the suffix added to the dataset names. F.e. the name of the newly updated train dataset will be ```train_dataset``` + ```run_output_sufix```


## Running the model training

```train_and_evaluate_models.sh```
