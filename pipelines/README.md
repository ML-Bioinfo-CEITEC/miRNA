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
The script takes the paths of the train and test dataset. It trains models, uses the trained models to make predictions, and saves the predictions to the ```/data/predictions``` folder.
The plotting part of the script takes the predictions and compares them to TS or other methods and feature sets and outputs comparison plots to ```/plots``` folder.

The following arguments are expected to run the script:

```train_dataset``` and ```test_dataset``` are the path to the specific files dataset files.

```run_output_sufix``` is the suffix added to the prediction file names. F.e. the name of the predictions output file will be ```test_dataset``` + ```run_output_sufix```

```run_name``` is the name of the folder for the prediction file. The resulting placement will be ```/data/predictions/run_name```

```comparison_name``` is the display name of the method in the resulting plots.

```prediction_folders_to_plot``` is a list of folders from ```/data/predictions``` in which the script will search for methods to plot in comparison with the currently evaluated method.

```methods_to_plot``` is a list of method names that will be used for comparison with the currently evaluated method. The method names are the same as the column names in dataframes with predictions in the respective folders in ```/data/predictions```


