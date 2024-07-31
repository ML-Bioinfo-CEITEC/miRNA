# Scanning the transcriptome for ago2:miRNA to mRNA binding sites and predicting fold change

## Running example models and evaluation
### Set up the environment
Create a conda env by running ```conda create --name YOUR_ENVIRONMENT_NAME --file spec-file.txt```

Activate an existing environment by ```conda activate YOUR_ENVIRONMENT_NAME```

### Download the data
Run ```./data/download_data_for_ML.sh``` which installs the ```gdown``` package (if not installed yet) through pip and downloads the datasets.

Installing the ```gdown``` through pip is temporary until we fix installing through conda 


### Example ML notebooks
After you download the datasets, you can play with example notebooks ```ML_training_and_evaluation.regression.ipynb``` and ```ML_training_and_evaluation.classification.ipynb```. Choose the environment ```YOUR_ENVIRONMENT_NAME``` you created before as your notebook kernel.