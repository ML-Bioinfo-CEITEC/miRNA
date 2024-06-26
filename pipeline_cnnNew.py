import random
from pytorch_lightning.loggers import CometLogger
import pandas as pd
from collect_binding_sites import collect_binding_sites
from feature_extraction import count_statistics, normalize_statistics, FEATURES, FEATURE_NAMES
from pad_input_data import pad_features
from utils import get_labels, get_baseline_metrics
from dataset import get_train_dataloader, get_val_dataloader, get_test_dataloader, split_train_test_bartel, predict
from model import Small_CNN
from pytorch_lightning import Trainer
from IPython.utils import io
import numpy as np
from statistics import mean
import json
import gdown
import os

#Run the following if its the first time
if not os.path.exists("explainability_scores_hsa-miR-106b-5p.json"):
    url = "https://drive.google.com/file/d/1ayyD1w6SHzLS8638eoBzUX3OMq4cxSUx/view?usp=sharing"
    output = "explainability_scores_hsa-miR-106b-5p.json"
    gdown.download(url=url, output=output, quiet=False, fuzzy=True)

random.seed(42)

mirna_FCs = pd.read_csv('modules/evaluation/mirna_fcs.csv',index_col=0, header=0, sep=',')
# mirna_FCs.columns.values
list(mirna_FCs)

toSkip = True
if not toSkip:
    mirna_sequences = ['UAGCAGCACGUAAAUAUUGGCG', 'UAAAGUGCUGACAGUGCAGAU', 'UAACACUGUCUGGUAACGAUGU', 'UAAUACUGCCUGGUAAUGAUGA', 'AUGACCUAUGAAUUGACAGAC', 'UGAGGUAGUAGGUUGUAUGGUU', 'AGCAGCAUUGUACAGGGCUAUGA']
    mirna_sequences = [rna_to_dna(x) for x in mirna_sequences]
    print(mirna_sequences)
    miRNA_names = ['hsa-miR-16-5p', 'hsa-miR-106b-5p', 'hsa-miR-200a-3p', 'hsa-miR-200b-3p', 'hsa-miR-215-5p', 'hsa-let-7c-5p', 'hsa-miR-103a-3p']
    miRNA_name_to_seq = {}
    for i in range(len(miRNA_names)):
        miRNA_name_to_seq[miRNA_names[i]] = mirna_sequences[i]
    miRNA_name_to_seq

mirna_name = 'hsa-miR-106b-5p'
mirna_seq = 'TAAAGTGCTGACAGTGCAGAT'

load_scores_path = "explainability_scores_{}.json".format(mirna_name)
binding_sites = collect_binding_sites(load_scores_path, mirna_seq)

# each item of binding_sites contains a triplet of arrays: ([starts],[ends],[lengths])
# 1st item in [starts] coresponds to 1st item in [ends] and [lengths] aswell, 2nd start to 2nd end and length, and so on
binding_sites[6:8]

input_data, input_data_genes, transcripts_with_no_bs = count_statistics(binding_sites, load_scores_path, mirna_seq)

input_data_normalized = normalize_statistics(input_data)

padded_data_tensor = pad_features(input_data_normalized, pad_to_length = (len(FEATURES) * 10))

len(padded_data_tensor), padded_data_tensor[0].size(), padded_data_tensor[0]

input_labels, padded_data_tensor, input_data_genes_filtered = get_labels(mirna_name, padded_data_tensor, input_data_genes)

# genes we can compare with Bartel are in test set
x_train, y_train, x_val, y_val, x_test, y_test, gene_names_train, gene_names_val, gene_names_test = split_train_test_bartel(
    padded_data_tensor, 
    input_labels, 
    input_data_genes_filtered, 
    mirna_FCs,
    mirna_name
)

print(len(y_train), len(y_val), len(y_test))
print(len(gene_names_train), len(gene_names_val), len(gene_names_test))

### Create pytorch dataset
BATCH_SIZE = 32
train_loader = get_train_dataloader(x_train, y_train, BATCH_SIZE)
val_loader = get_val_dataloader(x_val, y_val, BATCH_SIZE)
test_loader = get_test_dataloader(x_test, y_test, BATCH_SIZE)

comet_logger = CometLogger(
    api_key="EpKIINrla6U4B4LJhd9Sv4i0b",
    project_name="mirna",
)

model = Small_CNN(pooling='att')
# trainer = Trainer(max_epochs=1, gpus=1)  # Use GPU if available, train for X epochs
trainer = Trainer(logger=comet_logger, max_epochs=3)  # Use GPU if available, train for X epochs

# capture_output to have a cleaner notebook
# you can follow the training at the  https://www.comet.com/davidcechak/mirna/  see log of this cell
with io.capture_output() as captured:
    trainer.fit(model, train_loader, val_loader)

# Predict
gene_to_predictions, predictions = predict(model, x_test, gene_names_test)
print(list(gene_to_predictions.items())[:2])

results = {}
results['model'] = result[0]

# computes correlation of model predictions and true labels
model_corr = np.corrcoef(predictions, y_test)[0][1]
results['model']['corr'] = model_corr

print('Model metrics: ')
print(results)

# Compare with baselines

#### Baseline #1 mean of the training dataset labels

baseline_name = 'mean_baseline'
train_x_mean = mean(y_train)
baseline_mean = np.full((len(y_test),), train_x_mean)
results[baseline_name] = get_baseline_metrics(baseline_mean, y_test)

print_baseline_metrics(results, baseline_name)

#### Baseline #2 random in range(min_y_tran, max_y_train) of the training dataset labels
# Baseline: for each test sample returns a random item in range(min_y_tran, max_y_train) of the training dataset labels
baseline_name = 'mean_rnd'
baseline_max = max(y_train)
baseline_min = min(y_train)
np.random.seed(42)
print('min, max :', baseline_min, baseline_max)
baseline_rnd = np.random.uniform(baseline_min, baseline_max, [len(y_test)])
results[baseline_name] = get_baseline_metrics(baseline_rnd, y_test)

print_baseline_metrics(results, baseline_name)

# Log
with open('results.json', 'w') as fp:
    json.dump(results, fp)

### Compare with Bartel - correlation and top predictions plot (i.e. genes with highest predicted FC plot)
#Todo: metrics and comparison plots