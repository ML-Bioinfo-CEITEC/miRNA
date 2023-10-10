from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import pandas as pd


# genes we can compare with Bartel are in test set
def split_train_test_bartel(
    padded_data_tensor, input_labels, input_data_genes_filtered, mirna_FCs, mirna_name
):
    targetscan = pd.read_csv('Predicted_Targets_Context_Scores.default_predictions.txt',index_col=0, header=0, sep='\t')
    targetscan = targetscan[["context++ score","weighted context++ score","miRNA","Gene Symbol"]]
    targetscan = targetscan[targetscan['miRNA'] == mirna_name]

    bartel_gene_names = targetscan['Gene Symbol'].to_numpy().flatten()

    bartel_test_samples = set(bartel_gene_names).intersection(
        set(mirna_FCs[["Gene symbol"]].to_numpy().flatten())).intersection(
        set(input_data_genes_filtered))
    

    # x_train, x_test, y_train, y_test = train_test_split(padded_data_tensor, input_labels, test_size=0.2, random_state=42)
    # Split the data and gene names simultaneously
    indecis = []
    x_train, x_test, y_train, y_test, gene_names_train, gene_names_test = [],[],[],[],[],[]

    for sample, label, name in zip(padded_data_tensor, input_labels, input_data_genes_filtered):
        if name in bartel_test_samples:
            x_test.append(sample)
            y_test.append(label)
            gene_names_test.append(name)
        else:
            x_train.append(sample)
            y_train.append(label)
            gene_names_train.append(name)
    # print(len(x_train), len(x_test))

    # validation
    x_train, x_val, y_train, y_val, gene_names_train, gene_names_val = train_test_split(x_train, y_train, gene_names_train, test_size=0.1, random_state=42)

    # print(len(y_train), len(y_val), len(y_test))
    # print(len(gene_names_train), len(gene_names_val), len(gene_names_test))
    
    return x_train, y_train, x_val, y_val, x_test, y_test, gene_names_train, gene_names_val, gene_names_test

    
# default train-val-test split
def split_train_test(padded_data_tensor, input_labels, input_data_genes_filtered):
    # x_train, x_test, y_train, y_test = train_test_split(padded_data_tensor, input_labels, test_size=0.2, random_state=42)
    # Split the data and gene names simultaneously
    (x_train, x_test, y_train, y_test, gene_names_train, gene_names_test) = train_test_split(
        padded_data_tensor, input_labels, input_data_genes_filtered, test_size=0.2, random_state=42)

        #validation
    x_train, x_val, y_train, y_val, gene_names_train, gene_names_val = train_test_split(x_train, y_train, gene_names_train, test_size=0.1, random_state=42)

    # print(len(y_train), len(y_val), len(y_test))
    # print(len(gene_names_train), len(gene_names_val), len(gene_names_test))
    
    return x_train, y_train, x_val, y_val, x_test, y_test, gene_names_train, gene_names_val, gene_names_test


# pytorch datasets and dataloders
def get_test_dataloader(x_test, y_test, batch_size):
    test_dataset = TensorDataset(torch.stack(x_test).unsqueeze(1).float(), torch.tensor(y_test).unsqueeze(dim=1))    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

class TrainDataset(Dataset):
    def __init__(self, data, labels, exps):
        self.data = data
        self.labels = labels
        self.exps = exps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        exp = self.exps[idx]
        return x, y, exp
    
class ValDataset(Dataset):
    def __init__(self, data, labels, identifiers):
        self.data = data
        self.labels = labels
        self.identifiers = identifiers

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        identifier = self.identifiers[idx]
        return x, y, identifier

def get_train_dataloader(x_train, y_train, batch_size):
    #dummy exps and indentifiers
    exps = [1 for x in range(len(y_train))]
    train_dataset = TrainDataset(torch.stack(x_train).unsqueeze(1).float(), torch.tensor(y_train).unsqueeze(dim=1), exps)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def get_val_dataloader(x_val, y_val, batch_size):
    #dummy exps and indentifiers
    identifiers = []
    for label in y_val:
        item = {}
        item['readid']=0
        item['label']=label
        item['exp']=1
        identifiers.append(item)
    val_dataset = ValDataset(torch.stack(x_val).unsqueeze(1).float(), torch.tensor(y_val).unsqueeze(dim=1), identifiers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return val_loader 


# general function to predict fold change
# returns: 
# dict {gene_name: fold_change_prediction}
# array of predictions in the same order as gene_names_test
def predict(model, x_test, gene_names_test):
    with torch.no_grad(): # Disables gradient computation, as it's not needed during evaluation
        predictions = model(torch.stack(x_test).unsqueeze(1).float())

    gene_to_predictions = {}
    for i in range(len(gene_names_test)):
        gene_to_predictions[gene_names_test[i]] = predictions.squeeze(1)[i].item()
    
    predictions_scalar = [x.item() for x in predictions.squeeze(1)]
    
    return gene_to_predictions, predictions_scalar