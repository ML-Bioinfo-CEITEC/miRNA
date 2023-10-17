import torch
import torchmetrics
import pytorch_lightning as pl
from types import SimpleNamespace
import numpy as np
import re
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from modules.models.modules import ConvNet, RNNEncoder, MLP, Attention, Permute, SimpleCNN # , BigConvNet, ResConvNet
from sklearn.metrics import average_precision_score
import csv


class Small_CNN(pl.LightningModule):
    def __init__(
            self,
            pooling='max',
            lr=1e-3,
            warmup_steps=100,
            wd=0.01,
            logging_steps=1,
            pos_weight=1.0,
    ):

        super().__init__()
        self.lr = lr
        self.wd = wd
        self.warmup_steps=warmup_steps

        if(pooling=='rnn'):
            self.architecture = torch.nn.Sequential(
                SimpleCNN(num_layers=1),
                Permute(),
                RNNEncoder(input_size=8, hidden_size=16, num_layers=1, dropout=0.2),
                MLP(input_size=96, hidden_size=30),
            )
        if(pooling=='att'):
            self.architecture = torch.nn.Sequential(
                SimpleCNN(num_layers=1),
                Permute(),
                Attention(input_dim=8, len_limit=400000),
                MLP(8, 30),
            )
        if(pooling=='max'):
            self.architecture = torch.nn.Sequential(
                SimpleCNN(num_layers=1),
                torch.nn.AdaptiveMaxPool1d(1),
                torch.nn.Flatten(),
                MLP(8, 30),
            )
        
        # self.acc = torchmetrics.classification.Accuracy(task="binary")
        # self.ce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(self.device))
        self.ce = torch.nn.MSELoss()
            
        # Logging utils to fix, TODO resolve with a custom callback?
        self.training_step_counter = 0
        self.cumulative_loss = 0
        self.cumulative_acc = 0
        self.logging_steps = logging_steps
        
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()
        self.r2 = torchmetrics.R2Score()

    def forward(self, x):
        return self.architecture(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(), 
            lr=self.lr, 
            weight_decay=self.wd
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            total_iters=self.warmup_steps
        )
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx, dataloader_idx=None):
        x, y, exp = train_batch
        logits = self(x)
        loss = self.ce(logits, y)
        
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        # probs = self.logits_to_probs(logits)
        
        metrics = self.get_metrics(logits, y, exp)
        self.log_cumulative_train_metrics(loss, metrics['mse'])
        for metric, value in metrics.items():
            self.log(f'train {metric}', value, on_epoch=True, on_step=False)
        
        # print("Logits:", logits)
        # print("Labels:", y)
        # print("Loss:", loss)

        self.lr_schedulers().step()
        return loss
    
    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        x, y, identifier = val_batch
        exp = identifier['exp']
        logits=self(x)
        loss = self.ce(logits, y)
        
        self.log('valid_loss', loss, on_epoch=True, on_step=False)
        # probs = self.logits_to_probs(logits)
        metrics = self.get_metrics(logits, y, exp)
        for metric, value in metrics.items():
            self.log(f'valid {metric}', value, on_epoch=True, on_step=False)
            
        return {
            'preds': logits.detach().cpu().numpy(), 
            'identifier': {
                'readid':identifier['readid'],
                'label':identifier['label'].detach().cpu(),
                'exp':identifier['exp'],
            }
        }

    
    def predict_step(self, batch, batch_idx):
        xs, ids = batch
        logits = self.forward(xs)
        # res = self.logits_to_probs(logits)
        return logits, ids
    
    def get_metrics(self, predictions, labels, exps):
        metrics = {}
        
        mse = self.mse(predictions, labels)
        mae = self.mae(predictions, labels)
        r2 = self.r2(predictions, labels)
        rmse = torch.sqrt(mse)

        metrics['mse'] = mse
        metrics['mae'] = mae
        metrics['r2'] = r2
        metrics['rmse'] = rmse
        
        # exps = np.array(exps)
        # for e in np.unique(exps):
        #     indices = exps == e
        #     if (sum(indices) > 0): #If Non-empty
        #         metrics[f'{e} acc'] = self.acc(predictions[exps == e], labels[exps == e])
        return metrics

    def log_cumulative_train_metrics(self, loss, accuracy):
        self.training_step_counter += 1
        self.cumulative_loss += loss.item()
        self.cumulative_acc += accuracy

        if self.training_step_counter % self.logging_steps == 0:
            avg_loss = self.cumulative_loss / self.logging_steps
            avg_acc = self.cumulative_acc / self.logging_steps
            
            self.log(f'train_loss_cum', avg_loss, on_step=True, on_epoch=False)
            self.log(f'train_acc_cum', avg_acc, on_step=True, on_epoch=False)
            self.cumulative_acc = 0
            self.cumulative_loss = 0
            
#     def logits_to_probs(self, logits):
#         # return torch.sigmoid(logits)
#         return logits
    

    def validation_epoch_end(self, outputs):
        # Aggregate all validation predictions into auroc metrics
        read_to_pred, read_to_label, read_to_exp = self.aggregate_outputs(outputs)

        print(f"\n Validation Metrics: MSE={self.mse.compute()}, MAE={self.mae.compute()}, R2={self.r2.compute()}, RMSE={torch.sqrt(self.mse.compute())}")
        with open('validation_metrics.csv', mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([self.mse.compute().item(), self.mae.compute().item(), self.r2.compute().item(), torch.sqrt(self.mse.compute()).item()])
    
        # reset metrics (important for next epoch)
        self.mse.reset()
        self.mae.reset()
        self.r2.reset()

    def aggregate_outputs(self, outputs):
        read_to_preds = {}
        read_to_label = {}
        read_to_exp = {}
        for log in outputs:
            preds = log['preds']#.cpu().numpy()
            ids = log['identifier']

            for i, (readid, pred, label, exp) in enumerate(zip(ids['readid'], preds, ids['label'], ids['exp'])):
                read_to_label[readid] = label
                read_to_exp[readid] = exp
                #TODO remove for optimization
                assert len(pred) == 1 
                read_to_preds[readid] = pred[0]

        return read_to_preds, read_to_label, read_to_exp
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        print(x.shape, 'test_step')  # Debugging statement
        
        logits = self(x)
        loss = self.ce(logits, y)
        metrics = self.get_metrics(logits, y, None)  # Assuming you have a method to calculate metrics
        output = {'test_loss': loss, **metrics}
        return output

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_metrics = {key: torch.stack([x[key] for x in outputs]).mean() for key in outputs[0].keys() if key != 'test_loss'}
        results = {'test_loss': avg_loss, **avg_metrics}
        self.log_dict(results)
