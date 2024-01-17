import torch
from torch import nn
import torchmetrics
import lightning as L
from transformers import AutoModel, AutoTokenizer

POOL_OPTIONS = ['last', 'first', 'pool', 'sum']

class HyenaDNAEncoder(L.LightningModule):
    def __init__(self, model_path = None, revision = None, freeze = False):
        super().__init__()
        
        if model_path is None:
            print("model_path not provided, defaulting to LongSafari/hyenadna-tiny-16k-seqlen-d128-hf and revision e83c7caa155780f5f898017e736c3f6041e559cf")
        
            model_path = "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf"
            revision = "e83c7caa155780f5f898017e736c3f6041e559cf"
        
        self.encoder = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            revision=revision)
        
        if freeze:
            self.encoder.freeze()
            
        self.out_features = self.encoder.backbone.ln_f.normalized_shape[0]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, revision=revision)
        
        # add pad token id to model config - it is not there by default and CLS models meeds it
        self.encoder.config.pad_token_id = self.tokenizer._vocab_str_to_int[self.tokenizer.pad_token]
        
    def forward(self, x):
        return self.encoder(x).last_hidden_state

class Pooler(nn.Module):
    def __init__(
        self, l_output=None, use_lengths=False, mode="last"
    ):
        """
        :param 
        """
        super().__init__()

        if l_output is None:
            self.l_output = None
            self.squeeze = False
        elif l_output == 0:
            # Equivalent to getting an output of length 1 and then squeezing
            self.l_output = 1
            self.squeeze = True
        else:
            assert l_output > 0
            self.l_output = l_output
            self.squeeze = False

        self.use_lengths = use_lengths
        self.mode = mode

        if mode == 'ragged':
            assert not use_lengths

    def forward(self, x, state=None, lengths=None, l_output=None):
        """
        x: (n_batch, l_seq, d_model)
        Returns: (n_batch, l_output, d_output)
        """
        

        if self.l_output is None:
            if l_output is not None:
                assert isinstance(l_output, int)  # Override by pass in
            else:
                # Grab entire output
                l_output = x.size(-2)
            squeeze = False
        else:
            l_output = self.l_output
            squeeze = self.squeeze

        if self.mode == "last":
            restrict = lambda x: x[..., -l_output:, :]
        elif self.mode == "first":
            restrict = lambda x: x[..., :l_output, :]
        elif self.mode == "pool":
            restrict = lambda x: (
                torch.cumsum(x, dim=-2)
                / torch.arange(
                    1, 1 + x.size(-2), device=x.device, dtype=x.dtype
                ).unsqueeze(-1)
            )[..., -l_output:, :]

            def restrict(x):
                L = x.size(-2)
                s = x.sum(dim=-2, keepdim=True)
                if l_output > 1:
                    c = torch.cumsum(x[..., -(l_output - 1) :, :].flip(-2), dim=-2)
                    c = F.pad(c, (0, 0, 1, 0))
                    s = s - c  # (B, l_output, D)
                    s = s.flip(-2)
                denom = torch.arange(
                    L - l_output + 1, L + 1, dtype=x.dtype, device=x.device
                )
                s = s / denom
                return s

        elif self.mode == "sum":
            restrict = lambda x: torch.cumsum(x, dim=-2)[..., -l_output:, :]

        else:
            raise NotImplementedError(
                "Mode must be ", POOL_OPTIONS
            )

        # Restrict to actual length of sequence
        if self.use_lengths:
            assert lengths is not None
            x = torch.stack(
                [
                    restrict(out[..., :length, :])
                    for out, length in zip(torch.unbind(x, dim=0), lengths)
                ],
                dim=0,
            )
        else:
            x = restrict(x)

        if squeeze:
            assert x.size(-2) == 1
            x = x.squeeze(-2)

        return x
        
class HyenaDNABinaryCls(L.LightningModule):
    def __init__(
        self,
        model_path_encoder = None,
        revision_encoder = None,
        freeze_encoder = False,
        mode_pooler = 'last',
        classifier = None,
        lr = 1e-3,
        weight_decay = 0.01,
        warmup_steps = 100
    ):
        
        super().__init__()
        
        self.encoder = HyenaDNAEncoder(
            model_path = model_path_encoder,
            revision = revision_encoder,
            freeze = freeze_encoder)
        self.pooler = Pooler(mode = mode_pooler, l_output = 0) 

        if classifier is None:
            self.cls = nn.Sequential(
                nn.Linear(self.encoder.out_features, 1),
                nn.Sigmoid()
            )
        else:
            self.cls = classifier
        
        self.criterion = nn.BCELoss()

        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")
        
        self.lr = lr
        self.wd = weight_decay
        self.warmup_steps = warmup_steps

    def forward(self, x):
        hidden = self.encoder(x)
        embedd = self.pooler(hidden)

        return self.cls(embedd)
    
    def _shared_step(self, batch):
        
        x, y = batch
        
        y_hat = self(x)
        loss = self.criterion(y_hat, y.float())
        
        return loss, y_hat, y
        
    def training_step(self, batch, batch_idx):
        
        loss, y_hat, y = self._shared_step(batch)
        
        self.log("ptl/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_acc(y_hat, y)
        self.log('ptl/train_accuracy', self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        loss, y_hat, y = self._shared_step(batch)
        
        self.log("plt/val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.val_acc(y_hat, y)
        self.log('ptl/val_accuracy', self.val_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {"val_loss": loss, "val_accuracy": self.val_acc}
        
    def test_step(self, batch, batch_idx):
        
        _, y_hat, y = self._shared_step(batch)

        self.test_acc(y_hat, y)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return test_loss
    
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
