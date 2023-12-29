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
        
        #print("l_output", l_output)
        #print("self.l_output", self.l_output)

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
            # TODO use same restrict function as pool case
        # elif self.mode == 'ragged':
        #     assert lengths is not None, "lengths must be provided for ragged mode"
        #     # remove any additional padding (beyond max length of any sequence in the batch)
        #     restrict = lambda x: x[..., : max(lengths), :]
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
        #print(self.encoder.out_features)
        if classifier is None:
            self.cls = nn.Sequential(
                nn.Linear(self.encoder.out_features, 1),
                nn.Sigmoid()
            )
        else:
            self.cls = classifier
        
        self.criterion = nn.BCELoss()
        self.accuracy = torchmetrics.classification.Accuracy(task="binary")
        
        self.lr = lr
        self.wd = weight_decay
        self.warmup_steps = warmup_steps

    def forward(self, x):
        hidden = self.encoder(x)
        embedd = self.pooler(hidden)

        return self.cls(embedd)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        y_hat = self(x)
        loss = self.criterion(y_hat, y.float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        acc = self.accuracy(y_hat, y)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y.float())
        
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        acc = self.accuracy(y_hat, y)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return val_loss
        
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        
        y_hat = self(x)
        test_loss = self.criterion(y_hat, y.float())
        
        self.log("test_loss", test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        acc = self.accuracy(y_hat, y)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
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
