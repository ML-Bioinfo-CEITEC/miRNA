import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch import seed_everything
from sklearn.model_selection import train_test_split

from hyenadna import HyenaDNABinaryCls
from hyenadna_helpers import miRNA_Dataset
from utils import load_data_from_csv
from utils import MRNA_SEQ_COLUMN, MIRNA_SEQ_COLUMN, CLASSIFICATION_LABEL_COLUMN

def prepare_data(
    data_path,
    tokenizer,
    test_size = 0.2,
    seq1_col = MIRNA_SEQ_COLUMN,
    seq2_col = MRNA_SEQ_COLUMN,
    cls_label_col = CLASSIFICATION_LABEL_COLUMN,
    batch_size = 16,
    use_padding = True,
    add_eos = False
):
    
    data = load_data_from_csv(data_path)
    train_df, test_df = train_test_split(data, test_size=test_size)
    
    train_dset = miRNA_Dataset(
        train_df[seq1_col].reset_index(drop=True),
        train_df[seq2_col].reset_index(drop=True),
        train_df[cls_label_col].reset_index(drop=True),
        max_length = tokenizer.model_max_length,
        use_padding = use_padding,
        tokenizer=tokenizer,
        add_eos=add_eos,
    )
    
    test_dset = miRNA_Dataset(
        test_df[seq1_col].reset_index(drop=True),
        test_df[seq2_col].reset_index(drop=True),
        test_df[cls_label_col].reset_index(drop=True),
        max_length = tokenizer.model_max_length,
        use_padding = use_padding,
        tokenizer=tokenizer,
        add_eos=add_eos,
    )
    
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
    
if __name__ == '__main__':
    
    hyperparams = {
        "batch_size": 16,
        "mode_pooler": 'last',
        "max_epochs": 10,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "warmup_steps": 100,
        "data_path": "~/miRNA/HyenaDNA_experiment/debug/3utr.sequences.refseq_id.mirna_fc.pkl.nlp.clamped_to_zero.cls.seq_len_below.balanced_down",
        
    }
    
    seed_everything(42, workers=True)

    cls = HyenaDNABinaryCls(
        mode_pooler=hyperparams["mode_pooler"],
        lr=hyperparams["lr"],
        weight_decay=hyperparams["weight_decay"],
        warmup_steps=hyperparams["warmup_steps"]
    )
    
    # Arguments made to CometLogger are passed on to the comet_ml.Experiment class
    comet_logger = CometLogger(
        project_name="mirna-hyenadna",
        api_key='3NQhHgMmmlfnoqTcvkG03nYo9',
        log_code=True
    )
    
    comet_logger.log_hyperparams(hyperparams)
    
    train_loader, test_loader = prepare_data(
        hyperparams["data_path"],
        cls.encoder.tokenizer
    )
    
    trainer = L.Trainer(
        deterministic=True,
        accelerator="gpu",
        max_epochs=hyperparams["max_epochs"],
        logger=comet_logger
    )
    trainer.fit(
        model=cls,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader
    )