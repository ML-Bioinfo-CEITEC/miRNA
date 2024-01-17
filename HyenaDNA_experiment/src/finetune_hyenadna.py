import torch
from torch.utils.data import DataLoader, random_split
import lightning as L
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch import seed_everything
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from argparse import ArgumentParser

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
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=42)
    
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
    
    return train_dset, test_dset
    
def train_func(config):
    
    net = HyenaDNABinaryCls(
        mode_pooler=config["mode_pooler"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"]
    )
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    trainset, _ = prepare_data(
        config["data_path"],
        net.encoder.tokenizer
    )

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=1`
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=1`
    )
    
    # Arguments made to CometLogger are passed on to the comet_ml.Experiment class
    comet_logger = CometLogger(
        project_name=config["project_name"],
        api_key=config["api_key"],
        log_code=True
    )
    
    comet_logger.log_hyperparams(config)
    
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        logger=comet_logger,
        enable_progress_bar=False,
        deterministic=True
    )
    trainer.fit(
        model=net,
        train_dataloaders=trainloader,
        val_dataloaders=valloader
    )
    
    comet_logger.experiment.log_model("best-model", trainer.model)
    comet_logger.finalize()
    
    
if __name__ == '__main__':
   
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=Path, required=True)
    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()

    # Reproducibility
    seed_everything(42, workers=True)
    torch.manual_seed(42)

    config = {
        "mode_pooler": tune.choice(['last']),#"first", "last", "pool", "sum"]),
        "weight_decay": tune.loguniform(0.0000001, 0.1),
        "warmup_steps": tune.choice([100, 200, 300, 400, 500, 600, 700]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([8, 16]),
        "data_path": args.data_path,
        "project_name": args.project_name, 
        "api_key": args.api_key
    }
    
    # The maximum training epochs
    num_epochs = 10

    # Number of sampls from parameter space
    num_samples = 10
    
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=2, reduction_factor=2)
    
    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="ptl/val_accuracy",
            checkpoint_score_order="max",
        ),
    )
    
    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": config},
        tune_config=tune.TuneConfig(
            metric="ptl/val_accuracy",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    results = tuner.fit()
    
    best_result = results.get_best_result(metric="ptl/val_accuracy", mode="max")
    print("Best trial config: {}".format(best_result.config))
    
    best_model = best_result.get_best_checkpoint(metric="ptl/val_accuracy", mode="max")
    print("Saving best model to {}".format(args.output_dir))
    best_model.to_directory(args.output_dir)