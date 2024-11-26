import argparse
import os
import torch
import datetime
from dateutil import tz 
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint)
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
import yaml

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
current_workspace = os.path.dirname(current_dir)
models_dir = os.path.join(current_workspace, "models")
sys.path.append(models_dir)
from STNet_model import STNetModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_cards', type=str, default='', help='Comma-separated list of GPU card numbers')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--config_file_name', type=str, default='STNet.yaml', help='Path to the config file')
    parser.add_argument('--num_devices', type=int, default=4)
    parser.add_argument("--main_data_storage", type=str, default="/data1/zliang")
    parser.add_argument('--working_codespace', type=str, default="/home/zliang/BRIDGE_BIG_600K")
    parser.add_argument('--organ_selected', nargs='+')
    hyperparameters = parser.parse_args()
    seed_everything(hyperparameters.seed)

    print(hyperparameters.organ_selected)

    single_organ = hyperparameters.organ_selected[0]

    gpu_cards = hyperparameters.gpu_cards.split(",") if hyperparameters.gpu_cards else []
    gpu_cards_str = ",".join(gpu_cards)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_cards_str
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    current_time = datetime.datetime.now(tz.tzlocal()).strftime("%Y_%m_%d_%H_%M_%S")
    checkpoint_folder_path = os.path.join(hyperparameters.main_data_storage, "checkpoints",
                                          "STNet_training_results",
                                          f"STNet_{current_time}_{single_organ}",
                                          "checkpoints")
    os.makedirs(checkpoint_folder_path, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=checkpoint_folder_path,
            filename='{epoch}',
            verbose=False,
            save_last=True,
            # every_n_epochs=5,
            save_top_k=-1, # if save_top_k == -1, all models are saved
            mode="max",
            auto_insert_metric_name=False,
            save_weights_only=False,
        ),
    ]

    logger_folder_path = os.path.join(hyperparameters.main_data_storage, "checkpoints",
                                      "STNet_training_results",
                                      f"STNet_{current_time}_{single_organ}",
                                      "loggers")
    os.makedirs(logger_folder_path, exist_ok=True)
    wandb_logger = WandbLogger(
        name=f"STNet_{current_time}",
        save_dir=logger_folder_path,
        project="BRIDGE_BIG_600K"
    )

    config_file_path = os.path.join(hyperparameters.working_codespace, "config", hyperparameters.config_file_name)
    with open(config_file_path) as file:
        config = yaml.safe_load(file)
    config.update(vars(hyperparameters))
    config["device"] = device
    hyperparameters = argparse.Namespace(**config)
    model = STNetModel(**vars(hyperparameters))

    datamodule = model.datamodule
    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        devices=hyperparameters.num_devices,
        num_nodes=1, # default
        precision="bf16-mixed",
        logger=wandb_logger,
        callbacks=callbacks,
        fast_dev_run=False, # default
        max_epochs=hyperparameters.max_epochs,
        limit_train_batches=1.0, # default
        limit_val_batches=0.0,
        limit_test_batches=0.0,
        limit_predict_batches=0.0,
        overfit_batches=0.0, # default
        val_check_interval=0.0,
        check_val_every_n_epoch=1, # default
        num_sanity_val_steps=2, # default
        log_every_n_steps=50, # default
        enable_checkpointing=True, # default
        enable_progress_bar=True, # default
        accumulate_grad_batches=hyperparameters.accumulate_grad_batches,
        gradient_clip_val=None, # default
        gradient_clip_algorithm="norm", # default
        deterministic=None, # default
        benchmark=None, # default
    )
    trainer.fit(model, datamodule)
    # trainer.test(model, datamodule, ckpt_path="best")

if __name__ == "__main__":
    main()