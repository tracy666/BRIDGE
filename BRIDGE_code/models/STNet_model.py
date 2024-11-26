from typing import Any, Literal, List
from base_model import BaseModel
from torchvision.models import densenet121, DenseNet121_Weights
import torch
import torch.nn as nn
import os
import pandas as pd
from einops import rearrange
import torch.nn.functional as F

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
current_workspace = os.path.dirname(current_dir)
backbones_dir = os.path.join(current_workspace, "backbones")
sys.path.append(backbones_dir)
from encoders import ImageEncoder, GeneEncoder, Projection_Head, prepare_model
from lr_scheduler import linear_warmup_decay

dataset_dir = os.path.join(current_workspace, "dataset")
sys.path.append(dataset_dir)
from image_gene_data_module import DataModule

class STNetModel(BaseModel):
    def __init__(
            self,
            max_epochs: int,
            warmup_epochs: int,
            batch_size: int,
            num_workers: int,
            learning_rate: float,
            weight_decay: float,
            num_devices: int,
            accumulate_grad_batches: int,

            main_data_storage: str,
            raw_data_folder_name: str,
            project_data_folder_name: str,
            working_codespace: str,
            gene_csv_name: str,
            generation_date: str,
            organ_selected: str,

            input_gene_number: int,
            number_of_eval_HEG_genes: int,
            number_of_eval_HVG_genes: int,

            image_to_gene_generative_loss_weight: float,

            use_normed_patch: bool,

            scGPT_gene_mask_ratio: bool,

            image_encoder_name: Literal["DenseNet121", "CTransPath", "DINOv1", "DINOv2"],

            device,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.num_devices = num_devices
        self.accumulate_grad_batches = accumulate_grad_batches

        self.main_data_storage = main_data_storage
        self.raw_data_folder_name = raw_data_folder_name
        self.project_data_folder_name = project_data_folder_name
        self.working_codespace = working_codespace
        self.gene_csv_name = gene_csv_name
        self.generation_date = generation_date
        self.organ_selected = organ_selected

        self.input_gene_number = input_gene_number
        self.number_of_eval_HEG_genes = number_of_eval_HEG_genes
        self.number_of_eval_HVG_genes = number_of_eval_HVG_genes

        self.image_to_gene_generative_loss_weight = image_to_gene_generative_loss_weight

        self.use_normed_patch = use_normed_patch

        self.scGPT_gene_mask_ratio = scGPT_gene_mask_ratio

        self.image_encoder_name = image_encoder_name

        super().__init__(max_epochs, warmup_epochs, batch_size, num_workers, learning_rate,
                         weight_decay, num_devices, accumulate_grad_batches)
        self.save_hyperparameters()

        gene_csv_path = os.path.join(working_codespace, "data_raw_preprocessing", "selected_genes", gene_csv_name)
        self.selected_genes_list = pd.read_csv(gene_csv_path)["gene_names"].values.tolist()
        if self.number_of_eval_HEG_genes == 0 and self.number_of_eval_HVG_genes == 0:
            raise ValueError("Please select the number of eval HEG genes and eval HVG genes. They cannot be both 0.")
        else:
            if self.number_of_eval_HEG_genes == 0:
                eval_selected_genes_HEG_list = []
            else:
                eval_selected_genes_HEG_csv_path = os.path.join(self.working_codespace, "data_raw_preprocessing", "selected_genes", f"top_HEG_genes_number_{self.number_of_eval_HEG_genes}.csv")
                eval_selected_genes_HEG_list = pd.read_csv(eval_selected_genes_HEG_csv_path)["gene_names"].values.tolist()
            if self.number_of_eval_HVG_genes == 0:
                eval_selected_genes_HVG_list = []
            else:
                eval_selected_genes_HVG_csv_path = os.path.join(self.working_codespace, "data_raw_preprocessing", "selected_genes", f"top_HVG_genes_number_{self.number_of_eval_HVG_genes}.csv")
                eval_selected_genes_HVG_list = pd.read_csv(eval_selected_genes_HVG_csv_path)["gene_names"].values.tolist()
            self.eval_selected_genes_list = eval_selected_genes_HEG_list + eval_selected_genes_HVG_list

        if self.image_encoder_name == "DenseNet121":
            self.model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            self.model.classifier = nn.Linear(1024, self.input_gene_number)
        elif self.image_encoder_name == "CTransPath":
            self.image_encoder = ImageEncoder(
                model_name=self.image_encoder_name,
                latent_embedding_dimension=256,
                main_data_storage=main_data_storage,
                raw_data_folder_name=raw_data_folder_name,
                device=device,
            )
            self.image_to_gene_decoder = nn.Linear(768, self.input_gene_number)
        elif self.image_encoder_name == "DINOv1":
            self.image_encoder = ImageEncoder(
                model_name=self.image_encoder_name,
                latent_embedding_dimension=256,
                main_data_storage=main_data_storage,
                raw_data_folder_name=raw_data_folder_name,
                device=device,
            )
            self.image_to_gene_decoder = nn.Linear(2048, self.input_gene_number)
        elif self.image_encoder_name == "DINOv2":
            self.image_encoder = ImageEncoder(
                model_name=self.image_encoder_name,
                latent_embedding_dimension=256,
                main_data_storage=main_data_storage,
                raw_data_folder_name=raw_data_folder_name,
                device=device,
            )
            self.image_to_gene_decoder = nn.Linear(384, self.input_gene_number)
        else:
            raise ValueError(f"Invalid image encoder name: {self.image_encoder_name}")

    def forward(
            self,
            batch,
            split="train",
    ):
        if self.use_normed_patch:
            original_patch_view = batch["all_original_normed_patch"]
        else:
            original_patch_view = batch["all_original_patch"]

        input_gene_expression = batch["all_binned_gene_expression"]

        if self.image_encoder_name == "DenseNet121":        
            image_to_gene_prediction = self.model(original_patch_view)
        elif self.image_encoder_name == "CTransPath" or self.image_encoder_name == "DINOv1" or self.image_encoder_name == "DINOv2":
            original_image_feature = self.image_encoder(original_patch_view)
            image_to_gene_prediction = self.image_to_gene_decoder(original_image_feature)
        else:
            raise ValueError(f"Invalid image encoder name: {self.image_encoder_name}")

        image_to_gene_generative_loss = torch.mean(
            (image_to_gene_prediction - input_gene_expression) ** 2
        )
        image_to_gene_generative_loss *= self.image_to_gene_generative_loss_weight

        loss_dict = {
            "overall_loss": image_to_gene_generative_loss,
            "image_to_gene_generative_loss": image_to_gene_generative_loss,
        }
        metrics_dict = dict()
        return_dict = dict()
        return loss_dict, metrics_dict, return_dict

    def on_shared_epoch_end(
        self,
        step_outputs: List,
    ) -> None:
        metrics_dict = dict()
        return metrics_dict

    def setup_datamodule(self):
        self.datamodule = DataModule(
            main_data_storage=self.main_data_storage,
            project_data_folder_name=self.project_data_folder_name,
            working_codespace=self.working_codespace,
            gene_csv_name=self.gene_csv_name,
            generation_date=self.generation_date,
            scGPT_gene_mask_ratio=self.scGPT_gene_mask_ratio,
            organ_selected=self.organ_selected,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        self.train_iters_per_epoch = len(self.datamodule.train_dataloader()) // (self.num_devices * self.accumulate_grad_batches)