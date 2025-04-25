from typing import Any, Literal, List, Dict
from base_model import BaseModel
from torchvision.models import densenet121, DenseNet121_Weights
import torch
import torch.nn as nn
import os
import pandas as pd
from einops import rearrange
import torch.nn.functional as F
import timm
from lightning.pytorch.utilities.types import STEP_OUTPUT
from sklearn.neighbors import KDTree
import scipy
import numpy as np

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
current_workspace = os.path.dirname(current_dir)
backbones_dir = os.path.join(current_workspace, "backbones")
sys.path.append(backbones_dir)
from encoders import ImageEncoder, GeneEncoder, Projection_Head

dataset_dir = os.path.join(current_workspace, "dataset")
sys.path.append(dataset_dir)
from image_gene_data_module import DataModule

class BLEEPModel(BaseModel):
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
            latent_embedding_dimension: int,

            use_normed_patch: bool,
            use_hard_or_soft_contrastive_loss: Literal["hard", "soft"],

            scGPT_gene_mask_ratio: bool,
            temperature: float,
            retrieval_size: int,

            image_encoder_name: Literal["ResNet", "CTransPath"],
            gene_encoder_name: Literal["MLP", "TabNet", "scGPT"],

            device,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
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
        self.latent_embedding_dimension = latent_embedding_dimension

        self.use_normed_patch = use_normed_patch
        self.use_hard_or_soft_contrastive_loss = use_hard_or_soft_contrastive_loss

        self.scGPT_gene_mask_ratio = scGPT_gene_mask_ratio
        self.temperature = temperature
        self.retrieval_size = retrieval_size

        self.image_encoder_name = image_encoder_name
        self.gene_encoder_name = gene_encoder_name

        super().__init__(max_epochs, warmup_epochs, batch_size, num_workers, learning_rate,
                         weight_decay, num_devices, accumulate_grad_batches)
        self.save_hyperparameters()

        if self.image_encoder_name == "ResNet50":
            self.image_encoder = timm.create_model(
                model_name="resnet50",
                pretrained=True,
                num_classes=0,
                global_pool="avg"
            )
            self.image_projection_head = Projection_Head(
                embedding_dim=2048,
                projection_dim=self.latent_embedding_dimension,
            )
        elif self.image_encoder_name == "CTransPath":
            self.image_encoder = ImageEncoder(
                model_name="CTransPath",
                latent_embedding_dimension=latent_embedding_dimension,
                main_data_storage=main_data_storage,
                raw_data_folder_name=raw_data_folder_name,
                device=device,
            )
        elif self.image_encoder_name == "DenseNet121":
            self.image_encoder = ImageEncoder(
                model_name="DenseNet121",
                latent_embedding_dimension=latent_embedding_dimension,
                main_data_storage=main_data_storage,
                raw_data_folder_name=raw_data_folder_name,
                device=device,
            )
        else:
            raise ValueError(f"Invalid image encoder name: {self.image_encoder_name}")

        if self.gene_encoder_name == "MLP":
            self.gene_encoder = Projection_Head(
                embedding_dim=self.input_gene_number,
                projection_dim=self.latent_embedding_dimension,
            )
        elif self.gene_encoder_name == "TabNet":
            self.gene_encoder = GeneEncoder(
                model_name="TabNet",
                input_gene_dimension=input_gene_number,
                output_dimension=512,
                latent_embedding_dimension=latent_embedding_dimension,
                main_data_storage=main_data_storage,
                raw_data_folder_name=raw_data_folder_name,
                device=device,
            )
        elif self.gene_encoder_name == "scGPT":
            self.gene_encoder = GeneEncoder(
                model_name="scGPT",
                input_gene_dimension=input_gene_number,
                output_dimension=512,
                latent_embedding_dimension=latent_embedding_dimension,
                main_data_storage=main_data_storage,
                raw_data_folder_name=raw_data_folder_name,
                device=device,
            )
        else:
            raise ValueError(f"Invalid gene encoder name: {self.gene_encoder_name}")        

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

        if self.image_encoder_name == "ResNet50":
            original_image_feature = self.image_encoder(original_patch_view)
            original_image_embedding = self.image_projection_head(original_image_feature)
        elif self.image_encoder_name == "CTransPath":
            original_image_feature = self.image_encoder(original_patch_view)
            original_image_embedding = self.image_encoder.forward_head(original_image_feature)
        elif self.image_encoder_name == "DenseNet121":
            original_image_feature = self.image_encoder(original_patch_view)
            original_image_embedding = self.image_encoder.forward_head(original_image_feature)
        else:
            raise ValueError(f"Invalid image encoder name: {self.image_encoder_name}")

        if self.gene_encoder_name == "MLP":
            original_gene_embedding = self.gene_encoder(input_gene_expression.float())
        elif self.gene_encoder_name == "TabNet":
            original_gene_feature = self.gene_encoder(input_gene_expression.float())
            original_gene_embedding = self.gene_encoder.forward_head(original_gene_feature)
        elif self.gene_encoder_name == "scGPT":
            original_gene_feature, _ = self.gene_encoder(
                [
                    batch['all_scgpt_tokenized_gene_ids'],
                    batch['all_scgpt_tokenized_gene_values'].float(),
                ]
            )
            original_gene_embedding = self.gene_encoder.forward_head(original_gene_feature)
        else:
            raise ValueError(f"Invalid gene encoder name: {self.gene_encoder_name}")        

        if self.use_hard_or_soft_contrastive_loss == "hard":
            contrastive_loss = self.Hard_InfoNCE_loss(
                image_embedding=original_image_embedding,
                gene_embedding=original_gene_embedding,
                temperature=self.temperature,
            )
        elif self.use_hard_or_soft_contrastive_loss == "soft":
            contrastive_loss = self.Soft_InfoNCE_loss(
                image_embedding=original_image_embedding,
                gene_embedding=original_gene_embedding,
                temperature=self.temperature,
            )
        else:
            raise ValueError(f"Invalid contrastive loss type: {self.use_hard_or_soft_contrastive_loss}")

        loss_dict = {
            "overall_loss": contrastive_loss,
            "contrastive_loss": contrastive_loss,
        }

        metrics_dict = dict()
        logits = original_image_embedding @ original_gene_embedding.t()
        target_indices = torch.arange(original_patch_view.size()[0]).type_as(logits).long()
        batch_size = batch["all_original_patch"].size(0)
        if batch_size >= 20:
            recall_5, recall_10 = self.topk_recall(
                output=logits,
                target=target_indices,
                top_k_values=(5, 10),
            )
            metrics_dict["batch_recall_5"] = recall_5
            metrics_dict["batch_recall_10"] = recall_10

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