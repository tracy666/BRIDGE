from base_model import BaseModel
from typing import Any, List, Literal
import os
import torch.nn.functional as F
import torch
import random
from scgpt.loss import masked_mse_loss
from einops import rearrange
import numpy as np
import pandas as pd
from lightning.pytorch.utilities.types import STEP_OUTPUT
from sklearn.neighbors import KDTree
import scipy
import numpy as np
from typing import Any, Dict, List

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

class BRIDGEModel(BaseModel):
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

            image_encoder_model_name: str,
            freeze_image_encoder_parameter: bool,
            gene_encoder_model_name: str,
            freeze_gene_encoder_parameter: bool,

            input_gene_number: int,
            number_of_eval_HEG_genes: int,
            number_of_eval_HVG_genes: int,
            output_gene_dimension: int,
            latent_embedding_dimension: int,

            contrastive_loss_weight: float,
            masked_contrastive_loss_weight: float,
            image_to_gene_generative_loss_weight: float,
            gene_reconstruction_loss_weight: float,
            image_self_supervised_loss_weight: float,
            gene_self_supervised_loss_weight: float,

            use_normed_patch: bool,
            use_hard_or_soft_contrastive_loss: Literal["hard", "soft"],

            MAE_image_mask_ratio: float,
            TabNet_gene_mask_ratio: float,
            scGPT_gene_mask_ratio: float,
            temperature: float,

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

        self.image_encoder_model_name = image_encoder_model_name
        self.freeze_image_encoder_parameter = freeze_image_encoder_parameter
        self.gene_encoder_model_name = gene_encoder_model_name
        self.freeze_gene_encoder_parameter = freeze_gene_encoder_parameter

        self.input_gene_number = input_gene_number
        self.number_of_eval_HEG_genes = number_of_eval_HEG_genes
        self.number_of_eval_HVG_genes = number_of_eval_HVG_genes
        self.output_gene_dimension = output_gene_dimension
        self.latent_embedding_dimension = latent_embedding_dimension

        self.contrastive_loss_weight = contrastive_loss_weight
        self.masked_contrastive_loss_weight = masked_contrastive_loss_weight
        self.image_to_gene_generative_loss_weight = image_to_gene_generative_loss_weight
        self.gene_reconstruction_loss_weight = gene_reconstruction_loss_weight
        self.image_self_supervised_loss_weight = image_self_supervised_loss_weight
        self.gene_self_supervised_loss_weight = gene_self_supervised_loss_weight

        self.use_normed_patch = use_normed_patch
        self.use_hard_or_soft_contrastive_loss = use_hard_or_soft_contrastive_loss

        self.MAE_image_mask_ratio = MAE_image_mask_ratio
        self.TabNet_gene_mask_ratio = TabNet_gene_mask_ratio
        self.scGPT_gene_mask_ratio = scGPT_gene_mask_ratio
        self.temperature = temperature
        random.seed(3407)

        super().__init__(max_epochs, warmup_epochs, batch_size, num_workers, learning_rate, weight_decay, num_devices, accumulate_grad_batches)
        self.save_hyperparameters()

        self.image_encoder = ImageEncoder(
            model_name=image_encoder_model_name,
            latent_embedding_dimension=latent_embedding_dimension,
            main_data_storage=main_data_storage,
            raw_data_folder_name=raw_data_folder_name,
            device=device,
        )
        if freeze_image_encoder_parameter:
            for parameter in self.image_encoder.parameters():
                parameter.requires_grad = False
            self.image_encoder.model.eval()
        
        self.gene_encoder = GeneEncoder(
            model_name=gene_encoder_model_name,
            input_gene_dimension=input_gene_number,
            output_dimension=output_gene_dimension,
            latent_embedding_dimension=latent_embedding_dimension,
            main_data_storage=main_data_storage,
            raw_data_folder_name=raw_data_folder_name,
            device=device,
        )
        if freeze_gene_encoder_parameter:
            for parameter in self.gene_encoder.parameters():
                parameter.requires_grad = False
            self.gene_encoder.model.eval()

        self.image_to_gene_decoder = torch.nn.Linear(self.image_encoder.feature_size, input_gene_number)

        self.gene_to_gene_decoder = torch.nn.Linear(output_gene_dimension, input_gene_number)

        gene_csv_path = os.path.join(self.working_codespace, "data_raw_preprocessing", "selected_genes", self.gene_csv_name)
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
    
    def forward(
        self,
        batch,
        split="train",
    ):
        batch_size = batch["all_original_normed_patch"].size(0)
        if self.use_normed_patch:
            original_patch_view = batch["all_original_normed_patch"]
            augmented_patch_view = batch["all_augmented_normed_patch"]
        else:
            original_patch_view = batch["all_original_patch"]
            augmented_patch_view = batch["all_augmented_patch"]

        original_image_feature = self.image_encoder(original_patch_view)
        original_image_embedding = self.image_encoder.forward_head(original_image_feature)
        original_image_embedding = F.normalize(original_image_embedding, p=2, dim=1)
        augmented_image_feature = self.image_encoder(augmented_patch_view)
        augmented_image_embedding = self.image_encoder.forward_head(augmented_image_feature)
        augmented_image_embedding = F.normalize(augmented_image_embedding, p=2, dim=1)

        input_gene_expression = batch["all_binned_gene_expression"]

        if self.gene_encoder_model_name == "TabNet" or self.gene_encoder_model_name == "MLP":
            original_gene_feature = self.gene_encoder(input_gene_expression.float())
            original_gene_embedding = self.gene_encoder.forward_head(original_gene_feature)
            original_gene_embedding = F.normalize(original_gene_embedding, p=2, dim=1)

            nonzero_boolean_tensor = input_gene_expression.bool()
            true_indices_in_nonzero_boolean_tensor = torch.nonzero(nonzero_boolean_tensor)
            number_of_positions_to_be_masked = int(true_indices_in_nonzero_boolean_tensor.size(0) * (self.TabNet_gene_mask_ratio))
            gene_masked_indices = random.sample(range(true_indices_in_nonzero_boolean_tensor.size(0)), number_of_positions_to_be_masked)
            gene_mask = torch.ones_like(nonzero_boolean_tensor)
            gene_mask[true_indices_in_nonzero_boolean_tensor[gene_masked_indices].split(1, dim=1)] = 0
            masked_boolean_tensor = nonzero_boolean_tensor * gene_mask
            masked_gene_expression = input_gene_expression.clone()
            masked_gene_expression[masked_boolean_tensor] = 0
            masked_gene_feature = self.gene_encoder(masked_gene_expression.float())
            masked_gene_embedding = self.gene_encoder.forward_head(masked_gene_feature)
            masked_gene_embedding = F.normalize(masked_gene_embedding, p=2, dim=1)

        elif self.gene_encoder_model_name == "scGPT":
            original_gene_feature, _ = self.gene_encoder(
                [
                    batch['all_scgpt_tokenized_gene_ids'],
                    batch['all_scgpt_tokenized_gene_values'].float(),
                ]
            )
            original_gene_embedding = self.gene_encoder.forward_head(original_gene_feature)
            original_gene_embedding = F.normalize(original_gene_embedding, p=2, dim=1)
            
            masked_gene_feature, masked_gene_prediction = self.gene_encoder(
                [
                    batch['all_scgpt_tokenized_gene_ids'],
                    batch['all_scgpt_masked_tokenized_gene_values'].float(),
                ]
            )
            masked_positions = batch['all_scgpt_masked_tokenized_gene_values'].eq(-1)
            gene_self_supervised_loss = masked_mse_loss(
                masked_gene_prediction,
                batch['all_scgpt_tokenized_gene_values'].float(),
                masked_positions,
            )
            masked_gene_embedding = self.gene_encoder.forward_head(masked_gene_feature)
            masked_gene_embedding = F.normalize(masked_gene_embedding, p=2, dim=1)        
        else:
            raise NotImplementedError(f"Gene encoder model {self.gene_encoder_model_name} is not implemented")

        """
        Now comes to the loss part:
        1. image-gene contrastive loss -> contrastive_loss
        2. image-gene prediction loss -> image_to_gene_generative_loss
        3. masked gene-gene prediction loss -> gene_reconstruction_loss
        4. augmented image-image contrastive loss -> image_self_supervised_loss
        5. gene-masked gene contrastive loss -> gene_self_supervised_loss
        """

        if self.use_hard_or_soft_contrastive_loss == "hard":
            contrastive_loss = self.Hard_InfoNCE_loss(
                image_embedding=original_image_embedding,
                gene_embedding=original_gene_embedding,
                temperature=self.temperature,
            )
            if self.masked_contrastive_loss_weight > 0:
                masked_contrastive_loss = self.Hard_InfoNCE_loss(
                    image_embedding=augmented_image_embedding,
                    gene_embedding=masked_gene_embedding,
                    temperature=self.temperature,
                )
            else:
                masked_contrastive_loss = torch.tensor(0.).type_as(original_image_feature)
            if self.image_self_supervised_loss_weight > 0:
                image_self_supervised_loss = self.Hard_InfoNCE_loss(
                    original_image_embedding,
                    augmented_image_embedding,
                    temperature=self.temperature,
                )
            else:
                image_self_supervised_loss = torch.tensor(0.).type_as(original_image_feature)
            if self.gene_self_supervised_loss_weight > 0:
                if self.gene_encoder_model_name == "TabNet" or self.gene_encoder_model_name == "MLP":
                    gene_self_supervised_loss = self.Hard_InfoNCE_loss(
                        masked_gene_embedding,
                        original_gene_embedding,
                        temperature=self.temperature,
                    )
            else:
                gene_self_supervised_loss = torch.tensor(0.).type_as(original_gene_feature)

        elif self.use_hard_or_soft_contrastive_loss == "soft":

            contrastive_loss = self.Soft_InfoNCE_loss(
                image_embedding=original_image_embedding,
                gene_embedding=original_gene_embedding,
                temperature=self.temperature,
            )
            if self.masked_contrastive_loss_weight > 0:
                masked_contrastive_loss = self.Soft_InfoNCE_loss(
                    image_embedding=augmented_image_embedding,
                    gene_embedding=masked_gene_embedding,
                    temperature=self.temperature,
                )
            else:
                masked_contrastive_loss = torch.tensor(0.).type_as(original_image_feature)
            if self.image_self_supervised_loss_weight > 0:
                image_self_supervised_loss = self.Soft_InfoNCE_loss(
                    original_image_embedding,
                    augmented_image_embedding,
                    temperature=self.temperature,
                )
            else:
                image_self_supervised_loss = torch.tensor(0.).type_as(original_image_feature)
            if self.gene_self_supervised_loss_weight > 0:
                if self.gene_encoder_model_name == "TabNet" or self.gene_encoder_model_name == "MLP":
                    gene_self_supervised_loss = self.Soft_InfoNCE_loss(
                        masked_gene_embedding,
                        original_gene_embedding,
                        temperature=self.temperature,
                    )
            else:
                gene_self_supervised_loss = torch.tensor(0.).type_as(original_gene_feature)

        else:
            raise ValueError(f"Invalid contrastive loss type: {self.use_hard_or_soft_contrastive_loss}")     

        image_to_gene_prediction = self.image_to_gene_decoder(
            original_image_feature
        )
        image_to_gene_generative_loss = torch.mean(
            (image_to_gene_prediction - input_gene_expression) ** 2
        )

        if self.gene_reconstruction_loss_weight > 0:
            gene_to_gene_prediction = self.gene_to_gene_decoder(
                masked_gene_feature
        )
            gene_reconstruction_loss = torch.mean(
                    (gene_to_gene_prediction - input_gene_expression) ** 2
        )   
        else:
            gene_reconstruction_loss = torch.tensor(0.).type_as(original_gene_feature)  

        contrastive_loss *= self.contrastive_loss_weight
        masked_contrastive_loss *= self.masked_contrastive_loss_weight
        image_to_gene_generative_loss *= self.image_to_gene_generative_loss_weight
        gene_reconstruction_loss *= self.gene_reconstruction_loss_weight
        image_self_supervised_loss *= self.image_self_supervised_loss_weight
        gene_self_supervised_loss *= self.gene_self_supervised_loss_weight

        loss_dict = {
            "overall_loss": contrastive_loss + masked_contrastive_loss + image_to_gene_generative_loss + gene_reconstruction_loss + image_self_supervised_loss + gene_self_supervised_loss,
            "contrastive_loss": contrastive_loss,
            "masked_contrastive_loss": masked_contrastive_loss,
            "image_to_gene_generative_loss": image_to_gene_generative_loss,
            "gene_reconstruction_loss": gene_reconstruction_loss,
            "image_self_supervised_loss": image_self_supervised_loss,
            "gene_self_supervised_loss": gene_self_supervised_loss,
        }

        # compute retrieval metrics
        # metrics_dict = dict()
        # logits = original_image_embedding @ original_gene_embedding.t()
        # target_indices = torch.arange(batch_size).type_as(logits).long()
        # if batch_size >= 20:
        #     recall_5, recall_10 = self.topk_recall(
        #         output=logits,
        #         target=target_indices,
        #         top_k_values=(5, 10),
        #     )
        #     metrics_dict["batch_recall_5"] = recall_5
        #     metrics_dict["batch_recall_10"] = recall_10

        # return_dict = dict()
        # return loss_dict, metrics_dict, return_dict
    
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
