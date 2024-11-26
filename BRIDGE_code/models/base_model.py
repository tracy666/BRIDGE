from lightning import LightningModule
from typing import Any, Dict, List
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT
from abc import abstractmethod
import numpy as np
import scipy
import os

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
current_workspace = os.path.dirname(current_dir)
backbones_dir = os.path.join(current_workspace, "backbones")
sys.path.append(backbones_dir)
from lr_scheduler import linear_warmup_decay

class BaseModel(LightningModule):
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
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_devices = num_devices
        self.accumulate_grad_batches = accumulate_grad_batches
        self.setup_datamodule()
    
    @staticmethod
    def Hard_InfoNCE_loss(image_embedding, gene_embedding, temperature):
        assert image_embedding.size(0) == gene_embedding.size(0), f"Batch size mismatch: {image_embedding.size(0)} != {gene_embedding.size(0)}"
        batch_size = image_embedding.size(0)
        logits = image_embedding @ gene_embedding.t()
        target_indices = torch.arange(batch_size).type_as(logits).long()
        image_loss = F.cross_entropy(logits / temperature, target_indices)
        gene_loss = F.cross_entropy(logits.t() / temperature, target_indices)
        contrastive_loss = (image_loss + gene_loss)/2.
        return contrastive_loss

    @staticmethod
    def Soft_InfoNCE_loss(image_embedding, gene_embedding, temperature):
        assert image_embedding.size(0) == gene_embedding.size(0), f"Batch size mismatch: {image_embedding.size(0)} != {gene_embedding.size(0)}"
        batch_size = image_embedding.size(0)
        logits = image_embedding @ gene_embedding.t()
        image_image_similarity = image_embedding @ image_embedding.t()
        gene_gene_similarity = gene_embedding @ gene_embedding.t()
        soft_targets = F.softmax(
            (image_image_similarity + gene_gene_similarity) / 2 * temperature, dim=-1
        )
        image_loss = F.cross_entropy(logits / temperature, soft_targets)
        gene_loss = F.cross_entropy(logits.t() / temperature, soft_targets)
        contrastive_loss = (image_loss + gene_loss)/2.
        return contrastive_loss
    
    def training_step(
        self,
        batch: Dict,
        batch_idx: int,
    ) -> STEP_OUTPUT:
        batch_size = batch["all_original_normed_patch"].size(0)
        loss_dict, metrics_dict, _ = self(batch, split="train")
        log_dict = dict()
        for key, value in loss_dict.items():
            log_dict["train_" + key] = value
        for key, value in metrics_dict.items():
            log_dict["train_" + key] = value
        self.log_dict(log_dict, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True,
                      batch_size=batch_size, rank_zero_only=True)
        return loss_dict["overall_loss"]

    def validation_step(
        self,
        batch: Dict,
        batch_idx: int,
    ) -> STEP_OUTPUT:
        batch_size = batch["all_original_patch"].size(0)
        loss_dict, metrics_dict, return_dict = self(batch, split="valid")
        log_dict = dict()
        for key, value in loss_dict.items():
            log_dict["test_" + key] = value
        for key, value in metrics_dict.items():
            log_dict["test_" + key] = value
        self.log_dict(log_dict, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True,
                      batch_size=batch_size, rank_zero_only=True)
        self.val_step_output.append(return_dict)
        return loss_dict["overall_loss"]
    
    @abstractmethod
    def forward(self, batch, split="train"):
        raise NotImplementedError
    
    @abstractmethod
    def on_shared_epoch_end(self, step_outputs: List) -> None:
        raise NotImplementedError
    
    def on_validation_epoch_start(self) -> None:
        self.val_step_output = []

    def on_validation_epoch_end(self) -> None:
        metrics_dict = self.on_shared_epoch_end(self.val_step_output)
        new_metrics_dict = dict()
        for key, value in metrics_dict.items():
            new_metrics_dict["val_" + key] = value

        self.log_dict(new_metrics_dict, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True,
                      batch_size=len(self.val_step_output), rank_zero_only=True)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.learning_rate))
        
        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs
        
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
    
    @staticmethod
    def compute_topk_genes_pcc(
        gene_ground_truth: np.ndarray,
        gene_prediction: np.ndarray,
        selected_genes_list: List[str],
        eval_selected_gene_names_list: List[str],
    ):
        assert gene_ground_truth.shape == gene_prediction.shape, f"Mismatch of shape between gene ground truth and prediction, while ground truth is of shape {gene_ground_truth.shape} and prediction is of shape {gene_prediction.shape}"
        gene_names_indices_mapping = {gene_name: index for index, gene_name in enumerate(selected_genes_list)}
        column_numbers_selected = [gene_names_indices_mapping[gene_name] for gene_name in selected_genes_list if gene_name in eval_selected_gene_names_list]
        sorted_column_numbers_selected = sorted(column_numbers_selected, key=lambda x: eval_selected_gene_names_list.index(selected_genes_list[x]))

        pcc_values_list = list()
        for column_number in sorted_column_numbers_selected:
            specific_gene_ground_truth = gene_ground_truth[:, column_number]
            specific_gene_prediction = gene_prediction[:, column_number]
            pcc_results = scipy.stats.pearsonr(specific_gene_ground_truth, specific_gene_prediction)
            pcc_statistic = pcc_results[0]
            pcc_pvalue = pcc_results[1]
            if np.isnan(pcc_statistic):
                pcc_statistic = 0
            pcc_values_list.append(pcc_statistic)
        pcc_mean = np.mean(pcc_values_list)
        return pcc_mean

    @abstractmethod
    def setup_datamodule(self):
        raise NotImplementedError

    @staticmethod
    def topk_recall(
        output,
        target,
        top_k_values: tuple,
    ):
        with torch.no_grad():
            max_k_value = max(top_k_values)
            batch_size = target.size(0)
            _, prediction = output.topk(max_k_value, 1, True, True)
            prediction = prediction.t()
            correct_match = prediction.eq(target.view(1, -1).expand_as(prediction))
            result_list = list()
            for k_value in top_k_values:
                correct_match_with_k_value = (correct_match[:k_value].sum(dim=0) >= 1).float().sum() # the number of at least one prediction
                result_list.append(correct_match_with_k_value.mul_(100.0 / batch_size))
        return result_list
