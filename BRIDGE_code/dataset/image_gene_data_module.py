from lightning.pytorch import LightningDataModule
from typing import List, Optional, Union, Literal, Tuple
from image_gene_dataset import ImageGeneDataset
from torch.utils.data import DataLoader
import argparse
import time
import os

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
current_workspace = os.path.dirname(current_dir)
data_raw_preprocessing_dir = os.path.join(current_workspace, "data_raw_preprocessing")
sys.path.append(data_raw_preprocessing_dir)
from step0_preprocess_helper_functions import running_time_display

class DataModule(LightningDataModule):
    def __init__(
            self,
            main_data_storage: str,
            project_data_folder_name: str,
            working_codespace: str,
            gene_csv_name: str,
            generation_date: str,
            scGPT_gene_mask_ratio: float,
            organ_selected: str,
            batch_size: int,
            num_workers: int,
    ):
        super().__init__()
        self.main_data_storage = main_data_storage
        self.project_data_folder_name = project_data_folder_name
        self.working_codespace = working_codespace
        self.gene_csv_name = gene_csv_name
        self.generation_date = generation_date
        self.scGPT_gene_mask_ratio = scGPT_gene_mask_ratio
        self.organ_selected = organ_selected
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(
        self,
    ):
        train_dataset = ImageGeneDataset(
            main_data_storage=self.main_data_storage,
            project_data_folder_name=self.project_data_folder_name,
            working_codespace=self.working_codespace,
            gene_csv_name=self.gene_csv_name,
            generation_date=self.generation_date,
            train_or_test="train",
            scGPT_gene_mask_ratio=self.scGPT_gene_mask_ratio,
            organ_selected=self.organ_selected,
        )
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
        )
        
    def val_dataloader(
        self,
    ):
        test_dataset = ImageGeneDataset(
            main_data_storage=self.main_data_storage,
            project_data_folder_name=self.project_data_folder_name,
            working_codespace=self.working_codespace,
            gene_csv_name=self.gene_csv_name,
            generation_date=self.generation_date,
            train_or_test="valid",
            scGPT_gene_mask_ratio=self.scGPT_gene_mask_ratio,
            organ_selected=self.organ_selected,
        )
        return DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
        )
    
    def test_dataloader(
            self,
    ):
        test_dataset = ImageGeneDataset(
            main_data_storage=self.main_data_storage,
            project_data_folder_name=self.project_data_folder_name,
            working_codespace=self.working_codespace,
            gene_csv_name=self.gene_csv_name,
            generation_date=self.generation_date,
            train_or_test="test",
            scGPT_gene_mask_ratio=self.scGPT_gene_mask_ratio,
            organ_selected=self.organ_selected,
        )
        return DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
        )     
