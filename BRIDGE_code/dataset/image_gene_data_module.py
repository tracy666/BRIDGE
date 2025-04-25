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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_data_storage", type=str, default="/data1/zliang")
    parser.add_argument("--project_data_folder_name", type=str, default="BIG_600K")
    parser.add_argument("--working_codespace", type=str, default="/home/zliang/BRIDGE/BRIDGE_code")
    parser.add_argument("--gene_csv_name", type=str, default="all_intersection_genes_number_7730.csv")
    parser.add_argument("--generation_date", type=str, default="20240601")
    args = parser.parse_args()

    data_module = DataModule(
        main_data_storage=args.main_data_storage,
        project_data_folder_name=args.project_data_folder_name,
        working_codespace=args.working_codespace,
        gene_csv_name=args.gene_csv_name,
        generation_date=args.generation_date,
        scGPT_gene_mask_ratio=0.25,
        organ_selected=["all"],
        batch_size=16,
        num_workers=4,
    )
    print(len(data_module.train_dataloader()))
    print(len(data_module.val_dataloader()))
    print(len(data_module.test_dataloader()))

    for batch in data_module.val_dataloader():
        print(batch)
        break

if __name__ == "__main__":
    main()