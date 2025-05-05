import argparse
import os
import logging
from datetime import datetime
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from typing import List, Optional, Union, Literal, Tuple
import scipy
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from sklearn.neighbors import KDTree
import time
from torch.utils.data import Dataset

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
current_workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
dataset_dir = os.path.join(current_workspace, "dataset")
sys.path.append(dataset_dir)
from image_gene_dataset import ImageGeneDataset, SingleSlideEvalImageGeneDataset, get_image_transforms

models_dir = os.path.join(current_workspace, "models")
sys.path.append(models_dir)
from BLEEP_model import BLEEPModel

raw_preprocessing_dir = os.path.join(current_workspace, "data_raw_preprocessing")
sys.path.append(raw_preprocessing_dir)
from step0_preprocess_helper_functions import running_time_display

def get_single_cell_gene_embedding_for_retrieval(
        model,
        dataloader,
        device,
) -> np.ndarray:
    all_single_cell_gene_embedding_list = list()
    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Extract gene sampling pool embeddings for retrieval'):
        input_gene_expression = batch["all_binned_gene_expression"].to(device)

        if model.gene_encoder_name == "MLP":
            original_gene_embedding = model.gene_encoder(input_gene_expression.float())
        elif model.gene_encoder_name == "TabNet":
            original_gene_feature = model.gene_encoder(input_gene_expression.float())
            original_gene_embedding = model.gene_encoder.forward_head(original_gene_feature)
            original_gene_embedding = F.normalize(original_gene_embedding, p=2, dim=1)
        elif model.gene_encoder_name == "scGPT":
            scgpt_tokenized_gene_ids = batch['all_scgpt_tokenized_gene_ids'].to(device)
            scgpt_tokenized_gene_values = batch['all_scgpt_tokenized_gene_values'].float().to(device)
            with torch.cuda.amp.autocast(enabled=True): # Automatic mixed precision
                original_gene_feature, _ = model.gene_encoder(
                    [
                        scgpt_tokenized_gene_ids,
                        scgpt_tokenized_gene_values,
                    ]
                )
                original_gene_embedding = model.gene_encoder.forward_head(original_gene_feature)
            original_gene_embedding = F.normalize(original_gene_embedding, p=2, dim=1)
        else:
            raise ValueError(f"Unsupported gene encoder model: {model.gene_encoder.model_name}")
        
        all_single_cell_gene_embedding_list.append(original_gene_embedding.detach().cpu().numpy())
    all_single_cell_gene_embedding = np.concatenate(all_single_cell_gene_embedding_list, axis=0)
    return all_single_cell_gene_embedding

class SingleCellGeneDataset(Dataset):
    def __init__(
        self,
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
        working_codespace: str,
        gene_csv_name: str,
        generation_date: str,
        organ_type: str,
    ):
        super().__init__()
        self.main_data_storage = main_data_storage
        self.raw_data_folder_name = raw_data_folder_name
        self.project_data_folder_name = project_data_folder_name
        self.working_codespace = working_codespace
        self.gene_csv_name = gene_csv_name
        self.generation_date = generation_date
        self.organ_type = organ_type
        gene_csv_name_without_extension = os.path.splitext(gene_csv_name)[0]

        self.full_dataset_tokenized_gene_ids_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Cellxgene_single_cell_datasets", self.organ_type, "Single_cell_dataset_scGPT_preprocessed", "full_single_cell_dataset_tokenized_gene_ids")
        self.full_dataset_tokenized_gene_values_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Cellxgene_single_cell_datasets", self.organ_type, "Single_cell_dataset_scGPT_preprocessed", "full_single_cell_dataset_tokenized_gene_values")
        self.full_dataset_X_binned_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Cellxgene_single_cell_datasets", self.organ_type, "Single_cell_dataset_scGPT_preprocessed", "full_single_cell_dataset_X_binned")
        assert len(os.listdir(self.full_dataset_tokenized_gene_ids_folder_path)) == len(os.listdir(self.full_dataset_tokenized_gene_values_folder_path)) == len(os.listdir(self.full_dataset_X_binned_folder_path)), f"Number of files in {full_dataset_masked_tokenized_gene_values_folder_path}, {full_dataset_tokenized_gene_ids_folder_path}, {full_dataset_tokenized_gene_values_folder_path}, {full_dataset_X_binned_folder_path} are not equal"
    
    def __getitem__(
            self,
            index,
    ):
        # 2. full_dataset_tokenized_gene_ids
        scgpt_tokenized_gene_ids_path = os.path.join(self.full_dataset_tokenized_gene_ids_folder_path, f"index_{index}.pt")
        scgpt_tokenized_gene_ids = torch.load(scgpt_tokenized_gene_ids_path)

        # 3. full_dataset_tokenized_gene_values
        scgpt_tokenized_gene_values_path = os.path.join(self.full_dataset_tokenized_gene_values_folder_path, f"index_{index}.pt")
        scgpt_tokenized_gene_values = torch.load(scgpt_tokenized_gene_values_path)

        # 4. full_dataset_X_binned
        binned_gene_expression_path = os.path.join(self.full_dataset_X_binned_folder_path, f"index_{index}.pt")
        binned_gene_expression = torch.load(binned_gene_expression_path)

        return {
            "all_binned_gene_expression": binned_gene_expression,
            "all_scgpt_tokenized_gene_ids": scgpt_tokenized_gene_ids,
            "all_scgpt_tokenized_gene_values": scgpt_tokenized_gene_values,
        }

    def __len__(
            self,
    ):
        return len(os.listdir(self.full_dataset_tokenized_gene_ids_folder_path))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--main_data_storage', type=str, default="/data1/zliang")
    parser.add_argument('--raw_data_folder_name', type=str, default="Histo_ST_raw")
    parser.add_argument('--project_data_folder_name', type=str, default="BIG_600K")
    parser.add_argument('--working_codespace', type=str, default="/home/zliang/BRIDGE/BRIDGE_code")

    parser.add_argument("--generation_date", type=str, default="20240601")
    parser.add_argument('--gene_csv_name', type=str, default="all_intersection_genes_number_7730.csv")

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--gpu_cards', type=str, default='', help='Comma-separated list of GPU card numbers')
    args = parser.parse_args()

    gpu_cards = args.gpu_cards.split(",") if args.gpu_cards else []
    gpu_cards_str = ",".join(gpu_cards)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_cards_str
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    downstream_tasks_saving_dir = current_dir.replace(
        args.working_codespace,
        args.main_data_storage,
    )
    os.makedirs(downstream_tasks_saving_dir, exist_ok=True)

    for organ_type in [
        "brain",
        "breast",
        "heart",
        "liver",
        "lung",
        "nose",
        "ovary",
        "prostate",
        "skin",
        "small_and_large_intestine",
    ]:
        single_cell_gene_dataset = SingleCellGeneDataset(
            main_data_storage=args.main_data_storage,
            raw_data_folder_name=args.raw_data_folder_name,
            project_data_folder_name=args.project_data_folder_name,
            working_codespace=args.working_codespace,
            gene_csv_name=args.gene_csv_name,
            generation_date=args.generation_date,
            organ_type=organ_type,
        )
        single_cell_gene_dataloader = torch.utils.data.DataLoader(
            single_cell_gene_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
        )
        
        checkpoint = os.path.join(args.main_data_storage, "finalized_checkpoints", "BLEEP", "single_organ", f"BLEEP_{organ_type}", "19.ckpt")
        assert os.path.exists(checkpoint)

        try:
            epoch_number = int(checkpoint.split("/")[-1].split(".")[0]) + 1
        except ValueError:
            epoch_number = int((checkpoint.split("/")[-1].split(".")[0]).split("_")[-1]) + 1

        pretrained_BLEEP_model = BLEEPModel.load_from_checkpoint(
            checkpoint,
            strict=False,
            main_data_storage=args.main_data_storage,
            working_codespace=args.working_codespace,
            weight_decay=1e-3,
        )
        pretrained_BLEEP_model.eval()
        print(f"Evaluating single-organ BLEEP model checkpoint: {checkpoint} for organ {organ_type}")
        print("---------------------------------------------------------------------------------------------------------------")

        SC_extraction_start_time = time.time()
        whole_single_cell_dataset_gene_embedding = get_single_cell_gene_embedding_for_retrieval(
            model=pretrained_BLEEP_model,
            dataloader=single_cell_gene_dataloader,
            device=device,
        )
        SC_extraction_end_time = time.time()
        print(f"single-organ BLEEP single-cell reference dataset extraction time: {running_time_display(SC_extraction_end_time - SC_extraction_start_time)}")
        print("---------------------------------------------------------------------------------------------------------------")

        sc_embedding_saving_dir = os.path.join(
            downstream_tasks_saving_dir,
            f"{organ_type}_at_epoch_{epoch_number}"
        )
        os.makedirs(sc_embedding_saving_dir, exist_ok=True)

        saving_start_time = time.time()
        torch.save(torch.from_numpy(whole_single_cell_dataset_gene_embedding), os.path.join(sc_embedding_saving_dir, f"{organ_type}_whole_single_cell_dataset_gene_embedding.pt"))
        starting_end_time = time.time()
        print(f"single-organ BLEEP single-cell reference dataset saving time: {running_time_display(starting_end_time - saving_start_time)}")
        print("---------------------------------------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()