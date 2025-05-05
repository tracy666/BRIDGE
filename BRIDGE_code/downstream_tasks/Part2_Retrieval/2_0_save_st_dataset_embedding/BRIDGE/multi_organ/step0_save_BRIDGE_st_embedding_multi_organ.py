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
current_workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
dataset_dir = os.path.join(current_workspace, "dataset")
sys.path.append(dataset_dir)
from image_gene_dataset import ImageGeneDataset, SingleSlideEvalImageGeneDataset, get_image_transforms

models_dir = os.path.join(current_workspace, "models")
sys.path.append(models_dir)
from BRIDGE_model import BRIDGEModel

raw_preprocessing_dir = os.path.join(current_workspace, "data_raw_preprocessing")
sys.path.append(raw_preprocessing_dir)
from step0_preprocess_helper_functions import running_time_display

def get_image_gene_embedding_for_retrieval(
        model,
        dataloader,
        device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_image_embedding_list = list()
    all_gene_embedding_list = list()
    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Extract gene sampling pool embeddings for retrieval'):
        original_patch_view = batch["all_original_normed_patch"].to(device)
        input_gene_expression = batch["all_binned_gene_expression"].to(device)

        if model.image_encoder_model_name == "ResNet50":
            original_image_feature = model.image_encoder(original_patch_view)
            original_image_embedding = model.image_projection_head(original_image_feature)
        elif model.image_encoder_model_name == "CTransPath" or model.image_encoder_model_name == "DenseNet121":
            original_image_feature = model.image_encoder(original_patch_view)
            original_image_embedding = model.image_encoder.forward_head(original_image_feature)
        else:
            raise ValueError(f"Unsupported image encoder model: {model.image_encoder_name}")

        if model.gene_encoder_model_name == "MLP":
            original_gene_embedding = model.gene_encoder(input_gene_expression.float())
        elif model.gene_encoder_model_name == "TabNet":
            original_gene_feature = model.gene_encoder(input_gene_expression.float())
            original_gene_embedding = model.gene_encoder.forward_head(original_gene_feature)
        elif model.gene_encoder_model_name == "scGPT":
            scgpt_tokenized_gene_ids = batch['all_scgpt_tokenized_gene_ids'].to(device)
            scgpt_tokenized_gene_values = batch['all_scgpt_tokenized_gene_values'].float().to(device)
            original_gene_feature, _ = model.gene_encoder(
                [
                    scgpt_tokenized_gene_ids,
                    scgpt_tokenized_gene_values,
                ]
            )
            original_gene_embedding = model.gene_encoder.forward_head(original_gene_feature)
        else:
            raise ValueError(f"Unsupported gene encoder model: {model.gene_encoder_name}")

        all_image_embedding_list.append(original_image_embedding.detach().cpu().numpy())
        all_gene_embedding_list.append(original_gene_embedding.detach().cpu().numpy())
    all_image_embedding = np.concatenate(all_image_embedding_list, axis=0)
    all_gene_embedding = np.concatenate(all_gene_embedding_list, axis=0)
    return all_image_embedding, all_gene_embedding

class Whole600KDataset(Dataset):
    def __init__(
        self,
        main_data_storage: str,
        project_data_folder_name: str,
        working_codespace: str,
        gene_csv_name: str,
        generation_date: str,
        scGPT_gene_mask_ratio: float,
    ):
        super().__init__()
        self.main_data_storage = main_data_storage
        self.project_data_folder_name = project_data_folder_name
        self.working_codespace = working_codespace
        self.gene_csv_name = gene_csv_name
        self.generation_date = generation_date
        gene_mask_ratio_str = str(scGPT_gene_mask_ratio).split(".")[1]
        gene_csv_name_without_extension = os.path.splitext(gene_csv_name)[0]

        patient_dir_numerical_pt_path = os.path.join(working_codespace, "generated_files", generation_date, gene_csv_name_without_extension, f"full_dataset_patient_dir_numerical.pt")
        self.all_patient_dir_numerical = torch.load(patient_dir_numerical_pt_path)
        patch_path_npy_path = os.path.join(working_codespace, "generated_files", generation_date, gene_csv_name_without_extension, f"full_dataset_patch_path.npy")
        self.all_patch_path = np.load(patch_path_npy_path, allow_pickle=True)
        normed_patch_path_npy_path = os.path.join(working_codespace, "generated_files", generation_date, gene_csv_name_without_extension, f"full_dataset_normed_patch_path.npy")
        self.all_normed_patch_path = np.load(normed_patch_path_npy_path, allow_pickle=True)

        self.full_dataset_masked_tokenized_gene_values_folder_path = os.path.join(working_codespace, "generated_files", generation_date, gene_csv_name_without_extension, f"full_dataset_masked_tokenized_gene_values_with_mask_ratio_{gene_mask_ratio_str}")
        self.full_dataset_tokenized_gene_ids_folder_path = os.path.join(working_codespace, "generated_files", generation_date, gene_csv_name_without_extension, "full_dataset_tokenized_gene_ids")
        self.full_dataset_tokenized_gene_values_folder_path = os.path.join(working_codespace, "generated_files", generation_date, gene_csv_name_without_extension, "full_dataset_tokenized_gene_values")
        self.full_dataset_X_binned_folder_path = os.path.join(working_codespace, "generated_files", generation_date, gene_csv_name_without_extension, "full_dataset_X_binned")
        assert len(os.listdir(self.full_dataset_masked_tokenized_gene_values_folder_path)) == len(os.listdir(self.full_dataset_tokenized_gene_ids_folder_path)) == len(os.listdir(self.full_dataset_tokenized_gene_values_folder_path)) == len(os.listdir(self.full_dataset_X_binned_folder_path)), f"Number of files in {full_dataset_masked_tokenized_gene_values_folder_path}, {full_dataset_tokenized_gene_ids_folder_path}, {full_dataset_tokenized_gene_values_folder_path}, {full_dataset_X_binned_folder_path} are not equal"

        full_dataset_index_list = [i for i in range(0, len(os.listdir(self.full_dataset_masked_tokenized_gene_values_folder_path)))]
        self.full_dataset_index_list = full_dataset_index_list

        self.test_image_transform = get_image_transforms(train_or_test="test", image_size=224, crop_size=224)

    def __getitem__(
        self,
        index,
    ):
        patient_dir_numerical = self.all_patient_dir_numerical[index]
        patch_path = os.path.join(self.main_data_storage, self.project_data_folder_name, self.all_patch_path[index])
        normed_patch_path = os.path.join(self.main_data_storage, self.project_data_folder_name, self.all_normed_patch_path[index])

        # 1. full_dataset_masked_tokenized_gene_values_with_mask_ratio_xx
        scgpt_masked_tokenized_gene_values_path = os.path.join(self.full_dataset_masked_tokenized_gene_values_folder_path, f"index_{index}.pt")
        scgpt_masked_tokenized_gene_values = torch.load(scgpt_masked_tokenized_gene_values_path)

        # 2. full_dataset_tokenized_gene_ids
        scgpt_tokenized_gene_ids_path = os.path.join(self.full_dataset_tokenized_gene_ids_folder_path, f"index_{index}.pt")
        scgpt_tokenized_gene_ids = torch.load(scgpt_tokenized_gene_ids_path)

        # 3. full_dataset_tokenized_gene_values
        scgpt_tokenized_gene_values_path = os.path.join(self.full_dataset_tokenized_gene_values_folder_path, f"index_{index}.pt")
        scgpt_tokenized_gene_values = torch.load(scgpt_tokenized_gene_values_path)

        # 4. full_dataset_X_binned
        binned_gene_expression_path = os.path.join(self.full_dataset_X_binned_folder_path, f"index_{index}.pt")
        binned_gene_expression = torch.load(binned_gene_expression_path)

        raw_patch = Image.open(patch_path).convert('RGB')
        raw_normed_patch = Image.open(normed_patch_path).convert('RGB')
        original_patch = self.test_image_transform(raw_patch)
        original_normed_patch = self.test_image_transform(raw_normed_patch)

        return {
            "all_patient_dir_numerical": patient_dir_numerical,
            "all_binned_gene_expression": binned_gene_expression,
            "all_scgpt_tokenized_gene_ids": scgpt_tokenized_gene_ids,
            "all_scgpt_tokenized_gene_values": scgpt_tokenized_gene_values,
            "all_scgpt_masked_tokenized_gene_values": scgpt_masked_tokenized_gene_values,
            "all_original_patch": original_patch,
            "all_original_normed_patch": original_normed_patch,
        }
    
    def __len__(self):
        return len(self.full_dataset_index_list)

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

    whole_dataset = Whole600KDataset(
        main_data_storage=args.main_data_storage,
        project_data_folder_name=args.project_data_folder_name,
        working_codespace=args.working_codespace,
        gene_csv_name=args.gene_csv_name,
        generation_date=args.generation_date,
        scGPT_gene_mask_ratio=0.25,
    )
    whole_dataloader = torch.utils.data.DataLoader(
        whole_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )

    checkpoint = os.path.join(args.main_data_storage, "finalized_checkpoints", "BRIDGE", "multi_organ", "49.ckpt")
    assert os.path.exists(checkpoint)

    try:
        epoch_number = int(checkpoint.split("/")[-1].split(".")[0]) + 1
    except ValueError:
        epoch_number = int((checkpoint.split("/")[-1].split(".")[0]).split("_")[-1]) + 1

    pretrained_BRIDGE_model = BRIDGEModel.load_from_checkpoint(
        checkpoint,
        strict=False,
        main_data_storage=args.main_data_storage,
        working_codespace=args.working_codespace,
        weight_decay=1e-3,
    )
    pretrained_BRIDGE_model.eval()
    print(f"Evaluating multi-organ BRIDGE model checkpoint: {checkpoint}")
    print("---------------------------------------------------------------------------------------------------------------")

    ST_extraction_start_time = time.time()
    whole_dataset_image_embedding, whole_dataset_gene_embedding = get_image_gene_embedding_for_retrieval(
        model=pretrained_BRIDGE_model,
        dataloader=whole_dataloader,
        device=device,
    )
    ST_extraction_end_time = time.time()
    print(f"multi-organ BRIDGE ST reference dataset extraction time: {running_time_display(ST_extraction_end_time - ST_extraction_start_time)}")
    print("---------------------------------------------------------------------------------------------------------------")

    st_embedding_saving_dir = os.path.join(
        downstream_tasks_saving_dir,
        f"multi_organ_at_epoch_{epoch_number}"
    )
    os.makedirs(st_embedding_saving_dir, exist_ok=True)

    saving_start_time = time.time()
    torch.save(torch.from_numpy(whole_dataset_image_embedding), os.path.join(st_embedding_saving_dir, f"multi_organ_whole_st_dataset_image_embedding.pt"))
    torch.save(torch.from_numpy(whole_dataset_gene_embedding), os.path.join(st_embedding_saving_dir, f"multi_organ_whole_st_dataset_gene_embedding.pt"))
    starting_end_time = time.time()
    print(f"multi-organ BRIDGE ST reference dataset saving time: {running_time_display(starting_end_time - saving_start_time)}")
    print("---------------------------------------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()