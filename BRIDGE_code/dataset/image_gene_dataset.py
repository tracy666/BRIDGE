import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Optional, Union, Literal, Tuple
import os
import anndata
import time
import argparse
import pandas as pd
import numpy as np
from scgpt.tokenizer import random_mask_value
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import time
from tqdm import tqdm

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
current_workspace = os.path.dirname(current_dir)
data_raw_preprocessing_dir = os.path.join(current_workspace, "data_raw_preprocessing")
sys.path.append(data_raw_preprocessing_dir)
from step0_preprocess_helper_functions import running_time_display
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='anndata')
warnings.filterwarnings(action="ignore", category=ResourceWarning) # ResourceWarning: Implicitly cleaning up <TemporaryDirectory>

def get_image_transforms(
        train_or_test: str,
        image_size: int,
        crop_size: int,
):
    if train_or_test == "train":
        img_transforms = transforms.Compose([
            transforms.Resize(size=image_size),
            transforms.RandomResizedCrop(size=crop_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation((90, 90))]),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, 
                                                            sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        img_transforms = transforms.Compose([
            transforms.Resize(size=image_size),
            transforms.CenterCrop(size=image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return img_transforms

class ImageGeneDataset(Dataset):
    def __init__(
            self,
            main_data_storage: str,
            project_data_folder_name: str,
            working_codespace: str,
            gene_csv_name: str,
            generation_date: str,
            train_or_test: Literal["train", "valid", "test"],
            scGPT_gene_mask_ratio: float,
            organ_selected: List[str],
    ):
        super().__init__()
        self.main_data_storage = main_data_storage
        self.project_data_folder_name = project_data_folder_name
        self.working_codespace = working_codespace
        self.generation_date = generation_date
        self.train_or_test = train_or_test
        self.gene_mask_ratio_str = str(scGPT_gene_mask_ratio).split(".")[1]
        self.organ_selected = organ_selected

        gene_csv_path = os.path.join(working_codespace, "data_raw_preprocessing", "selected_genes", gene_csv_name)
        selected_genes_list = pd.read_csv(gene_csv_path)["gene_names"].values.tolist()
        self.selected_genes_list = selected_genes_list
        self.gene_csv_name_without_extension = os.path.splitext(gene_csv_name)[0]
        check_scgpt_masked_input_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_masked_tokenized_gene_values_with_mask_ratio_{self.gene_mask_ratio_str}")
        assert os.path.exists(check_scgpt_masked_input_path), f"Haven't generate masked tokenized values for mask ratio {scGPT_gene_mask_ratio} under path {check_scgpt_masked_input_path}"

        patient_dir_numerical_pt_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_patient_dir_numerical.pt")
        self.all_patient_dir_numerical = torch.load(patient_dir_numerical_pt_path)
        patch_path_npy_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_patch_path.npy")
        self.all_patch_path = np.load(patch_path_npy_path, allow_pickle=True)
        normed_patch_path_npy_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_normed_patch_path.npy")
        self.all_normed_patch_path = np.load(normed_patch_path_npy_path, allow_pickle=True)

        train_df_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, f"train_df.csv")
        test_df_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, f"test_df.csv")
        train_df = pd.read_csv(train_df_path)
        test_df = pd.read_csv(test_df_path)
        
        if len(self.organ_selected) == 1:
            organ_selected = self.organ_selected[0]
            if organ_selected == "all":
                train_patient_dir_numerical_list = train_df["patient_dir_numerical"].tolist()
                test_patient_dir_numerical_list = test_df["patient_dir_numerical"].tolist()
            else:
                train_patient_dir_numerical_list = train_df[train_df["organ_type"] == organ_selected]["patient_dir_numerical"].tolist()
                test_patient_dir_numerical_list = test_df[test_df["organ_type"] == organ_selected]["patient_dir_numerical"].tolist()
        
        elif len(self.organ_selected) > 1:            
            assert "all" not in self.organ_selected, f"Please select only one organ or select all organs."
            train_patient_dir_numerical_list = list()
            test_patient_dir_numerical_list = list()
            for organ_selected in self.organ_selected:
                train_patient_dir_numerical_list += train_df[train_df["organ_type"] == organ_selected]["patient_dir_numerical"].tolist()
                test_patient_dir_numerical_list += test_df[test_df["organ_type"] == organ_selected]["patient_dir_numerical"].tolist()
        else:
            raise ValueError(f"Please select at least one organ.")
                
        train_mask = np.isin(self.all_patient_dir_numerical, train_patient_dir_numerical_list)
        test_mask = np.isin(self.all_patient_dir_numerical, test_patient_dir_numerical_list)

        train_index_list = list()
        valid_index_list = list()
        test_index_list = list()
        for index, boolean_value in enumerate(train_mask):
            if boolean_value:
                train_index_list.append(index)
        for index, boolean_value in enumerate(test_mask):
            if boolean_value:
                test_index_list.append(index)
        
        if self.train_or_test == "train":
            self.selected_index_list = train_index_list
            self.train_image_transform1, self.train_image_transform2 = [get_image_transforms(train_or_test=self.train_or_test, image_size=224, crop_size=224)] * 2
        elif self.train_or_test == "test":
            self.selected_index_list = test_index_list
            self.test_image_transform = get_image_transforms(train_or_test=self.train_or_test, image_size=224, crop_size=224)
        else:
            raise ValueError(f"The train_or_test should be either 'train' or 'valid' or 'test' or 'downstream, but now the value is {self.train_or_test}.")
        
        assert len(self.selected_index_list) > 0, f"The length of the dataset is zero. Please check whether the organ selected is valid."
    
    def __getitem__(
        self,
        index
    ):
        start_time = time.time()
        selected_index = self.selected_index_list[index]
        patient_dir_numerical = self.all_patient_dir_numerical[selected_index]
        patch_path = os.path.join(self.main_data_storage, self.project_data_folder_name, self.all_patch_path[selected_index])
        normed_patch_path = os.path.join(self.main_data_storage, self.project_data_folder_name, self.all_normed_patch_path[selected_index])

        # 1. full_dataset_masked_tokenized_gene_values_with_mask_ratio_xx
        scgpt_masked_tokenized_gene_values_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_masked_tokenized_gene_values_with_mask_ratio_{self.gene_mask_ratio_str}", f"index_{selected_index}.pt")
        scgpt_masked_tokenized_gene_values = torch.load(scgpt_masked_tokenized_gene_values_path)

        # 2. full_dataset_tokenized_gene_ids
        scgpt_tokenized_gene_ids_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_tokenized_gene_ids", f"index_{selected_index}.pt")
        scgpt_tokenized_gene_ids = torch.load(scgpt_tokenized_gene_ids_path)

        # 3. full_dataset_tokenized_gene_values
        scgpt_tokenized_gene_values_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_tokenized_gene_values", f"index_{selected_index}.pt")
        scgpt_tokenized_gene_values = torch.load(scgpt_tokenized_gene_values_path)
        
        # 4. full_dataset_X_binned
        binned_gene_expression_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_X_binned", f"index_{selected_index}.pt")
        binned_gene_expression = torch.load(binned_gene_expression_path)
        # binned_gene_expression = binned_gene_expression.float()

        raw_patch = Image.open(patch_path).convert('RGB')
        raw_normed_patch = Image.open(normed_patch_path).convert('RGB')
        if (self.train_or_test == "train") or (self.train_or_test == "valid"):
            original_patch = self.train_image_transform1(raw_patch)
            augmented_patch = self.train_image_transform2(raw_patch)
            original_normed_patch = self.train_image_transform1(raw_normed_patch)
            augmented_normed_patch = self.train_image_transform2(raw_normed_patch)
        elif (self.train_or_test == "test"):
            original_patch = self.test_image_transform(raw_patch)
            augmented_patch = original_patch
            original_normed_patch = self.test_image_transform(raw_normed_patch)
            augmented_normed_patch = original_normed_patch
        end_time = time.time()

        return {
            "all_patient_dir_numerical": patient_dir_numerical,
            "all_binned_gene_expression": binned_gene_expression,
            "all_scgpt_tokenized_gene_ids": scgpt_tokenized_gene_ids,
            "all_scgpt_tokenized_gene_values": scgpt_tokenized_gene_values,
            "all_scgpt_masked_tokenized_gene_values": scgpt_masked_tokenized_gene_values,
            "all_original_patch": original_patch,
            "all_augmented_patch": augmented_patch,
            "all_original_normed_patch": original_normed_patch,
            "all_augmented_normed_patch": augmented_normed_patch,
        }
    
    
    def __len__(
            self
    ):
        return len(self.selected_index_list)
    
class SingleSlideEvalImageGeneDataset(Dataset):
    def __init__(
            self,
            main_data_storage: str,
            project_data_folder_name: str,
            working_codespace: str,
            patient_dir: str,
            gene_csv_name: str,
            generation_date: str,
            train_or_test: Literal["train", "test"],
            scGPT_gene_mask_ratio: float,
    ):
        super().__init__()
        self.main_data_storage = main_data_storage
        self.project_data_folder_name = project_data_folder_name
        self.working_codespace = working_codespace
        self.patient_dir = patient_dir
        self.generation_date = generation_date
        self.train_or_test = train_or_test
        self.gene_mask_ratio_str = str(scGPT_gene_mask_ratio).split(".")[1]

        gene_csv_path = os.path.join(working_codespace, "data_raw_preprocessing", "selected_genes", gene_csv_name)
        selected_genes_list = pd.read_csv(gene_csv_path)["gene_names"].values.tolist()
        self.selected_genes_list = selected_genes_list
        self.gene_csv_name_without_extension = os.path.splitext(gene_csv_name)[0]
        assert os.path.exists(os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_masked_tokenized_gene_values_with_mask_ratio_{self.gene_mask_ratio_str}")), f"Haven't generate masked tokenized values for mask ratio {scGPT_gene_mask_ratio}"

        patient_dir_numerical_pt_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_patient_dir_numerical.pt")
        self.all_patient_dir_numerical = torch.load(patient_dir_numerical_pt_path)
        patch_path_npy_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_patch_path.npy")
        self.all_patch_path = np.load(patch_path_npy_path, allow_pickle=True)
        normed_patch_path_npy_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_normed_patch_path.npy")
        self.all_normed_patch_path = np.load(normed_patch_path_npy_path, allow_pickle=True)

        patient_dir_organ_mapping_df_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, "patient_dir_organ_mapping_df.csv")
        patient_dir_organ_mapping_df = pd.read_csv(patient_dir_organ_mapping_df_path)
        patient_dir_num_mapping_dict = dict(zip(patient_dir_organ_mapping_df["patient_dir"], patient_dir_organ_mapping_df.index))
        single_patient_dir_numerical = patient_dir_num_mapping_dict[self.patient_dir]
        mask = np.isin(self.all_patient_dir_numerical, single_patient_dir_numerical)

        index_list = list()
        for index, boolean_value in enumerate(mask):
            if boolean_value:
                index_list.append(index)
        self.selected_index_list = index_list
        
        if self.train_or_test == "train":
            self.train_image_transform1, self.train_image_transform2 = [get_image_transforms(train_or_test=self.train_or_test, image_size=224, crop_size=224)] * 2
        elif self.train_or_test == "test":
            self.test_image_transform = get_image_transforms(train_or_test=self.train_or_test, image_size=224, crop_size=224)
        else:
            raise ValueError(f"The train_or_test should be either 'train' or 'test', but now the value is {self.train_or_test}.")
    
    def __getitem__(
        self,
        index
    ):
        selected_index = self.selected_index_list[index]
        patient_dir_numerical = self.all_patient_dir_numerical[selected_index]
        patch_path = os.path.join(self.main_data_storage, self.project_data_folder_name, self.all_patch_path[selected_index])
        normed_patch_path = os.path.join(self.main_data_storage, self.project_data_folder_name, self.all_normed_patch_path[selected_index])

        # 1. full_dataset_masked_tokenized_gene_values_with_mask_ratio_xx
        scgpt_masked_tokenized_gene_values_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_masked_tokenized_gene_values_with_mask_ratio_{self.gene_mask_ratio_str}", f"index_{selected_index}.pt")
        scgpt_masked_tokenized_gene_values = torch.load(scgpt_masked_tokenized_gene_values_path)
        # 2. full_dataset_tokenized_gene_ids
        scgpt_tokenized_gene_ids_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_tokenized_gene_ids", f"index_{selected_index}.pt")
        scgpt_tokenized_gene_ids = torch.load(scgpt_tokenized_gene_ids_path)
        # 3. full_dataset_tokenized_gene_values
        scgpt_tokenized_gene_values_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_tokenized_gene_values", f"index_{selected_index}.pt")
        scgpt_tokenized_gene_values = torch.load(scgpt_tokenized_gene_values_path)
        # 4. full_dataset_X_binned
        binned_gene_expression_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_X_binned", f"index_{selected_index}.pt")
        binned_gene_expression = torch.load(binned_gene_expression_path)

        raw_patch = Image.open(patch_path).convert('RGB')
        raw_normed_patch = Image.open(normed_patch_path).convert('RGB')
        if self.train_or_test == "train":
            original_patch = self.train_image_transform1(raw_patch)
            augmented_patch = self.train_image_transform2(raw_patch)
            original_normed_patch = self.train_image_transform1(raw_normed_patch)
            augmented_normed_patch = self.train_image_transform2(raw_normed_patch)
        elif self.train_or_test == "test":
            original_patch = self.test_image_transform(raw_patch)
            augmented_patch = original_patch
            original_normed_patch = self.test_image_transform(raw_normed_patch)
            augmented_normed_patch = original_normed_patch

        return {
            "all_patient_dir_numerical": patient_dir_numerical,
            "all_binned_gene_expression": binned_gene_expression,
            "all_scgpt_tokenized_gene_ids": scgpt_tokenized_gene_ids,
            "all_scgpt_tokenized_gene_values": scgpt_tokenized_gene_values,
            "all_scgpt_masked_tokenized_gene_values": scgpt_masked_tokenized_gene_values,
            "all_original_patch": original_patch,
            "all_augmented_patch": augmented_patch,
            "all_original_normed_patch": original_normed_patch,
            "all_augmented_normed_patch": augmented_normed_patch,
        }
    
    def __len__(
            self
    ):
        return len(self.selected_index_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_data_storage", type=str, default="/data1/zliang")
    parser.add_argument("--project_data_folder_name", type=str, default="BIG_600K")
    parser.add_argument("--working_codespace", type=str, default="/home/zliang/BRIDGE/BRIDGE_code")
    parser.add_argument("--gene_csv_name", type=str, default="all_intersection_genes_number_7730.csv")
    parser.add_argument("--generation_date", type=str, default="20240601")
    args = parser.parse_args()

    train_dataset = ImageGeneDataset(
        main_data_storage=args.main_data_storage,
        project_data_folder_name=args.project_data_folder_name,
        working_codespace=args.working_codespace,
        gene_csv_name=args.gene_csv_name,
        generation_date=args.generation_date,
        train_or_test="train",
        scGPT_gene_mask_ratio=0.25,
        organ_selected=["all"],
    )
    print(len(train_dataset))
    
    test_dataset = ImageGeneDataset(
        main_data_storage=args.main_data_storage,
        project_data_folder_name=args.project_data_folder_name,
        working_codespace=args.working_codespace,
        gene_csv_name=args.gene_csv_name,
        generation_date=args.generation_date,
        train_or_test="test",
        scGPT_gene_mask_ratio=0.25,
        organ_selected=["all"],
    )
    print(len(test_dataset))

    single_slide_eval_dataset = SingleSlideEvalImageGeneDataset(
        main_data_storage=args.main_data_storage,
        project_data_folder_name=args.project_data_folder_name,
        working_codespace=args.working_codespace,
        patient_dir="5_locations_lung/WSA_LngSP8759313",
        gene_csv_name=args.gene_csv_name,
        generation_date=args.generation_date,
        train_or_test="test",
        scGPT_gene_mask_ratio=0.25,
    )
    print(len(single_slide_eval_dataset))



if __name__ == "__main__":
    main()