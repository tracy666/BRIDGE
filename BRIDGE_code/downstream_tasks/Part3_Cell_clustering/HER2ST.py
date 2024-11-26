from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Union, Literal, Tuple
import pandas as pd
import numpy as np
import anndata
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
current_workspace = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
dataset_dir = os.path.join(current_workspace, "dataset")
sys.path.append(dataset_dir)
from image_gene_dataset import get_image_transforms

class SingleSlideEvalImageGeneDataset_HER2ST(Dataset):
    def __init__(
            self,
            main_data_storage: str,
            raw_data_folder_name: str,
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
        self.raw_data_folder_name = raw_data_folder_name
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

        patient_stdata_path = os.path.join(self.main_data_storage, self.project_data_folder_name, self.patient_dir, "preprocessed_data", f"stdata.h5ad")
        patient_stdata = anndata.read_h5ad(patient_stdata_path)
        patient_stdata.obs_names_make_unique()
        patient_stdata.var_names_make_unique()

        # filter rows such that patches does not exist (due to inaccurate coordinates)
        existing_patch_mask = [os.path.exists(os.path.join(self.main_data_storage, self.project_data_folder_name, single_patch_path)) for single_patch_path in patient_stdata.obs["patch_path"].values.tolist()]
        patient_stdata_with_patches = patient_stdata[existing_patch_mask]
        patient_stdata_obs_df = (patient_stdata_with_patches.obs).reset_index()

        patient_label_df_path = os.path.join(self.main_data_storage, self.raw_data_folder_name, "Specific_datasets/HER2ST_Version_3_0/Spatial_deconvolution_of_HER2-positive_Breast_cancer_delineates_tumor-associated_cell_type_interactions/meta", f"{self.patient_dir.split('/')[1]}_labeled_coordinates.tsv")
        assert os.path.exists(patient_label_df_path), f"Patient label df path: {patient_label_df_path} does not exist, please check if {self.patient_dir} is valid with annotation"
        patient_label_df = pd.read_csv(patient_label_df_path, sep="\t")

        patient_full_information_df = pd.merge(patient_stdata_obs_df, patient_label_df, left_on=['array_row', 'array_col'], right_on=['x', 'y'])
        HER2ST_patient_label_mapping_dict = {
            'adipose tissue': 0,
            'breast glands': 1,
            'cancer in situ': 2,
            'connective tissue': 3,
            'immune infiltrate': 4,
            'invasive cancer': 5,
            'undetermined': 6,
        }
        # assert sorted(patient_full_information_df["label"].unique().tolist()) == sorted(list(HER2ST_patient_label_mapping_dict.keys())), f"Patient label df unique values: {patient_full_information_df['label'].unique().tolist()}) does not match with HER2ST_patient_label_mapping_dict keys: {list(HER2ST_patient_label_mapping_dict.keys())}"
        patient_full_information_df["label_numerical"] = patient_full_information_df["label"].map(HER2ST_patient_label_mapping_dict)
        patient_full_information_df = patient_full_information_df.set_index("spot_barcode")
        self.patient_full_information_df = patient_full_information_df

        patient_dir_numerical_pt_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_patient_dir_numerical.pt")
        self.all_patient_dir_numerical = torch.load(patient_dir_numerical_pt_path)
        patch_path_npy_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_patch_path.npy")
        self.all_patch_path = np.load(patch_path_npy_path, allow_pickle=True)
        normed_patch_path_npy_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_normed_patch_path.npy")
        self.all_normed_patch_path = np.load(normed_patch_path_npy_path, allow_pickle=True)
        spot_barcode_npy_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_spot_barcode.npy")
        self.all_spot_barcode = np.load(spot_barcode_npy_path, allow_pickle=True)

        patient_dir_organ_mapping_df_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, "patient_dir_organ_mapping_df.csv")
        patient_dir_organ_mapping_df = pd.read_csv(patient_dir_organ_mapping_df_path)
        patient_dir_num_mapping_dict = dict(zip(patient_dir_organ_mapping_df["patient_dir"], patient_dir_organ_mapping_df.index))
        single_patient_dir_numerical = patient_dir_num_mapping_dict[self.patient_dir]
        mask = np.isin(self.all_patient_dir_numerical, single_patient_dir_numerical)

        index_list = list()
        for index, boolean_value in enumerate(mask):
            if boolean_value:
                if self.all_spot_barcode[index] in self.patient_full_information_df.index.tolist():
                    index_list.append(index)
        self.selected_index_list = index_list

        selected_spot_barcode_list = self.all_spot_barcode[self.selected_index_list].tolist()
        assert sorted(selected_spot_barcode_list) == sorted(patient_full_information_df.index.tolist()), f"Selected spot barcode list: {selected_spot_barcode_list} does not match with patient full information df index: {patient_full_information_df.index.tolist()}"
        
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

        spot_barcode = self.all_spot_barcode[selected_index]
        patient_label_numerical = self.patient_full_information_df.loc[spot_barcode, "label_numerical"]

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
            "all_patient_label_numerical": patient_label_numerical,
            "all_patient_spot_barcode": spot_barcode,
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