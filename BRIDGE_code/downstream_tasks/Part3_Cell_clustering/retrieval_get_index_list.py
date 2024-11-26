import argparse
import logging
from datetime import datetime
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from typing import List, Optional, Union, Literal, Tuple
import scipy
import pickle
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import time
from sklearn.neighbors import KDTree
import h5py

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
current_workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
dataset_dir = os.path.join(current_workspace, "dataset")
sys.path.append(dataset_dir)
from image_gene_dataset import get_image_transforms

def get_index_list(
        retrieval_embedding_train_or_test: Literal["train", "test"],
        main_data_storage: str,
        project_data_folder_name: str,
        working_codespace: str,
        gene_csv_name: str,
        generation_date: str,
        data_source_train_or_test: Literal["train", "valid", "test", "train_valid", "train_test", "valid_test", "train_valid_test"],
        scGPT_gene_mask_ratio: float,
        organ_selected: List[str],
        patient_dir: str,
        exclude_patient_dir: bool,
):
    gene_mask_ratio_str = str(scGPT_gene_mask_ratio).split(".")[1]
    gene_csv_name_without_extension = os.path.splitext(gene_csv_name)[0]
    check_scgpt_masked_input_path = os.path.join(working_codespace, "generated_files", generation_date, gene_csv_name_without_extension, f"full_dataset_masked_tokenized_gene_values_with_mask_ratio_{gene_mask_ratio_str}")
    assert os.path.exists(check_scgpt_masked_input_path), f"Haven't generate masked tokenized values for mask ratio {scGPT_gene_mask_ratio} under path {check_scgpt_masked_input_path}"

    patient_dir_numerical_pt_path = os.path.join(working_codespace, "generated_files", generation_date, gene_csv_name_without_extension, f"full_dataset_patient_dir_numerical.pt")
    all_patient_dir_numerical = torch.load(patient_dir_numerical_pt_path)

    if retrieval_embedding_train_or_test == "train":
        train_df_path = os.path.join(working_codespace, "generated_files", generation_date, f"train_df.csv")
        valid_df_path = os.path.join(working_codespace, "generated_files", generation_date, f"valid_df.csv")
        test_df_path = os.path.join(working_codespace, "generated_files", generation_date, f"test_df.csv")
        train_df = pd.read_csv(train_df_path)
        valid_df = pd.read_csv(valid_df_path)
        test_df = pd.read_csv(test_df_path)

        if len(organ_selected) == 1:
            organ_selected = organ_selected[0]
            if organ_selected == "all":
                train_patient_dir_numerical_list = train_df["patient_dir_numerical"].tolist()
                valid_patient_dir_numerical_list = valid_df["patient_dir_numerical"].tolist()
                test_patient_dir_numerical_list = test_df["patient_dir_numerical"].tolist()
            else:
                train_patient_dir_numerical_list = train_df[train_df["organ_type"] == organ_selected]["patient_dir_numerical"].tolist()
                valid_patient_dir_numerical_list = valid_df[valid_df["organ_type"] == organ_selected]["patient_dir_numerical"].tolist()
                test_patient_dir_numerical_list = test_df[test_df["organ_type"] == organ_selected]["patient_dir_numerical"].tolist()
        
        elif len(organ_selected) > 1:            
            assert "all" not in organ_selected, f"Please select only one organ or select all organs."
            train_patient_dir_numerical_list = list()
            valid_patient_dir_numerical_list = list()
            test_patient_dir_numerical_list = list()
            for organ_selected in organ_selected:
                train_patient_dir_numerical_list += train_df[train_df["organ_type"] == organ_selected]["patient_dir_numerical"].tolist()
                valid_patient_dir_numerical_list += valid_df[valid_df["organ_type"] == organ_selected]["patient_dir_numerical"].tolist()
                test_patient_dir_numerical_list += test_df[test_df["organ_type"] == organ_selected]["patient_dir_numerical"].tolist()
        else:
            raise ValueError(f"Please select at least one organ.")
                
        train_mask = np.isin(all_patient_dir_numerical, train_patient_dir_numerical_list)
        valid_mask = np.isin(all_patient_dir_numerical, valid_patient_dir_numerical_list)
        test_mask = np.isin(all_patient_dir_numerical, test_patient_dir_numerical_list)

        patient_dir_organ_mapping_df_path = os.path.join(working_codespace, "generated_files", generation_date, "patient_dir_organ_mapping_df.csv")
        patient_dir_organ_mapping_df = pd.read_csv(patient_dir_organ_mapping_df_path)
        patient_dir_num_mapping_dict = dict(zip(patient_dir_organ_mapping_df["patient_dir"], patient_dir_organ_mapping_df.index))
        single_patient_dir_numerical = patient_dir_num_mapping_dict[patient_dir]
        single_patient_mask = np.isin(all_patient_dir_numerical, single_patient_dir_numerical)

        train_index_array = np.where(train_mask)[0]
        valid_index_array = np.where(valid_mask)[0]
        test_index_array = np.where(test_mask)[0]
        single_patient_index_array = np.where(single_patient_mask)[0]

        train_index_list = train_index_array.tolist()
        valid_index_list = valid_index_array.tolist()
        test_index_list = test_index_array.tolist()
        single_patient_index_list = single_patient_index_array.tolist()

        if data_source_train_or_test == "train":
            selected_index_list = train_index_list
        elif data_source_train_or_test == "valid":
            selected_index_list = valid_index_list
        elif data_source_train_or_test == "test":
            selected_index_list = test_index_list
        elif data_source_train_or_test == "train_valid":
            selected_index_list = train_index_list + valid_index_list
        elif data_source_train_or_test == "train_test":
            selected_index_list = train_index_list + test_index_list
        elif data_source_train_or_test == "valid_test":
            selected_index_list = valid_index_list + test_index_list
        elif data_source_train_or_test == "train_valid_test":
            selected_index_list = train_index_list + valid_index_list + test_index_list
        else:
            raise ValueError(f"The train_or_test should be either 'train' or 'valid' or 'test' or 'downstream, but now the value is {train_or_test}.")
        
        if exclude_patient_dir:
            selected_index_list = list(set(selected_index_list) - set(single_patient_index_list))
        
        assert len(selected_index_list) > 0, f"The length of the dataset is zero. Please check whether the organ selected is valid."
    
    elif retrieval_embedding_train_or_test == "test":

        patient_dir_organ_mapping_df_path = os.path.join(working_codespace, "generated_files", generation_date, "patient_dir_organ_mapping_df.csv")
        patient_dir_organ_mapping_df = pd.read_csv(patient_dir_organ_mapping_df_path)
        patient_dir_num_mapping_dict = dict(zip(patient_dir_organ_mapping_df["patient_dir"], patient_dir_organ_mapping_df.index))
        single_patient_dir_numerical = patient_dir_num_mapping_dict[patient_dir]
        single_patient_mask = np.isin(all_patient_dir_numerical, single_patient_dir_numerical)

        single_patient_index_array = np.where(single_patient_mask)[0]
        single_patient_index_list = single_patient_index_array.tolist()

        selected_index_list = single_patient_index_list
        assert len(selected_index_list) > 0, f"The length of the dataset is zero. Please check whether the organ selected is valid."

    else:
        raise ValueError(f"The retrieval_embedding_train_or_test should be either 'train' or 'test', but now the value is {retrieval_embedding_train_or_test}.")
    return selected_index_list