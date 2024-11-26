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
# from image_gene_dataset import SingleSlideEvalImageGeneDataset
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
        
def get_direct_partial_image_gene_embedding_for_retrieval(
        whole_dataset_embedding_folder_path: str,
        selected_index_list: List[int],
):
    whole_dataset_image_embedding_path = os.path.join(whole_dataset_embedding_folder_path, "whole_st_dataset_image_embedding.pt")
    whole_dataset_gene_embedding_path = os.path.join(whole_dataset_embedding_folder_path, "whole_st_dataset_gene_embedding.pt")
    whole_dataset_image_embedding = torch.load(whole_dataset_image_embedding_path)
    whole_dataset_gene_embedding = torch.load(whole_dataset_gene_embedding_path)
    assert whole_dataset_image_embedding.size()[0] == whole_dataset_gene_embedding.size()[0], f"The number of rows in whole_dataset_image_embedding.npy and whole_dataset_gene_embedding.npy should be the same."

    selected_image_embedding = whole_dataset_image_embedding[selected_index_list, :]
    selected_gene_embedding = whole_dataset_gene_embedding[selected_index_list, :]

    assert selected_image_embedding.size()[0] == selected_gene_embedding.size()[0], f"The number of rows in selected_image_embedding.npy and selected_gene_embedding.npy should be the same."
    return selected_image_embedding, selected_gene_embedding

def get_retrieval_indexes(
    all_train_gene_embedding: torch.Tensor,
    all_test_image_embedding: torch.Tensor,
    retrieval_size: int,
) -> torch.Tensor:
    all_train_gene_embedding_array = all_train_gene_embedding.numpy()
    all_test_image_embedding_array = all_test_image_embedding.numpy()
    kdtree = KDTree(all_train_gene_embedding_array)

    whole_distance, whole_indices = kdtree.query(all_test_image_embedding_array, k=retrieval_size)
    return whole_distance, whole_indices # each is a numpy array of shape (spot size, retrieval size)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--main_data_storage', type=str, default="/disk1/zliang")
    parser.add_argument('--project_data_folder_name', type=str, default="BIG_600K")
    parser.add_argument('--working_codespace', type=str, default="/home/zliang/BRIDGE_BIG_600K")

    parser.add_argument("--generation_date", type=str, default="20240601")
    parser.add_argument('--gene_csv_name', type=str, default="all_intersection_genes_number_7730.csv")
    parser.add_argument('--eval_set', type=str, default="valid", choices=["all", "valid", "test"])

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--retrieval_size', type=int, default=256) # 1024 # 2048

    parser.add_argument('--st_embedding_folder_name', type=str)
    parser.add_argument('--single_organ_epoch_num', type=int, default=20)
    args = parser.parse_args()

    downstream_tasks_saving_dir = current_dir.replace(
        args.working_codespace,
        args.main_data_storage,
    )
    retrieval_saving_dir = os.path.dirname(os.path.dirname(os.path.dirname(downstream_tasks_saving_dir)))

    valid_df_path = os.path.join(args.working_codespace, "generated_files", args.generation_date, f"valid_df.csv")
    valid_df = pd.read_csv(valid_df_path)

    for row in tqdm(valid_df.itertuples(), total=len(valid_df)):
        print(f"Processing {row.patient_dir} with {row.organ_type}. Start with training indexes")

        training_index_start_time = time.time()
        multi_organ_training_index_list = get_index_list(
            retrieval_embedding_train_or_test="train",
            main_data_storage=args.main_data_storage,
            project_data_folder_name=args.project_data_folder_name,
            working_codespace=args.working_codespace,
            gene_csv_name=args.gene_csv_name,
            generation_date=args.generation_date,
            data_source_train_or_test="train",
            scGPT_gene_mask_ratio=0.25,
            organ_selected=["all"],
            patient_dir=row.patient_dir,
            exclude_patient_dir=True,
        )
        training_index_end_time = time.time()
        print(f"Training index list takes {training_index_end_time - training_index_start_time} seconds")

        print(f"Start with testing indexes")
        testing_index_start_time = time.time()
        testing_index_list = get_index_list(
            retrieval_embedding_train_or_test="test",
            main_data_storage=args.main_data_storage,
            project_data_folder_name=args.project_data_folder_name,
            working_codespace=args.working_codespace,
            gene_csv_name=args.gene_csv_name,
            generation_date=args.generation_date,
            data_source_train_or_test="train",
            scGPT_gene_mask_ratio=0.25,
            organ_selected=["all"],
            patient_dir=row.patient_dir,
            exclude_patient_dir=True,
        )
        testing_index_end_time = time.time()
        print(f"Testing index list takes {testing_index_end_time - testing_index_start_time} seconds")

        single_organ_whole_dataset_embedding_folder_path = os.path.join(
            retrieval_saving_dir,
            "2_0_save_whole_st_training_data_embedding",
            "BRIDGE",
            args.st_embedding_folder_name,
            f"{row.organ_type}_at_epoch_{args.single_organ_epoch_num}",          
        )
        assert os.path.exists(single_organ_whole_dataset_embedding_folder_path)

        single_organ_training_image_embedding, single_organ_training_gene_embedding = get_direct_partial_image_gene_embedding_for_retrieval(
            whole_dataset_embedding_folder_path=single_organ_whole_dataset_embedding_folder_path,
            selected_index_list=multi_organ_training_index_list,
        )
        single_organ_testing_image_embedding, single_organ_testing_gene_embedding = get_direct_partial_image_gene_embedding_for_retrieval(
            whole_dataset_embedding_folder_path=single_organ_whole_dataset_embedding_folder_path,
            selected_index_list=testing_index_list,
        )

        retrieval_indexes_start_time = time.time()
        single_organ_whole_distance, single_organ_whole_indices = get_retrieval_indexes(
            all_train_gene_embedding=single_organ_training_gene_embedding,
            all_test_image_embedding=single_organ_testing_image_embedding,
            retrieval_size=args.retrieval_size,
        )
        retrieval_indexes_end_time = time.time()
        print(f"Get retrieval indexes takes {retrieval_indexes_end_time - retrieval_indexes_start_time} seconds")

        print(f"Start saving single organ data")
        single_organ_saving_start_time = time.time()
        patient_dir_without_slash = (row.patient_dir).replace("/", "_")
        single_organ_pred_data_folder_saving_path = os.path.join(
            downstream_tasks_saving_dir,
            f"retrieval_size_{args.retrieval_size}",
            patient_dir_without_slash,
            args.st_embedding_folder_name,
            f"{row.organ_type}_at_epoch_{args.single_organ_epoch_num}",
        )
        os.makedirs(single_organ_pred_data_folder_saving_path, exist_ok=True)
        single_organ_retrieval_distance_saving_path = os.path.join(single_organ_pred_data_folder_saving_path, f"{patient_dir_without_slash}_whole_distance.npy")
        single_organ_retrieval_indices_saving_path = os.path.join(single_organ_pred_data_folder_saving_path, f"{patient_dir_without_slash}_whole_indices.npy")
        if not os.path.exists(single_organ_retrieval_distance_saving_path):
            np.save(single_organ_retrieval_distance_saving_path, single_organ_whole_distance)
            print(f"{row.patient_dir} single organ whole distance saved")
        if not os.path.exists(single_organ_retrieval_indices_saving_path):
            np.save(single_organ_retrieval_indices_saving_path, single_organ_whole_indices)
            print(f"{row.patient_dir} single organ whole indices saved")
        single_organ_saving_end_time = time.time()
        print(f"Finished saving single organ data, takes {single_organ_saving_end_time - single_organ_saving_start_time} seconds")
        print(f"single organ all finished!!")

if __name__ == "__main__":
    main()