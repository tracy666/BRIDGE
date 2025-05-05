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
current_workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))))
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
        test_df_path = os.path.join(working_codespace, "generated_files", generation_date, f"test_df.csv")
        
        train_df = pd.read_csv(train_df_path)
        test_df = pd.read_csv(test_df_path)

        if len(organ_selected) == 1:
            organ_selected = organ_selected[0]
            if organ_selected == "all":
                train_patient_dir_numerical_list = train_df["patient_dir_numerical"].tolist()
                test_patient_dir_numerical_list = test_df["patient_dir_numerical"].tolist()
            else:
                train_patient_dir_numerical_list = train_df[train_df["organ_type"] == organ_selected]["patient_dir_numerical"].tolist()
                test_patient_dir_numerical_list = test_df[test_df["organ_type"] == organ_selected]["patient_dir_numerical"].tolist()
        
        elif len(organ_selected) > 1:            
            assert "all" not in organ_selected, f"Please select only one organ or select all organs."
            train_patient_dir_numerical_list = list()
            test_patient_dir_numerical_list = list()
            for organ_selected in organ_selected:
                train_patient_dir_numerical_list += train_df[train_df["organ_type"] == organ_selected]["patient_dir_numerical"].tolist()
                test_patient_dir_numerical_list += test_df[test_df["organ_type"] == organ_selected]["patient_dir_numerical"].tolist()
        else:
            raise ValueError(f"Please select at least one organ.")
                
        train_mask = np.isin(all_patient_dir_numerical, train_patient_dir_numerical_list)
        test_mask = np.isin(all_patient_dir_numerical, test_patient_dir_numerical_list)

        patient_dir_organ_mapping_df_path = os.path.join(working_codespace, "generated_files", generation_date, "patient_dir_organ_mapping_df.csv")
        patient_dir_organ_mapping_df = pd.read_csv(patient_dir_organ_mapping_df_path)
        patient_dir_num_mapping_dict = dict(zip(patient_dir_organ_mapping_df["patient_dir"], patient_dir_organ_mapping_df.index))
        single_patient_dir_numerical = patient_dir_num_mapping_dict[patient_dir]
        single_patient_mask = np.isin(all_patient_dir_numerical, single_patient_dir_numerical)

        train_index_array = np.where(train_mask)[0]
        test_index_array = np.where(test_mask)[0]
        single_patient_index_array = np.where(single_patient_mask)[0]

        train_index_list = train_index_array.tolist()
        test_index_list = test_index_array.tolist()
        single_patient_index_list = single_patient_index_array.tolist()

        if data_source_train_or_test == "train":
            selected_index_list = train_index_list
        elif data_source_train_or_test == "test":
            selected_index_list = test_index_list
        elif data_source_train_or_test == "train_test":
            selected_index_list = train_index_list + test_index_list
        else:
            raise ValueError(f"The train_or_test should be either 'train' or 'valid' or 'test' or 'downstream, but now the value is {data_source_train_or_test}.")
        
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
        organ_type: str,
        selected_index_list: List[int],
):
    whole_dataset_image_embedding_path = os.path.join(whole_dataset_embedding_folder_path, f"{organ_type}_whole_st_dataset_image_embedding.pt")
    whole_dataset_gene_embedding_path = os.path.join(whole_dataset_embedding_folder_path, f"{organ_type}_whole_st_dataset_gene_embedding.pt")
    whole_dataset_image_embedding = torch.load(whole_dataset_image_embedding_path)
    whole_dataset_gene_embedding = torch.load(whole_dataset_gene_embedding_path)
    assert whole_dataset_image_embedding.size()[0] == whole_dataset_gene_embedding.size()[0], f"The number of rows in whole_dataset_image_embedding.npy and whole_dataset_gene_embedding.npy should be the same."

    selected_image_embedding = whole_dataset_image_embedding[selected_index_list, :]
    selected_gene_embedding = whole_dataset_gene_embedding[selected_index_list, :]

    assert selected_image_embedding.size()[0] == selected_gene_embedding.size()[0], f"The number of rows in selected_image_embedding.npy and selected_gene_embedding.npy should be the same."
    return selected_image_embedding, selected_gene_embedding

def get_single_cell_gene_embedding_of_specific_organ_type_for_retrieval(
        sc_dataset_embedding_folder_path: str,
        main_data_storage: str,
        raw_data_folder_name: str,
        organ_type: str,
):
    whole_single_cell_dataset_gene_embedding_path = os.path.join(sc_dataset_embedding_folder_path, f"{organ_type}_whole_single_cell_dataset_gene_embedding.pt")
    whole_single_cell_dataset_gene_embedding = torch.load(whole_single_cell_dataset_gene_embedding_path)
    
    whole_single_cell_dataset_folder_path = os.path.join(
        main_data_storage,
        raw_data_folder_name,
        "Cellxgene_single_cell_datasets",
        organ_type,
        "Single_cell_dataset_scGPT_preprocessed",
    )
    whole_single_cell_dataset_gene_ground_truth_expression_path = os.path.join(whole_single_cell_dataset_folder_path, "full_single_cell_dataset_X_binned.pt")
    whole_single_cell_dataset_gene_ground_truth_expression_with_selected_genes_path = os.path.join(whole_single_cell_dataset_folder_path, "full_single_cell_dataset_X_binned_with_selected_genes.pt")
    if os.path.exists(whole_single_cell_dataset_gene_ground_truth_expression_with_selected_genes_path):
        whole_single_cell_dataset_gene_ground_truth_expression = torch.load(whole_single_cell_dataset_gene_ground_truth_expression_with_selected_genes_path)
    else:
        whole_single_cell_dataset_gene_ground_truth_expression = torch.load(whole_single_cell_dataset_gene_ground_truth_expression_path)
    return whole_single_cell_dataset_gene_embedding, whole_single_cell_dataset_gene_ground_truth_expression

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
    parser.add_argument('--main_data_storage', type=str, default="/data1/zliang")
    parser.add_argument('--raw_data_folder_name', type=str, default="Histo_ST_raw")
    parser.add_argument('--project_data_folder_name', type=str, default="BIG_600K")
    parser.add_argument('--working_codespace', type=str, default="/home/zliang/BRIDGE/BRIDGE_code")

    parser.add_argument("--generation_date", type=str, default="20240601")
    parser.add_argument('--gene_csv_name', type=str, default="all_intersection_genes_number_7730.csv")

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--retrieval_size', type=int, default=256) # 1024 # 2048

    parser.add_argument('--epoch_number', type=int, default=50)

    parser.add_argument('--gpu_cards', type=str, default='', help='Comma-separated list of GPU card numbers')
    # parser.add_argument('--model_settings', type=str, default='multi_organ_setting')
    args = parser.parse_args()

    gpu_cards = args.gpu_cards.split(",") if args.gpu_cards else []
    gpu_cards_str = ",".join(gpu_cards)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_cards_str
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    retrieval_tasks_saving_dir = current_dir.replace(
        args.working_codespace,
        args.main_data_storage,
    )
    os.makedirs(retrieval_tasks_saving_dir, exist_ok=True)

    downstream_tasks_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(retrieval_tasks_saving_dir)))))

    test_df_path = os.path.join(args.working_codespace, "generated_files", args.generation_date, f"test_df.csv")
    test_df = pd.read_csv(test_df_path)

    for row in tqdm(test_df.itertuples(), total=len(test_df)):
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

        multi_organ_BRIDGE_ST_reference_dataset_embedding_folder_path = os.path.join(
            downstream_tasks_folder,
            f"Part2_Retrieval",
            f"2_0_save_st_dataset_embedding",
            f"BRIDGE",
            f"multi_organ",
            f"multi_organ_at_epoch_{args.epoch_number}",
        )
        assert os.path.exists(multi_organ_BRIDGE_ST_reference_dataset_embedding_folder_path)
        
        multi_organ_BRIDGE_SC_reference_dataset_embedding_folder_path = os.path.join(
            downstream_tasks_folder,
            f"Part2_Retrieval",
            f"2_0_save_single_cell_dataset_embedding",
            f"BRIDGE",
            f"multi_organ",
            f"{row.organ_type}_at_epoch_{args.epoch_number}",
        )
        assert os.path.exists(multi_organ_BRIDGE_SC_reference_dataset_embedding_folder_path)
        
        multi_organ_BRIDGE_training_single_cell_gene_embedding, multi_organ_BRIDGE_training_single_cell_gene_ground_truth_expression = get_single_cell_gene_embedding_of_specific_organ_type_for_retrieval(            
            sc_dataset_embedding_folder_path=multi_organ_BRIDGE_SC_reference_dataset_embedding_folder_path,
            main_data_storage=args.main_data_storage,
            raw_data_folder_name=args.raw_data_folder_name,
            organ_type=row.organ_type,
        )

        # multi_organ_BRIDGE_training_image_embedding, multi_organ_BRIDGE_training_gene_embedding = get_direct_partial_image_gene_embedding_for_retrieval(
        #     whole_dataset_embedding_folder_path=multi_organ_BRIDGE_ST_reference_dataset_embedding_folder_path,
        #     organ_type="multi_organ",
        #     selected_index_list=multi_organ_training_index_list,
        # )
        multi_organ_BRIDGE_testing_image_embedding, multi_organ_BRIDGE_testing_gene_embedding = get_direct_partial_image_gene_embedding_for_retrieval(
            whole_dataset_embedding_folder_path=multi_organ_BRIDGE_ST_reference_dataset_embedding_folder_path,
            organ_type="multi_organ",
            selected_index_list=testing_index_list,
        )

        retrieval_indices_start_time = time.time()
        multi_organ_BRIDGE_whole_distance, multi_organ_BRIDGE_whole_indices = get_retrieval_indexes(
            all_train_gene_embedding=multi_organ_BRIDGE_training_single_cell_gene_embedding,
            all_test_image_embedding=multi_organ_BRIDGE_testing_image_embedding,
            retrieval_size=args.retrieval_size,
        )
        retrieval_indices_end_time = time.time()
        print(f"Get retrieval indices takes {retrieval_indices_end_time - retrieval_indices_start_time} seconds")

        print(f"Start saving multi-organ BRIDGE distance and indices")
        patient_dir_without_slash = (row.patient_dir).replace("/", "_")
        result_saving_path = os.path.join(
            retrieval_tasks_saving_dir,
            patient_dir_without_slash,
            f"distance_and_indices",
        )
        os.makedirs(result_saving_path, exist_ok=True)

        distance_saving_path = os.path.join(result_saving_path, f"{patient_dir_without_slash}_whole_distance.npy")
        if not os.path.exists(distance_saving_path):
            np.save(distance_saving_path, multi_organ_BRIDGE_whole_distance)
            print(f"{row.patient_dir}: multi-organ BRIDGE retrieval distance saved to {distance_saving_path}")

        indices_saving_path = os.path.join(result_saving_path, f"{patient_dir_without_slash}_whole_indices.npy")
        if not os.path.exists(indices_saving_path):
            np.save(indices_saving_path, multi_organ_BRIDGE_whole_indices)
            print(f"{row.patient_dir}: multi-organ BRIDGE retrieval indices saved to {indices_saving_path}")



if __name__ == "__main__":
    main()