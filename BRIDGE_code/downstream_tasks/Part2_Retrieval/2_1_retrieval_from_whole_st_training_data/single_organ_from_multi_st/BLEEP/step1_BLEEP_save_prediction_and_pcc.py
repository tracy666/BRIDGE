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
        
def compute_avg_pcc_given_indices_for_retrieval(
    all_train_gene_ground_truth_expression: np.ndarray,
    all_test_gene_ground_truth_expression: np.ndarray,
    whole_distance: np.ndarray,
    whole_indices: np.ndarray,
    retrieval_size: int,
    selected_genes_list: List[str],
    eval_selected_genes_list: List[str],
    weighted: bool,
):
    all_train_gene_ground_truth_expression = all_train_gene_ground_truth_expression.numpy()
    all_test_gene_prediction_from_retrieval_list = list()
    assert all_test_gene_ground_truth_expression.shape[0] == whole_distance.shape[0] == whole_indices.shape[0]
    for index in tqdm(range(len(all_test_gene_ground_truth_expression)), total=len(all_test_gene_ground_truth_expression), desc=f"Retrieve gene prediction"):
        distance = whole_distance[index]
        indices = whole_indices[index]
        retrieval_train_gene_ground_truth_expressions = all_train_gene_ground_truth_expression[indices]
        if weighted:
            test_gene_prediction_from_retrieval = (scipy.special.softmax(1 - distance).reshape(-1, 1) * retrieval_train_gene_ground_truth_expressions).sum(axis=0)
        else:
            test_gene_prediction_from_retrieval = np.mean(retrieval_train_gene_ground_truth_expressions, axis=0)
        all_test_gene_prediction_from_retrieval_list.append(test_gene_prediction_from_retrieval.reshape(1, -1))

    all_test_gene_prediction_from_retrieval = np.concatenate(all_test_gene_prediction_from_retrieval_list, axis=0)

    gene_names_indices_mapping = {gene_name: index for index, gene_name in enumerate(selected_genes_list)}
    column_numbers_selected = [gene_names_indices_mapping[gene_name] for gene_name in selected_genes_list if gene_name in eval_selected_genes_list]
    sorted_column_numbers_selected = sorted(column_numbers_selected, key=lambda x: eval_selected_genes_list.index(selected_genes_list[x]))
    
    pcc_values_list = list()
    pcc_values_dict = dict()
    for column_number in sorted_column_numbers_selected:
        specific_gene_ground_truth = all_test_gene_ground_truth_expression[:, column_number]
        specific_gene_prediction = all_test_gene_prediction_from_retrieval[:, column_number]
        pcc_results = scipy.stats.pearsonr(specific_gene_ground_truth, specific_gene_prediction)
        pcc_statistic = pcc_results[0]
        pcc_pvalue = pcc_results[1]
        if np.isnan(pcc_statistic):
            pcc_statistic = 0
        pcc_values_list.append(pcc_statistic)
        pcc_values_dict[selected_genes_list[column_number]] = pcc_statistic
    pcc_mean = np.mean(pcc_values_list)
    return pcc_mean, pcc_values_dict, all_test_gene_prediction_from_retrieval

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
    parser.add_argument('--weighted_sum', action='store_true', help='Whether to use weighted sum for retrieval')
    args = parser.parse_args()

    if args.weighted_sum:
        weighted_method = "similarity_weight"
        weighted = True
    else:
        weighted_method = "mean"
        weighted = False
    print(weighted_method, weighted)

    downstream_tasks_saving_dir = current_dir.replace(
        args.working_codespace,
        args.main_data_storage,
    )
    retrieval_saving_dir = os.path.dirname(os.path.dirname(os.path.dirname(downstream_tasks_saving_dir)))

    logger_dir = os.path.join(current_dir, "logger")
    os.makedirs(logger_dir, exist_ok=True)

    logger_name = datetime.now().strftime('%Y%m%d%H%M%S') + f"_BLEEP_single_organ_for_{args.st_embedding_folder_name}_of_retrieval_size{args.retrieval_size}_and_weighted_method_{weighted_method}"
    # Set up logger
    logger = logging.getLogger(__name__) # Create a custom logger
    logger.setLevel(logging.INFO)
    # Create a file handler and set its level to INFO
    file_handler = logging.FileHandler(os.path.join(logger_dir, logger_name), 'w')
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout) # Create a stream handler and set its level to DEBUG
    stream_handler.setLevel(logging.DEBUG)
    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler) # Add the handlers to the logger
    logger.addHandler(stream_handler)
    logger = logging.getLogger(__name__)
    logger.info(f"Current evaluation single-organ folder is {args.st_embedding_folder_name}")
    logger.info(f"retrieval size: {args.retrieval_size}, weighted method: {weighted_method}")

    gene_csv_path = os.path.join(args.working_codespace, "data_raw_preprocessing", "selected_genes", args.gene_csv_name)
    selected_genes_list = pd.read_csv(gene_csv_path)["gene_names"].tolist()

    valid_df_path = os.path.join(args.working_codespace, "generated_files", args.generation_date, f"valid_df.csv")
    valid_df = pd.read_csv(valid_df_path)

    eval_selected_genes_list = list(np.load("/data1/zliang/Histo_ST_raw/Supplementary_data/human_biomarkers_80_genes.npy", allow_pickle=True))

    valid_df_path = os.path.join(args.working_codespace, "generated_files", args.generation_date, f"valid_df.csv")
    valid_df = pd.read_csv(valid_df_path)

    ground_truth_start_time = time.time()
    whole_dataset_ground_truth_path = os.path.join(
        args.working_codespace,
        "generated_files",
        args.generation_date,
        args.gene_csv_name.split(".")[0],
        "full_dataset_X_binned.h5",
    )
    # To read the data back as a PyTorch tensor:
    with h5py.File(whole_dataset_ground_truth_path, 'r') as hf:
        # Access the dataset in the HDF5 file
        dataset = hf['dataset_name']
        
        # Load the dataset as a NumPy array
        data_array = dataset[()]
        
        # Convert the NumPy array back to a PyTorch tensor
        whole_dataset_gene_ground_truth_expression = torch.from_numpy(data_array)
    
    ground_truth_end_time = time.time()
    print(f"Ground truth loading takes {ground_truth_end_time - ground_truth_start_time} seconds")

    single_organ_full_gene_slide_pcc_list = list()
    single_organ_eval_gene_slide_pcc_list = list()

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

        patient_dir_without_slash = (row.patient_dir).replace("/", "_")

        single_organ_pred_data_folder_saving_path = os.path.join(
            downstream_tasks_saving_dir,
            f"retrieval_size_{args.retrieval_size}",
            patient_dir_without_slash,
            args.st_embedding_folder_name,
            f"{row.organ_type}_at_epoch_{args.single_organ_epoch_num}",
        )
        single_organ_whole_distance_file_path = os.path.join(
            single_organ_pred_data_folder_saving_path,
            f"{patient_dir_without_slash}_whole_distance.npy",
        )
        single_organ_whole_indices_file_path = os.path.join(
            single_organ_pred_data_folder_saving_path,
            f"{patient_dir_without_slash}_whole_indices.npy",
        )
        assert os.path.exists(single_organ_whole_distance_file_path) and os.path.exists(single_organ_whole_indices_file_path)

        single_organ_whole_distance = np.load(single_organ_whole_distance_file_path, allow_pickle=True)
        single_organ_whole_indices = np.load(single_organ_whole_indices_file_path, allow_pickle=True)

        print(f"Start with single organ pcc calculation")
        single_organ_full_gene_pcc_start_time = time.time()
        single_organ_full_gene_slide_pcc, single_organ_full_gene_slide_genes_pcc_dict, single_organ_full_gene_all_test_gene_prediction_from_retrieval = compute_avg_pcc_given_indices_for_retrieval(
            all_train_gene_ground_truth_expression=whole_dataset_gene_ground_truth_expression[multi_organ_training_index_list],
            all_test_gene_ground_truth_expression=whole_dataset_gene_ground_truth_expression[testing_index_list],
            whole_distance=single_organ_whole_distance,
            whole_indices=single_organ_whole_indices,
            retrieval_size=args.retrieval_size,
            selected_genes_list=selected_genes_list,
            eval_selected_genes_list=selected_genes_list,
            weighted=weighted,
        )
        single_organ_full_gene_pcc_end_time = time.time()
        print(f"Finished calculating single organ full gene pcc, takes {single_organ_full_gene_pcc_end_time - single_organ_full_gene_pcc_start_time} seconds")

        single_organ_eval_gene_slide_pcc_start_time = time.time()
        single_organ_eval_gene_slide_pcc, single_organ_eval_gene_slide_genes_pcc_dict, single_organ_eval_gene_all_test_gene_prediction_from_retrieval = compute_avg_pcc_given_indices_for_retrieval(
            all_train_gene_ground_truth_expression=whole_dataset_gene_ground_truth_expression[multi_organ_training_index_list],
            all_test_gene_ground_truth_expression=whole_dataset_gene_ground_truth_expression[testing_index_list],
            whole_distance=single_organ_whole_distance,
            whole_indices=single_organ_whole_indices,
            retrieval_size=args.retrieval_size,
            selected_genes_list=selected_genes_list,
            eval_selected_genes_list=eval_selected_genes_list,
            weighted=weighted,
        )
        single_organ_eval_gene_slide_pcc_end_time = time.time()
        print(f"Finished calculating single organ eval gene pcc, takes {single_organ_eval_gene_slide_pcc_end_time - single_organ_eval_gene_slide_pcc_start_time} seconds")

        single_organ_full_gene_slide_pcc_list.append(single_organ_full_gene_slide_pcc)
        single_organ_eval_gene_slide_pcc_list.append(single_organ_eval_gene_slide_pcc)
        logger.info(f"For {row.patient_dir} with {row.organ_type}, single organ full gene performance: {single_organ_full_gene_slide_pcc}")
        logger.info(f"For {row.patient_dir} with {row.organ_type}, single organ eval gene performance: {single_organ_eval_gene_slide_pcc}")

        print(f"Start saving single organ data")
        single_organ_saving_start_time = time.time()
        single_organ_full_gene_slide_genes_pcc_dict = {k: v for k, v in sorted(single_organ_full_gene_slide_genes_pcc_dict.items(), key=lambda item: item[1], reverse = True)}

        single_organ_retrieval_saving_path = os.path.join(
            single_organ_pred_data_folder_saving_path,
            f"weighted_method_{weighted_method}",
        )
        os.makedirs(single_organ_retrieval_saving_path, exist_ok=True)
        
        single_organ_ground_truth_saving_path = os.path.join(single_organ_retrieval_saving_path, f"{patient_dir_without_slash}_ground_truth.npy")
        single_organ_prediction_saving_path = os.path.join(single_organ_retrieval_saving_path, f"{patient_dir_without_slash}_prediction.npy")
        single_organ_pcc_dict_saving_path = os.path.join(single_organ_retrieval_saving_path, f"{patient_dir_without_slash}_pcc_dict.npy")
        if not os.path.exists(single_organ_ground_truth_saving_path):
            np.save(single_organ_ground_truth_saving_path, whole_dataset_gene_ground_truth_expression[testing_index_list])
            print(f"{row.patient_dir} single organ ground truth saved")
        if not os.path.exists(single_organ_prediction_saving_path):
            np.save(single_organ_prediction_saving_path, single_organ_full_gene_all_test_gene_prediction_from_retrieval)
            print(f"{row.patient_dir} single organ prediction saved")
        if not os.path.exists(single_organ_pcc_dict_saving_path):
            with open(single_organ_pcc_dict_saving_path, "wb") as f:
                pickle.dump(single_organ_full_gene_slide_genes_pcc_dict, f)
            print(f"{row.patient_dir} single organ pcc dict saved")
        single_organ_saving_end_time = time.time()
        print(f"Finished saving single organ data, takes {single_organ_saving_end_time - single_organ_saving_start_time} seconds")
        print(f"single organ all finished!!")

    assert len(single_organ_eval_gene_slide_pcc_list) == len(single_organ_full_gene_slide_pcc_list)  == len(valid_df)
    single_organ_eval_gene_pcc_avg = sum(single_organ_eval_gene_slide_pcc_list)/len(single_organ_eval_gene_slide_pcc_list)
    single_organ_full_gene_pcc_avg = sum(single_organ_full_gene_slide_pcc_list)/len(single_organ_full_gene_slide_pcc_list)

    logger.info(f"Avg eval gene single organ pcc avg: {single_organ_eval_gene_pcc_avg}")
    logger.info(f"Avg full gene single organ pcc avg: {single_organ_full_gene_pcc_avg}")

if __name__ == "__main__":
    main()