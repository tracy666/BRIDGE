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
    parser.add_argument('--main_data_storage', type=str, default="/data1/zliang")
    parser.add_argument('--raw_data_folder_name', type=str, default="Histo_ST_raw")
    parser.add_argument('--project_data_folder_name', type=str, default="BIG_600K")
    parser.add_argument('--working_codespace', type=str, default="/home/zliang/BRIDGE/BRIDGE_code")

    parser.add_argument("--generation_date", type=str, default="20240601")
    parser.add_argument('--gene_csv_name', type=str, default="all_intersection_genes_number_7730.csv")
    parser.add_argument('--num_of_eval_genes', type=int, default=1000)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--retrieval_size', type=int, default=256) # 1024 # 2048

    parser.add_argument('--epoch_number', type=int, default=20)

    parser.add_argument('--gpu_cards', type=str, default='', help='Comma-separated list of GPU card numbers')
    parser.add_argument('--model_settings', type=str, default='multi_organ_setting')
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
    
    downstream_tasks_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(retrieval_tasks_saving_dir))))

    logger_dir = os.path.join(current_dir, "logger")
    os.makedirs(logger_dir, exist_ok=True)
    
    logger_name = datetime.now().strftime('%Y%m%d%H%M%S') + f"_BLEEP_retrieval_from_SC_dataset_for_{args.model_settings}"
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

    test_df_path = os.path.join(args.working_codespace, "generated_files", args.generation_date, f"test_df.csv")
    test_df = pd.read_csv(test_df_path)

    full_genes_csv_path = os.path.join(args.working_codespace, "data_raw_preprocessing", "selected_genes", args.gene_csv_name)
    full_genes_list = pd.read_csv(full_genes_csv_path)["gene_names"].values.tolist()
    
    selected_genes_HEG_csv_path = os.path.join(args.working_codespace, "data_raw_preprocessing", "selected_genes", f"top_HEG_genes_number_{args.num_of_eval_genes}.csv")
    selected_genes_HEG_list = pd.read_csv(selected_genes_HEG_csv_path)["gene_names"].values.tolist()
    selected_genes_HVG_csv_path = os.path.join(args.working_codespace, "data_raw_preprocessing", "selected_genes", f"top_HVG_genes_number_{args.num_of_eval_genes}.csv")
    selected_genes_HVG_list = pd.read_csv(selected_genes_HVG_csv_path)["gene_names"].values.tolist()
    selected_genes_list = selected_genes_HEG_list + selected_genes_HVG_list
    selected_genes_list = list(set(selected_genes_list))
    
    biomarker_genes_csv_path = os.path.join(args.working_codespace, "data_raw_preprocessing", "selected_genes", f"selected_biomarkers_number_80.csv")
    biomarker_genes_list = pd.read_csv(biomarker_genes_csv_path)["gene_names"].values.tolist()
    logger.info(f"Biomarker genes setting is of {len(biomarker_genes_list)} involved.")

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

    full_gene_slide_pcc_list = list()
    selected_gene_slide_pcc_list = list()
    biomarker_gene_slide_pcc_list = list()

    for row in tqdm(test_df.itertuples(), total=len(test_df)):
        patient_dir_without_slash = (row.patient_dir).replace("/", "_")
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

        distance_file_path = os.path.join(
            retrieval_tasks_saving_dir,
            patient_dir_without_slash,
            f"distance_and_indices",
            f"{patient_dir_without_slash}_whole_distance.npy"
        )
        indices_file_path = os.path.join(
            retrieval_tasks_saving_dir,
            patient_dir_without_slash,
            f"distance_and_indices",
            f"{patient_dir_without_slash}_whole_indices.npy"
        )
        assert os.path.exists(distance_file_path) and os.path.exists(indices_file_path)

        whole_distance = np.load(distance_file_path, allow_pickle=True)
        whole_indices = np.load(indices_file_path, allow_pickle=True)
        
        BLEEP_ST_reference_dataset_embedding_folder_path = os.path.join(
            downstream_tasks_folder,
            f"Part2_Retrieval",
            f"2_0_save_st_dataset_embedding",
            f"BLEEP",
            f"{row.organ_type}_at_epoch_{args.epoch_number}",
        )
        assert os.path.exists(BLEEP_ST_reference_dataset_embedding_folder_path)
        
        BLEEP_SC_reference_dataset_embedding_folder_path = os.path.join(
            downstream_tasks_folder,
            f"Part2_Retrieval",
            f"2_0_save_single_cell_dataset_embedding",
            f"BLEEP",
            f"{row.organ_type}_at_epoch_{args.epoch_number}",
        )
        assert os.path.exists(BLEEP_SC_reference_dataset_embedding_folder_path)
        
        BLEEP_training_single_cell_gene_embedding, BLEEP_training_single_cell_gene_ground_truth_expression = get_single_cell_gene_embedding_of_specific_organ_type_for_retrieval(            
            sc_dataset_embedding_folder_path=BLEEP_SC_reference_dataset_embedding_folder_path,
            main_data_storage=args.main_data_storage,
            raw_data_folder_name=args.raw_data_folder_name,
            organ_type=row.organ_type,
        )

        full_gene_slide_pcc, full_gene_slide_genes_pcc_dict, full_gene_all_test_gene_prediction_from_retrieval = compute_avg_pcc_given_indices_for_retrieval(
            # all_train_gene_ground_truth_expression=whole_dataset_gene_ground_truth_expression[multi_organ_training_index_list],
            all_train_gene_ground_truth_expression=BLEEP_training_single_cell_gene_ground_truth_expression,
            all_test_gene_ground_truth_expression=whole_dataset_gene_ground_truth_expression[testing_index_list],
            whole_distance=whole_distance,
            whole_indices=whole_indices,
            retrieval_size=args.retrieval_size,
            selected_genes_list=full_genes_list,
            eval_selected_genes_list=full_genes_list,
            weighted=True,
        )

        selected_gene_slide_pcc, selected_gene_slide_genes_pcc_dict, selected_gene_all_test_gene_prediction_from_retrieval = compute_avg_pcc_given_indices_for_retrieval(
            # all_train_gene_ground_truth_expression=whole_dataset_gene_ground_truth_expression[multi_organ_training_index_list],
            all_train_gene_ground_truth_expression=BLEEP_training_single_cell_gene_ground_truth_expression,
            all_test_gene_ground_truth_expression=whole_dataset_gene_ground_truth_expression[testing_index_list],
            whole_distance=whole_distance,
            whole_indices=whole_indices,
            retrieval_size=args.retrieval_size,
            selected_genes_list=full_genes_list,
            eval_selected_genes_list=selected_genes_list,
            weighted=True,
        )

        biomarker_gene_slide_pcc, biomarker_gene_slide_genes_pcc_dict, biomarker_gene_all_test_gene_prediction_from_retrieval = compute_avg_pcc_given_indices_for_retrieval(
            # all_train_gene_ground_truth_expression=whole_dataset_gene_ground_truth_expression[multi_organ_training_index_list],
            all_train_gene_ground_truth_expression=BLEEP_training_single_cell_gene_ground_truth_expression,
            all_test_gene_ground_truth_expression=whole_dataset_gene_ground_truth_expression[testing_index_list],
            whole_distance=whole_distance,
            whole_indices=whole_indices,
            retrieval_size=args.retrieval_size,
            selected_genes_list=full_genes_list,
            eval_selected_genes_list=biomarker_genes_list,
            weighted=True,
        )

        full_gene_slide_pcc_list.append(full_gene_slide_pcc)
        selected_gene_slide_pcc_list.append(selected_gene_slide_pcc)
        biomarker_gene_slide_pcc_list.append(biomarker_gene_slide_pcc)
        logger.info(f"For {row.patient_dir} with {row.organ_type}, BLEEP biomarker gene performance: {biomarker_gene_slide_pcc}")

        full_gene_slide_genes_pcc_dict = {k: v for k, v in sorted(full_gene_slide_genes_pcc_dict.items(), key=lambda item: item[1], reverse = True)}

        result_saving_path = os.path.join(
            retrieval_tasks_saving_dir,
            patient_dir_without_slash,
            f"prediction_and_pcc",
        )
        os.makedirs(result_saving_path, exist_ok=True)

        ground_truth_saving_path = os.path.join(result_saving_path, f"{patient_dir_without_slash}_ground_truth.npy")
        prediction_saving_path = os.path.join(result_saving_path, f"{patient_dir_without_slash}_prediction.npy")
        pcc_dict_saving_path = os.path.join(result_saving_path, f"{patient_dir_without_slash}_pcc_dict.npy")

        if not os.path.exists(ground_truth_saving_path):
            np.save(ground_truth_saving_path, whole_dataset_gene_ground_truth_expression[testing_index_list])
            print(f"{row.patient_dir}: BLEEP retrieval ground truth saved to {ground_truth_saving_path}")

        if not os.path.exists(prediction_saving_path):
            np.save(prediction_saving_path, full_gene_all_test_gene_prediction_from_retrieval)
            print(f"{row.patient_dir}: BLEEP retrieval prediction saved to {prediction_saving_path}")

        if not os.path.exists(pcc_dict_saving_path):
            with open(pcc_dict_saving_path, "wb") as f:
                pickle.dump(full_gene_slide_genes_pcc_dict, f)
            print(f"{row.patient_dir}: BLEEP retrieval pcc dict saved to {pcc_dict_saving_path}")

    biomarker_gene_pcc_avg = sum(biomarker_gene_slide_pcc_list)/len(biomarker_gene_slide_pcc_list)
    logger.info(f"For BLEEP under {args.model_settings}, Average biomarker gene performance: {biomarker_gene_pcc_avg}")



if __name__ == "__main__":
    main()