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
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import time
from tqdm import tqdm
import scanpy as sc
from multiprocessing import Pool
from functools import partial

from step0_preprocess_helper_functions import running_time_display, find_all_stdata_h5ad
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='anndata')
warnings.filterwarnings(action="ignore", category=ResourceWarning) # ResourceWarning: Implicitly cleaning up <TemporaryDirectory>

def generate_single_slide_further_preprocessed_stdata(
        h5ad_path: str,
        main_data_storage: str,
        project_data_folder_name: str,
        selected_genes_list: List[str],
        patient_dir_mapping_dict: dict,
        patient_dir_organ_num_dict: dict,
):
        generation_start_time = time.time()
        stdata = anndata.read_h5ad(h5ad_path)
        stdata.obs_names_make_unique()
        stdata.var_names_make_unique()
        # step 1. filter rows such that patches does not exist (due to inaccurate coordinates)
        existing_patch_mask = [os.path.exists(os.path.join(main_data_storage, project_data_folder_name, single_patch_path)) for single_patch_path in stdata.obs["patch_path"].values.tolist()]
        stdata_with_patches = stdata[existing_patch_mask]
        # step 2. select columns based on the selected_genes_list (added: also follow the same order of the given list)
        gene_names_list_per_slide = stdata_with_patches.var.index.tolist()
        gene_names_indices_mapping = {gene_name: index for index, gene_name in enumerate(gene_names_list_per_slide)}
        column_numbers_selected = [gene_names_indices_mapping[gene_name] for gene_name in gene_names_list_per_slide if gene_name in selected_genes_list]
        sorted_column_numbers_selected = sorted(column_numbers_selected, key=lambda x: selected_genes_list.index(gene_names_list_per_slide[x]))
        stdata_with_selected_genes = stdata_with_patches[:, sorted_column_numbers_selected]
        assert stdata_with_selected_genes.var.index.tolist() == selected_genes_list, f"There is some missing genes or the order does not match."
        assert stdata_with_selected_genes.shape[1] == len(selected_genes_list), f"Missing genes or unmatched variable number in the stdata {h5ad_path}."
        # step 3. create a new column in obs named patient_dir_numerical with numerical values of patient_dir
        stdata_copy = stdata_with_selected_genes.copy()
        stdata_copy.obs["patient_dir_numerical"] = stdata_with_selected_genes.obs["patient_dir"].map(patient_dir_mapping_dict)
        stdata_copy.obs["organ_type_numerical"] = stdata_with_selected_genes.obs["patient_dir"].map(patient_dir_organ_num_dict)
        generation_end_time = time.time()
        print(f"The generation time for {h5ad_path} is: {running_time_display(generation_end_time - generation_start_time)}.")
        return stdata_copy

def save_single_row_data_to_pt_file(
          index: int,
          dataset_inputs_folder_path: str,
          input_big_tensor: torch.tensor,
          input_big_tensor_name: str,
) -> None:
    input_big_tensor_name_without_extension = os.path.splitext(input_big_tensor_name)[0]
    output_folder_path = os.path.join(dataset_inputs_folder_path, input_big_tensor_name_without_extension)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    single_row_of_tensor = input_big_tensor[index, :].clone().detach()
    single_row_of_tensor_saving_path = os.path.join(output_folder_path, f"index_{index}.pt")
    if not os.path.exists(single_row_of_tensor_saving_path):
        torch.save(single_row_of_tensor, single_row_of_tensor_saving_path)
    print(f"Finished saving the {index}th row of the {input_big_tensor_name} to the individual pt files.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_data_storage", type=str, default="/data1/zliang")
    parser.add_argument("--raw_data_folder_name", type=str, default="Histo_ST_raw")
    parser.add_argument("--project_data_folder_name", type=str, default="BIG_600K")
    parser.add_argument("--working_codespace", type=str, default="/home/zliang/BRIDGE/BRIDGE_code")
    parser.add_argument("--multiprocessing_pool_num", type=int, default=5)
    parser.add_argument("--gene_csv_name", type=str, default="all_intersection_genes_number_7730.csv")
    parser.add_argument("--generation_date", type=str, default="20240601")
    args = parser.parse_args()

    '''
    Files need to be saved:
    1. patient_dir_numerical.pt ☑
    2. X_binned.pt ☑
    3. tokenized_gene_ids.pt ☑
    4. tokenized_gene_values.pt ☑
    5. masked_tokenized_gene_values.pt (* with different ratio, 0.25 & 0.75) ☑
    6. patch_patch.npy ☑
    7. normed_patch_path.npy ☑
    '''

    train_and_test_stdata_list = find_all_stdata_h5ad(
        main_data_storage=args.main_data_storage,
        project_data_folder_name=args.project_data_folder_name,
        stdata_file_names=["stdata.h5ad"],
        include_downstream_task_data=True,
    )
    print(f"The number of stdata files found is: {len(train_and_test_stdata_list)}.")

    selected_genes_csv_path = os.path.join(args.working_codespace, "data_raw_preprocessing", "selected_genes", args.gene_csv_name)
    selected_genes_list = pd.read_csv(selected_genes_csv_path)["gene_names"].tolist()
    gene_csv_name_without_extension = os.path.splitext(args.gene_csv_name)[0]

    patient_dir_mapping_csv_path = os.path.join(args.working_codespace, "generated_files", args.generation_date, "patient_dir_organ_mapping_df.csv")
    patient_dir_mapping_df = pd.read_csv(patient_dir_mapping_csv_path)
    patient_dir_mapping_dict = dict(zip(patient_dir_mapping_df["patient_dir"], patient_dir_mapping_df.index))

    patient_dir_organ_name_mapping_dict = dict(zip(patient_dir_mapping_df["patient_dir"], patient_dir_mapping_df["organ_type"]))
    organ_name_num_mapping_dict = {
        "brain": 0,
        "breast": 1,
        "cervix": 2,
        "heart": 3,
        "kidney": 4,
        "liver": 5,
        "lung": 6,
        "lymph_node": 7,
        "nose": 8,
        "ovary": 9,
        "prostate": 10,
        "skin": 11,
        "small_and_large_intestine": 12,
    }
    patient_dir_organ_num_dict = dict()
    for patient_dir, organ_name in patient_dir_organ_name_mapping_dict.items():
        patient_dir_organ_num_dict[patient_dir] = organ_name_num_mapping_dict[organ_name]

    dataset_inputs_folder_path = os.path.join(args.working_codespace, "generated_files", args.generation_date, gene_csv_name_without_extension)
    if not os.path.exists(dataset_inputs_folder_path):
        os.makedirs(dataset_inputs_folder_path)

    ############################################################
    # Part 1. full_dataset_stdata.h5ad
    
    read_single_stdata_start_time = time.time()
    total_stdata_list = list()
    pool = Pool(processes=args.multiprocessing_pool_num)
    with tqdm(total=len(train_and_test_stdata_list)) as pbar:
        for stdata in pool.imap(partial(generate_single_slide_further_preprocessed_stdata, main_data_storage=args.main_data_storage, project_data_folder_name=args.project_data_folder_name, selected_genes_list=selected_genes_list, patient_dir_mapping_dict=patient_dir_mapping_dict, patient_dir_organ_num_dict=patient_dir_organ_num_dict), train_and_test_stdata_list):
            total_stdata_list.append(stdata)
            pbar.update()
    pool.close()
    pool.join()
    read_single_stdata_end_time = time.time()
    print(f"The reading of single stdata is finished. Time Used: {running_time_display(read_single_stdata_end_time - read_single_stdata_start_time)}.")

    concatenate_all_single_stdata_start_time = time.time()
    full_dataset_big_stdata = anndata.concat(total_stdata_list)
    concatenate_all_single_stdata_end_time = time.time()
    print(f"The concatenation of all single stdata is finished. Time Used: {running_time_display(concatenate_all_single_stdata_end_time - concatenate_all_single_stdata_start_time)}.")

    write_big_h5ad_start_time = time.time()
    full_dataset_big_stdata_str_copy = full_dataset_big_stdata.copy()
    full_dataset_big_stdata_str_copy.obs = full_dataset_big_stdata_str_copy.obs.astype(str)
    full_dataset_big_stdata_str_copy.var = full_dataset_big_stdata_str_copy.var.astype(str)
    if not os.path.exists(os.path.join(dataset_inputs_folder_path, "full_dataset_stdata.h5ad")):
        full_dataset_big_stdata_str_copy.write_h5ad(os.path.join(dataset_inputs_folder_path, "full_dataset_stdata.h5ad"))
    write_big_h5ad_end_time = time.time()
    print(f"The writing of the full dataset stdata is finished. Time Used: {running_time_display(write_big_h5ad_end_time - write_big_h5ad_start_time)}.")
    ############################################################

    ############################################################
    # Part 2. X_binned.pt & patient_dir_numerical.pt & organ_type_numerical.pt & patch_patch.npy & normed_patch_path.npy & full_dataset_spot_barcode.npy

    full_dataset_stdata = anndata.read_h5ad(os.path.join(dataset_inputs_folder_path, "full_dataset_stdata.h5ad"))

    scgpt_preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        # filter_gene_by_counts=3,  # step 1
        filter_gene_by_counts=False,  # I modified since it affects the generation of selected genes h5ad
        # filter_cell_by_counts=False,  # step 2
        filter_cell_by_counts=1,  # I modified since there are empty cells that cause errors for binning procedure
        normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=True,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        # subset_hvg=n_hvg,  # 5. whether to subset the raw data to highly variable genes
        # hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=51,  # 6. whether to bin the raw data and to what number of bins # Following the setting of scGPT
        # https://github.com/bowang-lab/scGPT/blob/main/tutorials/Tutorial_Annotation.ipynb
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    full_dataset_stdata_scgpt_preprocess = full_dataset_stdata.copy()

    scgpt_preprocess_start_time = time.time()
    scgpt_preprocessor(full_dataset_stdata_scgpt_preprocess, batch_key=None)
    scgpt_preprocess_end_time = time.time()
    print(f"The scgpt preprocess of the full dataset stdata is finished. Time Used: {running_time_display(scgpt_preprocess_end_time - scgpt_preprocess_start_time)}.")

    write_big_h5ad_start_time = time.time()
    full_dataset_big_stdata_str_copy = full_dataset_stdata_scgpt_preprocess.copy()
    full_dataset_big_stdata_str_copy.obs = full_dataset_big_stdata_str_copy.obs.astype(str)
    full_dataset_big_stdata_str_copy.var = full_dataset_big_stdata_str_copy.var.astype(str)
    if not os.path.exists(os.path.join(dataset_inputs_folder_path, "full_dataset_stdata_scgpt_preprocess.h5ad")):
        full_dataset_big_stdata_str_copy.write_h5ad(os.path.join(dataset_inputs_folder_path, "full_dataset_stdata_scgpt_preprocess.h5ad"))
    write_big_h5ad_end_time = time.time()
    print(f"The writing of the full dataset stdata is finished. Time Used: {running_time_display(write_big_h5ad_end_time - write_big_h5ad_start_time)}.")

    X_binned_array = full_dataset_stdata_scgpt_preprocess.layers["X_binned"]
    X_binned_tensor = torch.from_numpy(X_binned_array)
    if not os.path.exists(os.path.join(dataset_inputs_folder_path, "full_dataset_X_binned.pt")):
        torch.save(X_binned_tensor, os.path.join(dataset_inputs_folder_path, "full_dataset_X_binned.pt"))

    patient_dir_numerical = full_dataset_stdata_scgpt_preprocess.obs["patient_dir_numerical"].astype('int32')
    patient_dir_numerical_tensor = torch.tensor(patient_dir_numerical, dtype=torch.int)
    patient_dir_numerical_2d_tensor = torch.unsqueeze(input=patient_dir_numerical_tensor, dim=1)
    if not os.path.exists(os.path.join(dataset_inputs_folder_path, "full_dataset_patient_dir_numerical.pt")):
        torch.save(patient_dir_numerical_2d_tensor, os.path.join(dataset_inputs_folder_path, "full_dataset_patient_dir_numerical.pt"))
    
    organ_type_numerical = full_dataset_stdata_scgpt_preprocess.obs["organ_type_numerical"].astype('int32')
    organ_type_numerical_tensor = torch.tensor(organ_type_numerical, dtype=torch.int)
    organ_type_numerical_2d_tensor = torch.unsqueeze(input=organ_type_numerical_tensor, dim=1)
    if not os.path.exists(os.path.join(dataset_inputs_folder_path, "full_dataset_organ_type_numerical.pt")):
        torch.save(organ_type_numerical_2d_tensor, os.path.join(dataset_inputs_folder_path, "full_dataset_organ_type_numerical.pt"))

    patch_path_array = full_dataset_stdata_scgpt_preprocess.obs["patch_path"].values
    normed_patch_path_array = full_dataset_stdata_scgpt_preprocess.obs["normed_patch_path"].values
    spot_barcode_array = np.array(full_dataset_stdata_scgpt_preprocess.obs.index)
    if not os.path.exists(os.path.join(dataset_inputs_folder_path, "full_dataset_patch_path.npy")):
        np.save(os.path.join(dataset_inputs_folder_path, "full_dataset_patch_path.npy"), patch_path_array)
    if not os.path.exists(os.path.join(dataset_inputs_folder_path, "full_dataset_normed_patch_path.npy")):
        np.save(os.path.join(dataset_inputs_folder_path, "full_dataset_normed_patch_path.npy"), normed_patch_path_array)
    if not os.path.exists(os.path.join(dataset_inputs_folder_path, "full_dataset_spot_barcode.npy")):
        np.save(os.path.join(dataset_inputs_folder_path, "full_dataset_spot_barcode.npy"), spot_barcode_array)
    ############################################################  

    ############################################################
    # Part 3. tokenized_gene_ids.pt & tokenized_gene_values.pt

    X_binned_tensor = torch.load(os.path.join(dataset_inputs_folder_path, "full_dataset_X_binned.pt"))
    X_binned_array = X_binned_tensor.numpy()

    scgpt_human_vocab_path = os.path.join(args.main_data_storage, args.raw_data_folder_name, "Supplementary_data", "pretrained", "scGPT", "scGPT_human", "vocab.json")
    scgpt_human_vocab = GeneVocab.from_file(scgpt_human_vocab_path)
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    for token in special_tokens:
         if token not in scgpt_human_vocab:
             scgpt_human_vocab.append_token(token)
    scgpt_human_vocab.set_default_index(scgpt_human_vocab["<pad>"])

    scgpt_tokenize_start_time = time.time()
    print("Start tokenizing!")
    scgpt_tokenized_data = tokenize_and_pad_batch(
            data = X_binned_array,
            gene_ids = np.array(scgpt_human_vocab(selected_genes_list), dtype=int),
            max_len = len(selected_genes_list),
            vocab = scgpt_human_vocab,
            pad_token = "<pad>",
            pad_value = -2,
            append_cls = True, # append <cls> token at the beginning
            include_zero_gene = False,
            cls_token = "<cls>",
    )
    scgpt_tokenize_end_time = time.time()
    print(f"The scgpt tokenization of the full dataset X_binned is finished. Time Used: {running_time_display(scgpt_tokenize_end_time - scgpt_tokenize_start_time)}.")

    if not os.path.exists(os.path.join(dataset_inputs_folder_path, "full_dataset_tokenized_data.pt")):
        torch.save(scgpt_tokenized_data, os.path.join(dataset_inputs_folder_path, "full_dataset_tokenized_data.pt"))

    tokenized_data = torch.load(os.path.join(dataset_inputs_folder_path, "full_dataset_tokenized_data.pt"))
    tokenized_gene_ids_tensor = tokenized_data["genes"]
    tokenized_gene_values_tensor = tokenized_data["values"]
    if not os.path.exists(os.path.join(dataset_inputs_folder_path, "full_dataset_tokenized_gene_ids.pt")):
        torch.save(tokenized_gene_ids_tensor, os.path.join(dataset_inputs_folder_path, "full_dataset_tokenized_gene_ids.pt"))
    if not os.path.exists(os.path.join(dataset_inputs_folder_path, "full_dataset_tokenized_gene_values.pt")):
        torch.save(tokenized_gene_values_tensor, os.path.join(dataset_inputs_folder_path, "full_dataset_tokenized_gene_values.pt"))
    ############################################################

    ############################################################
    # Part 4. masked_tokenized_gene_values.pt

    tokenized_gene_values_tensor = torch.load(os.path.join(dataset_inputs_folder_path, "full_dataset_tokenized_gene_values.pt"))

    masked_tokenized_gene_values_tensor_ratio_25 = random_mask_value(
        values = tokenized_gene_values_tensor,
        mask_ratio = 0.25,
        mask_value = -1,
        pad_value = -2,
    )
    masked_tokenized_gene_values_tensor_ratio_75 = random_mask_value(
        values = tokenized_gene_values_tensor,
        mask_ratio = 0.75,
        mask_value = -1,
        pad_value = -2,
    )
    if not os.path.exists(os.path.join(dataset_inputs_folder_path, "full_dataset_masked_tokenized_gene_values_with_mask_ratio_25.pt")):
        torch.save(masked_tokenized_gene_values_tensor_ratio_25, os.path.join(dataset_inputs_folder_path, "full_dataset_masked_tokenized_gene_values_with_mask_ratio_25.pt"))
    if not os.path.exists(os.path.join(dataset_inputs_folder_path, "full_dataset_masked_tokenized_gene_values_with_mask_ratio_75.pt")):
        torch.save(masked_tokenized_gene_values_tensor_ratio_75, os.path.join(dataset_inputs_folder_path, "full_dataset_masked_tokenized_gene_values_with_mask_ratio_75.pt")) 
    ############################################################

    ############################################################
    # Part 5. Create individual pt files for each row of the big tensor

    # 1. full_dataset_masked_tokenized_gene_values_with_mask_ratio_25.pt
    full_dataset_masked_tokenized_gene_values_with_mask_ratio_25_tensor = torch.load(os.path.join(dataset_inputs_folder_path, "full_dataset_masked_tokenized_gene_values_with_mask_ratio_25.pt"))
    pool = Pool(processes=args.multiprocessing_pool_num)
    with tqdm(total=len(list(range(full_dataset_masked_tokenized_gene_values_with_mask_ratio_25_tensor.size()[0])))) as pbar:
        for _ in pool.imap(partial(save_single_row_data_to_pt_file, dataset_inputs_folder_path=dataset_inputs_folder_path, input_big_tensor=full_dataset_masked_tokenized_gene_values_with_mask_ratio_25_tensor, input_big_tensor_name="full_dataset_masked_tokenized_gene_values_with_mask_ratio_25.pt"), list(range(full_dataset_masked_tokenized_gene_values_with_mask_ratio_25_tensor.size()[0]))):
            pbar.update()
    pool.close()
    pool.join()

    # 2. full_dataset_masked_tokenized_gene_values_with_mask_ratio_75.pt
    full_dataset_masked_tokenized_gene_values_with_mask_ratio_75_tensor = torch.load(os.path.join(dataset_inputs_folder_path, "full_dataset_masked_tokenized_gene_values_with_mask_ratio_75.pt"))
    pool = Pool(processes=args.multiprocessing_pool_num)
    with tqdm(total=len(list(range(full_dataset_masked_tokenized_gene_values_with_mask_ratio_75_tensor.size()[0])))) as pbar:
        for _ in pool.imap(partial(save_single_row_data_to_pt_file, dataset_inputs_folder_path=dataset_inputs_folder_path, input_big_tensor=full_dataset_masked_tokenized_gene_values_with_mask_ratio_75_tensor, input_big_tensor_name="full_dataset_masked_tokenized_gene_values_with_mask_ratio_75.pt"), list(range(full_dataset_masked_tokenized_gene_values_with_mask_ratio_75_tensor.size()[0]))):
            pbar.update()
    pool.close()
    pool.join()

    # 3. full_dataset_tokenized_gene_ids.pt
    full_dataset_tokenized_gene_ids_tensor = torch.load(os.path.join(dataset_inputs_folder_path, "full_dataset_tokenized_gene_ids.pt"))
    pool = Pool(processes=args.multiprocessing_pool_num)
    with tqdm(total=len(list(range(full_dataset_tokenized_gene_ids_tensor.size()[0])))) as pbar:
        for _ in pool.imap(partial(save_single_row_data_to_pt_file, dataset_inputs_folder_path=dataset_inputs_folder_path, input_big_tensor=full_dataset_tokenized_gene_ids_tensor, input_big_tensor_name="full_dataset_tokenized_gene_ids.pt"), list(range(full_dataset_tokenized_gene_ids_tensor.size()[0]))):
            pbar.update()
    pool.close()
    pool.join()

    # 4. full_dataset_tokenized_gene_values.pt
    full_dataset_tokenized_gene_values_tensor = torch.load(os.path.join(dataset_inputs_folder_path, "full_dataset_tokenized_gene_values.pt"))
    pool = Pool(processes=args.multiprocessing_pool_num)
    with tqdm(total=len(list(range(full_dataset_tokenized_gene_values_tensor.size()[0])))) as pbar:
        for _ in pool.imap(partial(save_single_row_data_to_pt_file, dataset_inputs_folder_path=dataset_inputs_folder_path, input_big_tensor=full_dataset_tokenized_gene_values_tensor, input_big_tensor_name="full_dataset_tokenized_gene_values.pt"), list(range(full_dataset_tokenized_gene_values_tensor.size()[0]))):
            pbar.update()
    pool.close()
    pool.join()

    # 5. full_dataset_X_binned.pt
    full_dataset_X_binned_tensor = torch.load(os.path.join(dataset_inputs_folder_path, "full_dataset_X_binned.pt"))
    pool = Pool(processes=args.multiprocessing_pool_num)
    with tqdm(total=len(list(range(full_dataset_X_binned_tensor.size()[0])))) as pbar:
        for _ in pool.imap(partial(save_single_row_data_to_pt_file, dataset_inputs_folder_path=dataset_inputs_folder_path, input_big_tensor=full_dataset_X_binned_tensor, input_big_tensor_name="full_dataset_X_binned.pt"), list(range(full_dataset_X_binned_tensor.size()[0]))):
            pbar.update()
    pool.close()
    pool.join()
    ############################################################

if __name__ == "__main__":
    main()