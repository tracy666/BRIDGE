from typing import List, Optional, Union, Literal, Tuple
import os
import tarfile
import gzip
import zipfile
from collections import defaultdict
import pandas as pd
import shutil
import anndata
import numpy as np
import math
import scanpy as sc

def Genomics_dataset_num_to_string(
    num: int,
) -> str:
    if len(str(num)) <= 3:
        string = "0" * (3 - len(str(num))) + str(num)
        return string
    else:
        raise ValueError(f"Dataset no.{str(num)} has a numerical name that is too large.")  # Corrected raise statement
    
def double_unzip_tar_gz(
    project_dataset_raw_data_path: str,
    project_dataset_raw_unzipped_data_path: str,
    filetype: Optional[Union[Literal["matrix", "spatial"], str]],
    already_in_folder_format: Optional[bool] = False,
    tar_gz_path: Optional[str] = None,
    already_folder_name: Optional[str] = None,
) -> None:
    if filetype == "matrix":
        if already_in_folder_format == False:
            assert os.path.exists(tar_gz_path), f"The .tar.gz file at {tar_gz_path} does not exists, please check."
            # Task 1: Unzip base tar.gz but keep components in gz format for Seurat analysis
            filtered_feature_bc_matrix_path = os.path.join(project_dataset_raw_unzipped_data_path, "filtered_feature_bc_matrix")
            if not os.path.exists(filtered_feature_bc_matrix_path):
                if os.path.getsize(tar_gz_path) == 0:
                    print(f"The file at {tar_gz_path} is empty (occupy 0 in size) and cannot be opened via tarfile.open.")
                with tarfile.open(tar_gz_path, 'r:gz') as tar_gz:
                    tar_gz.extractall(project_dataset_raw_unzipped_data_path, filter='data')
            assert "filtered_feature_bc_matrix" in os.listdir(project_dataset_raw_unzipped_data_path), f"The {tar_gz_path} is not extracted successfully."
            # Task 2: Unzip base tar.gz and unzip the components further
            filtered_feature_bc_matrix_content_list = os.listdir(filtered_feature_bc_matrix_path)
            filtered_feature_bc_matrix_content_paths_list = [os.path.join(filtered_feature_bc_matrix_path, content) for content in filtered_feature_bc_matrix_content_list]
            filtered_feature_bc_matrix_all_unzipped_path = os.path.join(project_dataset_raw_unzipped_data_path, "filtered_feature_bc_matrix_all_unzipped")
            if not os.path.exists(filtered_feature_bc_matrix_all_unzipped_path):
                os.makedirs(filtered_feature_bc_matrix_all_unzipped_path)
            for content_path in filtered_feature_bc_matrix_content_paths_list:
                unzipped_filename = os.path.splitext(os.path.basename(content_path))[0]
                unzipped_file_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_path, unzipped_filename)
                # Open the .gz file and write the unzipped contents to the new file
                with gzip.open(content_path, 'rb') as gz_file:
                    with open(unzipped_file_path, 'wb') as output_file:
                        output_file.write(gz_file.read())
            assert len(os.listdir(filtered_feature_bc_matrix_all_unzipped_path))==3, f"Not three files in unzipped folder {filtered_feature_bc_matrix_all_unzipped_path}, please double check."
        else:
            project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
            assert already_folder_name in project_dataset_raw_data_content_list, f"There is no {already_folder_name} inside {project_dataset_raw_data_path}, set the wrong value for already_in_folder_format? It is {already_in_folder_format} now"
            filtered_feature_bc_matrix_path = os.path.join(project_dataset_raw_data_path, already_folder_name)
            filtered_feature_bc_matrix_content_list = os.listdir(filtered_feature_bc_matrix_path)
            filtered_feature_bc_matrix_content_paths_list = [os.path.join(filtered_feature_bc_matrix_path, content) for content in filtered_feature_bc_matrix_content_list]
            filtered_feature_bc_matrix_all_unzipped_path = os.path.join(project_dataset_raw_unzipped_data_path, "filtered_feature_bc_matrix_all_unzipped")
            if not os.path.exists(filtered_feature_bc_matrix_all_unzipped_path):
                os.makedirs(filtered_feature_bc_matrix_all_unzipped_path)
            for content_path in filtered_feature_bc_matrix_content_paths_list:
                unzipped_filename = os.path.splitext(os.path.basename(content_path))[0]
                unzipped_file_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_path, unzipped_filename)
                # Open the .gz file and write the unzipped contents to the new file
                with gzip.open(content_path, 'rb') as gz_file:
                    with open(unzipped_file_path, 'wb') as output_file:
                        output_file.write(gz_file.read())
            assert len(os.listdir(filtered_feature_bc_matrix_all_unzipped_path))==3, f"Not three files in unzipped folder {filtered_feature_bc_matrix_all_unzipped_path}, please double check."           

    elif filetype == "spatial":
        spatial_path = os.path.join(project_dataset_raw_unzipped_data_path, "spatial")
        if not os.path.exists(spatial_path):
            if os.path.getsize(tar_gz_path) == 0:
                print(f"The file at {tar_gz_path} is empty (occupy 0 in size) and cannot be opened via tarfile.open.")
            with tarfile.open(tar_gz_path, 'r:gz') as tar_gz:
                tar_gz.extractall(project_dataset_raw_unzipped_data_path, filter='data')
        assert "spatial" in os.listdir(project_dataset_raw_unzipped_data_path), f"The {tar_gz_path} is not extracted successfully."
    else:
        raise ValueError(f"Encountering situation other than 'matrix' or 'spatial' for {filetype} file type.")
        
def running_time_display(
    elapsed_time: float,
) -> str:
    if elapsed_time < 60:
        return f"{elapsed_time:.2f} seconds"
    elif elapsed_time < 3600:
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        return f"{minutes} minutes {seconds} seconds"
    else:
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        return f"{hours} hours {minutes} minutes {seconds} seconds"
    
def unzip_zip(
    zip_path: str,
    unzip_folder_path: str,
) -> None:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            file_path = os.path.join(unzip_folder_path, file)
            if not os.path.exists(file_path):
                zip_ref.extract(file, unzip_folder_path)

def HCA001_manual_image_pcw_h5ad_mapping(
        
) -> dict:
    manual_image_pcw_h5ad_mapping = {
        "V10S24-031 A1.jpg": ["pcw20", "6332STDY10289520"],
        "V10S24-031 B1.jpg": ["pcw20", "6332STDY10289521"],
        "V10S24-031 C1.jpg": ["pcw20", "6332STDY10289522"],
        "V10S24-031 D1.jpg": ["pcw19", "6332STDY10289523"],
        "V19N20-080_15413-LNG--FO-3_C1.tif": ["pcw16", "6332STDY9479168"],
        "V19N20-080_15415-LNG--FO-2_D1.tif": ["pcw17", "6332STDY9479169"],
        "V19N20-080_Hst4-LNG--FO-1_A1.tif": ["pcw12", "6332STDY9479166"],
        "V19N20-080_Hst7-LNG--FO-2_B1.tif": ["pcw14", "6332STDY9479167"],
        "V19N20-081_15417-LNG-0-FO-4_D1.tif": ["pcw17", "6332STDY9479173"],
        "V19N20-081_15424-LNG-0-FO-3_C1.tif": ["pcw16", "6332STDY9479172"],
        "V19N20-081_15428-LNG-0-FO-2_B1.tif": ["pcw14", "6332STDY9479171"],
        "V19N20-081_Hst5-LNG--FO-1_A1.tif": ["pcw12", "6332STDY9479170"],
    }
    return manual_image_pcw_h5ad_mapping

def unzip_tar(
        tar_path: str,
        untar_folder_path: str,
) -> None:
    if not os.path.exists(untar_folder_path):
        os.makedirs(untar_folder_path)
    # Check if the tar file has already been extracted
    if len(os.listdir(untar_folder_path)) != 0:
        return
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(tar_path, filter='data')

def unzip_gz(
        gz_path: str,
        ungz_folder_path: str,
) -> None:
    unzipped_file_name = os.path.splitext(os.path.basename(gz_path))[0]  # Removing the '.gz' extension
    unzipped_file_path = os.path.join(ungz_folder_path, unzipped_file_name)
    if not os.path.exists(unzipped_file_path):
        with gzip.open(gz_path, 'rb') as gz_file:
            with open(unzipped_file_path, 'wb') as unzipped_file:
                unzipped_file.write(gz_file.read())

def unzip_tar_gz(
        tar_gz_path: str,
        untar_gz_path: str,
) -> None:
    if not os.path.exists(untar_gz_path):
        if os.path.getsize(tar_gz_path) == 0:
            print(f"The file at {tar_gz_path} is empty (occupy 0 in size) and cannot be opened via tarfile.open.")
        with tarfile.open(tar_gz_path, 'r:gz') as tar_gz:
            tar_gz.extractall(untar_gz_path, filter='data')

def NCBI003_image_model_mapping(
        
) -> dict:
    image_model_mapping = {
        "A1": "Human_Nephrectomy",
        "B1": "Sham_Model",
        "C1": "IRI_Model",
        "D1": "CLP_Model",
    }
    return image_model_mapping

def NCBI007_dataset_name_image_name_mapping(
    
) -> dict:
    dataset_name_image_name_mapping = {
        "A1": "A1",
        "A2": "A2",
        "A3": "A3",
        "A4": "A4",
        "A6": "ST200320A",
        "A7": "ST200320B",
        "A8": "ST200320C",
        "A9": "ST200320D",
    }
    return dataset_name_image_name_mapping

def find_folders_for_super_long_path(
    directory: str,
) -> Optional[List[str]]:
    subdirectories = [os.path.join(directory, item) for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))]
    if len(subdirectories) == 2:
        return subdirectories
    elif len(subdirectories) == 1:
        return find_folders_for_super_long_path(subdirectories[0])
    else:
        return None

def gene_barcode_name_mapping(
    main_data_storage: str,
    project_data_folder_name: str,
) -> dict:
    features_tsv_paths_list = list()
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name)
    for root, dirs, files in os.walk(project_data_folder_path):
        if "filtered_feature_bc_matrix_all_unzipped" in dirs:
            filtered_matrix_unzipped_folder_path = os.path.join(root, "filtered_feature_bc_matrix_all_unzipped")
            features_tsv_path = os.path.join(filtered_matrix_unzipped_folder_path, "features.tsv")
            if os.path.exists(features_tsv_path) and os.path.isfile(features_tsv_path):
                features_tsv_paths_list.append(features_tsv_path)

    gene_barcode_to_name_mapping = defaultdict(dict)
    gene_barcodes_10x = list()
    for tsv_path in features_tsv_paths_list:
        features = pd.read_csv(tsv_path, header=None, delimiter='\t', names=["gene_barcode", "gene_name", "gene_expression"])
        for index, row in features.iterrows():
            barcode = row["gene_barcode"]
            gene_barcodes_10x.append(barcode)
            name = row["gene_name"]
            if name not in gene_barcode_to_name_mapping[barcode]:
                gene_barcode_to_name_mapping[barcode][name] = 1
            else:
                gene_barcode_to_name_mapping[barcode][name] += 1

    gene_barcode_to_name_mapping_df = pd.DataFrame([(key, sub_key, value) for key, sub_dict in gene_barcode_to_name_mapping.items() for sub_key, value in sub_dict.items()], columns=['Gene_barcode', 'Gene_name', 'frequency'])
    gene_barcode_to_name_mapping_df['frequency'] = pd.to_numeric(gene_barcode_to_name_mapping_df['frequency']) # Convert the 'frequency' column to numeric
    idx = gene_barcode_to_name_mapping_df.groupby('Gene_barcode')['frequency'].idxmax() # Find the index of rows with the highest frequency for each unique value of 'Gene_barcode'
    gene_barcode_to_name_mapping_df_filtered = gene_barcode_to_name_mapping_df.loc[idx] # Keep only the rows with the highest frequency for each unique value of 'Gene_barcode'
    gene_barcode_to_name_mapping_df_filtered = gene_barcode_to_name_mapping_df_filtered.dropna(subset=["Gene_name"]) # Filter out rows with empty values in the 'Gene_name' column
    whether_barcode_is_unique = gene_barcode_to_name_mapping_df_filtered['Gene_barcode'].is_unique
    assert whether_barcode_is_unique == True, f"There are duplicated barcodes in gene_barcode_to_name_mapping_df_filtered. Please fix."
    gene_barcode_to_name_dictionary_for_stdata = dict(zip(gene_barcode_to_name_mapping_df_filtered['Gene_barcode'], gene_barcode_to_name_mapping_df_filtered['Gene_name']))
    return gene_barcode_to_name_dictionary_for_stdata

def delete_with_keywords(
        root_dir: str,
        keywords: List[str],
        file_or_dir: Literal["file", "dir"],
) -> None:
    for root, dirs, files in os.walk(root_dir):
        if file_or_dir == "file":
            for file in files:
                file_path = os.path.join(root, file)
                if any(keyword in file for keyword in keywords):
                    print("Deleting file:", file_path)
                    os.remove(file_path)
        elif file_or_dir == "dir":
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if any(keyword in dir for keyword in keywords):
                    print("Deleting dir:", dir_path)
                    shutil.rmtree(dir_path)
        else:
            raise ValueError(f"file_or_dir should be either 'file' or 'dir', but it is {file_or_dir} now.")

def Genomics_project_dataset_to_organ(
    
) -> dict:
    project_dataset_to_organ = {
        "001": "brain",
        "002": "lung",
        "003": "kidney",
        "004": "small_and_large_intestine",
        "005": "breast",
        "006": "ovary", 
        "007": "lung",
        "011": "breast", 
        "012": "ovary",
        "013": "cervix",
        "014": "small_and_large_intestine",
        "015": "prostate",
        "017": "prostate",
        "018": "prostate",
        "019": "breast",
        "027": "brain",
        "029": "small_and_large_intestine",
        "031": "brain",
        "034": "breast",
        "036": "lymph_node",
        "037": "heart",
        "038": "breast",
        "039": "breast",
        "046_Control_Replicate_1": "small_and_large_intestine",
        "046_Control_Replicate_2": "small_and_large_intestine",
        "046_Post-Xenium_Replicate_1": "small_and_large_intestine",
        "046_Post-Xenium_Replicate_2": "small_and_large_intestine",
        "101": "breast",
    }
    return project_dataset_to_organ

def find_all_patches(
        main_data_storage: str,
        project_data_folder_name: str,
        patches_folder_names: List[str],
) -> List:
    patches_list = list()
    for root, dirs, files in os.walk(os.path.join(main_data_storage, project_data_folder_name)):
        for folder_name in patches_folder_names:
            if folder_name in dirs:
                folder_path = os.path.join(root, folder_name)
                normed_folder_path = folder_path.replace("patches", "normed_patches")
                if not os.path.exists(normed_folder_path):
                    os.makedirs(normed_folder_path)
                for dir_root, dir_dirs, dir_files in os.walk(folder_path):
                    for file in dir_files:
                        patch_path = os.path.join(dir_root, file)
                        patches_list.append(patch_path)
    return patches_list

def find_all_stdata_h5ad(
        main_data_storage: str,
        project_data_folder_name: str,
        stdata_file_names: List[str],
        include_downstream_task_data: bool,
) -> List:
    stdata_list = list()
    for root, dirs, files in os.walk(os.path.join(main_data_storage, project_data_folder_name)):
        for file_name in stdata_file_names:
            if file_name in files:
                stdata_path = os.path.join(root, file_name)
                if ("preprocessed_data" in stdata_path) and ("downstream_task_data" not in stdata_path):
                    stdata_list.append(stdata_path)
                if include_downstream_task_data:
                    if ("preprocessed_data" in stdata_path) and ("DLPFC" in stdata_path):
                        stdata_list.append(stdata_path)
    return stdata_list

def get_genes_of_each_slide(
        stdata_path: str,
) -> List:
    stdata_anndata = anndata.read_h5ad(stdata_path)
    # stdata_anndata.var_names_make_unique()
    gene_names_list = stdata_anndata.var.index.tolist()
    return gene_names_list

def get_spot_number_of_each_slide(
        stdata_path: str,
) -> Tuple[str, int]:
    stdata_anndata = anndata.read_h5ad(stdata_path)
    return stdata_path, len(stdata_anndata.obs)

def get_mean_std_of_intersection_genes_of_each_slide(
        stdata_path: str,
        intersection_genes_list: List[str],
) -> Tuple[List[float], List[float]]:
    stdata_anndata = anndata.read_h5ad(stdata_path)
    stdata_anndata.var_names_make_unique()
    gene_counts_matrix = stdata_anndata.X
    gene_names_list_per_slide = stdata_anndata.var.index.tolist()
    gene_names_indices_mapping = {gene_name: index for index, gene_name in enumerate(gene_names_list_per_slide)}
    column_numbers_selected = [gene_names_indices_mapping[gene_name] for gene_name in gene_names_list_per_slide if gene_name in intersection_genes_list]
    sorted_column_numbers_selected = sorted(column_numbers_selected, key=lambda x: intersection_genes_list.index(gene_names_list_per_slide[x]))
    gene_counts_matrix_with_intersection_genes = gene_counts_matrix[:, sorted_column_numbers_selected]
    gene_counts_matrix_with_intersection_genes_square = gene_counts_matrix_with_intersection_genes.copy()
    gene_counts_matrix_with_intersection_genes_square.data **= 2

    column_mean_value_list = gene_counts_matrix_with_intersection_genes.mean(axis=0).tolist()[0]
    column_square_mean_value_list = gene_counts_matrix_with_intersection_genes_square.mean(axis=0).tolist()[0]
    column_variance_value_list = [mean_squared - mean**2 for mean, mean_squared in zip(column_mean_value_list, column_square_mean_value_list)]
    column_standard_deviation_list = [math.sqrt(variance) for variance in column_variance_value_list]
    assert len(column_mean_value_list) == len(column_standard_deviation_list) == len(intersection_genes_list), f"The length of column_mean_value_list, column_standard_deviation_list, and intersection_genes_list should be the same."
    return stdata_path, (column_mean_value_list, column_standard_deviation_list)

def calculate_weighted_sum_std_of_intersection_genes(
        index: int,
        spot_number_dict: dict,
        total_slide_mean_for_each_gene_dict: dict,
        total_slide_std_for_each_gene_dict: dict,
) -> Tuple[int, Tuple[float, float]]:
    total_spot_number = sum(spot_number_dict.values())
    index_gene_total_mean, index_gene_total_std = 0, 0
    for stdata_path in list(spot_number_dict.keys()):
        single_slide_spot_number = spot_number_dict[stdata_path]
        single_slide_mean_list = total_slide_mean_for_each_gene_dict[stdata_path]
        specific_gene_mean_value = single_slide_mean_list[index]
        single_slide_std_list = total_slide_std_for_each_gene_dict[stdata_path]
        specific_gene_std_value = single_slide_std_list[index]
        index_gene_total_mean += single_slide_spot_number * specific_gene_mean_value
        index_gene_total_std += single_slide_spot_number * specific_gene_std_value
    index_gene_avg_mean = index_gene_total_mean / total_spot_number
    index_gene_avg_std = index_gene_total_std / total_spot_number
    return index, (index_gene_avg_mean, index_gene_avg_std)

def get_top_k_genes(
        weighted_value_for_each_gene_dict: List[float],
        top_k_genes: int,
        intersection_genes_list: List[str],
) -> List[str]:
    top_k_indices_sorted_by_value = sorted(range(len(weighted_value_for_each_gene_dict)), key=lambda i: weighted_value_for_each_gene_dict[i], reverse=True)
    top_k_indices = top_k_indices_sorted_by_value[:top_k_genes]
    top_k_genes = [intersection_genes_list[index] for index in top_k_indices]
    return top_k_indices, top_k_genes
    
def get_top_k_HEG_genes_per_slide(
        stdata_path: str,
        top_k_HEG_value: int,
) -> List[str]:
    stdata_anndata = anndata.read_h5ad(stdata_path)
    # stdata_anndata.var_names_make_unique()
    gene_counts_matrix = stdata_anndata.X
    gene_names_list_per_slide = stdata_anndata.var.index.tolist()

    gene_counts_matrix_copy = gene_counts_matrix.copy()
    column_mean_value_list = gene_counts_matrix_copy.mean(axis=0).tolist()[0]
    top_k_indices = sorted(range(len(column_mean_value_list)), key=lambda i: column_mean_value_list[i], reverse=True)[:top_k_HEG_value]
    top_k_genes = [gene_names_list_per_slide[index] for index in top_k_indices]
    return top_k_genes

def find_all_stdata_h5ad_200K(
        main_data_storage: str,
        project_data_folder_name: str,
        working_codespace: str,
        generation_date: str,
) -> List:
    stdata_list = list()
    train_df_path = os.path.join(working_codespace, "generated_files", generation_date, "train_df.csv")
    train_patient_dir_list = pd.read_csv(train_df_path)["patient_dir"].tolist()
    valid_df_path = os.path.join(working_codespace, "generated_files", generation_date, "valid_df.csv")
    valid_patient_dir_list = pd.read_csv(valid_df_path)["patient_dir"].tolist()
    test_df_path = os.path.join(working_codespace, "generated_files", generation_date, "test_df.csv")
    test_patient_dir_list = pd.read_csv(test_df_path)["patient_dir"].tolist()
    total_patient_dir_list = train_patient_dir_list + valid_patient_dir_list + test_patient_dir_list

    for patient_dir in total_patient_dir_list:
        stdata_path = os.path.join(main_data_storage, project_data_folder_name, patient_dir, "preprocessed_data", "stdata_MICCAI.h5ad")
        assert os.path.exists(stdata_path), f"The stdata file at {stdata_path} does not exist."
        stdata_list.append(stdata_path)
    assert len(stdata_list) == 167, f"The number of stdata files is not 167, please check."
    return stdata_list

def find_all_stdata_h5ad_200K_with_DLPFC(
        main_data_storage: str,
        project_data_folder_name: str,
        working_codespace: str,
        generation_date: str,
) -> List:
    stdata_list = list()
    train_df_path = os.path.join(working_codespace, "generated_files", generation_date, "train_df.csv")
    train_patient_dir_list = pd.read_csv(train_df_path)["patient_dir"].tolist()
    valid_df_path = os.path.join(working_codespace, "generated_files", generation_date, "valid_df.csv")
    valid_patient_dir_list = pd.read_csv(valid_df_path)["patient_dir"].tolist()
    test_df_path = os.path.join(working_codespace, "generated_files", generation_date, "test_df.csv")
    test_patient_dir_list = pd.read_csv(test_df_path)["patient_dir"].tolist()

    # newly added
    downstream_df_path = os.path.join(working_codespace, "generated_files", generation_date, "downstream_df.csv")
    downstream_task_data_patient_dir_list = pd.read_csv(downstream_df_path)["patient_dir"].tolist()
    total_patient_dir_list = train_patient_dir_list + valid_patient_dir_list + test_patient_dir_list + downstream_task_data_patient_dir_list

    for patient_dir in total_patient_dir_list:
        stdata_path = os.path.join(main_data_storage, project_data_folder_name, patient_dir, "preprocessed_data", "stdata_MICCAI.h5ad")
        assert os.path.exists(stdata_path), f"The stdata file at {stdata_path} does not exist."
        stdata_list.append(stdata_path)
    assert len(stdata_list) == 179, f"The number of stdata files is not 179, please check."
    return stdata_list

def find_path_with_keywords(
        root_dir: str,
        keywords: List[str],
        file_or_dir: Literal["file", "dir"],
) -> None:
    return_list = list()
    for root, dirs, files in os.walk(root_dir):
        if file_or_dir == "file":
            for file in files:
                file_path = os.path.join(root, file)
                if any(keyword in file for keyword in keywords):
                    return_list.append(file_path)
        elif file_or_dir == "dir":
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if any(keyword in dir for keyword in keywords):
                    return_list.append(dir_path)
        else:
            raise ValueError(f"file_or_dir should be either 'file' or 'dir', but it is {file_or_dir} now.")
    return return_list




