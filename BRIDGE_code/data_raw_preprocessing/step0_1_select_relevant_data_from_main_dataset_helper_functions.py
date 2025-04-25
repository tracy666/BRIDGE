from typing import List, Optional, Union, Literal
import os
import time
import shutil
import pandas as pd
import anndata
import pyzipper
from step0_preprocess_helper_functions import Genomics_dataset_num_to_string, double_unzip_tar_gz, running_time_display, unzip_zip, HCA001_manual_image_pcw_h5ad_mapping, unzip_tar, unzip_gz, unzip_tar_gz, NCBI003_image_model_mapping, NCBI007_dataset_name_image_name_mapping, find_folders_for_super_long_path

def Genomics_project_list(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "10xGenomics_human")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "10xGenomics")
    raw_datasets = os.listdir(raw_data_folder_path)
    excluded_10x_dataset_num = [
        8, # different staining (IF Stained), greyscale image
        9, # different staining (IF Stained), greyscale image
        10, # repeated dataset with 031
        16, # different staining (IF Stained), greyscale image
        20, # different staining (Fluorescent Stained), greyscale image
        21, # different staining (DAPI Stained), greyscale image
        22, # different staining (DAPI Stained), greyscale image
        23, # different staining (DAPI Stained), greyscale image
        24, # different staining (DAPI Stained), greyscale image
        25, # different staining (DAPI Stained), greyscale image
        26, # different staining (DAPI Stained), greyscale image
        28, # repeated dataset with 027
        30, # repeated dataset with 029
        32, # repeated dataset with 031
        33, # repeated dataset with 031
        35, # repeated dataset with 034
        40, # different staining (Anti-SNAP25 Stained), greyscale image
        41, # different staining (Anti-GFAP Stained), greyscale image
        42, # repeated dataset with 036
        43, # repeated dataset with 037
        44, # repeated dataset with 038
        45, # repeated dataset with 039
        # 46, # newest dataset released 2023.10.6
        ]
    excluded_10x_dataset_str = ["10xGenomics" + Genomics_dataset_num_to_string(num) for num in excluded_10x_dataset_num]
    project_datasets = [dataset for dataset in raw_datasets if dataset not in excluded_10x_dataset_str]
    return project_datasets

def Genomics_select_data_single_helper(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "10xGenomics_human")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "10xGenomics")
    raw_dataset_path = os.path.join(raw_data_folder_path, project_dataset)
    raw_dataset_long_title_list = os.listdir(raw_dataset_path)
    raw_dataset_long_title_without_DS_list = [title for title in raw_dataset_long_title_list if os.path.isdir(os.path.join(raw_dataset_path, title))]
    assert len(raw_dataset_long_title_without_DS_list) == 1, f"10xGenomics dataset {project_dataset} has more than one long title"
    raw_dataset_long_title_without_DS_name = raw_dataset_long_title_without_DS_list[0]
    raw_dataset_long_title_without_DS_path = os.path.join(raw_dataset_path, raw_dataset_long_title_without_DS_name)
    raw_dataset_long_title_without_DS_content_list = os.listdir(raw_dataset_long_title_without_DS_path)
    
    if project_dataset == "10xGenomics046":
        start_time = time.time()
        raw_dataset_long_title_without_DS_content_list_filtered_DS = [content for content in raw_dataset_long_title_without_DS_content_list if os.path.isdir(os.path.join(raw_dataset_long_title_without_DS_path, content))]
        assert len(raw_dataset_long_title_without_DS_content_list_filtered_DS) == 4, f"10xGenomics dataset 046 has more than/ less than 4 subfolders"
        for subfolder in raw_dataset_long_title_without_DS_content_list_filtered_DS:
            raw_dataset_long_title_filtered_subfolder_path = os.path.join(raw_dataset_long_title_without_DS_path, subfolder)
            raw_dataset_long_title_filtered_subfolder_content_list = os.listdir(raw_dataset_long_title_filtered_subfolder_path)
            extended_dataset_name = f"{project_dataset}_{subfolder}"
            project_dataset_raw_data_path = os.path.join(project_data_folder_path, extended_dataset_name, "raw_data")
            if not os.path.exists(project_dataset_raw_data_path):
                os.makedirs(project_dataset_raw_data_path)
            matrix, spatial, image = False, False, False
            for raw_data_item in raw_dataset_long_title_filtered_subfolder_content_list:
                if "filtered_feature_bc_matrix.tar.gz" in raw_data_item:
                    raw_matrix_path = os.path.join(raw_dataset_long_title_filtered_subfolder_path, raw_data_item)
                    project_matrix_path = os.path.join(project_dataset_raw_data_path, raw_data_item)
                    if not os.path.exists(project_matrix_path):
                        shutil.copy2(raw_matrix_path, project_matrix_path)
                    matrix = True
                elif "spatial.tar.gz" in raw_data_item:
                    raw_spatial_path = os.path.join(raw_dataset_long_title_filtered_subfolder_path, raw_data_item)
                    project_spatial_path = os.path.join(project_dataset_raw_data_path, raw_data_item)
                    if not os.path.exists(project_spatial_path):
                        shutil.copy2(raw_spatial_path, project_spatial_path)
                    spatial = True
                elif "tissue_image.btf" in raw_data_item:
                    raw_tissue_image_path = os.path.join(raw_dataset_long_title_filtered_subfolder_path, raw_data_item)
                    project_tissue_image_path = os.path.join(project_dataset_raw_data_path, raw_data_item)
                    if not os.path.exists(project_tissue_image_path):
                        shutil.copy2(raw_tissue_image_path, project_tissue_image_path)
                    image = True
            assert (matrix & spatial & image) == True, f"Missing filetype for 10xGenomics dataset {extended_dataset_name}: {'matrix' if not matrix else ''} {'spatial' if not spatial else ''} {'image' if not image else ''} is missing."
        
            project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
            project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, extended_dataset_name, "raw_unzipped_data")
            if not os.path.exists(project_dataset_raw_unzipped_data_path):
                os.makedirs(project_dataset_raw_unzipped_data_path)
            for raw_data_item in project_dataset_raw_data_content_list:
                tar_gz_path = os.path.join(project_dataset_raw_data_path, raw_data_item)
                if "filtered_feature_bc_matrix.tar.gz" in raw_data_item:
                    double_unzip_tar_gz(
                        project_dataset_raw_data_path=project_dataset_raw_data_path,
                        project_dataset_raw_unzipped_data_path=project_dataset_raw_unzipped_data_path,
                        filetype="matrix",
                        already_in_folder_format=False,
                        tar_gz_path=tar_gz_path,
                    )
                elif "spatial.tar.gz" in raw_data_item:
                    double_unzip_tar_gz(
                        project_dataset_raw_data_path=project_dataset_raw_data_path,
                        project_dataset_raw_unzipped_data_path=project_dataset_raw_unzipped_data_path,
                        filetype="spatial",
                        tar_gz_path=tar_gz_path,
                    )
            end_time = time.time()
            print(f"Selecting data for 10xGenomics dataset {extended_dataset_name} is finished. Time used: {running_time_display(end_time - start_time)}.")
            
    else:
        start_time = time.time()
        project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
        if not os.path.exists(project_dataset_raw_data_path):
            os.makedirs(project_dataset_raw_data_path)
        matrix, spatial, image = False, False, False
        for raw_data_item in raw_dataset_long_title_without_DS_content_list:
            if "filtered_feature_bc_matrix.tar.gz" in raw_data_item:
                raw_matrix_path = os.path.join(raw_dataset_long_title_without_DS_path, raw_data_item)
                project_matrix_path = os.path.join(project_dataset_raw_data_path, raw_data_item)
                if not os.path.exists(project_matrix_path):
                    shutil.copy2(raw_matrix_path, project_matrix_path)
                matrix = True
            elif "spatial.tar.gz" in raw_data_item:
                raw_spatial_path = os.path.join(raw_dataset_long_title_without_DS_path, raw_data_item)
                project_spatial_path = os.path.join(project_dataset_raw_data_path, raw_data_item)
                if not os.path.exists(project_spatial_path):
                    shutil.copy2(raw_spatial_path, project_spatial_path)
                spatial = True
            elif "tissue_image.tif" in raw_data_item:
                raw_tissue_image_path = os.path.join(raw_dataset_long_title_without_DS_path, raw_data_item)
                project_tissue_image_path = os.path.join(project_dataset_raw_data_path, raw_data_item)
                if not os.path.exists(project_tissue_image_path):
                    shutil.copy2(raw_tissue_image_path, project_tissue_image_path)
                image = True
        if image != True:
            for raw_data_item in raw_dataset_long_title_without_DS_content_list:
                if "image.tif" in raw_data_item:
                    raw_tissue_image_path = os.path.join(raw_dataset_long_title_without_DS_path, raw_data_item)
                    project_tissue_image_path = os.path.join(project_dataset_raw_data_path, raw_data_item)
                    if not os.path.exists(project_tissue_image_path):
                        shutil.copy2(raw_tissue_image_path, project_tissue_image_path)
                    image = True
        if image != True:
            for raw_data_item in raw_dataset_long_title_without_DS_content_list:
                if "image.jpg" in raw_data_item:
                    raw_tissue_image_path = os.path.join(raw_dataset_long_title_without_DS_path, raw_data_item)
                    project_tissue_image_path = os.path.join(project_dataset_raw_data_path, raw_data_item)
                    if not os.path.exists(project_tissue_image_path):
                        shutil.copy2(raw_tissue_image_path, project_tissue_image_path)
                    image = True
        assert (matrix & spatial & image) == True, f"Missing filetype for 10xGenomics dataset {project_dataset}: {'matrix' if not matrix else ''} {'spatial' if not spatial else ''} {'image' if not image else ''} is missing."
        
        project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
        project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
        if not os.path.exists(project_dataset_raw_unzipped_data_path):
            os.makedirs(project_dataset_raw_unzipped_data_path)
        for raw_data_item in project_dataset_raw_data_content_list:
            tar_gz_path = os.path.join(project_dataset_raw_data_path, raw_data_item)
            if "filtered_feature_bc_matrix.tar.gz" in raw_data_item:
                double_unzip_tar_gz(
                    project_dataset_raw_data_path=project_dataset_raw_data_path,
                    project_dataset_raw_unzipped_data_path=project_dataset_raw_unzipped_data_path,
                    filetype="matrix",
                    already_in_folder_format=False,
                    tar_gz_path=tar_gz_path,
                )
            elif "spatial.tar.gz" in raw_data_item:
                double_unzip_tar_gz(
                    project_dataset_raw_data_path=project_dataset_raw_data_path,
                    project_dataset_raw_unzipped_data_path=project_dataset_raw_unzipped_data_path,
                    filetype="spatial",
                    tar_gz_path=tar_gz_path,
                )
        end_time = time.time()
        print(f"Selecting data for 10xGenomics dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def DLPFC_project_list(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "downstream_task_data", "DLPFC")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "downstream_task_data", "DLPFC")
    raw_dataset_long_title_list = os.listdir(raw_data_folder_path)
    raw_dataset_long_title_list = [title for title in raw_dataset_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, title))]
    assert len(raw_dataset_long_title_list) == 1, f"DLPFC dataset has more than one long title"
    raw_dataset_long_title_name = raw_dataset_long_title_list[0]
    raw_dataset_long_title_path = os.path.join(raw_data_folder_path, raw_dataset_long_title_name)
    raw_dataset_long_title_content_list = os.listdir(raw_dataset_long_title_path)
    raw_dataset_long_title_content_zip_file_list = [content for content in raw_dataset_long_title_content_list if content.endswith(".zip")]
    assert len(raw_dataset_long_title_content_zip_file_list) == 1, f"DLPFC dataset has more than one zip file"
    raw_dataset_long_title_content_zip_file_name = raw_dataset_long_title_content_zip_file_list[0]
    raw_dataset_long_title_content_zip_file_path = os.path.join(raw_dataset_long_title_path, raw_dataset_long_title_content_zip_file_name)
    unzip_zip(
        zip_path=raw_dataset_long_title_content_zip_file_path,
        unzip_folder_path=raw_dataset_long_title_path
    )

    raw_dataset_long_title_unzipped_folder_path = os.path.join(raw_dataset_long_title_path, "10x_DLPFC")
    raw_dataset_long_title_unzipped_folder_content_list = os.listdir(raw_dataset_long_title_unzipped_folder_path)
    project_datasets = [dataset for dataset in raw_dataset_long_title_unzipped_folder_content_list if (dataset + "_full_image.tif") in raw_dataset_long_title_content_list]
    assert len(project_datasets) == len(raw_dataset_long_title_unzipped_folder_content_list), f"There is unmatching between tif images and other files for DLPFC dataset"
    return project_datasets

def DLPFC_select_data_single_helper(
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
        project_dataset: str,
) -> None:
    start_time = time.time()
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "downstream_task_data", "DLPFC")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "downstream_task_data", "DLPFC")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [title for title in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, title))]
    assert len(raw_data_folder_long_title_list) == 1, f"DLPFC dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_unzipped_folder_path = os.path.join(raw_data_folder_long_title_path, "10x_DLPFC")
    raw_data_folder_long_title_unzipped_folder_content_list = os.listdir(raw_data_folder_long_title_unzipped_folder_path)

    image, other_info = False, False
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    assert project_dataset in raw_data_folder_long_title_unzipped_folder_content_list
    raw_other_info_path = os.path.join(raw_data_folder_long_title_unzipped_folder_path, project_dataset)
    if not os.path.exists(project_dataset_raw_data_path):
        shutil.copytree(raw_other_info_path, project_dataset_raw_data_path)
    other_info = True
    raw_image_file = [file for file in raw_data_folder_long_title_content_list if (project_dataset + "_full_image.tif") in file]
    assert len(raw_image_file) == 1, f"More than one image file for DLPFC dataset {project_dataset}"
    raw_image_file = raw_image_file[0]
    raw_image_path = os.path.join(raw_data_folder_long_title_path, raw_image_file)
    project_image_path = os.path.join(project_dataset_raw_data_path, raw_image_file)
    if not os.path.exists(project_image_path):
        shutil.copy2(raw_image_path, project_image_path)
    image = True
    assert (image & other_info) == True, f"Missing filetype for DLPFC dataset {project_dataset}: {'image' if not image else ''} {'other_info' if not other_info else ''} is missing."

    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert "filtered_feature_bc_matrix" in project_dataset_raw_data_content_list, f"Missing matrix for DLPFC dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    double_unzip_tar_gz(
        project_dataset_raw_data_path=project_dataset_raw_data_path,
        project_dataset_raw_unzipped_data_path=project_dataset_raw_unzipped_data_path,
        filetype="matrix",
        already_in_folder_format=True,
        already_folder_name="filtered_feature_bc_matrix",
    )
    end_time = time.time()
    print(f"Selecting data for DLPFC dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def TCGA_BRCA_project_list(
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "downstream_task_data", "TCGA_BRCA")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "downstream_task_data", "TCGA_BRCA")
    raw_data_folder_content_list = os.listdir(raw_data_folder_path)
    project_datasets = [content for content in raw_data_folder_content_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    raw_data_folder_content_file_list = [content for content in raw_data_folder_content_list if os.path.isfile(os.path.join(raw_data_folder_path, content))]
    assert len(project_datasets) + len(raw_data_folder_content_file_list) == len(raw_data_folder_content_list), f"Unmatching between folders and files for TCGA_BRCA dataset"
    if not os.path.exists(project_data_folder_path):
        os.makedirs(project_data_folder_path)
    for file in raw_data_folder_content_file_list:
        raw_file_path = os.path.join(raw_data_folder_path, file)
        project_file_path = os.path.join(project_data_folder_path, file)
        if not os.path.exists(project_file_path):
            shutil.copy2(raw_file_path, project_file_path)
    return project_datasets

def TCGA_BRCA_select_data_single_helper(
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
        project_dataset: str,
) -> None:
    start_time = time.time()
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "downstream_task_data", "TCGA_BRCA")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "downstream_task_data", "TCGA_BRCA")
    patches_folder, counts, spots = False, False, False
    raw_dataset_path = os.path.join(raw_data_folder_path, project_dataset)
    raw_dataset_content_list = os.listdir(raw_dataset_path)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)
    for content in raw_dataset_content_list:
        if "patches_mag10_thre25" in content:
            raw_patches_folder_path = os.path.join(raw_dataset_path, content)
            project_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, content)
            if not os.path.exists(project_patches_folder_path):
                shutil.copytree(raw_patches_folder_path, project_patches_folder_path)
            patches_folder = True
        elif "augmented_star_gene_counts.tsv" in content:
            raw_counts_path = os.path.join(raw_dataset_path, content)
            project_counts_path = os.path.join(project_dataset_raw_data_path, content)
            if not os.path.exists(project_counts_path):
                shutil.copy2(raw_counts_path, project_counts_path)
            counts = True
        elif "spots_mag10_thre25.csv" in content:
            raw_spots_path = os.path.join(raw_dataset_path, content)
            project_spots_path = os.path.join(project_dataset_raw_data_path, content)
            if not os.path.exists(project_spots_path):
                shutil.copy2(raw_spots_path, project_spots_path)
            spots = True
    assert (patches_folder & counts & spots) == True, f"Missing filetype for TCGA_BRCA dataset {project_dataset}: {'patches_folder' if not patches_folder else ''} {'counts' if not counts else ''} {'spots' if not spots else ''} is missing."
    end_time = time.time()
    print(f"Selecting data for TCGA_BRCA dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def DRYAD001_project_list(
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "DRYAD", "DRYAD001")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "DRYAD", "DRYAD001")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"DRYAD001 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_zip_file_list = [content for content in raw_data_folder_long_title_content_list if content.endswith(".zip")]
    assert len(raw_data_folder_long_title_content_zip_file_list) == 1, f"DRYAD001 dataset has more than one zip file"
    raw_data_folder_long_title_content_zip_file_name = raw_data_folder_long_title_content_zip_file_list[0]
    raw_data_folder_long_title_content_zip_file_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_zip_file_name)
    unzip_zip(
        zip_path=raw_data_folder_long_title_content_zip_file_path,
        unzip_folder_path=os.path.join(raw_data_folder_long_title_path, "dryad"),
    )
    dryad001_zip_filename = "10XVisium_2.zip" # "10XVisium_2.zip" is the only relevant zip file inside the above unzipped folder
    dryad001_zip_basename, _ = os.path.splitext(dryad001_zip_filename)
    unzip_zip(
        zip_path=os.path.join(raw_data_folder_long_title_path, "dryad", dryad001_zip_filename),
        unzip_folder_path=os.path.join(raw_data_folder_long_title_path, "dryad", dryad001_zip_basename),
    )
    raw_dataset_path = os.path.join(raw_data_folder_long_title_path, "dryad", dryad001_zip_basename)
    raw_dataset_content_list = os.listdir(raw_dataset_path)
    raw_dataset_without_MAC_content_list = [content for content in raw_dataset_content_list if not content.startswith("__MACOSX")]
    assert len(raw_dataset_without_MAC_content_list) == 1, f"DRYAD001 dataset has more than one effective folder except 10XVisium 2"
    raw_dataset_without_MAC_name = raw_dataset_without_MAC_content_list[0]
    raw_dataset_without_MAC_path = os.path.join(raw_dataset_path, raw_dataset_without_MAC_name)
    raw_dataset_without_MAC_content_list = os.listdir(raw_dataset_without_MAC_path)
    raw_dataset_without_MAC_content_list = [content for content in raw_dataset_without_MAC_content_list if os.path.isdir(os.path.join(raw_dataset_without_MAC_path, content))]

    raw_dataset_with_HE_content_list = list()
    for content in raw_dataset_without_MAC_content_list:
        raw_dataset_content_path = os.path.join(raw_dataset_without_MAC_path, content)
        raw_dataset_content_list = os.listdir(raw_dataset_content_path)
        if ("H&E" in raw_dataset_content_list) and ("outs" in raw_dataset_content_list):
            raw_dataset_with_HE_content_list.append(content)

    # a specific outlier
    if "#UKF265_C_ST" in raw_dataset_with_HE_content_list:
        raw_dataset_with_HE_content_list.remove("#UKF265_C_ST")

    project_datasets = list()
    for content in raw_dataset_with_HE_content_list:
        raw_dataset_HE_path = os.path.join(raw_dataset_without_MAC_path, content, "H&E")
        raw_dataset_HE_content_list = os.listdir(raw_dataset_HE_path)
        raw_dataset_HE_image_file = [content for content in raw_dataset_HE_content_list if (".jpeg" in content) or (".TIF" in content) or (".jpg" in content) or (".tif" in content)]
        assert len(raw_dataset_HE_image_file) == 1, f"More than one image file for DRYAD001 dataset {content}"
        raw_dataset_outs_path = os.path.join(raw_dataset_without_MAC_path, content, "outs")
        raw_dataset_outs_content_list = os.listdir(raw_dataset_outs_path)
        assert ('filtered_feature_bc_matrix' in raw_dataset_outs_content_list) and ('spatial' in raw_dataset_outs_content_list), f"Missing filetype for DRYAD001 dataset {content}: {'matrix' if not ('filtered_feature_bc_matrix' in raw_dataset_outs_content_list) else ''} {'spatial' if not ('spatial' in raw_dataset_outs_content_list) else ''} is missing."
        project_datasets.append(content)
    return project_datasets

def DRYAD001_select_data_single_helper(
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
        project_dataset: str,
) -> None:
    start_time = time.time()
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "DRYAD", "DRYAD001")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "DRYAD", "DRYAD001")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"DRYAD001 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    dryad001_zip_filename = "10XVisium_2.zip" # "10XVisium_2.zip" is the only relevant zip file inside the above unzipped folder
    dryad001_zip_basename, _ = os.path.splitext(dryad001_zip_filename)
    raw_dataset_path = os.path.join(raw_data_folder_long_title_path, "dryad", dryad001_zip_basename)
    raw_dataset_content_list = os.listdir(raw_dataset_path)
    raw_dataset_without_MAC_content_list = [content for content in raw_dataset_content_list if not content.startswith("__MACOSX")]
    assert len(raw_dataset_without_MAC_content_list) == 1, f"DRYAD001 dataset has more than one effective folder except 10XVisium 2"
    raw_dataset_without_MAC_name = raw_dataset_without_MAC_content_list[0]
    raw_dataset_without_MAC_path = os.path.join(raw_dataset_path, raw_dataset_without_MAC_name)

    raw_dataset_path = os.path.join(raw_dataset_without_MAC_path, project_dataset)
    image, matrix, spatial = False, False, False
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    raw_dataset_HE_path = os.path.join(raw_dataset_path, "H&E")
    raw_dataset_HE_content_list = os.listdir(raw_dataset_HE_path)
    raw_dataset_HE_image_file = [content for content in raw_dataset_HE_content_list if (".jpeg" in content) or (".TIF" in content) or (".jpg" in content) or (".tif" in content)]
    assert len(raw_dataset_HE_image_file) == 1, f"More than one image file for DRYAD001 dataset {project_dataset}"
    raw_image_file = raw_dataset_HE_image_file[0]
    raw_image_path = os.path.join(raw_dataset_HE_path, raw_image_file)
    project_image_path = os.path.join(project_dataset_raw_data_path, raw_image_file)
    if not os.path.exists(project_image_path):
        shutil.copy2(raw_image_path, project_image_path)
    image = True

    raw_dataset_outs_path = os.path.join(raw_dataset_path, "outs")
    raw_dataset_outs_content_list = os.listdir(raw_dataset_outs_path)
    raw_dataset_filtered_feature_bc_matrix_folder = [content for content in raw_dataset_outs_content_list if os.path.isdir(os.path.join(raw_dataset_outs_path, content)) and ("filtered_feature_bc_matrix" in content)]
    assert len(raw_dataset_filtered_feature_bc_matrix_folder) == 1, f"More than one filtered_feature_bc_matrix folder for DRYAD001 dataset {project_dataset}"
    raw_matrix_folder = raw_dataset_filtered_feature_bc_matrix_folder[0]
    raw_matrix_path = os.path.join(raw_dataset_outs_path, raw_matrix_folder)
    project_matrix_path = os.path.join(project_dataset_raw_data_path, raw_matrix_folder)
    if not os.path.exists(project_matrix_path):
        shutil.copytree(raw_matrix_path, project_matrix_path)
    matrix = True

    raw_dataset_spatial_folder = [content for content in raw_dataset_outs_content_list if os.path.isdir(os.path.join(raw_dataset_outs_path, content)) and ("spatial" in content)]
    assert len(raw_dataset_spatial_folder) == 1, f"More than one spatial folder for DRYAD001 dataset {project_dataset}"
    raw_spatial_folder = raw_dataset_spatial_folder[0]
    raw_spatial_path = os.path.join(raw_dataset_outs_path, raw_spatial_folder)
    project_spatial_path = os.path.join(project_dataset_raw_data_path, raw_spatial_folder)
    if not os.path.exists(project_spatial_path):
        shutil.copytree(raw_spatial_path, project_spatial_path)
    spatial = True
    assert (image & matrix & spatial) == True, f"Missing filetype for DRYAD001 dataset {project_dataset}: {'image' if not image else ''} {'matrix' if not matrix else ''} {'spatial' if not spatial else ''} is missing."
    
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for DRYAD001 dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    for raw_data_item in project_dataset_raw_data_content_list:
        if "filtered_feature_bc_matrix" in raw_data_item:
            double_unzip_tar_gz(
                project_dataset_raw_data_path=project_dataset_raw_data_path,
                project_dataset_raw_unzipped_data_path=project_dataset_raw_unzipped_data_path,
                filetype="matrix",
                already_in_folder_format=True,
                already_folder_name="filtered_feature_bc_matrix",
            )
    end_time = time.time()
    print(f"Selecting data for DRYAD001 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def HCA001_project_list(
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Human_cell_atlas_human", "Human_cell_atlas001")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "Human_cell_atlas", "Human_cell_atlas001")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"Human_cell_atlas001 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_folder_list = [content for content in raw_data_folder_long_title_content_list if os.path.isdir(os.path.join(raw_data_folder_long_title_path, content))]
    raw_data_folder_long_title_pure_raw_path = os.path.join(raw_data_folder_long_title_path, "Pure_Raw_Direct_Downloads")
    raw_data_folder_long_title_gathering_path = os.path.join(raw_data_folder_long_title_path, "Files_Gathering")
    if (not os.path.exists(raw_data_folder_long_title_pure_raw_path)) and (not os.path.exists(raw_data_folder_long_title_gathering_path)):
        os.makedirs(raw_data_folder_long_title_pure_raw_path)
        os.makedirs(raw_data_folder_long_title_gathering_path)
        for folder in raw_data_folder_long_title_content_folder_list:
            folder_path = os.path.join(raw_data_folder_long_title_path, folder)
            folder_content_list = os.listdir(folder_path)
            assert len(folder_content_list) == 1, f"More than one file in folder {folder}"
            folder_content = folder_content_list[0]
            raw_original_path = os.path.join(raw_data_folder_long_title_pure_raw_path, folder_content)
            raw_gathering_path = os.path.join(raw_data_folder_long_title_gathering_path, folder_content)
            if not os.path.exists(raw_gathering_path):
                shutil.copy2(raw_original_path, raw_gathering_path)
                shutil.move(folder_path, raw_data_folder_long_title_pure_raw_path)
    raw_data_folder_long_title_pure_raw_content_list = os.listdir(raw_data_folder_long_title_pure_raw_path)
    raw_data_folder_long_title_gathering_content_list = os.listdir(raw_data_folder_long_title_gathering_path)
    assert len(raw_data_folder_long_title_pure_raw_content_list) == len(raw_data_folder_long_title_gathering_content_list) == 28, f"Unmatching between pure raw and gathering files for Human_cell_atlas001 dataset"

    manual_image_pcw_h5ad_mapping = HCA001_manual_image_pcw_h5ad_mapping()
    project_datasets = list()
    for image_file in list(manual_image_pcw_h5ad_mapping.keys()):
        h5ad_file_name = [file for file in manual_image_pcw_h5ad_mapping[image_file] if "STDY" in file]
        assert len(h5ad_file_name) == 1, f"More than one h5ad file for image {image_file} in Human_cell_atlas001 dataset"
        h5ad_file_name = h5ad_file_name[0]
        project_datasets.append(h5ad_file_name)
    return project_datasets

def HCA001_select_data_single_helper(
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
        project_dataset: str,
) -> None:
    start_time = time.time()
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Human_cell_atlas_human", "Human_cell_atlas001")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "Human_cell_atlas", "Human_cell_atlas001")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"Human_cell_atlas001 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_gathering_path = os.path.join(raw_data_folder_long_title_path, "Files_Gathering")
    manual_image_pcw_h5ad_mapping = HCA001_manual_image_pcw_h5ad_mapping()

    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    image, h5ad= False, False
    for key, value in manual_image_pcw_h5ad_mapping.items():
        if value[1] == project_dataset:
            image_file = key
            break
        else:
            image_file = None
    assert image_file is not None, f"Missing image file for HCA001 dataset {project_dataset}"

    raw_image_path = os.path.join(raw_data_folder_long_title_gathering_path, image_file)
    assert os.path.exists(raw_image_path), f"Missing image path for HCA001 dataset {project_dataset}"
    project_image_path = os.path.join(project_dataset_raw_data_path, image_file)
    if not os.path.exists(project_image_path):
        shutil.copy2(raw_image_path, project_image_path)
    image = True

    raw_h5ad_path = os.path.join(raw_data_folder_long_title_gathering_path, project_dataset + ".h5ad")
    assert os.path.exists(raw_h5ad_path), f"Missing h5ad path for HCA001 dataset {project_dataset}"
    project_h5ad_path = os.path.join(project_dataset_raw_data_path, project_dataset + ".h5ad")
    if not os.path.exists(project_h5ad_path):
        shutil.copy2(raw_h5ad_path, project_h5ad_path)
    h5ad = True
    assert (image & h5ad) == True, f"Missing filetype for HCA001 dataset {project_dataset}: {'image' if not image else ''} {'h5ad' if not h5ad else ''} is missing."

    end_time = time.time()
    print(f"Selecting data for HCA001 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def Mendeley_data001_Patient_1_1k_project_list(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Mendeley_data", "Mendeley_data001")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "Mendeley_data", "Mendeley_data001")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"Mendeley_data001 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_zip_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".zip")]
    assert len(raw_data_folder_long_title_content_zip_file) == 1, f"Mendeley_data001 dataset has more than one zip file"
    raw_data_folder_long_title_content_zip_name = raw_data_folder_long_title_content_zip_file[0]
    raw_data_folder_long_title_content_zip_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_zip_name)
    unzip_zip(
        zip_path=raw_data_folder_long_title_content_zip_path,
        unzip_folder_path=raw_data_folder_long_title_path,
    )
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_list = [content for content in raw_data_folder_long_title_content_list if os.path.isdir(os.path.join(raw_data_folder_long_title_path, content))]
    assert len(raw_data_folder_long_title_content_list) == 1, f"Mendeley_data001 dataset has more than one effective folder"
    raw_data_folder_long_title_content_name = raw_data_folder_long_title_content_list[0]
    raw_data_folder_long_title_content_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_name)

    raw_data_folder_long_title_Patient_1_1k_matrices_path = os.path.join(raw_data_folder_long_title_content_path, "Count_matrices", "Patient 1", "1k_arrays")
    raw_data_folder_long_title_Patient_1_1k_images_path = os.path.join(raw_data_folder_long_title_content_path, "Histological_images", "Patient_1", "1k-array")
    raw_data_folder_long_title_Patient_1_1k_matrices_content_list = os.listdir(raw_data_folder_long_title_Patient_1_1k_matrices_path)
    raw_data_folder_long_title_Patient_1_1k_images_content_list = os.listdir(raw_data_folder_long_title_Patient_1_1k_images_path)
    assert len(raw_data_folder_long_title_Patient_1_1k_matrices_content_list) == len(raw_data_folder_long_title_Patient_1_1k_images_content_list), f"Unmatching between the number of files and the expected number of files for Mendeley_data001 dataset Patient 1 1k arrays"
    project_datasets = raw_data_folder_long_title_Patient_1_1k_matrices_content_list
    WQ_adjusted_coordinate_path = os.path.join(main_data_storage, raw_data_folder_name, "Supplementary_data", "Mendeley_data001_Patient_1_1k_WQ_coordinates")
    WQ_adjusted_coordinate_content_list = os.listdir(WQ_adjusted_coordinate_path)
    assert len(WQ_adjusted_coordinate_content_list) == 2 * len(project_datasets), f"Unmatching between the number of files and the expected number of files for Mendeley_data001 dataset Patient 1 1k arrays"
    return project_datasets

def Mendeley_data001_Patient_1_1k_single_helper(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time() 
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Mendeley_data", "Mendeley_data001")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "Mendeley_data", "Mendeley_data001")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"Mendeley_data001 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_list = [content for content in raw_data_folder_long_title_content_list if os.path.isdir(os.path.join(raw_data_folder_long_title_path, content))]
    assert len(raw_data_folder_long_title_content_list) == 1, f"Mendeley_data001 dataset has more than one effective folder"
    raw_data_folder_long_title_content_name = raw_data_folder_long_title_content_list[0]
    raw_data_folder_long_title_content_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_name)

    raw_data_folder_long_title_Patient_1_1k_matrices_path = os.path.join(raw_data_folder_long_title_content_path, "Count_matrices", "Patient 1", "1k_arrays")
    raw_data_folder_long_title_Patient_1_1k_images_path = os.path.join(raw_data_folder_long_title_content_path, "Histological_images", "Patient_1", "1k-array")
    raw_data_folder_long_title_Patient_1_1k_matrices_content_list = os.listdir(raw_data_folder_long_title_Patient_1_1k_matrices_path)
    raw_data_folder_long_title_Patient_1_1k_images_content_list = os.listdir(raw_data_folder_long_title_Patient_1_1k_images_path)
    assert len(raw_data_folder_long_title_Patient_1_1k_matrices_content_list) == len(raw_data_folder_long_title_Patient_1_1k_images_content_list), f"Unmatching between the number of files and the expected number of files for Mendeley_data001 dataset Patient 1 1k arrays"
    project_datasets = raw_data_folder_long_title_Patient_1_1k_matrices_content_list
    WQ_adjusted_coordinate_path = os.path.join(main_data_storage, raw_data_folder_name, "Supplementary_data", "Mendeley_data001_Patient_1_1k_WQ_coordinates")
    WQ_adjusted_coordinate_content_list = os.listdir(WQ_adjusted_coordinate_path)
    assert len(WQ_adjusted_coordinate_content_list) == 2 * len(project_datasets), f"Unmatching between the number of files and the expected number of files for Mendeley_data001 dataset Patient 1 1k arrays"
       
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, "Patient_1_1k_array_" + project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    image, stdata, spots = False, False, False
    
    raw_image_file = [image for image in raw_data_folder_long_title_Patient_1_1k_images_content_list if project_dataset in image]
    assert len(raw_image_file) == 1, f"More than one image file for Mendeley_data001 dataset Patient 1 1k arrays {project_dataset}"
    raw_image_file = raw_image_file[0]
    raw_image_path = os.path.join(raw_data_folder_long_title_Patient_1_1k_images_path, raw_image_file)
    project_image_path = os.path.join(project_dataset_raw_data_path, raw_image_file)
    if not os.path.exists(project_image_path):
        shutil.copy2(raw_image_path, project_image_path)
    image = True
    
    raw_matrices_folder = [content for content in raw_data_folder_long_title_Patient_1_1k_matrices_content_list if project_dataset in content]
    assert len(raw_matrices_folder) == 1, f"More than one matrix folder for Mendeley_data001 dataset Patient 1 1k arrays {project_dataset}"
    raw_matrices_folder = raw_matrices_folder[0]
    raw_matrices_folder_path = os.path.join(raw_data_folder_long_title_Patient_1_1k_matrices_path, raw_matrices_folder)
    raw_matrices_content_list = os.listdir(raw_matrices_folder_path)
    assert len(raw_matrices_content_list) == 2, f"Unmatching between the number of files and the expected number of files for Mendeley_data001 dataset Patient 1 1k arrays {project_dataset}"
    stdata_file = [content for content in raw_matrices_content_list if "stdata" in content]
    assert len(stdata_file) == 1, f"More than one stdata file for Mendeley_data001 dataset Patient 1 1k arrays {project_dataset}"
    stdata_file = stdata_file[0]
    raw_stdata_path = os.path.join(raw_matrices_folder_path, stdata_file)
    project_stdata_path = os.path.join(project_dataset_raw_data_path, stdata_file)
    if not os.path.exists(project_stdata_path):
        shutil.copy2(raw_stdata_path, project_stdata_path)
    stdata = True
    
    spots_file = [content for content in WQ_adjusted_coordinate_content_list if (project_dataset in content) and ("spot_label" not in content)]
    assert len(spots_file) == 1, f"More than one spots file for Mendeley_data001 dataset Patient 1 1k arrays {project_dataset}"
    spots_file = spots_file[0]
    raw_spots_path = os.path.join(WQ_adjusted_coordinate_path, spots_file)
    project_spots_path = os.path.join(project_dataset_raw_data_path, spots_file)
    if not os.path.exists(project_spots_path):
        shutil.copy2(raw_spots_path, project_spots_path)
    spots = True
    assert (image & stdata & spots) == True, f"Missing filetype for Mendeley_data001 dataset Patient 1 1k arrays {project_dataset}: {'image' if not image else ''} {'stdata' if not stdata else ''} {'spots' if not spots else ''} is missing."
    end_time = time.time()
    print(f"Selecting data for Mendeley_data001 dataset Patient 1 1k arrays {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def Mendeley_data001_Patient_1_Visium_project_list(
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Mendeley_data", "Mendeley_data001")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "Mendeley_data", "Mendeley_data001")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"Mendeley_data001 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_zip_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".zip")]
    assert len(raw_data_folder_long_title_content_zip_file) == 1, f"Mendeley_data001 dataset has more than one zip file"
    raw_data_folder_long_title_content_zip_name = raw_data_folder_long_title_content_zip_file[0]
    raw_data_folder_long_title_content_zip_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_zip_name)
    unzip_zip(
        zip_path=raw_data_folder_long_title_content_zip_path,
        unzip_folder_path=raw_data_folder_long_title_path,
    )
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_list = [content for content in raw_data_folder_long_title_content_list if os.path.isdir(os.path.join(raw_data_folder_long_title_path, content))]
    assert len(raw_data_folder_long_title_content_list) == 1, f"Mendeley_data001 dataset has more than one effective folder"
    raw_data_folder_long_title_content_name = raw_data_folder_long_title_content_list[0]
    raw_data_folder_long_title_content_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_name)

    raw_data_folder_long_title_Patient_1_matrices_path = os.path.join(raw_data_folder_long_title_content_path, "Count_matrices", "Patient 1", "Visium_with_annotation")
    raw_data_folder_long_title_Patient_1_matrices_content_list = os.listdir(raw_data_folder_long_title_Patient_1_matrices_path)
    project_datasets = raw_data_folder_long_title_Patient_1_matrices_content_list
    return project_datasets

def Mendeley_data001_Patient_1_Visium_single_helper(
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
        project_dataset: str,
) -> None:
    start_time = time.time()

    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Mendeley_data", "Mendeley_data001")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "Mendeley_data", "Mendeley_data001")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"Mendeley_data001 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_list = [content for content in raw_data_folder_long_title_content_list if os.path.isdir(os.path.join(raw_data_folder_long_title_path, content))]
    assert len(raw_data_folder_long_title_content_list) == 1, f"Mendeley_data001 dataset has more than one effective folder"
    raw_data_folder_long_title_content_name = raw_data_folder_long_title_content_list[0]
    raw_data_folder_long_title_content_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_name)

    raw_data_folder_long_title_Patient_1_matrices_path = os.path.join(raw_data_folder_long_title_content_path, "Count_matrices", "Patient 1", "Visium_with_annotation")
    raw_data_folder_long_title_Patient_1_images_path = os.path.join(raw_data_folder_long_title_content_path, "Histological_images", "Patient_1", "Visium")
    raw_data_folder_long_title_Patient_1_images_content_list = os.listdir(raw_data_folder_long_title_Patient_1_images_path)

    project_dataset_raw_data_path = os.path.join(project_data_folder_path, "Patient_1_Visium_" + project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    image, matrix = False, False

    raw_image_file = [image for image in raw_data_folder_long_title_Patient_1_images_content_list if project_dataset in image]
    assert len(raw_image_file) == 1, f"More than one image file for Mendeley_data001 dataset Patient 1 Visium {project_dataset}"
    raw_image_name = raw_image_file[0]
    raw_image_path = os.path.join(raw_data_folder_long_title_Patient_1_images_path, raw_image_name)
    project_image_path = os.path.join(project_dataset_raw_data_path, raw_image_name)
    if not os.path.exists(project_image_path):
        shutil.copy2(raw_image_path, project_image_path)
    image = True

    raw_matrices_folder_path = os.path.join(raw_data_folder_long_title_Patient_1_matrices_path, project_dataset)
    raw_matrices_content_list = os.listdir(raw_matrices_folder_path)
    assert len(raw_matrices_content_list) == 4, f" Unmatching between the number of files and the expected number of files for Mendeley_data001 dataset Patient 1 Visium {project_dataset}"
    for matrix in raw_matrices_content_list:
        raw_matrix_path = os.path.join(raw_matrices_folder_path, matrix)
        project_matrix_path = os.path.join(project_dataset_raw_data_path, matrix)
        if not os.path.exists(project_matrix_path):
            shutil.copy2(raw_matrix_path, project_matrix_path)
    matrix = True
    assert (image & matrix) == True, f"Missing filetype for Mendeley_data001 dataset Patient 1 Visium {project_dataset}: {'image' if not image else ''} {'matrix' if not matrix else ''} is missing."

    end_time = time.time()
    print(f"Selecting data for Mendeley_data001 dataset Patient 1 Visium {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def Mendeley_data001_Patient_2_project_list(
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Mendeley_data", "Mendeley_data001")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "Mendeley_data", "Mendeley_data001")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"Mendeley_data001 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_zip_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".zip")]
    assert len(raw_data_folder_long_title_content_zip_file) == 1, f"Mendeley_data001 dataset has more than one zip file"
    raw_data_folder_long_title_content_zip_name = raw_data_folder_long_title_content_zip_file[0]
    raw_data_folder_long_title_content_zip_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_zip_name)
    unzip_zip(
        zip_path=raw_data_folder_long_title_content_zip_path,
        unzip_folder_path=raw_data_folder_long_title_path,
    )
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_list = [content for content in raw_data_folder_long_title_content_list if os.path.isdir(os.path.join(raw_data_folder_long_title_path, content))]
    assert len(raw_data_folder_long_title_content_list) == 1, f"Mendeley_data001 dataset has more than one effective folder"
    raw_data_folder_long_title_content_name = raw_data_folder_long_title_content_list[0]
    raw_data_folder_long_title_content_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_name)

    raw_data_folder_long_title_Patient_2_matrices_path = os.path.join(raw_data_folder_long_title_content_path, "Count_matrices", "Patient 2")
    raw_data_folder_long_title_Patient_2_images_path = os.path.join(raw_data_folder_long_title_content_path, "Histological_images", "Patient_2")
    raw_data_folder_long_title_Patient_2_matrices_content_list = os.listdir(raw_data_folder_long_title_Patient_2_matrices_path)
    raw_data_folder_long_title_Patient_2_images_content_list = os.listdir(raw_data_folder_long_title_Patient_2_images_path)
    project_datasets = raw_data_folder_long_title_Patient_2_matrices_content_list
    return project_datasets

def Mendeley_data001_Patient_2_single_helper(
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
        project_dataset: str,
) -> None:
    start_time = time.time()

    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Mendeley_data", "Mendeley_data001")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "Mendeley_data", "Mendeley_data001")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"Mendeley_data001 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_list = [content for content in raw_data_folder_long_title_content_list if os.path.isdir(os.path.join(raw_data_folder_long_title_path, content))]
    assert len(raw_data_folder_long_title_content_list) == 1, f"Mendeley_data001 dataset has more than one effective folder"
    raw_data_folder_long_title_content_name = raw_data_folder_long_title_content_list[0]
    raw_data_folder_long_title_content_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_name)

    raw_data_folder_long_title_Patient_2_matrices_path = os.path.join(raw_data_folder_long_title_content_path, "Count_matrices", "Patient 2")
    raw_data_folder_long_title_Patient_2_images_path = os.path.join(raw_data_folder_long_title_content_path, "Histological_images", "Patient_2")
    raw_data_folder_long_title_Patient_2_images_content_list = os.listdir(raw_data_folder_long_title_Patient_2_images_path)

    project_dataset_raw_data_path = os.path.join(project_data_folder_path, "Patient_2_" + project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    image, matrix = False, False

    raw_image_file = [image for image in raw_data_folder_long_title_Patient_2_images_content_list if project_dataset in image]
    assert len(raw_image_file) == 1, f"More than one image file for Mendeley_data001 dataset Patient 2 {project_dataset}"
    raw_image_name = raw_image_file[0]
    raw_image_path = os.path.join(raw_data_folder_long_title_Patient_2_images_path, raw_image_name)
    project_image_path = os.path.join(project_dataset_raw_data_path, raw_image_name)
    if not os.path.exists(project_image_path):
        shutil.copy2(raw_image_path, project_image_path)
    image = True

    raw_matrices_folder_path = os.path.join(raw_data_folder_long_title_Patient_2_matrices_path, project_dataset)
    raw_matrices_content_list = os.listdir(raw_matrices_folder_path)
    assert len(raw_matrices_content_list) == 3, f" Unmatching between the number of files and the expected number of files for Mendeley_data001 dataset Patient 2 {project_dataset}"
    for matrix in raw_matrices_content_list:
        raw_matrix_path = os.path.join(raw_matrices_folder_path, matrix)
        project_matrix_path = os.path.join(project_dataset_raw_data_path, matrix)
        if not os.path.exists(project_matrix_path):
            shutil.copy2(raw_matrix_path, project_matrix_path)
    matrix = True
    assert (image & matrix) == True, f"Missing filetype for Mendeley_data001 dataset Patient 2 {project_dataset}: {'image' if not image else ''} {'matrix' if not matrix else ''} is missing."

    end_time = time.time()
    print(f"Selecting data for Mendeley_data001 dataset Patient 2 {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def Mendeley_data002_project_list(
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Mendeley_data", "Mendeley_data002")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "Mendeley_data", "Mendeley_data002")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"Mendeley_data002 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_zip_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".zip")]
    assert len(raw_data_folder_long_title_content_zip_file) == 1, f"Mendeley_data002 dataset has more than one zip file"
    raw_data_folder_long_title_content_zip_name = raw_data_folder_long_title_content_zip_file[0]
    raw_data_folder_long_title_content_zip_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_zip_name)
    unzip_zip(
        zip_path=raw_data_folder_long_title_content_zip_path,
        unzip_folder_path=raw_data_folder_long_title_path,
    )
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_list = [content for content in raw_data_folder_long_title_content_list if os.path.isdir(os.path.join(raw_data_folder_long_title_path, content))]
    assert len(raw_data_folder_long_title_content_list) == 4, f"Mendeley_data002 dataset has more than four effective folder"
    project_datasets = raw_data_folder_long_title_content_list
    return project_datasets

def Mendeley_data002_single_helper(
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
        project_dataset: str,
) -> None:
    start_time = time.time()
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Mendeley_data", "Mendeley_data002")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "Mendeley_data", "Mendeley_data002")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"Mendeley_data002 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)

    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    raw_dataset_path = os.path.join(raw_data_folder_long_title_path, project_dataset)
    raw_dataset_content_list = os.listdir(raw_dataset_path)
    image, matrix, spatial = False, False, False

    for content in raw_dataset_content_list:
        if ".jpg" in content:
            raw_image_path = os.path.join(raw_dataset_path, content)
            project_image_path = os.path.join(project_dataset_raw_data_path, content)
            if not os.path.exists(project_image_path):
                shutil.copy2(raw_image_path, project_image_path)
            image = True
        elif "filtered_feature_bc_matrix" in content:
            raw_matrix_path = os.path.join(raw_dataset_path, content)
            project_matrix_path = os.path.join(project_dataset_raw_data_path, content)
            if not os.path.exists(project_matrix_path):
                shutil.copy2(raw_matrix_path, project_matrix_path)
            matrix = True
        elif "spatial" in content:
            raw_spatial_folder_path = os.path.join(raw_dataset_path, content)
            project_spatial_folder_path = os.path.join(project_dataset_raw_data_path, content)
            if not os.path.exists(project_spatial_folder_path):
                shutil.copytree(raw_spatial_folder_path, project_spatial_folder_path)
            spatial = True
    assert (image & matrix & spatial) == True, f"Missing filetype for Mendeley_data002 dataset {project_dataset}: {'image' if not image else ''} {'matrix' if not matrix else ''} {'spatial' if not spatial else ''} is missing."

    end_time = time.time()
    print(f"Selecting data for Mendeley_data002 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def Mendeley_data003_project_list(
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Mendeley_data", "Mendeley_data003")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "Mendeley_data", "Mendeley_data003")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"Mendeley_data003 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_zip_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".zip")]
    assert len(raw_data_folder_long_title_content_zip_file) == 1, f"Mendeley_data003 dataset has more than one zip file"
    raw_data_folder_long_title_content_zip_name = raw_data_folder_long_title_content_zip_file[0]
    raw_data_folder_long_title_content_zip_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_zip_name)
    unzip_zip(
        zip_path=raw_data_folder_long_title_content_zip_path,
        unzip_folder_path=raw_data_folder_long_title_path,
    )
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_list = [content for content in raw_data_folder_long_title_content_list if os.path.isdir(os.path.join(raw_data_folder_long_title_path, content))]
    assert len(raw_data_folder_long_title_content_list) == 4, f"Mendeley_data003 dataset has more than four effective folder"
    project_datasets = raw_data_folder_long_title_content_list
    return project_datasets

def Mendeley_data003_single_helper(
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
        project_dataset: str,
) -> None:
    start_time = time.time()
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Mendeley_data", "Mendeley_data003")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "Mendeley_data", "Mendeley_data003")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"Mendeley_data003 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)

    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    raw_dataset_path = os.path.join(raw_data_folder_long_title_path, project_dataset)
    raw_dataset_content_list = os.listdir(raw_dataset_path)
    image, matrix, spatial = False, False, False

    for content in raw_dataset_content_list:
        if ".jpg" in content:
            raw_image_path = os.path.join(raw_dataset_path, content)
            project_image_path = os.path.join(project_dataset_raw_data_path, content)
            if not os.path.exists(project_image_path):
                shutil.copy2(raw_image_path, project_image_path)
            image = True
        elif "filtered_feature_bc_matrix" in content:
            raw_matrix_path = os.path.join(raw_dataset_path, content)
            project_matrix_path = os.path.join(project_dataset_raw_data_path, content)
            if not os.path.exists(project_matrix_path):
                shutil.copy2(raw_matrix_path, project_matrix_path)
            matrix = True
        elif "spatial" in content:
            raw_spatial_folder_path = os.path.join(raw_dataset_path, content)
            project_spatial_folder_path = os.path.join(project_dataset_raw_data_path, content)
            if not os.path.exists(project_spatial_folder_path):
                shutil.copytree(raw_spatial_folder_path, project_spatial_folder_path)
            spatial = True
    assert (image & matrix & spatial) == True, f"Missing filetype for Mendeley_data003 dataset {project_dataset}: {'image' if not image else ''} {'matrix' if not matrix else ''} {'spatial' if not spatial else ''} is missing."

    end_time = time.time()
    print(f"Selecting data for Mendeley_data003 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def NCBI001_project_list(
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "NCBI", "NCBI001")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "NCBI", "NCBI001")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"NCBI001 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_tar_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".tar")]
    assert len(raw_data_folder_long_title_content_tar_file) == 1, f"NCBI001 dataset has more than one tar file"
    raw_data_folder_long_title_content_tar_name = raw_data_folder_long_title_content_tar_file[0]
    raw_data_folder_long_title_content_tar_basename, _ = os.path.splitext(raw_data_folder_long_title_content_tar_name)
    raw_data_folder_long_title_content_tar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_name)
    raw_data_folder_long_title_content_unzipped_tar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_basename)
    unzip_tar(
        tar_path=raw_data_folder_long_title_content_tar_path,
        untar_folder_path=raw_data_folder_long_title_content_unzipped_tar_path,
    )

    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_images_list = [content for content in raw_data_folder_long_title_content_list if content.endswith(".tif")]
    raw_data_folder_long_title_content_folder_list = [content for content in raw_data_folder_long_title_content_list if os.path.isdir(os.path.join(raw_data_folder_long_title_path, content))]
    assert len(raw_data_folder_long_title_content_images_list) == 2, f"NCBI001 dataset has more than two image file"
    assert len(raw_data_folder_long_title_content_folder_list) == 1, f"NCBI001 dataset has more than one effective folder"
    project_datasets = [(image.split("-")[2]).split(".")[0] for image in raw_data_folder_long_title_content_images_list]
    raw_data_folder_long_title_content_folder_name = raw_data_folder_long_title_content_folder_list[0]
    raw_data_folder_long_title_content_folder_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_folder_name)
    raw_data_folder_long_title_content_folder_content_list = os.listdir(raw_data_folder_long_title_content_folder_path)
    return project_datasets

def NCBI001_single_helper(
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
        project_dataset: str,
) -> None:
    start_time = time.time()
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "NCBI", "NCBI001")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "NCBI", "NCBI001")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"NCBI001 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_images_list = [content for content in raw_data_folder_long_title_content_list if content.endswith(".tif")]
    raw_data_folder_long_title_content_folder_list = [content for content in raw_data_folder_long_title_content_list if os.path.isdir(os.path.join(raw_data_folder_long_title_path, content))]
    assert len(raw_data_folder_long_title_content_images_list) == 2, f"NCBI001 dataset has more than two image file"
    assert len(raw_data_folder_long_title_content_folder_list) == 1, f"NCBI001 dataset has more than one effective folder"
    raw_data_folder_long_title_content_folder_name = raw_data_folder_long_title_content_folder_list[0]
    raw_data_folder_long_title_content_folder_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_folder_name)
    raw_data_folder_long_title_content_folder_content_list = os.listdir(raw_data_folder_long_title_content_folder_path)

    if "AH" in project_dataset:
        patient_num = 1
    elif "AJ" in project_dataset:
        patient_num = 2
    else:
        raise ValueError(f"Unrecognized patient number for NCBI001 dataset {project_dataset}")
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    image, matrix, spatial = False, False, False

    for image_file in raw_data_folder_long_title_content_images_list:
        if project_dataset in image_file:
            raw_image_path = os.path.join(raw_data_folder_long_title_path, image_file)
            project_image_path = os.path.join(project_dataset_raw_data_path, image_file)
            if not os.path.exists(project_image_path):
                shutil.copy2(raw_image_path, project_image_path)
            image = True
    matrix_patient_name = "patient_" + str(patient_num)
    spatial_patient_name = "patient" + str(patient_num)
    for raw_data_item in raw_data_folder_long_title_content_folder_content_list:
        if (matrix_patient_name in raw_data_item) and (raw_data_item.endswith(".h5")):
            raw_matrix_path = os.path.join(raw_data_folder_long_title_content_folder_path, raw_data_item)
            project_matrix_path = os.path.join(project_dataset_raw_data_path, raw_data_item)
            if not os.path.exists(project_matrix_path):
                shutil.copy2(raw_matrix_path, project_matrix_path)
            matrix = True
        if (spatial_patient_name in raw_data_item) and ("spatial" in raw_data_item):
            raw_spatial_path = os.path.join(raw_data_folder_long_title_content_folder_path, raw_data_item)
            project_spatial_path = os.path.join(project_dataset_raw_data_path, raw_data_item)
            if not os.path.exists(project_spatial_path):
                shutil.copy2(raw_spatial_path, project_spatial_path)
            spatial = True
    assert (image & matrix & spatial) == True, f"Missing filetype for NCBI001 dataset {project_dataset}: {'image' if not image else ''} {'matrix' if not matrix else ''} {'spatial' if not spatial else ''} is missing."

    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for NCBI001 dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    if not os.path.exists(project_dataset_raw_unzipped_data_path):
        os.makedirs(project_dataset_raw_unzipped_data_path)
    for content in project_dataset_raw_data_content_list:
        if content.endswith(".tar.gz"):
            tar_gz_path = os.path.join(project_dataset_raw_data_path, content)
            double_unzip_tar_gz(
                project_dataset_raw_data_path=project_dataset_raw_data_path,
                project_dataset_raw_unzipped_data_path=project_dataset_raw_unzipped_data_path,
                filetype="spatial",
                already_in_folder_format=False,
                tar_gz_path=tar_gz_path
            )
    end_time = time.time()
    print(f"Selecting data for NCBI001 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def NCBI002_project_list(
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "NCBI", "NCBI002")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "NCBI", "NCBI002")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"NCBI001 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_tar_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".tar")]
    assert len(raw_data_folder_long_title_content_tar_file) == 1, f"NCBI002 dataset has more than one tar file"
    raw_data_folder_long_title_content_tar_name = raw_data_folder_long_title_content_tar_file[0]
    raw_data_folder_long_title_content_tar_basename, _ = os.path.splitext(raw_data_folder_long_title_content_tar_name)
    raw_data_folder_long_title_content_tar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_name)
    raw_data_folder_long_title_content_unzipped_tar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_basename)
    unzip_tar(
        tar_path=raw_data_folder_long_title_content_tar_path,
        untar_folder_path=raw_data_folder_long_title_content_unzipped_tar_path,
    )

    raw_data_folder_long_title_content_unzipped_tar_content_list = os.listdir(raw_data_folder_long_title_content_unzipped_tar_path)
    raw_data_folder_GSM_code_list = [content.split("_")[0] for content in raw_data_folder_long_title_content_unzipped_tar_content_list]
    raw_data_folder_GSM_code_list = list(set(raw_data_folder_GSM_code_list))
    project_datasets = list()
    for GSM_code in raw_data_folder_GSM_code_list:
        GSM_code_file = [content for content in raw_data_folder_long_title_content_unzipped_tar_content_list if GSM_code in content]
        GSM_code_image_file = [content for content in GSM_code_file if ".jpg" in content]
        if len(GSM_code_image_file) > 0:
            project_datasets.append(GSM_code)
    return project_datasets

def NCBI002_single_helper(
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
        project_dataset: str,
) -> None:
    start_time = time.time()
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "NCBI", "NCBI002")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "NCBI", "NCBI002")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"NCBI001 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_tar_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".tar")]
    assert len(raw_data_folder_long_title_content_tar_file) == 1, f"NCBI002 dataset has more than one tar file"
    raw_data_folder_long_title_content_tar_name = raw_data_folder_long_title_content_tar_file[0]
    raw_data_folder_long_title_content_tar_basename, _ = os.path.splitext(raw_data_folder_long_title_content_tar_name)
    raw_data_folder_long_title_content_unzipped_tar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_basename)
    raw_data_folder_long_title_content_unzipped_tar_content_list = os.listdir(raw_data_folder_long_title_content_unzipped_tar_path)

    raw_dataset_content_list = [content for content in raw_data_folder_long_title_content_unzipped_tar_content_list if project_dataset in content]
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    image, stdata, spot_data = False, False, False

    for raw_data_item in raw_dataset_content_list:
        if ".jpg" in raw_data_item:
            raw_image_path = os.path.join(raw_data_folder_long_title_content_unzipped_tar_path, raw_data_item)
            project_image_path = os.path.join(project_dataset_raw_data_path, raw_data_item)
            if not os.path.exists(project_image_path):
                shutil.copy2(raw_image_path, project_image_path)
            image = True
        elif "stdata" in raw_data_item:
            raw_stdata_path = os.path.join(raw_data_folder_long_title_content_unzipped_tar_path, raw_data_item)
            project_stdata_path = os.path.join(project_dataset_raw_data_path, raw_data_item)
            if not os.path.exists(project_stdata_path):
                shutil.copy2(raw_stdata_path, project_stdata_path)
            stdata = True
        elif "spot_data" in raw_data_item:
            raw_spots_path = os.path.join(raw_data_folder_long_title_content_unzipped_tar_path, raw_data_item)
            project_spots_path = os.path.join(project_dataset_raw_data_path, raw_data_item)
            if not os.path.exists(project_spots_path):
                shutil.copy2(raw_spots_path, project_spots_path)
            spot_data = True
    assert (image & stdata & spot_data) == True, f"Missing filetype for NCBI002 dataset {project_dataset}: {'image' if not image else ''} {'stdata' if not stdata else ''} {'spot_data' if not spot_data else ''} is missing."

    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for NCBI002 dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    if not os.path.exists(project_dataset_raw_unzipped_data_path):
        os.makedirs(project_dataset_raw_unzipped_data_path)
    for content in project_dataset_raw_data_content_list:
        gz_path = os.path.join(project_dataset_raw_data_path, content)
        unzip_gz(
            gz_path=gz_path,
            ungz_folder_path=project_dataset_raw_unzipped_data_path,
        )
    
    end_time = time.time()
    print(f"Selecting data for NCBI002 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def NCBI003_project_list(
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "NCBI", "NCBI003")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "NCBI", "NCBI003")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"NCBI003 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_tar_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".tar")]
    assert len(raw_data_folder_long_title_content_tar_file) == 1, f"NCBI003 dataset has more than one tar file"
    raw_data_folder_long_title_content_tar_name = raw_data_folder_long_title_content_tar_file[0]
    raw_data_folder_long_title_content_tar_basename, _ = os.path.splitext(raw_data_folder_long_title_content_tar_name)
    raw_data_folder_long_title_content_tar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_name)
    raw_data_folder_long_title_content_unzipped_tar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_basename)
    unzip_tar(
        tar_path=raw_data_folder_long_title_content_tar_path,
        untar_folder_path=raw_data_folder_long_title_content_unzipped_tar_path,
    )
    raw_data_folder_long_title_content_unzipped_tar_content_list = os.listdir(raw_data_folder_long_title_content_unzipped_tar_path)

    raw_data_folder_long_title_content_tar_gz_files = [content for content in raw_data_folder_long_title_content_list if content.endswith(".tar.gz")]
    assert len(raw_data_folder_long_title_content_tar_gz_files) == 4, f"NCBI003 dataset has more than four tar.gz file"
    for file in raw_data_folder_long_title_content_tar_gz_files:
        tar_gz_path = os.path.join(raw_data_folder_long_title_path, file)
        tar_gz_name = file.split(".")[0]
        untar_gz_path = os.path.join(raw_data_folder_long_title_path, tar_gz_name)
        unzip_tar_gz(
            tar_gz_path=tar_gz_path,
            untar_gz_path=untar_gz_path,
        )
    image_model_mapping = NCBI003_image_model_mapping()
    project_datasets = ["A1"] # The only one human slide
    return project_datasets

def NCBI003_single_helper(
        main_data_storage: str,
        raw_data_folder_name: str,
        project_data_folder_name: str,
        project_dataset: str,
) -> None:
    start_time = time.time()

    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "NCBI", "NCBI003")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "NCBI", "NCBI003")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"NCBI003 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_tar_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".tar")]
    assert len(raw_data_folder_long_title_content_tar_file) == 1, f"NCBI003 dataset has more than one tar file"
    raw_data_folder_long_title_content_tar_name = raw_data_folder_long_title_content_tar_file[0]
    raw_data_folder_long_title_content_tar_basename, _ = os.path.splitext(raw_data_folder_long_title_content_tar_name)
    raw_data_folder_long_title_content_unzipped_tar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_basename)
    raw_data_folder_long_title_content_unzipped_tar_content_list = os.listdir(raw_data_folder_long_title_content_unzipped_tar_path)
    image_model_mapping = NCBI003_image_model_mapping()

    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    image, stdata, spatial = False, False, False

    raw_image_file = [content for content in raw_data_folder_long_title_content_unzipped_tar_content_list if (project_dataset in content) and (".tif" in content)]
    assert len(raw_image_file) == 1, f"More than one image file for NCBI003 dataset {project_dataset}"
    raw_image_file = raw_image_file[0]
    raw_image_path = os.path.join(raw_data_folder_long_title_content_unzipped_tar_path, raw_image_file)
    project_image_path = os.path.join(project_dataset_raw_data_path, raw_image_file)
    if not os.path.exists(project_image_path):
        shutil.copy2(raw_image_path, project_image_path)
    image = True
    
    raw_stdata_file = [content for content in raw_data_folder_long_title_content_unzipped_tar_content_list if (project_dataset in content) and (".csv" in content)]
    assert len(raw_stdata_file) == 1, f"More than one stdata file for NCBI003 dataset {project_dataset}"
    raw_stdata_file = raw_stdata_file[0]
    raw_stdata_path = os.path.join(raw_data_folder_long_title_content_unzipped_tar_path, raw_stdata_file)
    project_stdata_path = os.path.join(project_dataset_raw_data_path, raw_stdata_file)
    if not os.path.exists(project_stdata_path):
        shutil.copy2(raw_stdata_path, project_stdata_path)
    stdata = True
    
    raw_folder_name = image_model_mapping[project_dataset]
    raw_folder = [content for content in raw_data_folder_long_title_content_list if (raw_folder_name in content) and (not content.endswith(".tar.gz"))]
    assert len(raw_folder) == 1, f"More than one model folder for NCBI003 dataset {project_dataset}"
    raw_folder_full_name = raw_folder[0]
    raw_folder_path = os.path.join(raw_data_folder_long_title_path, raw_folder_full_name)
    raw_folder_content_list = os.listdir(raw_folder_path)
    raw_spatial_folder = [content for content in raw_folder_content_list if "spatial" in content]
    assert len(raw_spatial_folder) == 1, f"More than one spatial folder for NCBI003 dataset {project_dataset}"
    raw_spatial_folder_name = raw_spatial_folder[0]
    raw_spatial_folder_path = os.path.join(raw_folder_path, raw_spatial_folder_name)
    project_spatial_folder_path = os.path.join(project_dataset_raw_data_path, raw_spatial_folder_name)
    if not os.path.exists(project_spatial_folder_path):
        shutil.copytree(raw_spatial_folder_path, project_spatial_folder_path)
    spatial = True
    assert (image & stdata & spatial) == True, f"Missing filetype for NCBI003 dataset {project_dataset}: {'image' if not image else ''} {'stdata' if not stdata else ''} {'spatial' if not spatial else ''} is missing."

    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for NCBI003 dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    if not os.path.exists(project_dataset_raw_unzipped_data_path):
        os.makedirs(project_dataset_raw_unzipped_data_path)
    for content in project_dataset_raw_data_content_list:
        if content.endswith(".gz"):
            gz_path = os.path.join(project_dataset_raw_data_path, content)
            unzip_gz(
                gz_path=gz_path,
                ungz_folder_path=project_dataset_raw_unzipped_data_path,
            )
    end_time = time.time()
    print(f"Selecting data for NCBI003 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def NCBI004_project_list(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "NCBI", "NCBI004")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "NCBI", "NCBI004")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"NCBI004 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_tar_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".tar")]
    assert len(raw_data_folder_long_title_content_tar_file) == 1, f"NCBI004 dataset has more than one tar file"
    raw_data_folder_long_title_content_tar_name = raw_data_folder_long_title_content_tar_file[0]
    raw_data_folder_long_title_content_tar_basename, _ = os.path.splitext(raw_data_folder_long_title_content_tar_name)
    raw_data_folder_long_title_content_tar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_name)
    raw_data_folder_long_title_content_unzipped_tar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_basename)
    unzip_tar(
        tar_path=raw_data_folder_long_title_content_tar_path,
        untar_folder_path=raw_data_folder_long_title_content_unzipped_tar_path,
    )
    raw_data_folder_long_title_content_unzipped_tar_content_list = os.listdir(raw_data_folder_long_title_content_unzipped_tar_path)

    raw_data_folder_long_title_content_tar_gz_files = [content for content in raw_data_folder_long_title_content_list if content.endswith(".tar.gz")]
    assert len(raw_data_folder_long_title_content_tar_gz_files) == 1, f"NCBI004 dataset has more than one tar.gz file"
    for file in raw_data_folder_long_title_content_tar_gz_files:
        tar_gz_path = os.path.join(raw_data_folder_long_title_path, file)
        tar_gz_name = file.split(".")[0]
        untar_gz_path = os.path.join(raw_data_folder_long_title_path, tar_gz_name)
        unzip_tar_gz(
            tar_gz_path=tar_gz_path,
            untar_gz_path=untar_gz_path,
        )

    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_image_folder_layer_1 = [content for content in raw_data_folder_long_title_content_list if ("images" in content) and (not content.endswith(".tar.gz"))]
    assert len(raw_data_folder_long_title_content_image_folder_layer_1) == 1, f"NCBI004 dataset has more than one image folder"
    raw_data_folder_long_title_content_image_folder_layer_1_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_image_folder_layer_1[0])
    raw_data_folder_long_title_content_image_folder_layer_2 = os.listdir(raw_data_folder_long_title_content_image_folder_layer_1_path)
    assert len(raw_data_folder_long_title_content_image_folder_layer_2) == 1, f"NCBI004 dataset has more than one image folder"
    raw_data_folder_long_title_content_image_folder_path = os.path.join(raw_data_folder_long_title_content_image_folder_layer_1_path, raw_data_folder_long_title_content_image_folder_layer_2[0])
    raw_data_folder_long_title_content_image_folder_content_list = os.listdir(raw_data_folder_long_title_content_image_folder_path)
    raw_data_folder_long_title_content_image_folder_content_list = [content for content in raw_data_folder_long_title_content_image_folder_content_list if not content.startswith("._")]
    assert len(raw_data_folder_long_title_content_image_folder_content_list) == 6, f"NCBI004 dataset has more than six image files"
    project_datasets = [content.split(".")[0] for content in raw_data_folder_long_title_content_image_folder_content_list]
    raw_RAW_folder = [content for content in raw_data_folder_long_title_content_list if ("RAW" in content) and (not content.endswith(".tar"))]
    assert len(raw_RAW_folder) == 1, f"NCBI004 dataset has more than one RAW folder"
    raw_RAW_folder = raw_RAW_folder[0]
    raw_RAW_folder_path = os.path.join(raw_data_folder_long_title_path, raw_RAW_folder)
    raw_RAW_folder_content_list = os.listdir(raw_RAW_folder_path)
    return project_datasets

def NCBI004_single_helper(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "NCBI", "NCBI004")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "NCBI", "NCBI004")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"NCBI004 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_image_folder_layer_1 = [content for content in raw_data_folder_long_title_content_list if ("images" in content) and (not content.endswith(".tar.gz"))]
    assert len(raw_data_folder_long_title_content_image_folder_layer_1) == 1, f"NCBI004 dataset has more than one image folder"
    raw_data_folder_long_title_content_image_folder_layer_1_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_image_folder_layer_1[0])
    raw_data_folder_long_title_content_image_folder_layer_2 = os.listdir(raw_data_folder_long_title_content_image_folder_layer_1_path)
    assert len(raw_data_folder_long_title_content_image_folder_layer_2) == 1, f"NCBI004 dataset has more than one image folder"
    raw_data_folder_long_title_content_image_folder_path = os.path.join(raw_data_folder_long_title_content_image_folder_layer_1_path, raw_data_folder_long_title_content_image_folder_layer_2[0])
    raw_data_folder_long_title_content_image_folder_content_list = os.listdir(raw_data_folder_long_title_content_image_folder_path)
    raw_data_folder_long_title_content_image_folder_content_list = [content for content in raw_data_folder_long_title_content_image_folder_content_list if not content.startswith("._")]
    assert len(raw_data_folder_long_title_content_image_folder_content_list) == 6, f"NCBI004 dataset has more than six image files"
    raw_RAW_folder = [content for content in raw_data_folder_long_title_content_list if ("RAW" in content) and (not content.endswith(".tar"))]
    assert len(raw_RAW_folder) == 1, f"NCBI004 dataset has more than one RAW folder"
    raw_RAW_folder = raw_RAW_folder[0]
    raw_RAW_folder_path = os.path.join(raw_data_folder_long_title_path, raw_RAW_folder)
    raw_RAW_folder_content_list = os.listdir(raw_RAW_folder_path)
    
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    image, stdata, spatial = False, False, False
    
    raw_image_file = [content for content in raw_data_folder_long_title_content_image_folder_content_list if project_dataset in content]
    assert len(raw_image_file) == 1, f"More than one image file for NCBI004 dataset {project_dataset}"
    raw_image_path = os.path.join(raw_data_folder_long_title_content_image_folder_path, raw_image_file[0])
    project_image_path = os.path.join(project_dataset_raw_data_path, raw_image_file[0])
    if not os.path.exists(project_image_path):
        shutil.copy2(raw_image_path, project_image_path)
    image = True
    
    pt_code_file_list = [content for content in raw_RAW_folder_content_list if project_dataset in content]
    GSE_code_list = [content.split("_")[0] for content in pt_code_file_list]
    GSE_code_list = list(set(GSE_code_list))
    assert len(GSE_code_list) == 1, f"More than one GSE code for NCBI004 dataset {project_dataset}"
    GSE_code = GSE_code_list[0]
    GSE_code_file_list = [content for content in raw_RAW_folder_content_list if GSE_code in content]
    assert len(GSE_code_file_list) == 5, f"Unmatching between the number of files and the expected number of files for NCBI004 dataset {project_dataset}"
    raw_barcodes_file = [file for file in GSE_code_file_list if "barcodes" in file]
    assert len(raw_barcodes_file) == 1, f"More than one barcodes file for NCBI004 dataset {project_dataset}"
    raw_barcodes_file = raw_barcodes_file[0]
    raw_barcodes_path = os.path.join(raw_RAW_folder_path, raw_barcodes_file)
    raw_features_file = [file for file in GSE_code_file_list if "features" in file]
    assert len(raw_features_file) == 1, f"More than one features file for NCBI004 dataset {project_dataset}"
    raw_features_file = raw_features_file[0]
    raw_features_path = os.path.join(raw_RAW_folder_path, raw_features_file)
    raw_matrix_file = [file for file in GSE_code_file_list if "matrix.mtx" in file]
    assert len(raw_matrix_file) == 1, f"More than one matrix file for NCBI004 dataset {project_dataset}"
    raw_matrix_file = raw_matrix_file[0]
    raw_matrix_path = os.path.join(raw_RAW_folder_path, raw_matrix_file)
    project_dataset_raw_data_filtered_feature_bc_matrix_path = os.path.join(project_dataset_raw_data_path, "filtered_feature_bc_matrix")
    if not os.path.exists(project_dataset_raw_data_filtered_feature_bc_matrix_path):
        os.makedirs(project_dataset_raw_data_filtered_feature_bc_matrix_path)
        shutil.copy2(raw_barcodes_path, project_dataset_raw_data_filtered_feature_bc_matrix_path)
        shutil.copy2(raw_features_path, project_dataset_raw_data_filtered_feature_bc_matrix_path)
        shutil.copy2(raw_matrix_path, project_dataset_raw_data_filtered_feature_bc_matrix_path)
    stdata = True
    
    raw_spatial_file = [file for file in GSE_code_file_list if "spatial" in file]
    raw_spatial_file = raw_spatial_file[0]
    raw_spatial_path = os.path.join(raw_RAW_folder_path, raw_spatial_file)
    project_spatial_path = os.path.join(project_dataset_raw_data_path, raw_spatial_file)
    if not os.path.exists(project_spatial_path):
        shutil.copy2(raw_spatial_path, project_spatial_path)
    spatial = True
    assert (image & stdata & spatial) == True, f"Missing filetype for NCBI004 dataset {project_dataset}: {'image' if not image else ''} {'stdata' if not stdata else ''} {'spatial' if not spatial else ''} is missing."
    
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for NCBI004 dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    if not os.path.exists(project_dataset_raw_unzipped_data_path):
        os.makedirs(project_dataset_raw_unzipped_data_path)
    for content in project_dataset_raw_data_content_list:
        if content.endswith(".tar.gz"):
            tar_gz_path = os.path.join(project_dataset_raw_data_path, content)
            double_unzip_tar_gz(
                project_dataset_raw_data_path=project_dataset_raw_data_path,
                project_dataset_raw_unzipped_data_path=project_dataset_raw_unzipped_data_path,
                filetype="spatial",
                already_in_folder_format=False,
                tar_gz_path=tar_gz_path
            )
        elif "filtered_feature_bc_matrix" in content:
            double_unzip_tar_gz(
                project_dataset_raw_data_path=project_dataset_raw_data_path,
                project_dataset_raw_unzipped_data_path=project_dataset_raw_unzipped_data_path,
                filetype="matrix",
                already_in_folder_format=True,
                already_folder_name=content,
            )
    end_time = time.time()
    print(f"Selecting data for NCBI004 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def NCBI005_project_list(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "NCBI", "NCBI005")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "NCBI", "NCBI005")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"NCBI005 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_tar_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".tar")]
    assert len(raw_data_folder_long_title_content_tar_file) == 1, f"NCBI005 dataset has more than one tar file"
    raw_data_folder_long_title_content_tar_name = raw_data_folder_long_title_content_tar_file[0]
    raw_data_folder_long_title_content_tar_basename, _ = os.path.splitext(raw_data_folder_long_title_content_tar_name)
    raw_data_folder_long_title_content_tar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_name)
    raw_data_folder_long_title_content_unzipped_tar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_basename)
    unzip_tar(
        tar_path=raw_data_folder_long_title_content_tar_path,
        untar_folder_path=raw_data_folder_long_title_content_unzipped_tar_path,
    )
    raw_data_folder_long_title_content_unzipped_tar_content_list = os.listdir(raw_data_folder_long_title_content_unzipped_tar_path)
    GSM_code_list = [content.split("_")[0] for content in raw_data_folder_long_title_content_unzipped_tar_content_list]
    GSM_code_list = list(set(GSM_code_list))
    GSM_code_mouse_start_value, GSM_code_human_end_value = 5621972, 5621978
    GSM_code_mouse_list = [f"GSM{str(value)}" for value in range(GSM_code_mouse_start_value, GSM_code_human_end_value + 1)]
    project_datasets = [code for code in GSM_code_list if code not in GSM_code_mouse_list]
    return project_datasets

def NCBI005_single_helper(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "NCBI", "NCBI005")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "NCBI", "NCBI005")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"NCBI005 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_tar_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".tar")]
    assert len(raw_data_folder_long_title_content_tar_file) == 1, f"NCBI005 dataset has more than one tar file"
    raw_data_folder_long_title_content_tar_name = raw_data_folder_long_title_content_tar_file[0]
    raw_data_folder_long_title_content_tar_basename, _ = os.path.splitext(raw_data_folder_long_title_content_tar_name)
    raw_data_folder_long_title_content_unzipped_tar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_basename)
    raw_data_folder_long_title_content_unzipped_tar_content_list = os.listdir(raw_data_folder_long_title_content_unzipped_tar_path)

    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    image, stdata, spots = False, False, False
    
    raw_image_file = [content for content in raw_data_folder_long_title_content_unzipped_tar_content_list if (project_dataset in content) and (".jpg" in content)]
    assert len(raw_image_file) == 1, f"More than one image file for NCBI005 dataset {project_dataset}"
    raw_image_file = raw_image_file[0]
    raw_image_path = os.path.join(raw_data_folder_long_title_content_unzipped_tar_path, raw_image_file)
    project_image_path = os.path.join(project_dataset_raw_data_path, raw_image_file)
    if not os.path.exists(project_image_path):
        shutil.copy2(raw_image_path, project_image_path)
    image = True
    
    raw_stdata_file = [content for content in raw_data_folder_long_title_content_unzipped_tar_content_list if (project_dataset in content) and ("filtered_feature_bc_matrix" in content)]
    assert len(raw_stdata_file) == 1, f"More than one stdata file for NCBI005 dataset {project_dataset}"
    raw_stdata_file = raw_stdata_file[0]
    raw_stdata_path = os.path.join(raw_data_folder_long_title_content_unzipped_tar_path, raw_stdata_file)
    project_stdata_path = os.path.join(project_dataset_raw_data_path, raw_stdata_file)
    if not os.path.exists(project_stdata_path):
        shutil.copy2(raw_stdata_path, project_stdata_path)
    stdata = True
    
    raw_spots_file = [content for content in raw_data_folder_long_title_content_unzipped_tar_content_list if (project_dataset in content) and ("positions_list" in content)]
    assert len(raw_spots_file) == 1, f"More than one spots file for NCBI005 dataset {project_dataset}"
    raw_spots_file = raw_spots_file[0]
    raw_spots_path = os.path.join(raw_data_folder_long_title_content_unzipped_tar_path, raw_spots_file)
    project_spots_path = os.path.join(project_dataset_raw_data_path, raw_spots_file)
    if not os.path.exists(project_spots_path):
        shutil.copy2(raw_spots_path, project_spots_path)
    spots = True
    assert (image & stdata & spots) == True, f"Missing filetype for NCBI005 dataset {project_dataset}: {'image' if not image else ''} {'stdata' if not stdata else ''} {'spots' if not spots else ''} is missing."
    
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for NCBI005 dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    if not os.path.exists(project_dataset_raw_unzipped_data_path):
        os.makedirs(project_dataset_raw_unzipped_data_path)
    for content in project_dataset_raw_data_content_list:
        if content.endswith(".gz"):
            gz_path = os.path.join(project_dataset_raw_data_path, content)
            unzip_gz(
                gz_path=gz_path,
                ungz_folder_path=project_dataset_raw_unzipped_data_path,
            )
    end_time = time.time()
    print(f"Selecting data for NCBI005 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def NCBI007_project_list(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "NCBI", "NCBI007")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "NCBI", "NCBI007")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"NCBI005 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_zip_files = [content for content in raw_data_folder_long_title_content_list if content.endswith(".zip")]
    assert len(raw_data_folder_long_title_content_zip_files) == 2, f"NCBI007 dataset has more than two zip files"
    for zip_file in raw_data_folder_long_title_content_zip_files:
        zip_path = os.path.join(raw_data_folder_long_title_path, zip_file)
        unzip_zip(
            zip_path=zip_path,
            unzip_folder_path=raw_data_folder_long_title_path,
        )
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    project_datasets = [content for content in raw_data_folder_long_title_content_list if content.startswith("A")]
    return project_datasets

def NCBI007_single_helper(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "NCBI", "NCBI007")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "NCBI", "NCBI007")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"NCBI007 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_image_folder = [content for content in raw_data_folder_long_title_content_list if ("STorig" in content) and (not content.endswith(".zip"))]
    assert len(raw_data_folder_long_title_image_folder) == 1, f"NCBI007 dataset has more than one image folder"
    raw_data_folder_long_title_image_folder_name = raw_data_folder_long_title_image_folder[0]
    raw_data_folder_long_title_image_folder_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_image_folder_name)
    raw_data_folder_long_title_image_folder_content_list = os.listdir(raw_data_folder_long_title_image_folder_path)
    dataset_name_image_name_mapping = NCBI007_dataset_name_image_name_mapping()
    
    raw_dataset_path = os.path.join(raw_data_folder_long_title_path, project_dataset)
    raw_dataset_content_list = os.listdir(raw_dataset_path)
    assert len(raw_dataset_content_list) == 10, f"Unmatching between the number of files and the expected number of files for NCBI007 dataset {project_dataset}"
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    image, matrix, spatial = False, False, False
    
    raw_image_file = [content for content in raw_data_folder_long_title_image_folder_content_list if dataset_name_image_name_mapping[project_dataset] in content]
    assert len(raw_image_file) == 1, f"More than one image file for NCBI007 dataset {project_dataset}"
    raw_image_file = raw_image_file[0]
    raw_image_path = os.path.join(raw_data_folder_long_title_image_folder_path, raw_image_file)
    project_image_path = os.path.join(project_dataset_raw_data_path, raw_image_file)
    if not os.path.exists(project_image_path):
        shutil.copy2(raw_image_path, project_image_path)
    image = True
    
    raw_matrix_folder = [content for content in raw_dataset_content_list if ("filtered_feature_bc_matrix" in content) and (not content.endswith(".h5"))]
    assert len(raw_matrix_folder) == 1, f"More than one matrix folder for NCBI007 dataset {project_dataset}"
    raw_matrix_folder = raw_matrix_folder[0]
    raw_matrix_folder_path = os.path.join(raw_dataset_path, raw_matrix_folder)
    project_matrix_folder_path = os.path.join(project_dataset_raw_data_path, raw_matrix_folder)
    if not os.path.exists(project_matrix_folder_path):
        shutil.copytree(raw_matrix_folder_path, project_matrix_folder_path)
    matrix = True
    
    raw_spatial_folder = [content for content in raw_dataset_content_list if "spatial" in content]
    assert len(raw_spatial_folder) == 1, f"More than one spatial folder for NCBI007 dataset {project_dataset}"
    raw_spatial_folder = raw_spatial_folder[0]
    raw_spatial_folder_path = os.path.join(raw_dataset_path, raw_spatial_folder)
    project_spatial_folder_path = os.path.join(project_dataset_raw_data_path, raw_spatial_folder)
    if not os.path.exists(project_spatial_folder_path):
        shutil.copytree(raw_spatial_folder_path, project_spatial_folder_path)
    spatial = True
    assert (image & matrix & spatial) == True, f"Missing filetype for NCBI007 dataset {project_dataset}: {'image' if not image else ''} {'matrix' if not matrix else ''} {'spatial' if not spatial else ''} is missing."
    
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for NCBI007 dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    if not os.path.exists(project_dataset_raw_unzipped_data_path):
        os.makedirs(project_dataset_raw_unzipped_data_path)
    for content in project_dataset_raw_data_content_list:
        if "filtered_feature_bc_matrix" in content:
            double_unzip_tar_gz(
                project_dataset_raw_data_path=project_dataset_raw_data_path,
                project_dataset_raw_unzipped_data_path=project_dataset_raw_unzipped_data_path,
                filetype="matrix",
                already_in_folder_format=True,
                already_folder_name=content,
            )
    end_time = time.time()
    print(f"Selecting data for NCBI007 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def NCBI008_project_list(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "NCBI", "NCBI008")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "NCBI", "NCBI008")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"NCBI008 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)

    raw_data_folder_long_title_content_download_file_new_zip_path = os.path.join(raw_data_folder_long_title_path, "download.zip")
    if not os.path.exists(raw_data_folder_long_title_content_download_file_new_zip_path):
        raw_data_folder_long_title_content_download_file = [content for content in raw_data_folder_long_title_content_list if content.endswith("download")]
        assert len(raw_data_folder_long_title_content_download_file) == 1, f"NCBI008 dataset has more than one download file"
        raw_data_folder_long_title_content_download_file = raw_data_folder_long_title_content_download_file[0]
        assert raw_data_folder_long_title_content_download_file == "download", f"NCBI008 dataset has a download file with a different name"
        raw_data_folder_long_title_content_download_file_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_download_file)
        os.rename(raw_data_folder_long_title_content_download_file_path, raw_data_folder_long_title_content_download_file_new_zip_path)
    unzip_zip(
        zip_path=raw_data_folder_long_title_content_download_file_new_zip_path,
        unzip_folder_path=raw_data_folder_long_title_path,
    )
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_tar_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".tar")]
    assert len(raw_data_folder_long_title_content_tar_file) == 1, f"NCBI008 dataset has more than one tar files"
    raw_data_folder_long_title_content_tar_file = raw_data_folder_long_title_content_tar_file[0]
    raw_data_folder_long_title_content_tar_basename, _ = os.path.splitext(raw_data_folder_long_title_content_tar_file)
    raw_data_folder_long_title_content_tar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_file)
    raw_data_folder_long_title_content_untar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_basename)
    unzip_tar(
        tar_path=raw_data_folder_long_title_content_tar_path,
        untar_folder_path=raw_data_folder_long_title_content_untar_path,
    )
    raw_data_folder_long_title_content_untar_content_list = os.listdir(raw_data_folder_long_title_content_untar_path)

    raw_data_folder_long_title_content_gz_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".gz")]
    assert len(raw_data_folder_long_title_content_gz_file) == 1, f"NCBI008 dataset has more than one gz files"
    raw_data_folder_long_title_content_gz_file = raw_data_folder_long_title_content_gz_file[0]
    raw_data_folder_long_title_content_gz_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_gz_file)
    unzip_gz(
        gz_path=raw_data_folder_long_title_content_gz_path,
        ungz_folder_path=raw_data_folder_long_title_path,
    )
    raw_data_folder_long_title_content_tsv_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".tsv")]
    assert len(raw_data_folder_long_title_content_tsv_file) == 1, f"NCBI008 dataset has more than one tsv files"
    raw_data_folder_long_title_content_tsv_file = raw_data_folder_long_title_content_tsv_file[0]
    raw_data_folder_long_title_content_tsv_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tsv_file)
    raw_data_folder_long_title_content_tsv_df = pd.read_csv(raw_data_folder_long_title_content_tsv_path, sep="\t")
    is_human_mask = raw_data_folder_long_title_content_tsv_df['Sample name'].str.contains('Human', case=False)
    raw_data_folder_long_title_content_tsv_df_human = raw_data_folder_long_title_content_tsv_df[is_human_mask]
    project_datasets = raw_data_folder_long_title_content_tsv_df_human['characteristics: shortFileName'].tolist()
    return project_datasets

def NCBI008_single_helper(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "NCBI", "NCBI008")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "NCBI", "NCBI008")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"NCBI008 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)

    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_tar_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".tar")]
    assert len(raw_data_folder_long_title_content_tar_file) == 1, f"NCBI008 dataset has more than one tar files"
    raw_data_folder_long_title_content_tar_file = raw_data_folder_long_title_content_tar_file[0]
    raw_data_folder_long_title_content_tar_basename, _ = os.path.splitext(raw_data_folder_long_title_content_tar_file)
    raw_data_folder_long_title_content_tar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_file)
    raw_data_folder_long_title_content_untar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_basename)
    raw_data_folder_long_title_content_untar_content_list = os.listdir(raw_data_folder_long_title_content_untar_path)

    raw_data_folder_long_title_content_image_folder = [content for content in raw_data_folder_long_title_content_list if "images_Visium" in content]
    assert len(raw_data_folder_long_title_content_image_folder) == 1, f"NCBI008 dataset has more than one image folder"
    raw_data_folder_long_title_content_image_folder = raw_data_folder_long_title_content_image_folder[0]
    raw_data_folder_long_title_content_image_folder_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_image_folder)
    raw_data_folder_long_title_content_image_folder_content_list = os.listdir(raw_data_folder_long_title_content_image_folder_path)
    
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    raw_dataset_content_list = [content for content in raw_data_folder_long_title_content_untar_content_list if project_dataset in content]
    assert len(raw_dataset_content_list) == 5, f"More than five files for NCBI008 dataset {project_dataset}"
    image, matrix, spots = False, False, False
    
    raw_image_file = [content for content in raw_data_folder_long_title_content_image_folder_content_list if project_dataset in content]
    assert len(raw_image_file) == 1, f"More than one image file for NCBI008 dataset {project_dataset}"
    raw_image_file = raw_image_file[0]
    raw_image_path = os.path.join(raw_data_folder_long_title_content_image_folder_path, raw_image_file)
    project_image_path = os.path.join(project_dataset_raw_data_path, raw_image_file)
    if not os.path.exists(project_image_path):
        shutil.copy2(raw_image_path, project_image_path)
    image = True
    
    raw_matrix_file = [content for content in raw_dataset_content_list if "filtered_feature_bc_matrix" in content]
    assert len(raw_matrix_file) == 1, f"More than one matrix file for NCBI008 dataset {project_dataset}"
    raw_matrix_file = raw_matrix_file[0]
    raw_matrix_path = os.path.join(raw_data_folder_long_title_content_untar_path, raw_matrix_file)
    project_matrix_path = os.path.join(project_dataset_raw_data_path, raw_matrix_file)
    if not os.path.exists(project_matrix_path):
        shutil.copy2(raw_matrix_path, project_matrix_path)
    matrix = True
    
    raw_spots_file = [content for content in raw_dataset_content_list if "tissue_positions_list" in content]
    assert len(raw_spots_file) == 1, f"More than one spots file for NCBI008 dataset {project_dataset}"
    raw_spots_file = raw_spots_file[0]
    raw_spots_path = os.path.join(raw_data_folder_long_title_content_untar_path, raw_spots_file)
    project_spots_path = os.path.join(project_dataset_raw_data_path, raw_spots_file)
    if not os.path.exists(project_spots_path):
        shutil.copy2(raw_spots_path, project_spots_path)
    spots = True
    assert (image & matrix & spots) == True, f"Missing filetype for NCBI008 dataset {project_dataset}: {'image' if not image else ''} {'matrix' if not matrix else ''} {'spots' if not spots else ''} is missing."
    
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for NCBI008 dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    if not os.path.exists(project_dataset_raw_unzipped_data_path):
        os.makedirs(project_dataset_raw_unzipped_data_path)
    for content in project_dataset_raw_data_content_list:
        if content.endswith(".gz"):
            gz_path = os.path.join(project_dataset_raw_data_path, content)
            unzip_gz(
                gz_path=gz_path,
                ungz_folder_path=project_dataset_raw_unzipped_data_path,
            )
    end_time = time.time()
    print(f"Selecting data for NCBI008 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def NCBI009_project_list(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "NCBI", "NCBI009")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "NCBI", "NCBI009")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"NCBI009 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_tar_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".tar")]
    assert len(raw_data_folder_long_title_content_tar_file) == 1, f"NCBI009 dataset has more than one tar files"
    raw_data_folder_long_title_content_tar_file = raw_data_folder_long_title_content_tar_file[0]
    raw_data_folder_long_title_content_tar_basename, _ = os.path.splitext(raw_data_folder_long_title_content_tar_file)
    raw_data_folder_long_title_content_tar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_file)
    raw_data_folder_long_title_content_untar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_basename)
    unzip_tar(
        tar_path=raw_data_folder_long_title_content_tar_path,
        untar_folder_path=raw_data_folder_long_title_content_untar_path,
    )
    raw_data_folder_long_title_content_untar_content_list = os.listdir(raw_data_folder_long_title_content_untar_path)
    raw_data_folder_long_title_content_untar_content_tar_gz_files = [content for content in raw_data_folder_long_title_content_untar_content_list if (content.endswith(".tar.gz"))]
    assert len(raw_data_folder_long_title_content_untar_content_tar_gz_files) == 2, f"NCBI009 dataset has more than two tar.gz files"
    for tar_gz_file in raw_data_folder_long_title_content_untar_content_tar_gz_files:
        tar_gz_path = os.path.join(raw_data_folder_long_title_content_untar_path, tar_gz_file)
        tar_gz_name = tar_gz_file.split(".")[0]
        untar_gz_path = os.path.join(raw_data_folder_long_title_content_untar_path, tar_gz_name)
        unzip_tar_gz(
            tar_gz_path=tar_gz_path,
            untar_gz_path=untar_gz_path,
        )
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_image_files = [content for content in raw_data_folder_long_title_content_list if ".tif" in content]
    assert len(raw_data_folder_long_title_content_image_files) == 2, f"NCBI009 dataset has more than two image files"
    project_datasets = [content.split("_")[0] for content in raw_data_folder_long_title_content_image_files]
    return project_datasets

def NCBI009_single_helper(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "NCBI", "NCBI009")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "NCBI", "NCBI009")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"NCBI009 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_tar_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".tar")]
    assert len(raw_data_folder_long_title_content_tar_file) == 1, f"NCBI009 dataset has more than one tar files"
    raw_data_folder_long_title_content_tar_file = raw_data_folder_long_title_content_tar_file[0]
    raw_data_folder_long_title_content_tar_basename, _ = os.path.splitext(raw_data_folder_long_title_content_tar_file)
    raw_data_folder_long_title_content_tar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_file)
    raw_data_folder_long_title_content_untar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_basename)
    raw_data_folder_long_title_content_untar_content_list = os.listdir(raw_data_folder_long_title_content_untar_path)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_image_files = [content for content in raw_data_folder_long_title_content_list if ".tif" in content]
    assert len(raw_data_folder_long_title_content_image_files) == 2, f"NCBI009 dataset has more than two image files"

    raw_dataset_folder = [content for content in raw_data_folder_long_title_content_untar_content_list if (project_dataset in content) and (not content.endswith(".tar.gz"))]
    assert len(raw_dataset_folder) == 1, f"More than one folder for NCBI009 dataset {project_dataset}"
    raw_dataset_folder = raw_dataset_folder[0]
    raw_dataset_folder_path = os.path.join(raw_data_folder_long_title_content_untar_path, raw_dataset_folder)
    raw_dataset_folder_subfolder = os.listdir(raw_dataset_folder_path)
    assert len(raw_dataset_folder_subfolder) == 1, f"More than one subfolder for NCBI009 dataset {project_dataset}"
    raw_dataset_folder_subfolder = raw_dataset_folder_subfolder[0]
    raw_dataset_folder_subfolder_path = os.path.join(raw_dataset_folder_path, raw_dataset_folder_subfolder)
    raw_dataset_folder_content_list = os.listdir(raw_dataset_folder_subfolder_path)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    image, matrix, spatial = False, False, False
    
    raw_image_file = [content for content in raw_data_folder_long_title_content_image_files if (project_dataset in content)]
    assert len(raw_image_file) == 1, f"More than one image file for NCBI009 dataset {project_dataset}"
    raw_image_file = raw_image_file[0]
    raw_image_path = os.path.join(raw_data_folder_long_title_path, raw_image_file)
    project_image_path = os.path.join(project_dataset_raw_data_path, raw_image_file)
    if not os.path.exists(project_image_path):
        shutil.copy2(raw_image_path, project_image_path)
    image = True
    
    raw_matrix_file = [content for content in raw_dataset_folder_content_list if "filtered_feature_bc_matrix.h5" in content]
    assert len(raw_matrix_file) == 1, f"More than one matrix file for NCBI009 dataset {project_dataset}"
    raw_matrix_file = raw_matrix_file[0]
    raw_matrix_path = os.path.join(raw_dataset_folder_subfolder_path, raw_matrix_file)
    project_matrix_path = os.path.join(project_dataset_raw_data_path, raw_matrix_file)
    if not os.path.exists(project_matrix_path):
        shutil.copy2(raw_matrix_path, project_matrix_path)
    matrix = True
    
    raw_spatial_folder = [content for content in raw_dataset_folder_content_list if "spatial" in content]
    assert len(raw_spatial_folder) == 1, f"More than one spatial folder for NCBI009 dataset {project_dataset}"
    raw_spatial_folder = raw_spatial_folder[0]
    raw_spatial_folder_path = os.path.join(raw_dataset_folder_subfolder_path, raw_spatial_folder)
    project_spatial_folder_path = os.path.join(project_dataset_raw_data_path, raw_spatial_folder)
    if not os.path.exists(project_spatial_folder_path):
        shutil.copytree(raw_spatial_folder_path, project_spatial_folder_path)
    spatial = True
    assert (image & matrix & spatial) == True, f"Missing filetype for NCBI009 dataset {project_dataset}: {'image' if not image else ''} {'matrix' if not matrix else ''} {'spatial' if not spatial else ''} is missing."
    
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for NCBI009 dataset {project_dataset}"
    end_time = time.time()
    print(f"Selecting data for NCBI009 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def Specific_datasets_5_locations_lung_non_FFPE_project_list(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Specific_datasets", "5_locations_lung")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "5_locations_lung")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"5_locations_lung dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_folders_list = [content for content in raw_data_folder_long_title_content_list if os.path.isdir(os.path.join(raw_data_folder_long_title_path, content))]
    raw_data_folder_long_title_content_files_list = [content for content in raw_data_folder_long_title_content_list if os.path.isfile(os.path.join(raw_data_folder_long_title_path, content))]

    raw_data_folder_long_title_pure_raw_path = os.path.join(raw_data_folder_long_title_path, "Pure_Raw_Direct_Downloads")
    raw_data_folder_long_title_gathering_path = os.path.join(raw_data_folder_long_title_path, "Files_Gathering")
    if (not os.path.exists(raw_data_folder_long_title_pure_raw_path)) and (not os.path.exists(raw_data_folder_long_title_gathering_path)):
        os.makedirs(raw_data_folder_long_title_pure_raw_path)
        os.makedirs(raw_data_folder_long_title_gathering_path)
        for folder in raw_data_folder_long_title_content_folders_list:
            raw_data_folder_long_title_content_folder_path = os.path.join(raw_data_folder_long_title_path, folder)
            raw_data_folder_long_title_content_folder_content_list = os.listdir(raw_data_folder_long_title_content_folder_path)
            assert len(raw_data_folder_long_title_content_folder_content_list) == 1, f"More than one file in {folder} folder"
            raw_data_folder_long_title_content_folder_content = raw_data_folder_long_title_content_folder_content_list[0]
            raw_data_folder_long_title_content_folder_content_gathering_path = os.path.join(raw_data_folder_long_title_gathering_path, raw_data_folder_long_title_content_folder_content)
            if not os.path.exists(raw_data_folder_long_title_content_folder_content_gathering_path):
                raw_data_folder_long_title_content_folder_content_original_path = os.path.join(raw_data_folder_long_title_content_folder_path, raw_data_folder_long_title_content_folder_content)
                shutil.copy2(raw_data_folder_long_title_content_folder_content_original_path, raw_data_folder_long_title_content_folder_content_gathering_path)
                shutil.move(raw_data_folder_long_title_content_folder_path, raw_data_folder_long_title_pure_raw_path)
    raw_data_folder_long_title_pure_raw_content_list = os.listdir(raw_data_folder_long_title_pure_raw_path)
    raw_data_folder_long_title_gathering_content_list = os.listdir(raw_data_folder_long_title_gathering_path)
    assert len(raw_data_folder_long_title_pure_raw_content_list) == len(raw_data_folder_long_title_gathering_content_list) == 44, f"The creation of gathering under folder {raw_data_folder_long_title_gathering_path} lost some files."

    for file in raw_data_folder_long_title_content_files_list:
        if file.endswith(".zip"):
            zip_basename, _ = os.path.splitext(os.path.basename(file))
            zip_path = os.path.join(raw_data_folder_long_title_path, file)
            unzip_zip(
                zip_path=zip_path,
                unzip_folder_path=raw_data_folder_long_title_path,
            )
    raw_data_folder_long_title_gathering_content_WSA_images_list = [content for content in raw_data_folder_long_title_gathering_content_list if ("WSA" in content) and ((".tif" in content) or (".jpg" in content))]
    raw_data_folder_long_title_gathering_content_FFPE_images_list = [content for content in raw_data_folder_long_title_gathering_content_list if ("V11" in content) and ((".tif" in content) or (".jpg" in content))]
    assert len(raw_data_folder_long_title_gathering_content_WSA_images_list) + len(raw_data_folder_long_title_gathering_content_FFPE_images_list) == 20, f"The number of images in {raw_data_folder_long_title_gathering_path} is not 20."

    # WSA = non-FFPE
    raw_data_folder_long_title_gathering_content_WSA_image_basenames_list = [os.path.splitext(os.path.basename(content))[0] for content in raw_data_folder_long_title_gathering_content_WSA_images_list]
    sp_h5ad_for_WSA_path = os.path.join(raw_data_folder_long_title_path, "Cell2location_outputs", "sp.h5ad")
    anndata_for_WSA = anndata.read_h5ad(sp_h5ad_for_WSA_path)
    spots_included_in_anndata_for_WSA = anndata_for_WSA.obs["in_tissue"].index.tolist()
    basename_index_mapping = dict()
    for basename in raw_data_folder_long_title_gathering_content_WSA_image_basenames_list:
        index_list = [i for i, element in enumerate(spots_included_in_anndata_for_WSA) if element.startswith(basename)]
        if not len(index_list) == 0:
            basename_index_mapping[basename] = index_list
    # There are five images without corresponding sp data: WSA_LngSP9258465, WSA_LngSP9258466, WSA_LngSP9258469, WSA_LngSP8759310, WSA_LngSP9258462
    assert len(basename_index_mapping.keys()) == len(raw_data_folder_long_title_gathering_content_WSA_image_basenames_list) - 5,\
        f"The missing sp data is more than WSA_LngSP9258465, WSA_LngSP9258466, WSA_LngSP9258469, WSA_LngSP8759310, WSA_LngSP9258462. Please check."
    project_datasets = list(basename_index_mapping.keys())
    return project_datasets

def Specific_datasets_5_locations_lung_non_FFPE_single_helper(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Specific_datasets", "5_locations_lung")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "5_locations_lung")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"5_locations_lung dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)

    raw_data_folder_long_title_pure_raw_path = os.path.join(raw_data_folder_long_title_path, "Pure_Raw_Direct_Downloads")
    raw_data_folder_long_title_gathering_path = os.path.join(raw_data_folder_long_title_path, "Files_Gathering")
    raw_data_folder_long_title_pure_raw_content_list = os.listdir(raw_data_folder_long_title_pure_raw_path)
    raw_data_folder_long_title_gathering_content_list = os.listdir(raw_data_folder_long_title_gathering_path)
    assert len(raw_data_folder_long_title_pure_raw_content_list) == len(raw_data_folder_long_title_gathering_content_list) == 44, f"The creation of gathering under folder {raw_data_folder_long_title_gathering_path} lost some files."
    raw_data_folder_long_title_gathering_content_WSA_images_list = [content for content in raw_data_folder_long_title_gathering_content_list if ("WSA" in content) and ((".tif" in content) or (".jpg" in content))]
    raw_data_folder_long_title_gathering_content_FFPE_images_list = [content for content in raw_data_folder_long_title_gathering_content_list if ("V11" in content) and ((".tif" in content) or (".jpg" in content))]
    assert len(raw_data_folder_long_title_gathering_content_WSA_images_list) + len(raw_data_folder_long_title_gathering_content_FFPE_images_list) == 20, f"The number of images in {raw_data_folder_long_title_gathering_path} is not 20."

    # WSA = non-FFPE
    raw_data_folder_long_title_gathering_content_WSA_image_basenames_list = [os.path.splitext(os.path.basename(content))[0] for content in raw_data_folder_long_title_gathering_content_WSA_images_list]
    sp_h5ad_for_WSA_path = os.path.join(raw_data_folder_long_title_path, "Cell2location_outputs", "sp.h5ad")
    anndata_for_WSA = anndata.read_h5ad(sp_h5ad_for_WSA_path)
    spots_included_in_anndata_for_WSA = anndata_for_WSA.obs["in_tissue"].index.tolist()
    basename_index_mapping = dict()
    for basename in raw_data_folder_long_title_gathering_content_WSA_image_basenames_list:
        index_list = [i for i, element in enumerate(spots_included_in_anndata_for_WSA) if element.startswith(basename)]
        if not len(index_list) == 0:
            basename_index_mapping[basename] = index_list
    
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    image, h5ad = False, False
    
    raw_image_file = [image for image in raw_data_folder_long_title_gathering_content_WSA_images_list if project_dataset in image]
    assert len(raw_image_file) == 1, f"More than one image file for 5_locations_lung dataset non FFPE {project_dataset}"
    raw_image_file = raw_image_file[0]
    raw_image_path = os.path.join(raw_data_folder_long_title_gathering_path, raw_image_file)
    project_image_path = os.path.join(project_dataset_raw_data_path, raw_image_file)
    if not os.path.exists(project_image_path):
        shutil.copy2(raw_image_path, project_image_path)
    image = True
    
    h5ad_filename = "sp_" + project_dataset + ".h5ad"
    project_h5ad_path = os.path.join(project_dataset_raw_data_path, h5ad_filename)
    if not os.path.exists(project_h5ad_path):
        h5ad_file = anndata_for_WSA[basename_index_mapping[project_dataset], :]
        h5ad_file.write(project_h5ad_path)
    h5ad = True
    assert (image & h5ad) == True, f"Missing filetype for 5_locations_lung dataset non FFPE {project_dataset}: {'image' if not image else ''} {'h5ad' if not h5ad else ''} is missing."
    end_time = time.time()
    print(f"Selecting data for 5_locations_lung dataset non FFPE {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def Specific_datasets_5_locations_lung_FFPE_project_list(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Specific_datasets", "5_locations_lung")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "5_locations_lung")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"5_locations_lung dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_folders_list = [content for content in raw_data_folder_long_title_content_list if os.path.isdir(os.path.join(raw_data_folder_long_title_path, content))]
    raw_data_folder_long_title_content_files_list = [content for content in raw_data_folder_long_title_content_list if os.path.isfile(os.path.join(raw_data_folder_long_title_path, content))]

    raw_data_folder_long_title_pure_raw_path = os.path.join(raw_data_folder_long_title_path, "Pure_Raw_Direct_Downloads")
    raw_data_folder_long_title_gathering_path = os.path.join(raw_data_folder_long_title_path, "Files_Gathering")
    if (not os.path.exists(raw_data_folder_long_title_pure_raw_path)) and (not os.path.exists(raw_data_folder_long_title_gathering_path)):
        os.makedirs(raw_data_folder_long_title_pure_raw_path)
        os.makedirs(raw_data_folder_long_title_gathering_path)
        for folder in raw_data_folder_long_title_content_folders_list:
            raw_data_folder_long_title_content_folder_path = os.path.join(raw_data_folder_long_title_path, folder)
            raw_data_folder_long_title_content_folder_content_list = os.listdir(raw_data_folder_long_title_content_folder_path)
            assert len(raw_data_folder_long_title_content_folder_content_list) == 1, f"More than one file in {folder} folder"
            raw_data_folder_long_title_content_folder_content = raw_data_folder_long_title_content_folder_content_list[0]
            raw_data_folder_long_title_content_folder_content_gathering_path = os.path.join(raw_data_folder_long_title_gathering_path, raw_data_folder_long_title_content_folder_content)
            if not os.path.exists(raw_data_folder_long_title_content_folder_content_gathering_path):
                raw_data_folder_long_title_content_folder_content_original_path = os.path.join(raw_data_folder_long_title_content_folder_path, raw_data_folder_long_title_content_folder_content)
                shutil.copy2(raw_data_folder_long_title_content_folder_content_original_path, raw_data_folder_long_title_content_folder_content_gathering_path)
                shutil.move(raw_data_folder_long_title_content_folder_path, raw_data_folder_long_title_pure_raw_path)
    raw_data_folder_long_title_pure_raw_content_list = os.listdir(raw_data_folder_long_title_pure_raw_path)
    raw_data_folder_long_title_gathering_content_list = os.listdir(raw_data_folder_long_title_gathering_path)
    assert len(raw_data_folder_long_title_pure_raw_content_list) == len(raw_data_folder_long_title_gathering_content_list) == 44, f"The creation of gathering under folder {raw_data_folder_long_title_gathering_path} lost some files."

    for file in raw_data_folder_long_title_content_files_list:
        if file.endswith(".zip"):
            zip_basename, _ = os.path.splitext(os.path.basename(file))
            zip_path = os.path.join(raw_data_folder_long_title_path, file)
            unzip_zip(
                zip_path=zip_path,
                unzip_folder_path=raw_data_folder_long_title_path,
            )
    raw_data_folder_long_title_gathering_content_WSA_images_list = [content for content in raw_data_folder_long_title_gathering_content_list if ("WSA" in content) and ((".tif" in content) or (".jpg" in content))]
    raw_data_folder_long_title_gathering_content_FFPE_images_list = [content for content in raw_data_folder_long_title_gathering_content_list if ("V11" in content) and ((".tif" in content) or (".jpg" in content))]
    assert len(raw_data_folder_long_title_gathering_content_WSA_images_list) + len(raw_data_folder_long_title_gathering_content_FFPE_images_list) == 20, f"The number of images in {raw_data_folder_long_title_gathering_path} is not 20."

    # FFPE
    FFPE_meta_csv_path = os.path.join(raw_data_folder_long_title_path, "FFPE_Cell2location_outputs_20231103", "meta.csv")
    FFPE_meta_csv_df = pd.read_csv(FFPE_meta_csv_path, header=None)
    FFPE_meta_WSA_names = FFPE_meta_csv_df[0].tolist()
    FFPE_meta_tif_names = FFPE_meta_csv_df[1].tolist()
    FFPE_meta_jpg_names = [name.replace("t.tif", ".jpg") for name in FFPE_meta_tif_names]
    assert sorted(FFPE_meta_jpg_names) == sorted(raw_data_folder_long_title_gathering_content_FFPE_images_list), f"The FFPE meta.csv and the gathering folder {raw_data_folder_long_title_gathering_path} do not match."
    FFPE_image_WSA_name_mapping = dict()
    for i in range(len(FFPE_meta_csv_df)):
        FFPE_image_WSA_name_mapping[FFPE_meta_jpg_names[i]] = FFPE_meta_WSA_names[i]
    project_datasets = [FFPE_image_WSA_name_mapping[image] for image in raw_data_folder_long_title_gathering_content_FFPE_images_list]
    return project_datasets

def Specific_datasets_5_locations_lung_FFPE_single_helper(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Specific_datasets", "5_locations_lung")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "5_locations_lung")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"5_locations_lung dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_pure_raw_path = os.path.join(raw_data_folder_long_title_path, "Pure_Raw_Direct_Downloads")
    raw_data_folder_long_title_gathering_path = os.path.join(raw_data_folder_long_title_path, "Files_Gathering")
    raw_data_folder_long_title_pure_raw_content_list = os.listdir(raw_data_folder_long_title_pure_raw_path)
    raw_data_folder_long_title_gathering_content_list = os.listdir(raw_data_folder_long_title_gathering_path)
    assert len(raw_data_folder_long_title_pure_raw_content_list) == len(raw_data_folder_long_title_gathering_content_list) == 44, f"The creation of gathering under folder {raw_data_folder_long_title_gathering_path} lost some files."
    raw_data_folder_long_title_gathering_content_WSA_images_list = [content for content in raw_data_folder_long_title_gathering_content_list if ("WSA" in content) and ((".tif" in content) or (".jpg" in content))]
    raw_data_folder_long_title_gathering_content_FFPE_images_list = [content for content in raw_data_folder_long_title_gathering_content_list if ("V11" in content) and ((".tif" in content) or (".jpg" in content))]
    assert len(raw_data_folder_long_title_gathering_content_WSA_images_list) + len(raw_data_folder_long_title_gathering_content_FFPE_images_list) == 20, f"The number of images in {raw_data_folder_long_title_gathering_path} is not 20."

    # FFPE
    FFPE_meta_csv_path = os.path.join(raw_data_folder_long_title_path, "FFPE_Cell2location_outputs_20231103", "meta.csv")
    FFPE_meta_csv_df = pd.read_csv(FFPE_meta_csv_path, header=None)
    FFPE_meta_WSA_names = FFPE_meta_csv_df[0].tolist()
    FFPE_meta_tif_names = FFPE_meta_csv_df[1].tolist()
    FFPE_meta_jpg_names = [name.replace("t.tif", ".jpg") for name in FFPE_meta_tif_names]
    assert sorted(FFPE_meta_jpg_names) == sorted(raw_data_folder_long_title_gathering_content_FFPE_images_list), f"The FFPE meta.csv and the gathering folder {raw_data_folder_long_title_gathering_path} do not match."
    FFPE_image_WSA_name_mapping = dict()
    for i in range(len(FFPE_meta_csv_df)):
        FFPE_image_WSA_name_mapping[FFPE_meta_jpg_names[i]] = FFPE_meta_WSA_names[i]
    
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    image, matrix, spatial = False, False, False
    
    for key, value in FFPE_image_WSA_name_mapping.items():
        if value == project_dataset:
            raw_image_file = key
            break
    else:
        raw_image_file = None
    assert raw_image_file is not None, f"Cannot find the image file for 5_locations_lung dataset FFPE {project_dataset}"
    raw_image_path = os.path.join(raw_data_folder_long_title_gathering_path, raw_image_file)
    _, extension = os.path.splitext(os.path.basename(raw_image_file))
    project_image_path = os.path.join(project_dataset_raw_data_path, f"{project_dataset}{extension}")
    if not os.path.exists(project_image_path):
        shutil.copy2(raw_image_path, project_image_path)
    image = True
    
    raw_matrix_folder_path = os.path.join(raw_data_folder_long_title_path, "FFPE_Cell2location_outputs_20231103", project_dataset, "outs", "filtered_feature_bc_matrix")
    assert os.path.exists(raw_matrix_folder_path), f"Cannot find the matrix folder for 5_locations_lung dataset FFPE {project_dataset}"
    project_matrix_folder_path = os.path.join(project_dataset_raw_data_path, "filtered_feature_bc_matrix")
    if not os.path.exists(project_matrix_folder_path):
        shutil.copytree(raw_matrix_folder_path, project_matrix_folder_path)
    matrix = True
    
    raw_spatial_folder_path = os.path.join(raw_data_folder_long_title_path, "FFPE_Cell2location_outputs_20231103", project_dataset, "outs", "spatial")
    assert os.path.exists(raw_spatial_folder_path), f"Cannot find the spatial folder for 5_locations_lung dataset FFPE {project_dataset}"
    project_spatial_folder_path = os.path.join(project_dataset_raw_data_path, "spatial")
    if not os.path.exists(project_spatial_folder_path):
        shutil.copytree(raw_spatial_folder_path, project_spatial_folder_path)
    spatial = True
    assert (image & matrix & spatial) == True, f"Missing filetype for 5_locations_lung dataset FFPE {project_dataset}: {'image' if not image else ''} {'matrix' if not matrix else ''} {'spatial' if not spatial else ''} is missing."
    
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for 5_locations_lung dataset FFPE {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    if not os.path.exists(project_dataset_raw_unzipped_data_path):
        os.makedirs(project_dataset_raw_unzipped_data_path)
    for content in project_dataset_raw_data_content_list:
        if "filtered_feature_bc_matrix" in content:
            double_unzip_tar_gz(
                project_dataset_raw_data_path=project_dataset_raw_data_path,
                project_dataset_raw_unzipped_data_path=project_dataset_raw_unzipped_data_path,
                filetype="matrix",
                already_in_folder_format=True,
                already_folder_name=content
            )
    end_time = time.time()
    print(f"Selecting data for 5_locations_lung dataset FFPE {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def Specific_datasets_BLEEP_project_list(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Specific_datasets", "BLEEP")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "BLEEP")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"BLEEP dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_tar_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".tar")]
    assert len(raw_data_folder_long_title_content_tar_file) == 1, f"BLEEP dataset has more than one tar files"
    raw_data_folder_long_title_content_tar_file = raw_data_folder_long_title_content_tar_file[0]
    raw_data_folder_long_title_content_tar_basename, _ = os.path.splitext(raw_data_folder_long_title_content_tar_file)
    raw_data_folder_long_title_content_tar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_file)
    raw_data_folder_long_title_content_untar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_basename)
    unzip_tar(
        tar_path=raw_data_folder_long_title_content_tar_path,
        untar_folder_path=raw_data_folder_long_title_content_untar_path,
    )
    raw_data_folder_long_title_content_untar_content_list = os.listdir(raw_data_folder_long_title_content_untar_path)
    project_datasets = [content.split("_")[0] for content in raw_data_folder_long_title_content_untar_content_list]
    project_datasets = list(set(project_datasets))
    return project_datasets

def Specific_datasets_BLEEP_single_helper(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Specific_datasets", "BLEEP")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "BLEEP")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"BLEEP dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_tar_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".tar")]
    assert len(raw_data_folder_long_title_content_tar_file) == 1, f"BLEEP dataset has more than one tar files"
    raw_data_folder_long_title_content_tar_file = raw_data_folder_long_title_content_tar_file[0]
    raw_data_folder_long_title_content_tar_basename, _ = os.path.splitext(raw_data_folder_long_title_content_tar_file)
    raw_data_folder_long_title_content_untar_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_tar_basename)
    raw_data_folder_long_title_content_untar_content_list = os.listdir(raw_data_folder_long_title_content_untar_path)
    
    raw_dataset_content_list = [content for content in raw_data_folder_long_title_content_untar_content_list if (project_dataset in content)]
    assert len(raw_dataset_content_list) == 13, f"More than thirteen files for BLEEP dataset {project_dataset}"
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    image, stdata, spots = False, False, False
    
    raw_image_file = [content for content in raw_dataset_content_list if (".tiff" in content)]
    assert len(raw_image_file) == 1, f"More than one image file for BLEEP dataset {project_dataset}"
    raw_image_file = raw_image_file[0]
    raw_image_path = os.path.join(raw_data_folder_long_title_content_untar_path, raw_image_file)
    project_image_path = os.path.join(project_dataset_raw_data_path, raw_image_file)
    if not os.path.exists(project_image_path):
        shutil.copy2(raw_image_path, project_image_path)
    image = True
    
    raw_barcodes_file = [content for content in raw_dataset_content_list if ("barcodes" in content)]
    assert len(raw_barcodes_file) == 1, f"More than one barcodes file for BLEEP dataset {project_dataset}"
    raw_barcodes_file = raw_barcodes_file[0]
    raw_barcodes_path = os.path.join(raw_data_folder_long_title_content_untar_path, raw_barcodes_file)
    raw_matrix_file = [content for content in raw_dataset_content_list if ("matrix.mtx" in content)]
    assert len(raw_matrix_file) == 1, f"More than one matrix file for BLEEP dataset {project_dataset}"
    raw_matrix_file = raw_matrix_file[0]
    raw_matrix_path = os.path.join(raw_data_folder_long_title_content_untar_path, raw_matrix_file)
    raw_features_file = [content for content in raw_dataset_content_list if ("features" in content)]
    assert len(raw_features_file) == 1, f"More than one features file for BLEEP dataset {project_dataset}"
    raw_features_file = raw_features_file[0]
    raw_features_path = os.path.join(raw_data_folder_long_title_content_untar_path, raw_features_file)
    project_stdata_path = os.path.join(project_dataset_raw_data_path, "filtered_feature_bc_matrix")
    if not os.path.exists(project_stdata_path):
        os.makedirs(project_stdata_path)
        shutil.copy2(raw_barcodes_path, project_stdata_path)
        shutil.copy2(raw_matrix_path, project_stdata_path)
        shutil.copy2(raw_features_path, project_stdata_path)
    stdata = True
    
    raw_spots_file = [content for content in raw_dataset_content_list if ("tissue_positions_list" in content)]
    assert len(raw_spots_file) == 1, f"More than one spots file for BLEEP dataset {project_dataset}"
    raw_spots_file = raw_spots_file[0]
    raw_spots_path = os.path.join(raw_data_folder_long_title_content_untar_path, raw_spots_file)
    project_spots_path = os.path.join(project_dataset_raw_data_path, raw_spots_file)
    if not os.path.exists(project_spots_path):
        shutil.copy2(raw_spots_path, project_spots_path)
    spots = True
    assert (image & stdata & spots) == True, f"Missing filetype for BLEEP dataset {project_dataset}: {'image' if not image else ''} {'stdata' if not stdata else ''} {'spots' if not spots else ''} is missing."
    
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for BLEEP dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    if not os.path.exists(project_dataset_raw_unzipped_data_path):
        os.makedirs(project_dataset_raw_unzipped_data_path)
    for content in project_dataset_raw_data_content_list:
        if "filtered_feature_bc_matrix" in content:
            double_unzip_tar_gz(
                project_dataset_raw_data_path=project_dataset_raw_data_path,
                project_dataset_raw_unzipped_data_path=project_dataset_raw_unzipped_data_path,
                filetype="matrix",
                already_in_folder_format=True,
                already_folder_name=content
            )
        elif content.endswith(".gz"):
            gz_path = os.path.join(project_dataset_raw_data_path, content)
            unzip_gz(
                gz_path=gz_path,
                ungz_folder_path=project_dataset_raw_unzipped_data_path,
            )
    end_time = time.time()
    print(f"Selecting data for BLEEP dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def Specific_datasets_HER2ST_project_list(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Specific_datasets", "HER2ST_Version_3_0")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "HER2ST")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"HER2ST dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    for content in raw_data_folder_long_title_content_list:
        if content.endswith(".zip"):
            if "code" in content:
                if not os.path.exists(os.path.join(raw_data_folder_long_title_path, "her2st-master")):
                    zip_path = os.path.join(raw_data_folder_long_title_path, content)
                    unzip_zip(
                        zip_path=zip_path,
                        unzip_folder_path=raw_data_folder_long_title_path,
                    )
            else:
                content_basename, _ = os.path.splitext(os.path.basename(content))
                zip_path = os.path.join(raw_data_folder_long_title_path, content)
                unzip_path = os.path.join(raw_data_folder_long_title_path, content_basename)
                if not os.path.exists(unzip_path):
                    if "count-matrices" in content or "images" in content:
                        password = "zNLXkYk3Q9znUseS"
                        with pyzipper.AESZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(path=raw_data_folder_long_title_path, pwd=password.encode('utf-8'))
                    elif "meta" in content or "spot-selections" in content:
                        os.makedirs(unzip_path)
                        password = "yUx44SzG6NdB32gY"
                        with pyzipper.AESZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(path=unzip_path, pwd=password.encode('utf-8'))
    raw_data_folder_long_title_image_folder_path = os.path.join(raw_data_folder_long_title_path, "images", "HE")
    raw_data_folder_long_title_image_content_list = os.listdir(raw_data_folder_long_title_image_folder_path)
    raw_data_folder_long_title_annotation_folder_path = os.path.join(raw_data_folder_long_title_path, "her2st-master", "app", "www", "imgs")
    raw_data_folder_long_title_annotation_content_list = os.listdir(raw_data_folder_long_title_annotation_folder_path)
    raw_data_folder_long_title_count_matrices_folder_path = os.path.join(raw_data_folder_long_title_path, "count-matrices")
    raw_data_folder_long_title_count_matrices_content_list = os.listdir(raw_data_folder_long_title_count_matrices_folder_path)
    raw_data_folder_long_title_spots_folder_path = os.path.join(raw_data_folder_long_title_path, "spot-selections")
    raw_data_folder_long_title_spots_content_list = os.listdir(raw_data_folder_long_title_spots_folder_path)
    project_datasets = [os.path.splitext(image)[0] for image in raw_data_folder_long_title_image_content_list]
    return project_datasets

def Specific_datasets_HER2ST_single_helper(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Specific_datasets", "HER2ST_Version_3_0")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "HER2ST")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"HER2ST dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_image_folder_path = os.path.join(raw_data_folder_long_title_path, "images", "HE")
    raw_data_folder_long_title_image_content_list = os.listdir(raw_data_folder_long_title_image_folder_path)
    raw_data_folder_long_title_annotation_folder_path = os.path.join(raw_data_folder_long_title_path, "her2st-master", "app", "www", "imgs")
    raw_data_folder_long_title_annotation_content_list = os.listdir(raw_data_folder_long_title_annotation_folder_path)
    raw_data_folder_long_title_count_matrices_folder_path = os.path.join(raw_data_folder_long_title_path, "count-matrices")
    raw_data_folder_long_title_count_matrices_content_list = os.listdir(raw_data_folder_long_title_count_matrices_folder_path)
    raw_data_folder_long_title_spots_folder_path = os.path.join(raw_data_folder_long_title_path, "spot-selections")
    raw_data_folder_long_title_spots_content_list = os.listdir(raw_data_folder_long_title_spots_folder_path)
    
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    image, annotation, count_matrix, spots = False, False, False, False
    
    raw_image_file = [content for content in raw_data_folder_long_title_image_content_list if (project_dataset in content)]
    assert len(raw_image_file) == 1, f"More than one image file for HER2ST dataset {project_dataset}"
    raw_image_file = raw_image_file[0]
    raw_image_path = os.path.join(raw_data_folder_long_title_image_folder_path, raw_image_file)
    project_image_path = os.path.join(project_dataset_raw_data_path, raw_image_file)
    if not os.path.exists(project_image_path):
        shutil.copy2(raw_image_path, project_image_path)
    image = True
    
    raw_annotation_file = [content for content in raw_data_folder_long_title_annotation_content_list if (project_dataset in content)]
    assert len(raw_annotation_file) == 1, f"More than one annotation file for HER2ST dataset {project_dataset}"
    raw_annotation_file = raw_annotation_file[0]
    raw_annotation_path = os.path.join(raw_data_folder_long_title_annotation_folder_path, raw_annotation_file)
    project_annotation_path = os.path.join(project_dataset_raw_data_path, raw_annotation_file)
    if not os.path.exists(project_annotation_path):
        shutil.copy2(raw_annotation_path, project_annotation_path)
    annotation = True
    
    raw_count_matrix_file = [content for content in raw_data_folder_long_title_count_matrices_content_list if (project_dataset in content)]
    assert len(raw_count_matrix_file) == 1, f"More than one count matrix file for HER2ST dataset {project_dataset}"
    raw_count_matrix_file = raw_count_matrix_file[0]
    raw_count_matrix_path = os.path.join(raw_data_folder_long_title_count_matrices_folder_path, raw_count_matrix_file)
    project_count_matrix_path = os.path.join(project_dataset_raw_data_path, raw_count_matrix_file)
    if not os.path.exists(project_count_matrix_path):
        shutil.copy2(raw_count_matrix_path, project_count_matrix_path)
    count_matrix = True
    
    raw_spots_file = [content for content in raw_data_folder_long_title_spots_content_list if (project_dataset in content)]
    assert len(raw_spots_file) == 1, f"More than one spots file for HER2ST dataset {project_dataset}"
    raw_spots_file = raw_spots_file[0]
    raw_spots_path = os.path.join(raw_data_folder_long_title_spots_folder_path, raw_spots_file)
    project_spots_path = os.path.join(project_dataset_raw_data_path, raw_spots_file)
    if not os.path.exists(project_spots_path):
        shutil.copy2(raw_spots_path, project_spots_path)
    spots = True
    assert (image & annotation & count_matrix & spots) == True, f"Missing filetype for HER2ST dataset {project_dataset}: {'image' if not image else ''} {'annotation' if not annotation else ''} {'count_matrix' if not count_matrix else ''} {'spots' if not spots else ''} is missing."
    
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 4, f"Unmatching between the number of files and the expected number of files for HER2ST dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    if not os.path.exists(project_dataset_raw_unzipped_data_path):
        os.makedirs(project_dataset_raw_unzipped_data_path)
    for content in project_dataset_raw_data_content_list:
        if content.endswith(".gz"):
            gz_path = os.path.join(project_dataset_raw_data_path, content)
            unzip_gz(
                gz_path=gz_path,
                ungz_folder_path=project_dataset_raw_unzipped_data_path,
            )    
    end_time = time.time()
    print(f"Selecting data for HER2ST dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def Specific_datasets_STNet_project_list(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Specific_datasets", "STNet_Version_5_0")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "STNet")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"STNet dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_zip_file = [content for content in raw_data_folder_long_title_content_list if content.endswith(".zip")]
    assert len(raw_data_folder_long_title_content_zip_file) == 1, f"STNet dataset has more than one zip file"
    raw_data_folder_long_title_content_zip_file = raw_data_folder_long_title_content_zip_file[0]
    raw_data_folder_long_title_content_zip_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_zip_file)
    unzip_zip(
        zip_path=raw_data_folder_long_title_content_zip_path,
        unzip_folder_path=raw_data_folder_long_title_path,
    )
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_folder_list = [content for content in raw_data_folder_long_title_content_list if os.path.isdir(os.path.join(raw_data_folder_long_title_path, content))]
    assert len(raw_data_folder_long_title_content_folder_list) == 1, f"STNet dataset has more than one folder"
    raw_data_folder_long_title_content_folder = raw_data_folder_long_title_content_folder_list[0]
    raw_data_folder_long_title_content_folder_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_folder)
    raw_data_folder_long_title_content_folder_content_list = os.listdir(raw_data_folder_long_title_content_folder_path)
    patient_id_list = list()
    for file_name in raw_data_folder_long_title_content_folder_content_list:
        if "csv.gz" in file_name:        
            start_index = file_name.index("spots_") + len("spots_")
            end_index = file_name.index(".")
            patient_id = file_name[start_index:end_index]
            patient_id_list.append(patient_id)
        elif "tsv.gz" in file_name:
            if "Coords" in file_name:
                end_index = file_name.index('_Coords')
                patient_id = file_name[:end_index]
                patient_id_list.append(patient_id)
            elif "stdata" in file_name:
                end_index = file_name.index('_stdata')
                patient_id = file_name[:end_index]
                patient_id_list.append(patient_id)
        elif ".jpg" in file_name:
            start_index = file_name.index("HE_") + len("HE_")
            end_index = file_name.index(".")
            patient_id = file_name[start_index:end_index]
            patient_id_list.append(patient_id)
        elif "metadata" in file_name:
            continue
        else:
            raise ValueError(f"There is a special file type in STNet dataset, which is {file_name}")
    patient_id_list = list(set(patient_id_list))
    project_datasets = list(set([patient_id[2:] for patient_id in patient_id_list]))
    return project_datasets

def Specific_datasets_STNet_single_helper(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Specific_datasets", "STNet_Version_5_0")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "STNet")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"STNet dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_folder_list = [content for content in raw_data_folder_long_title_content_list if os.path.isdir(os.path.join(raw_data_folder_long_title_path, content))]
    assert len(raw_data_folder_long_title_content_folder_list) == 1, f"STNet dataset has more than one folder"
    raw_data_folder_long_title_content_folder = raw_data_folder_long_title_content_folder_list[0]
    raw_data_folder_long_title_content_folder_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_folder)
    raw_data_folder_long_title_content_folder_content_list = os.listdir(raw_data_folder_long_title_content_folder_path)
     
    raw_dataset_content_list = [content for content in raw_data_folder_long_title_content_folder_content_list if project_dataset in content]
    assert len(raw_dataset_content_list) == 4, f"More than four files for STNet dataset {project_dataset}"
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    image, stdata, coords, spots = False, False, False, False
    
    raw_image_file = [content for content in raw_dataset_content_list if (".jpg" in content)]
    assert len(raw_image_file) == 1, f"More than one image file for STNet dataset {project_dataset}"
    raw_image_file = raw_image_file[0]
    raw_image_path = os.path.join(raw_data_folder_long_title_content_folder_path, raw_image_file)
    project_image_path = os.path.join(project_dataset_raw_data_path, raw_image_file)
    if not os.path.exists(project_image_path):
        shutil.copy2(raw_image_path, project_image_path)
    image = True
    
    raw_stdata_file = [content for content in raw_dataset_content_list if ("stdata" in content)]
    assert len(raw_stdata_file) == 1, f"More than one stdata file for STNet dataset {project_dataset}"
    raw_stdata_file = raw_stdata_file[0]
    raw_stdata_path = os.path.join(raw_data_folder_long_title_content_folder_path, raw_stdata_file)
    project_stdata_path = os.path.join(project_dataset_raw_data_path, raw_stdata_file)
    if not os.path.exists(project_stdata_path):
        shutil.copy2(raw_stdata_path, project_stdata_path)
    stdata = True
    
    raw_coords_file = [content for content in raw_dataset_content_list if ("Coords" in content)]
    assert len(raw_coords_file) == 1, f"More than one coords file for STNet dataset {project_dataset}"
    raw_coords_file = raw_coords_file[0]
    raw_coords_path = os.path.join(raw_data_folder_long_title_content_folder_path, raw_coords_file)
    project_coords_path = os.path.join(project_dataset_raw_data_path, raw_coords_file)
    if not os.path.exists(project_coords_path):
        shutil.copy2(raw_coords_path, project_coords_path)
    coords = True
    
    raw_spots_file = [content for content in raw_dataset_content_list if ("spots" in content)]
    assert len(raw_spots_file) == 1, f"More than one spots file for STNet dataset {project_dataset}"
    raw_spots_file = raw_spots_file[0]
    raw_spots_path = os.path.join(raw_data_folder_long_title_content_folder_path, raw_spots_file)
    project_spots_path = os.path.join(project_dataset_raw_data_path, raw_spots_file)
    if not os.path.exists(project_spots_path):
        shutil.copy2(raw_spots_path, project_spots_path)
    spots = True
    assert (image & stdata & coords & spots) == True, f"Missing filetype for STNet dataset {project_dataset}: {'image' if not image else ''} {'stdata' if not stdata else ''} {'coords' if not coords else ''} {'spots' if not spots else ''} is missing."
    
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 4, f"Unmatching between the number of files and the expected number of files for STNet dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    if not os.path.exists(project_dataset_raw_unzipped_data_path):
        os.makedirs(project_dataset_raw_unzipped_data_path)
    for content in project_dataset_raw_data_content_list:
        if content.endswith(".gz"):
            gz_path = os.path.join(project_dataset_raw_data_path, content)
            unzip_gz(
                gz_path=gz_path,
                ungz_folder_path=project_dataset_raw_unzipped_data_path,
            )
    end_time = time.time()
    print(f"Selecting data for STNet dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def Zenodo001_project_list(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Zenodo", "Zenodo001")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "Zenodo", "Zenodo001")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"Zenodo001 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_zip_files = [content for content in raw_data_folder_long_title_content_list if content.endswith(".zip")]
    assert len(raw_data_folder_long_title_content_zip_files) == 4, f"Zenodo001 dataset has more than four zip files"
    for raw_data_folder_long_title_content_zip_file in raw_data_folder_long_title_content_zip_files:
        zip_basename, _ = os.path.splitext(raw_data_folder_long_title_content_zip_file)
        zip_path = os.path.join(raw_data_folder_long_title_path, raw_data_folder_long_title_content_zip_file)
        unzip_folder_path = os.path.join(raw_data_folder_long_title_path, zip_basename)
        unzip_zip(
            zip_path=zip_path,
            unzip_folder_path=unzip_folder_path,
        )
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)
    raw_data_folder_long_title_content_folder_list = [content for content in raw_data_folder_long_title_content_list if os.path.isdir(os.path.join(raw_data_folder_long_title_path, content))]
    assert len(raw_data_folder_long_title_content_folder_list) == 4, f"Zenodo001 dataset has more than four folders"
    project_datasets = raw_data_folder_long_title_content_folder_list
    return project_datasets

def Zenodo001_single_helper(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Zenodo", "Zenodo001")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "Zenodo", "Zenodo001")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"Zenodo001 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    
    raw_dataset_path = os.path.join(raw_data_folder_long_title_path, project_dataset)
    raw_dataset_title_list = os.listdir(raw_dataset_path)
    raw_dataset_title_list = [title for title in raw_dataset_title_list if not title.startswith("__")]
    assert len(raw_dataset_title_list) == 1, f"More than one title for Zenodo001 dataset {project_dataset}"
    raw_dataset_title = raw_dataset_title_list[0]
    raw_dataset_title_path = os.path.join(raw_dataset_path, raw_dataset_title)
    raw_dataset_title_content_list = os.listdir(raw_dataset_title_path)
    raw_dataset_title_content_list = [content for content in raw_dataset_title_content_list if not content.startswith(".")]
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    image, matrix, spots = False, False, False
    
    raw_image_file = [content for content in raw_dataset_title_content_list if (".svs" in content)]
    assert len(raw_image_file) == 1, f"More than one image file for Zenodo001 dataset {project_dataset}"
    raw_image_file = raw_image_file[0]
    raw_image_path = os.path.join(raw_dataset_title_path, raw_image_file)
    project_image_path = os.path.join(project_dataset_raw_data_path, raw_image_file)
    if not os.path.exists(project_image_path):
        shutil.copy2(raw_image_path, project_image_path)
    image = True
    
    raw_matrix_folder = [content for content in raw_dataset_title_content_list if ("filtered_feature_bc_matrix" in content)]
    assert len(raw_matrix_folder) == 1, f"More than one matrix folder for Zenodo001 dataset {project_dataset}"
    raw_matrix_folder = raw_matrix_folder[0]
    raw_matrix_folder_path = os.path.join(raw_dataset_title_path, raw_matrix_folder)
    project_matrix_folder_path = os.path.join(project_dataset_raw_data_path, "filtered_feature_bc_matrix")
    if not os.path.exists(project_matrix_folder_path):
        shutil.copytree(raw_matrix_folder_path, project_matrix_folder_path)
    matrix = True
    
    raw_spots_file = [content for content in raw_dataset_title_content_list if ("tissue_positions_list" in content)]
    assert len(raw_spots_file) == 1, f"More than one spots file for Zenodo001 dataset {project_dataset}"
    raw_spots_file = raw_spots_file[0]
    raw_spots_path = os.path.join(raw_dataset_title_path, raw_spots_file)
    project_spots_path = os.path.join(project_dataset_raw_data_path, raw_spots_file)
    if not os.path.exists(project_spots_path):
        shutil.copy2(raw_spots_path, project_spots_path)
    spots = True
    assert (image & matrix & spots) == True, f"Missing filetype for Zenodo001 dataset {project_dataset}: {'image' if not image else ''} {'matrix' if not matrix else ''} {'spots' if not spots else ''} is missing."
    
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for Zenodo001 dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    if not os.path.exists(project_dataset_raw_unzipped_data_path):
        os.makedirs(project_dataset_raw_unzipped_data_path)
    for content in project_dataset_raw_data_content_list:
        if "filtered_feature_bc_matrix" in content:
            double_unzip_tar_gz(
                project_dataset_raw_data_path=project_dataset_raw_data_path,
                project_dataset_raw_unzipped_data_path=project_dataset_raw_unzipped_data_path,
                filetype="matrix",
                already_in_folder_format=True,
                already_folder_name=content
            )
    end_time = time.time()
    print(f"Selecting data for Zenodo001 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def Zenodo002_project_list(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
) -> List:
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Zenodo", "Zenodo002")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "Zenodo", "Zenodo002")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"Zenodo002 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)

    raw_data_folder_long_title_tar_gz_unzipped_path = os.path.join(raw_data_folder_long_title_path, "10X_Visium_unzipped")
    if not os.path.exists(raw_data_folder_long_title_tar_gz_unzipped_path):
        os.makedirs(raw_data_folder_long_title_tar_gz_unzipped_path)
    raw_data_folder_long_title_content_tar_gz_files_list = [content for content in raw_data_folder_long_title_content_list if content.endswith(".tar.gz")]
    for tar_gz_file in raw_data_folder_long_title_content_tar_gz_files_list:
        tar_gz_file_basename = (tar_gz_file.split("_")[2]).split(".")[0]
        tar_gz_path = os.path.join(raw_data_folder_long_title_path, tar_gz_file)
        untar_gz_path = os.path.join(raw_data_folder_long_title_tar_gz_unzipped_path, tar_gz_file_basename)
        if not os.path.exists(untar_gz_path):
            unzip_tar_gz(
                tar_gz_path=tar_gz_path,
                untar_gz_folder_path=raw_data_folder_long_title_tar_gz_unzipped_path,
            )
        untar_gz_outs_path = os.path.join(raw_data_folder_long_title_tar_gz_unzipped_path, tar_gz_file_basename, "outs")
        untar_gz_outs_content_list = os.listdir(untar_gz_outs_path)
        untar_gz_outs_unzipped_path = os.path.join(raw_data_folder_long_title_tar_gz_unzipped_path, tar_gz_file_basename, "outs_unzipped")
        if not os.path.exists(untar_gz_outs_unzipped_path):
            os.makedirs(untar_gz_outs_unzipped_path)
        untar_gz_outs_content_matrix_zip_file = [content for content in untar_gz_outs_content_list if "filtered_feature_bc_matrix.zip" in content]
        assert len(untar_gz_outs_content_matrix_zip_file) == 1, f"More than one matrix zip file for Zenodo002 dataset {tar_gz_file_basename}"
        untar_gz_outs_content_matrix_zip_file = untar_gz_outs_content_matrix_zip_file[0]
        untar_gz_outs_content_matrix_zip_path = os.path.join(untar_gz_outs_path, untar_gz_outs_content_matrix_zip_file)
        unzip_zip(
            zip_path=untar_gz_outs_content_matrix_zip_path,
            unzip_folder_path=untar_gz_outs_unzipped_path,
        )
        untar_gz_outs_content_spatial_zip_file = [content for content in untar_gz_outs_content_list if ("spatial.zip" in content)]
        assert len(untar_gz_outs_content_spatial_zip_file) == 1, f"More than one spatial zip file for Zenodo002 dataset {tar_gz_file_basename}"
        untar_gz_outs_content_spatial_zip_file = untar_gz_outs_content_spatial_zip_file[0]
        untar_gz_outs_content_spatial_zip_path = os.path.join(untar_gz_outs_path, untar_gz_outs_content_spatial_zip_file)
        unzip_zip(
            zip_path=untar_gz_outs_content_spatial_zip_path,
            unzip_folder_path=untar_gz_outs_unzipped_path,
        )

    meta_csv_file = [content for content in raw_data_folder_long_title_content_list if ("metadata" in content) and (".csv" in content)]
    assert len(meta_csv_file) == 1, f"More than one metadata csv file for Zenodo002 dataset"
    meta_csv_file = meta_csv_file[0]
    meta_csv_path = os.path.join(raw_data_folder_long_title_path, meta_csv_file)
    meta_csv_df = pd.read_csv(meta_csv_path)
    meta_xlsx_file = [content for content in raw_data_folder_long_title_content_list if ("metadata" in content) and (".xlsx" in content)]
    assert len(meta_xlsx_file) == 1, f"More than one metadata xlsx file for Zenodo002 dataset"
    meta_xlsx_file = meta_xlsx_file[0]
    meta_xlsx_path = os.path.join(raw_data_folder_long_title_path, meta_xlsx_file)
    meta_xlsx_df = pd.read_excel(meta_xlsx_path)
    meta_df = pd.merge(meta_csv_df, meta_xlsx_df, on=["patient", "patient_region_id"])
    P6_filter_condition1 = (meta_df['patient'] == 'P6') & (meta_df['hca_sample_id'] == "ACH0021") & (meta_df['sample_id'] == "Visium_17_CK295")
    P6_filter_condition2 = (meta_df['patient'] == 'P6') & (meta_df['hca_sample_id'] == "ACH0022") & (meta_df['sample_id'] == "Visium_2_CK280")
    meta_df = meta_df.drop(meta_df[P6_filter_condition1].index)
    meta_df = meta_df.drop(meta_df.loc[P6_filter_condition2].index)
    mapping_info_to_image_dict = dict(zip(meta_df["hca_sample_id"], meta_df["sample_id"]))
    project_datasets = list(mapping_info_to_image_dict.keys())
    raw_data_folder_long_title_tar_gz_unzipped_content_list = os.listdir(raw_data_folder_long_title_tar_gz_unzipped_path)
    raw_data_folder_long_title_tar_gz_unzipped_image_files_list = [content for content in raw_data_folder_long_title_content_list if (".tif" in content)]
    for project_dataset in project_datasets:
        assert project_dataset in raw_data_folder_long_title_tar_gz_unzipped_content_list, f"Missing folder for Zenodo002 dataset {project_dataset}"
        image_name = mapping_info_to_image_dict[project_dataset]
        if "AKK" in image_name:
            AKK_part, num_part = image_name.split("_")
            image_name = num_part + "_2019" + AKK_part
        image_file = [c for c in raw_data_folder_long_title_tar_gz_unzipped_image_files_list if image_name in c]
        assert len(image_file) == 1, f"More than one image file for Zenodo002 dataset {project_dataset}"
    return project_datasets

def Zenodo002_single_helper(
    main_data_storage: str,
    raw_data_folder_name: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    raw_data_folder_path = os.path.join(main_data_storage, raw_data_folder_name, "Zenodo", "Zenodo002")
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, "Zenodo", "Zenodo002")
    raw_data_folder_long_title_list = os.listdir(raw_data_folder_path)
    raw_data_folder_long_title_list = [content for content in raw_data_folder_long_title_list if os.path.isdir(os.path.join(raw_data_folder_path, content))]
    assert len(raw_data_folder_long_title_list) == 1, f"Zenodo002 dataset has more than one long title"
    raw_data_folder_long_title_name = raw_data_folder_long_title_list[0]
    raw_data_folder_long_title_path = os.path.join(raw_data_folder_path, raw_data_folder_long_title_name)
    raw_data_folder_long_title_content_list = os.listdir(raw_data_folder_long_title_path)

    raw_data_folder_long_title_tar_gz_unzipped_path = os.path.join(raw_data_folder_long_title_path, "10X_Visium_unzipped")
    raw_data_folder_long_title_content_tar_gz_files_list = [content for content in raw_data_folder_long_title_content_list if content.endswith(".tar.gz")]
    for tar_gz_file in raw_data_folder_long_title_content_tar_gz_files_list:
        tar_gz_file_basename = (tar_gz_file.split("_")[2]).split(".")[0]
        untar_gz_outs_unzipped_path = os.path.join(raw_data_folder_long_title_tar_gz_unzipped_path, tar_gz_file_basename, "outs_unzipped")

    meta_csv_file = [content for content in raw_data_folder_long_title_content_list if ("metadata" in content) and (".csv" in content)]
    assert len(meta_csv_file) == 1, f"More than one metadata csv file for Zenodo002 dataset"
    meta_csv_file = meta_csv_file[0]
    meta_csv_path = os.path.join(raw_data_folder_long_title_path, meta_csv_file)
    meta_csv_df = pd.read_csv(meta_csv_path)
    meta_xlsx_file = [content for content in raw_data_folder_long_title_content_list if ("metadata" in content) and (".xlsx" in content)]
    assert len(meta_xlsx_file) == 1, f"More than one metadata xlsx file for Zenodo002 dataset"
    meta_xlsx_file = meta_xlsx_file[0]
    meta_xlsx_path = os.path.join(raw_data_folder_long_title_path, meta_xlsx_file)
    meta_xlsx_df = pd.read_excel(meta_xlsx_path)
    meta_df = pd.merge(meta_csv_df, meta_xlsx_df, on=["patient", "patient_region_id"])
    P6_filter_condition1 = (meta_df['patient'] == 'P6') & (meta_df['hca_sample_id'] == "ACH0021") & (meta_df['sample_id'] == "Visium_17_CK295")
    P6_filter_condition2 = (meta_df['patient'] == 'P6') & (meta_df['hca_sample_id'] == "ACH0022") & (meta_df['sample_id'] == "Visium_2_CK280")
    meta_df = meta_df.drop(meta_df[P6_filter_condition1].index)
    meta_df = meta_df.drop(meta_df.loc[P6_filter_condition2].index)
    mapping_info_to_image_dict = dict(zip(meta_df["hca_sample_id"], meta_df["sample_id"]))
    raw_data_folder_long_title_tar_gz_unzipped_image_files_list = [content for content in raw_data_folder_long_title_content_list if (".tif" in content)]
    
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    if not os.path.exists(project_dataset_raw_data_path):
        os.makedirs(project_dataset_raw_data_path)
    image, matrix, spatial = False, False, False
    
    image_name = mapping_info_to_image_dict[project_dataset]
    if "AKK" in image_name:
        AKK_part, num_part = image_name.split("_")
        image_name = num_part + "_2019" + AKK_part
    raw_image_file = [c for c in raw_data_folder_long_title_tar_gz_unzipped_image_files_list if image_name in c]
    assert len(raw_image_file) == 1
    raw_image_file = raw_image_file[0]
    raw_image_path = os.path.join(raw_data_folder_long_title_path, raw_image_file)
    project_image_path = os.path.join(project_dataset_raw_data_path, raw_image_file)
    if not os.path.exists(project_image_path):
        shutil.copy2(raw_image_path, project_image_path)
    image = True
    
    # Mistake discovered on 2024/04/15
    # matrix_and_spatial_folder_paths = find_folders_for_super_long_path(untar_gz_outs_unzipped_path)
    matrix_and_spatial_folder_paths = find_folders_for_super_long_path(os.path.join(raw_data_folder_long_title_tar_gz_unzipped_path, project_dataset, "outs_unzipped"))
    raw_matrix_folder_path = [path for path in matrix_and_spatial_folder_paths if "outs/filtered_feature_bc_matrix" in path]
    assert len(raw_matrix_folder_path) == 1, f"More than one matrix path for Zenodo002 dataset {project_dataset}"
    raw_matrix_folder_path = raw_matrix_folder_path[0]
    project_matrix_folder_path = os.path.join(project_dataset_raw_data_path, "filtered_feature_bc_matrix")
    if not os.path.exists(project_matrix_folder_path):
        shutil.copytree(raw_matrix_folder_path, project_matrix_folder_path)
    matrix = True
    
    raw_spatial_folder_path = [path for path in matrix_and_spatial_folder_paths if "outs/spatial" in path]
    assert len(raw_spatial_folder_path) == 1, f"More than one spatial path for Zenodo002 dataset {project_dataset}"
    raw_spatial_folder_path = raw_spatial_folder_path[0]
    project_spatial_folder_path = os.path.join(project_dataset_raw_data_path, "spatial")
    if not os.path.exists(project_spatial_folder_path):
        shutil.copytree(raw_spatial_folder_path, project_spatial_folder_path)
    spatial = True
    assert (image & matrix & spatial) == True, f"Missing filetype for Zenodo002 dataset {project_dataset}: {'image' if not image else ''} {'matrix' if not matrix else ''} {'spatial' if not spatial else ''} is missing."
    
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for Zenodo002 dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    if not os.path.exists(project_dataset_raw_unzipped_data_path):
        os.makedirs(project_dataset_raw_unzipped_data_path)
    for content in project_dataset_raw_data_content_list:
        if "filtered_feature_bc_matrix" in content:
            double_unzip_tar_gz(
                project_dataset_raw_data_path=project_dataset_raw_data_path,
                project_dataset_raw_unzipped_data_path=project_dataset_raw_unzipped_data_path,
                filetype="matrix",
                already_in_folder_format=True,
                already_folder_name=content
            )
    end_time = time.time()
    print(f"Selecting data for Zenodo002 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")





