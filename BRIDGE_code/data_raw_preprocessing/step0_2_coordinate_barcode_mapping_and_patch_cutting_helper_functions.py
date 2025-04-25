from typing import List, Optional, Union, Literal
import time
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import anndata
import shutil
from scipy.io import mmread
from scipy.sparse import csr_matrix, csc_matrix
import scanpy as sc
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
import tifffile
import h5py
import polars as pl
import openslide

from step0_preprocess_helper_functions import running_time_display, Genomics_project_dataset_to_organ

def locations_lung_non_FFPE_dataset_list(
    main_data_storage: str,
    project_data_folder_name: str,
) -> List:
    dataset_name = ["5_locations_lung"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    project_datasets_non_FFPE = [patient for patient in project_datasets if "1097207" not in patient]
    project_datasets_FFPE = [patient for patient in project_datasets if "1097207" in patient]
    assert len(project_datasets_FFPE) + len(project_datasets_non_FFPE) == len(project_datasets), f"Unmatching between the number of FFPE and non-FFPE 5 datasets and the total number of datasets in 5 locations lung"
    return project_datasets_non_FFPE

def locations_lung_non_FFPE_single_helper(
    main_data_storage: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["5_locations_lung"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 2, f"Unmatching between the number of files and the expected number of files for 5 locations lung dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)
    
    image, h5ad = False, False
    image_file = [content for content in project_dataset_raw_data_content_list if (".jpg" in content) or (".tif" in content)]
    assert len(image_file) == 1, f"More than one image file for 5 locations lung dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_data_path, image_file)
    image = True
    h5ad_file = [content for content in project_dataset_raw_data_content_list if ".h5ad" in content]
    assert len(h5ad_file) == 1, f"More than one h5ad file for 5 locations lung dataset {project_dataset}"
    h5ad_file = h5ad_file[0]
    raw_h5ad_path = os.path.join(project_dataset_raw_data_path, h5ad_file)
    h5ad = True
    assert (image & h5ad) == True, f"Missing filetype for 5 locations lung dataset {project_dataset}: {'image' if not image else ''} {'h5ad' if not h5ad else ''} is missing." 
    
    project_dataset_raw_stdata = anndata.read_h5ad(raw_h5ad_path)
    assert project_dataset_raw_stdata.obs["in_tissue"].unique() == [1], f"5 locations lung dataset {project_dataset} has spots that do not fall inside tissue."
    spot_barcode_df = (project_dataset_raw_stdata.obs)[["in_tissue", "array_row", "array_col"]]
    spot_barcode_df.index.name = "spot_barcode"
    spot_barcode_df_copy = spot_barcode_df.copy()
    spot_barcode_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    spot_barcode_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    spot_barcode_df_copy.loc[:, "organ_type"] = "lung"
    image_extension = os.path.splitext(raw_image_path)[1]
    spot_barcode_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in spot_barcode_df.index]
    spot_barcode_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in spot_barcode_df.index]
    spot_barcode_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in spot_barcode_df.index]
    spot_barcode_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in spot_barcode_df.index]
    positions_df = pd.DataFrame(project_dataset_raw_stdata.obsm["spatial"], columns=["pxl_row_in_fullres", "pxl_col_in_fullres"], index=spot_barcode_df.index)
    assert len(spot_barcode_df_copy) == len(positions_df), f"Unmatching between the number of spots in the barcode dataframe and the number of spots in the positions dataframe."
    total_obs_df = pd.concat([spot_barcode_df_copy, positions_df], axis=1)
    gene_feature_df = (project_dataset_raw_stdata.var)[["feature_types", "SYMBOL"]]
    gene_feature_df = gene_feature_df.reset_index()
    gene_feature_df = gene_feature_df.rename(columns={"ENSEMBL": "gene_barcode", "feature_types": "gene_expression", "SYMBOL": "gene_name"})
    gene_feature_df.set_index("gene_name", inplace=True)
    project_dataset_stdata = anndata.AnnData(project_dataset_raw_stdata.X)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (positions_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)
    # project_dataset_stdata.uns["spatial"] = {"WSI": {}}

    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)

    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    # image = cv2.imread(raw_image_path)
    # project_dataset_stdata.uns["spatial"]["WSI"]["images"] = {"hires": image}
    # project_dataset_stdata.uns["spatial"]["WSI"]["scalefactors"] = {"tissue_hires_scalef": 1, "spot_diameter_fullres": 224.0}
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path)
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image = Image.open(raw_image_path)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = row["pxl_row_in_fullres"]
            y_pixel = row["pxl_col_in_fullres"]
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))
        
    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for 5 locations lung dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def locations_lung_FFPE_dataset_list(
    main_data_storage: str,
    project_data_folder_name: str,
) -> List:
    dataset_name = ["5_locations_lung"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    project_datasets_non_FFPE = [patient for patient in project_datasets if "1097207" not in patient]
    project_datasets_FFPE = [patient for patient in project_datasets if "1097207" in patient]
    assert len(project_datasets_FFPE) + len(project_datasets_non_FFPE) == len(project_datasets), f"Unmatching between the number of FFPE and non-FFPE 5 datasets and the total number of datasets in 5 locations lung"
    return project_datasets_FFPE

def locations_lung_FFPE_single_helper(
    main_data_storage: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> List:
    start_time = time.time()
    dataset_name = ["5_locations_lung"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for 5 locations lung dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    project_dataset_raw_unzipped_data_content_list = os.listdir(project_dataset_raw_unzipped_data_path)
    assert len(project_dataset_raw_unzipped_data_content_list) == 1, f"Unmatching between the number of files and the expected number of files for 5 locations lung dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)
    
    image, positions, barcodes, features, matrix = False, False, False, False, False
    image_file = [content for content in project_dataset_raw_data_content_list if (".jpg" in content) or (".tif" in content)]
    assert len(image_file) == 1, f"More than one image file for 5 locations lung dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_data_path, image_file)
    image = True
    
    positions_folder = [content for content in project_dataset_raw_data_content_list if ("spatial" in content)]
    assert len(positions_folder) == 1, f"More than one positions folder for 5 locations lung dataset {project_dataset}"
    positions_folder = positions_folder[0]
    raw_positions_folder_path = os.path.join(project_dataset_raw_data_path, positions_folder)
    raw_positions_folder_content_list = os.listdir(raw_positions_folder_path)
    positions_file = [content for content in raw_positions_folder_content_list if ("tissue_positions_list" in content)]
    assert len(positions_file) == 1, f"More than one positions file for 5 locations lung dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(raw_positions_folder_path, positions_file)
    positions = True

    filtered_feature_bc_matrix_all_unzipped_folder = [content for content in project_dataset_raw_unzipped_data_content_list if "filtered_feature_bc_matrix_all_unzipped" in content]
    assert len(filtered_feature_bc_matrix_all_unzipped_folder) == 1, f"More than one filtered_feature_bc_matrix_all_unzipped folder for 5 locations lung dataset {project_dataset}"
    filtered_feature_bc_matrix_all_unzipped_folder = filtered_feature_bc_matrix_all_unzipped_folder[0]
    filtered_feature_bc_matrix_all_unzipped_folder_path = os.path.join(project_dataset_raw_unzipped_data_path, filtered_feature_bc_matrix_all_unzipped_folder)
    filtered_feature_bc_matrix_all_unzipped_folder_content_list = os.listdir(filtered_feature_bc_matrix_all_unzipped_folder_path)
    
    barcodes_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "barcodes.tsv" in content]
    assert len(barcodes_file) == 1, f"More than one barcodes file for 5 locations lung dataset {project_dataset}"
    barcodes_file = barcodes_file[0]
    raw_barcodes_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, barcodes_file)
    barcodes = True
    
    features_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "features.tsv" in content]
    assert len(features_file) == 1, f"More than one features file for 5 locations lung dataset {project_dataset}"
    features_file = features_file[0]
    raw_features_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, features_file)
    features = True
    
    matrix_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "matrix.mtx" in content]
    assert len(matrix_file) == 1, f"More than one matrix file for 5 locations lung dataset {project_dataset}"
    matrix_file = matrix_file[0]
    raw_matrix_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, matrix_file)
    matrix = True
    assert (image & positions & barcodes & features & matrix) == True, f"Missing filetype for 5 locations lung dataset {project_dataset}: {'image' if not image else ''} {'positions' if not positions else ''} {'barcodes' if not barcodes else ''} {'features' if not features else ''} {'matrix' if not matrix else ''} is missing."
    
    gene_counts_matrix = csr_matrix(mmread(raw_matrix_path).T)
    spot_barcode_df = pd.read_csv(raw_barcodes_path, header=None, names=["spot_barcode"])
    spot_barcode_df = spot_barcode_df.set_index("spot_barcode")
    positions_df = pd.read_csv(raw_positions_path, header=None, names=["spot_barcode", "in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"])
    positions_df = positions_df.set_index("spot_barcode")
    positions_df = positions_df[positions_df["in_tissue"] == 1]
    assert set(spot_barcode_df.index) == set(positions_df.index), f"Unmatching between the spot barcodes in barcodes.tsv and tissue_positions_list.csv, and the difference is {set(spot_barcode_df.index) - set(positions_df.index)}."
    concat_obs_df = pd.concat([spot_barcode_df, positions_df], axis=1)
    concat_obs_df_copy = concat_obs_df.copy()
    concat_obs_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    concat_obs_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    concat_obs_df_copy.loc[:, "organ_type"] = "lung"
    image_extension = os.path.splitext(raw_image_path)[1]
    concat_obs_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in concat_obs_df.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in concat_obs_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for 5 locations lung dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in concat_obs_df_copy.columns]}"
    concat_obs_df_copy = concat_obs_df_copy.reindex(columns=obs_columns_order)
    total_obs_df = concat_obs_df_copy
    gene_feature_df =  pd.read_csv(raw_features_path, header=None, delimiter='\t', names=["gene_barcode", "gene_name", "gene_expression"])
    gene_feature_df = gene_feature_df.set_index("gene_name")
    
    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (positions_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)
    # project_dataset_stdata.uns["spatial"] = {"WSI": {}}

    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)

    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    # image = cv2.imread(raw_image_path)
    # project_dataset_stdata.uns["spatial"]["WSI"]["images"] = {"hires": image}
    # project_dataset_stdata.uns["spatial"]["WSI"]["scalefactors"] = {"tissue_hires_scalef": 1, "spot_diameter_fullres": 224.0}
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path)
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image = Image.open(raw_image_path)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = row["pxl_row_in_fullres"]
            y_pixel = row["pxl_col_in_fullres"]
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))

    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for 5 locations lung dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def Genomics_dataset_list(
    main_data_storage: str,
    project_data_folder_name: str,
) -> List:
    dataset_name = ["10xGenomics"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    return project_datasets

def Genomics_single_helper(
    main_data_storage: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["10xGenomics"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for 10xGenomics dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    project_dataset_raw_unzipped_data_content_list = os.listdir(project_dataset_raw_unzipped_data_path)
    assert len(project_dataset_raw_unzipped_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for 10xGenomics dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)
    
    image, positions, barcodes, features, matrix = False, False, False, False, False
    image_file = [content for content in project_dataset_raw_data_content_list if (".jpg" in content) or (".tif" in content) or (".tiff" in content) or (".btf" in content)]
    assert len(image_file) == 1, f"More than one image file for 10xGenomics dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_data_path, image_file)
    image = True
    
    positions_folder = [content for content in project_dataset_raw_unzipped_data_content_list if ("spatial" in content)]
    assert len(positions_folder) == 1, f"More than one positions folder for 10xGenomics dataset {project_dataset}"
    positions_folder = positions_folder[0]
    raw_positions_folder_path = os.path.join(project_dataset_raw_unzipped_data_path, positions_folder)
    raw_positions_folder_content_list = os.listdir(raw_positions_folder_path)
    positions_file = [content for content in raw_positions_folder_content_list if ("tissue_positions_list" in content) or ("tissue_positions" in content)]
    assert len(positions_file) == 1, f"More than one positions file for 10xGenomics dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(raw_positions_folder_path, positions_file)
    positions = True

    filtered_feature_bc_matrix_all_unzipped_folder = [content for content in project_dataset_raw_unzipped_data_content_list if "filtered_feature_bc_matrix_all_unzipped" in content]
    assert len(filtered_feature_bc_matrix_all_unzipped_folder) == 1, f"More than one filtered_feature_bc_matrix_all_unzipped folder for 10xGenomics dataset {project_dataset}"
    filtered_feature_bc_matrix_all_unzipped_folder = filtered_feature_bc_matrix_all_unzipped_folder[0]
    filtered_feature_bc_matrix_all_unzipped_folder_path = os.path.join(project_dataset_raw_unzipped_data_path, filtered_feature_bc_matrix_all_unzipped_folder)
    filtered_feature_bc_matrix_all_unzipped_folder_content_list = os.listdir(filtered_feature_bc_matrix_all_unzipped_folder_path)
    
    barcodes_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "barcodes.tsv" in content]
    assert len(barcodes_file) == 1, f"More than one barcodes file for 10xGenomics dataset {project_dataset}"
    barcodes_file = barcodes_file[0]
    raw_barcodes_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, barcodes_file)
    barcodes = True
    
    features_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "features.tsv" in content]
    assert len(features_file) == 1, f"More than one features file for 10xGenomics dataset {project_dataset}"
    features_file = features_file[0]
    raw_features_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, features_file)
    features = True
    
    matrix_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "matrix.mtx" in content]
    assert len(matrix_file) == 1, f"More than one matrix file for 10xGenomics dataset {project_dataset}"
    matrix_file = matrix_file[0]
    raw_matrix_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, matrix_file)
    matrix = True
    assert (image & positions & barcodes & features & matrix) == True, f"Missing filetype for 10xGenomics dataset {project_dataset}: {'image' if not image else ''} {'positions' if not positions else ''} {'barcodes' if not barcodes else ''} {'features' if not features else ''} {'matrix' if not matrix else ''} is missing."
    
    gene_counts_matrix = csr_matrix(mmread(raw_matrix_path).T)
    spot_barcode_df = pd.read_csv(raw_barcodes_path, header=None, names=["spot_barcode"])
    spot_barcode_df = spot_barcode_df.set_index("spot_barcode")
    positions_df = pd.read_csv(raw_positions_path, header=None, names=["spot_barcode", "in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"])
    positions_df = positions_df.set_index("spot_barcode")
    # positions_df = positions_df[positions_df["in_tissue"] == 1]
    positions_df = positions_df[positions_df["in_tissue"].isin(["1", 1])]
    assert set(spot_barcode_df.index) == set(positions_df.index), f"Unmatching between the spot barcodes in barcodes.tsv and tissue_positions_list.csv, and the difference is {set(spot_barcode_df.index) - set(positions_df.index)}."
    concat_obs_df = pd.concat([spot_barcode_df, positions_df], axis=1)
    concat_obs_df_copy = concat_obs_df.copy()
    concat_obs_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    concat_obs_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    # concat_obs_df_copy.loc[:, "organ_type"] = "lung"
    Genomics_project_dataset_to_organ_dict = Genomics_project_dataset_to_organ()
    project_dataset_number = project_dataset.split("10xGenomics")[1]
    assert project_dataset_number in Genomics_project_dataset_to_organ_dict.keys(), f"Unmatching between the project dataset name and the expected project dataset name for 10xGenomics dataset {project_dataset}"
    concat_obs_df_copy.loc[:, "organ_type"] = Genomics_project_dataset_to_organ_dict[project_dataset_number]
    image_extension = os.path.splitext(raw_image_path)[1]
    if image_extension == ".btf":
        image_extension = ".tif"
    concat_obs_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in concat_obs_df.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in concat_obs_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for 10xGenomics dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in concat_obs_df_copy.columns]}"
    concat_obs_df_copy = concat_obs_df_copy.reindex(columns=obs_columns_order)
    total_obs_df = concat_obs_df_copy
    gene_feature_df =  pd.read_csv(raw_features_path, header=None, delimiter='\t', names=["gene_barcode", "gene_name", "gene_expression"])
    gene_feature_df = gene_feature_df.set_index("gene_name")
    
    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (positions_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)
    # project_dataset_stdata.uns["spatial"] = {"WSI": {}}

    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)

    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    # image = cv2.imread(raw_image_path)
    # project_dataset_stdata.uns["spatial"]["WSI"]["images"] = {"hires": image}
    # project_dataset_stdata.uns["spatial"]["WSI"]["scalefactors"] = {"tissue_hires_scalef": 1, "spot_diameter_fullres": 224.0}
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path)
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image_extension = os.path.splitext(raw_image_path)[1]
        if image_extension == ".btf":
            btf_image_data = tifffile.imread(raw_image_path)
            btf_image_data = np.uint8(btf_image_data)
            image = Image.fromarray(btf_image_data)
            image_extension = ".tif"
        else:
            image = Image.open(raw_image_path)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = int(row["pxl_row_in_fullres"])
            y_pixel = int(row["pxl_col_in_fullres"])
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))
        
        # for barcode, row in project_dataset_stdata.obs.iterrows():
        #     x_pixel = int(row["pxl_row_in_fullres"])
        #     y_pixel = int(row["pxl_col_in_fullres"])
        #     left = x_pixel - patch_size // 2
        #     right = x_pixel + patch_size // 2
        #     top = y_pixel - patch_size // 2
        #     bottom = y_pixel + patch_size // 2
        #     if (
        #         left < 0
        #         or top < 0
        #         or right > image.shape[1]
        #         or bottom > image.shape[0]
        #     ):
        #         continue
        #     patch = image[top:bottom, left:right]
        #     patch_path = os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}")
        #     cv2.imwrite(patch_path, patch)
    
    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for 10xGenomics dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def BLEEP_dataset_list(
    main_data_storage: str,
    project_data_folder_name: str,
) -> List:
    dataset_name = ["BLEEP"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    return project_datasets

def BLEEP_single_helper(
    main_data_storage: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["BLEEP"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    project_dataset_raw_unzipped_data_content_list = os.listdir(project_dataset_raw_unzipped_data_path)
    assert len(project_dataset_raw_unzipped_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for BLEEP dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)
        
    image, positions, barcodes, features, matrix = False, False, False, False, False
    image_file = [content for content in project_dataset_raw_unzipped_data_content_list if (".jpg" in content) or (".tif" in content) or (".tiff" in content) or (".btf" in content)]
    assert len(image_file) == 1, f"More than one image file for BLEEP dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_unzipped_data_path, image_file)
    image = True
    
    positions_file = [content for content in project_dataset_raw_unzipped_data_content_list if ("tissue_positions_list" in content) or ("tissue_positions" in content)]
    assert len(positions_file) == 1, f"More than one positions file for BLEEP dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(project_dataset_raw_unzipped_data_path, positions_file)
    positions = True

    filtered_feature_bc_matrix_all_unzipped_folder = [content for content in project_dataset_raw_unzipped_data_content_list if "filtered_feature_bc_matrix_all_unzipped" in content]
    assert len(filtered_feature_bc_matrix_all_unzipped_folder) == 1, f"More than one filtered_feature_bc_matrix_all_unzipped folder for BLEEP dataset {project_dataset}"
    filtered_feature_bc_matrix_all_unzipped_folder = filtered_feature_bc_matrix_all_unzipped_folder[0]
    filtered_feature_bc_matrix_all_unzipped_folder_path = os.path.join(project_dataset_raw_unzipped_data_path, filtered_feature_bc_matrix_all_unzipped_folder)
    filtered_feature_bc_matrix_all_unzipped_folder_content_list = os.listdir(filtered_feature_bc_matrix_all_unzipped_folder_path)
    
    barcodes_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "barcodes.tsv" in content]
    assert len(barcodes_file) == 1, f"More than one barcodes file for BLEEP dataset {project_dataset}"
    barcodes_file = barcodes_file[0]
    raw_barcodes_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, barcodes_file)
    barcodes = True
    
    features_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "features.tsv" in content]
    assert len(features_file) == 1, f"More than one features file for BLEEP dataset {project_dataset}"
    features_file = features_file[0]
    raw_features_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, features_file)
    features = True
    
    matrix_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "matrix.mtx" in content]
    assert len(matrix_file) == 1, f"More than one matrix file for BLEEP dataset {project_dataset}"
    matrix_file = matrix_file[0]
    raw_matrix_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, matrix_file)
    matrix = True
    assert (image & positions & barcodes & features & matrix) == True, f"Missing filetype for BLEEP dataset {project_dataset}: {'image' if not image else ''} {'positions' if not positions else ''} {'barcodes' if not barcodes else ''} {'features' if not features else ''} {'matrix' if not matrix else ''} is missing."
    
    gene_counts_matrix = csr_matrix(mmread(raw_matrix_path).T)
    spot_barcode_df = pd.read_csv(raw_barcodes_path, header=None, names=["spot_barcode"])
    spot_barcode_df = spot_barcode_df.set_index("spot_barcode")
    positions_df = pd.read_csv(raw_positions_path, header=None, names=["spot_barcode", "in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"])
    positions_df = positions_df.set_index("spot_barcode")
    positions_df = positions_df[positions_df["in_tissue"] == 1]
    # assert set(spot_barcode_df.index) == set(positions_df.index), f"Unmatching between the spot barcodes in barcodes.tsv and tissue_positions_list.csv, and the difference is {set(spot_barcode_df.index) - set(positions_df.index)}."
    spot_barcode_intersection = list(set(spot_barcode_df.index).intersection(positions_df.index))
    # Create a boolean mask for selecting the rows
    row_mask = np.isin(spot_barcode_df.index, spot_barcode_intersection)

    # Apply the row mask to select the portion of gene_counts_matrix
    gene_counts_matrix = gene_counts_matrix[row_mask, :]
    
    spot_barcode_df = spot_barcode_df.loc[spot_barcode_intersection]
    positions_df = positions_df.loc[spot_barcode_intersection]
    assert set(spot_barcode_df.index) == set(positions_df.index), f"Unmatching between the spot barcodes in barcodes.tsv and tissue_positions_list.csv, and the difference is {set(spot_barcode_df.index) - set(positions_df.index)}."

    concat_obs_df = pd.concat([spot_barcode_df, positions_df], axis=1)
    concat_obs_df_copy = concat_obs_df.copy()
    concat_obs_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    concat_obs_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    concat_obs_df_copy.loc[:, "organ_type"] = "liver"
    image_extension = os.path.splitext(raw_image_path)[1]
    concat_obs_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in concat_obs_df.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in concat_obs_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for BLEEP dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in concat_obs_df_copy.columns]}"
    concat_obs_df_copy = concat_obs_df_copy.reindex(columns=obs_columns_order)
    total_obs_df = concat_obs_df_copy
    gene_feature_df =  pd.read_csv(raw_features_path, header=None, delimiter='\t', names=["gene_barcode", "gene_name", "gene_expression"])
    gene_feature_df = gene_feature_df.set_index("gene_name")
    
    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (positions_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)

    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)

    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image = Image.open(raw_image_path)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = row["pxl_row_in_fullres"]
            y_pixel = row["pxl_col_in_fullres"]
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))

    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for BLEEP dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def DLPFC_dataset_list(
    main_data_storage: str,
    project_data_folder_name: str,
) -> List:
    dataset_name = ["downstream_task_data", "DLPFC"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    return project_datasets

def DLPFC_single_helper(
    main_data_storage: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["downstream_task_data", "DLPFC"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 6, f"Unmatching between the number of files and the expected number of files for DLPFC dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    project_dataset_raw_unzipped_data_content_list = os.listdir(project_dataset_raw_unzipped_data_path)
    assert len(project_dataset_raw_unzipped_data_content_list) == 1, f"Unmatching between the number of files and the expected number of files for DLPFC dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)
    
    image, positions, barcodes, features, matrix, label_ground_truth, metadata = False, False, False, False, False, False, False
    image_file = [content for content in project_dataset_raw_data_content_list if (".jpg" in content) or (".tif" in content) or (".tiff" in content) or (".btf" in content)]
    assert len(image_file) == 1, f"More than one image file for DLPFC dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_data_path, image_file)
    image = True
    
    positions_folder = [content for content in project_dataset_raw_data_content_list if ("spatial" in content)]
    assert len(positions_folder) == 1, f"More than one positions folder for DLPFC dataset {project_dataset}"
    positions_folder = positions_folder[0]
    raw_positions_folder_path = os.path.join(project_dataset_raw_data_path, positions_folder)
    raw_positions_folder_content_list = os.listdir(raw_positions_folder_path)
    positions_file = [content for content in raw_positions_folder_content_list if ("tissue_positions_list" in content) or ("tissue_positions" in content)]
    assert len(positions_file) == 1, f"More than one positions file for DLPFC dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(raw_positions_folder_path, positions_file)
    positions = True

    filtered_feature_bc_matrix_all_unzipped_folder = [content for content in project_dataset_raw_unzipped_data_content_list if "filtered_feature_bc_matrix_all_unzipped" in content]
    assert len(filtered_feature_bc_matrix_all_unzipped_folder) == 1, f"More than one filtered_feature_bc_matrix_all_unzipped folder for DLPFC dataset {project_dataset}"
    filtered_feature_bc_matrix_all_unzipped_folder = filtered_feature_bc_matrix_all_unzipped_folder[0]
    filtered_feature_bc_matrix_all_unzipped_folder_path = os.path.join(project_dataset_raw_unzipped_data_path, filtered_feature_bc_matrix_all_unzipped_folder)
    filtered_feature_bc_matrix_all_unzipped_folder_content_list = os.listdir(filtered_feature_bc_matrix_all_unzipped_folder_path)
    
    barcodes_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "barcodes.tsv" in content]
    assert len(barcodes_file) == 1, f"More than one barcodes file for DLPFC dataset {project_dataset}"
    barcodes_file = barcodes_file[0]
    raw_barcodes_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, barcodes_file)
    barcodes = True
    
    features_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "features.tsv" in content]
    assert len(features_file) == 1, f"More than one features file for DLPFC dataset {project_dataset}"
    features_file = features_file[0]
    raw_features_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, features_file)
    features = True
    
    matrix_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "matrix.mtx" in content]
    assert len(matrix_file) == 1, f"More than one matrix file for DLPFC dataset {project_dataset}"
    matrix_file = matrix_file[0]
    raw_matrix_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, matrix_file)
    matrix = True
    
    label_ground_truth_file = [content for content in project_dataset_raw_data_content_list if "truth.txt" in content]
    assert len(label_ground_truth_file) == 1, f"More than one label_ground_truth file for DLPFC dataset {project_dataset}"
    label_ground_truth_file = label_ground_truth_file[0]
    raw_label_ground_truth_path = os.path.join(project_dataset_raw_data_path, label_ground_truth_file)
    label_ground_truth = True
    
    metadata_file = [content for content in project_dataset_raw_data_content_list if "metadata.tsv" in content]
    assert len(metadata_file) == 1, f"More than one metadata file for DLPFC dataset {project_dataset}"
    metadata_file = metadata_file[0]
    raw_metadata_path = os.path.join(project_dataset_raw_data_path, metadata_file)
    metadata = True
    assert (image & positions & barcodes & features & matrix & label_ground_truth & metadata) == True, f"Missing filetype for DLPFC dataset {project_dataset}: {'image' if not image else ''} {'positions' if not positions else ''} {'barcodes' if not barcodes else ''} {'features' if not features else ''} {'matrix' if not matrix else ''} {'label_ground_truth' if not label_ground_truth else ''} {'metadata' if not metadata else ''} is missing."
    
    gene_counts_matrix = csr_matrix(mmread(raw_matrix_path).T)
    spot_barcode_df = pd.read_csv(raw_barcodes_path, header=None, names=["spot_barcode"])
    spot_barcode_df = spot_barcode_df.set_index("spot_barcode")
    positions_df = pd.read_csv(raw_positions_path, header=None, names=["spot_barcode", "in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"])
    positions_df = positions_df.set_index("spot_barcode")
    # positions_df = positions_df[positions_df["in_tissue"] == 1]
    positions_df = positions_df[positions_df["in_tissue"].isin(["1", 1])]
    label_ground_truth_df = pd.read_csv(raw_label_ground_truth_path, sep="\t", header=None, names=["spot_barcode", "label_ground_truth"])
    label_ground_truth_df = label_ground_truth_df.set_index("spot_barcode")
    metadata_df = pd.read_csv(raw_metadata_path, sep="\t")
    metadata_df.index.name = "spot_barcode"
    assert set(spot_barcode_df.index) == set(positions_df.index) == set(label_ground_truth_df.index) == set(metadata_df.index), f"Unmatching between the spot barcodes in barcodes.tsv and tissue_positions_list.csv, and the difference is {set(spot_barcode_df.index) - set(positions_df.index)}."
    concat_obs_df = pd.concat([spot_barcode_df, positions_df], axis=1)
    concat_obs_df_copy = concat_obs_df.copy()
    concat_obs_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    concat_obs_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    concat_obs_df_copy.loc[:, "organ_type"] = "brain"
    image_extension = os.path.splitext(raw_image_path)[1]
    concat_obs_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in concat_obs_df.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in concat_obs_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for DLPFC dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in concat_obs_df_copy.columns]}"
    concat_obs_df_copy = concat_obs_df_copy.reindex(columns=obs_columns_order)
    full_concat_df = pd.concat([concat_obs_df_copy, label_ground_truth_df, metadata_df], axis=1)
    total_obs_df = full_concat_df
    gene_feature_df =  pd.read_csv(raw_features_path, header=None, delimiter='\t', names=["gene_barcode", "gene_name", "gene_expression"])
    gene_feature_df = gene_feature_df.set_index("gene_name")

    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (positions_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)

    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)

    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path)
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image = Image.open(raw_image_path)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = row["pxl_row_in_fullres"]
            y_pixel = row["pxl_col_in_fullres"]
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))
    
    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for DLPFC dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def DRYAD001_dataset_list(
    main_data_storage: str,
    project_data_folder_name: str,
) -> List:
    dataset_name = ["DRYAD", "DRYAD001"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    return project_datasets

def DRYAD001_single_helper(
    main_data_storage: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["DRYAD", "DRYAD001"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for DRYAD001 dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    project_dataset_raw_unzipped_data_content_list = os.listdir(project_dataset_raw_unzipped_data_path)
    assert len(project_dataset_raw_unzipped_data_content_list) == 1, f"Unmatching between the number of files and the expected number of files for DRYAD001 dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)
    
    image, positions, barcodes, features, matrix = False, False, False, False, False
    image_file = [content for content in project_dataset_raw_data_content_list if (".jpg" in content) or (".jpeg" in content) or (".tif" in content) or (".TIF" in content) or (".tiff" in content) or (".btf" in content)]
    assert len(image_file) == 1, f"More than one image file for DRYAD001 dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_data_path, image_file)
    image = True
    
    positions_folder = [content for content in project_dataset_raw_data_content_list if ("spatial" in content)]
    assert len(positions_folder) == 1, f"More than one positions folder for DRYAD001 dataset {project_dataset}"
    positions_folder = positions_folder[0]
    raw_positions_folder_path = os.path.join(project_dataset_raw_data_path, positions_folder)
    raw_positions_folder_content_list = os.listdir(raw_positions_folder_path)
    positions_file = [content for content in raw_positions_folder_content_list if ("tissue_positions_list" in content) or ("tissue_positions" in content)]
    assert len(positions_file) == 1, f"More than one positions file for DRYAD001 dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(raw_positions_folder_path, positions_file)
    positions = True

    filtered_feature_bc_matrix_all_unzipped_folder = [content for content in project_dataset_raw_unzipped_data_content_list if "filtered_feature_bc_matrix_all_unzipped" in content]
    assert len(filtered_feature_bc_matrix_all_unzipped_folder) == 1, f"More than one filtered_feature_bc_matrix_all_unzipped folder for 10xGenomics dataset {project_dataset}"
    filtered_feature_bc_matrix_all_unzipped_folder = filtered_feature_bc_matrix_all_unzipped_folder[0]
    filtered_feature_bc_matrix_all_unzipped_folder_path = os.path.join(project_dataset_raw_unzipped_data_path, filtered_feature_bc_matrix_all_unzipped_folder)
    filtered_feature_bc_matrix_all_unzipped_folder_content_list = os.listdir(filtered_feature_bc_matrix_all_unzipped_folder_path)
    
    barcodes_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "barcodes.tsv" in content]
    assert len(barcodes_file) == 1, f"More than one barcodes file for DRYAD001 dataset {project_dataset}"
    barcodes_file = barcodes_file[0]
    raw_barcodes_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, barcodes_file)
    barcodes = True
    
    features_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "features.tsv" in content]
    assert len(features_file) == 1, f"More than one features file for DRYAD001 dataset {project_dataset}"
    features_file = features_file[0]
    raw_features_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, features_file)
    features = True
    
    matrix_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "matrix.mtx" in content]
    assert len(matrix_file) == 1, f"More than one matrix file for DRYAD001 dataset {project_dataset}"
    matrix_file = matrix_file[0]
    raw_matrix_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, matrix_file)
    matrix = True
    assert (image & positions & barcodes & features & matrix) == True, f"Missing filetype for DRYAD001 dataset {project_dataset}: {'image' if not image else ''} {'positions' if not positions else ''} {'barcodes' if not barcodes else ''} {'features' if not features else ''} {'matrix' if not matrix else ''} is missing."
    
    gene_counts_matrix = csr_matrix(mmread(raw_matrix_path).T)
    spot_barcode_df = pd.read_csv(raw_barcodes_path, header=None, names=["spot_barcode"])
    spot_barcode_df = spot_barcode_df.set_index("spot_barcode")
    positions_df = pd.read_csv(raw_positions_path, header=None, names=["spot_barcode", "in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"])
    positions_df = positions_df.set_index("spot_barcode")
    positions_df = positions_df[positions_df["in_tissue"].isin(["1", 1])]
    if project_dataset == "#UKF242_T_ST":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] + 80
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] - 80
    elif project_dataset == "#UKF243_T_ST":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] - 275
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] + 275
    elif project_dataset == "#UKF248_C_ST":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] - 275
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] + 275
    elif project_dataset == "#UKF248_T_ST":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] + 200
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] - 200
    elif project_dataset == "#UKF251_T_ST":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] + 30
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] - 30
    elif project_dataset == "#UKF255_T_ST":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] + 30
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] - 30
    elif project_dataset == "#UKF256_C_ST":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] + 15
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] - 15
    elif project_dataset == "#UKF256_TC_ST":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] + 75
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] - 75
    elif project_dataset == "#UKF256_TI_ST":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] + 50
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] - 50
    elif project_dataset == "#UKF259_C_ST":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] - 30
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] + 50       
    elif project_dataset == "#UKF259_T_ST":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] + 75
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] - 75
    elif project_dataset == "#UKF260_T_ST":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] + 15
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] - 15
    elif project_dataset == "#UKF262_T_ST":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] + 50
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] - 50
    elif project_dataset == "#UKF265_T_ST":       
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] + 50
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] - 50
    elif project_dataset == "#UKF266_T_ST":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] + 30
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] - 30 
    elif project_dataset == "#UKF268_IDHMutant_T_ST":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] + 50
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] - 50
    elif project_dataset == "#UKF269_T_ST":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] + 75
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] - 75     
    elif project_dataset == "#UKF270_IDHMutant_T_ST":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] - 30
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] + 30  
    elif project_dataset == "#UKF275_T_ST":         
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] + 100
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] - 100 
    elif project_dataset in ["#UKF296_T_ST", "#UKF304_T_ST", "#UKF313_C_ST", "#UKF313_T_ST", "#UKF334_C_ST", "#UKF334_T_ST"]:
        pass
    else:
        raise ValueError(f"Unknown project dataset for DRYAD001 {project_dataset}")
    assert set(spot_barcode_df.index) == set(positions_df.index), f"Unmatching between the spot barcodes in barcodes.tsv and tissue_positions_list.csv, and the difference is {set(spot_barcode_df.index) - set(positions_df.index)}."
    concat_obs_df = pd.concat([spot_barcode_df, positions_df], axis=1)
    concat_obs_df_copy = concat_obs_df.copy()
    concat_obs_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    concat_obs_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    concat_obs_df_copy.loc[:, "organ_type"] = "brain"
    image_extension = os.path.splitext(raw_image_path)[1]
    concat_obs_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in concat_obs_df.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in concat_obs_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for DRYAD001 dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in concat_obs_df_copy.columns]}"
    concat_obs_df_copy = concat_obs_df_copy.reindex(columns=obs_columns_order)
    total_obs_df = concat_obs_df_copy
    gene_feature_df =  pd.read_csv(raw_features_path, header=None, delimiter='\t', names=["gene_barcode", "gene_name", "gene_expression"])
    gene_feature_df = gene_feature_df.set_index("gene_name")
    
    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (positions_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)
    
    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)
        
    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        if project_dataset == "#UKF242_T_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF243_T_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF248_C_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF248_T_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF251_T_ST":
            image = image.rotate(-90) # 90
        elif project_dataset == "#UKF255_T_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF256_C_ST":
            image = image.rotate(-90) # 90
        elif project_dataset == "#UKF256_TC_ST":
            image = image.rotate(-90) # 90
        elif project_dataset == "#UKF256_TI_ST":
            image = image.rotate(-90) # 90
        elif project_dataset == "#UKF259_C_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF259_T_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF260_T_ST":
            image = image.rotate(90) # 90 
        elif project_dataset == "#UKF262_T_ST":                          
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF265_T_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF266_T_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF268_IDHMutant_T_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF269_T_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF270_IDHMutant_T_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF275_T_ST": 
            image = image.rotate(90) # 90
        elif project_dataset in ["#UKF296_T_ST", "#UKF304_T_ST", "#UKF313_C_ST", "#UKF313_T_ST", "#UKF334_C_ST", "#UKF334_T_ST"]:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            raise ValueError(f"Unknown project dataset for DRYAD001 {project_dataset}")
        draw = ImageDraw.Draw(image)
        dot_size = 32
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
        
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image_extension = os.path.splitext(raw_image_path)[1]
        image = Image.open(raw_image_path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        if project_dataset == "#UKF242_T_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF243_T_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF248_C_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF248_T_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF251_T_ST":
            image = image.rotate(-90) # 90
        elif project_dataset == "#UKF255_T_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF256_C_ST":
            image = image.rotate(-90) # 90
        elif project_dataset == "#UKF256_TC_ST":
            image = image.rotate(-90) # 90
        elif project_dataset == "#UKF256_TI_ST":
            image = image.rotate(-90) # 90
        elif project_dataset == "#UKF259_C_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF259_T_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF260_T_ST":
            image = image.rotate(90) # 90 
        elif project_dataset == "#UKF262_T_ST":                          
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF265_T_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF266_T_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF268_IDHMutant_T_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF269_T_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF270_IDHMutant_T_ST":
            image = image.rotate(90) # 90
        elif project_dataset == "#UKF275_T_ST": 
            image = image.rotate(90) # 90
        elif project_dataset in ["#UKF296_T_ST", "#UKF304_T_ST", "#UKF313_C_ST", "#UKF313_T_ST", "#UKF334_C_ST", "#UKF334_T_ST"]:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            raise ValueError(f"Unknown project dataset for DRYAD001 {project_dataset}")
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = int(row["pxl_row_in_fullres"])
            y_pixel = int(row["pxl_col_in_fullres"])
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))

    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for DRYAD001 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def HER2ST_dataset_list(
    main_data_storage: str,
    project_data_folder_name: str,
) -> List:
    dataset_name = ["HER2ST"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    return project_datasets

def HER2ST_single_helper(
    main_data_storage: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["HER2ST"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 4, f"Unmatching between the number of files and the expected number of files for HER2ST dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    project_dataset_raw_unzipped_data_content_list = os.listdir(project_dataset_raw_unzipped_data_path)
    assert len(project_dataset_raw_unzipped_data_content_list) == 2, f"Unmatching between the number of files and the expected number of files for HER2ST dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)
    
    image, stdata, positions = False, False, False
    image_file = [content for content in project_dataset_raw_data_content_list if (".jpg" in content) or (".tif" in content) or (".tiff" in content) or (".btf" in content) and ("dark" not in content)]
    assert len(image_file) == 1, f"More than one image file for HER2ST dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_data_path, image_file)
    image = True

    stdata_file = [content for content in project_dataset_raw_unzipped_data_content_list if (".tsv" in content) and ("selection" not in content)]
    assert len(stdata_file) == 1, f"More than one stdata file for HER2ST dataset {project_dataset}"
    stdata_file = stdata_file[0]
    raw_stdata_path = os.path.join(project_dataset_raw_unzipped_data_path, stdata_file)
    stdata = True
    
    positions_file = [content for content in project_dataset_raw_unzipped_data_content_list if (".tsv" in content) and ("selection" in content)]
    assert len(positions_file) == 1, f"More than one positions file for HER2ST dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(project_dataset_raw_unzipped_data_path, positions_file)
    positions = True
    assert (image & stdata & positions) == True, f"Missing filetype for HER2ST dataset {project_dataset}: {'image' if not image else ''} {'stdata' if not stdata else ''} {'positions' if not positions else ''} is missing."
    
    gene_counts_df = pd.read_csv(raw_stdata_path, sep="\t")
    gene_counts_df = gene_counts_df.rename(columns={"Unnamed: 0": "spot_barcode"})
    gene_counts_df = gene_counts_df.set_index("spot_barcode")
    gene_counts_matrix = csr_matrix(gene_counts_df.values)
    
    spot_barcode_df = pd.read_csv(raw_positions_path, sep="\t")
    assert all(spot_barcode_df["selected"] == 1), f"Unmatching between the selected spots in the positions file for HER2ST dataset {project_dataset}"
    spot_barcode_df_copy = spot_barcode_df.copy()
    spot_barcode_df_copy.loc[:, "spot_barcode"] = spot_barcode_df_copy.apply(lambda row: str(int(row['x'])) + "x" + str(int(row['y'])), axis=1)
    spot_barcode_df_copy = spot_barcode_df_copy.set_index("spot_barcode")
    spot_barcode_df_copy = spot_barcode_df_copy.loc[:, ["new_x", "new_y", "pixel_x", "pixel_y", "selected"]]
    spot_barcode_df_copy = spot_barcode_df_copy.rename(columns={"new_x": "array_row", "new_y": "array_col", "pixel_x": "pxl_row_in_fullres", "pixel_y": "pxl_col_in_fullres", "selected": "in_tissue"})
    
    spot_barcode_intersection = list(set(spot_barcode_df_copy.index).intersection(gene_counts_df.index))
    row_mask = np.isin(gene_counts_df.index, spot_barcode_intersection)
    gene_counts_matrix = gene_counts_matrix[row_mask, :]
    spot_barcode_df_copy = spot_barcode_df_copy.loc[spot_barcode_intersection]
    gene_counts_df = gene_counts_df.loc[spot_barcode_intersection]
    assert set(spot_barcode_df_copy.index) == set(gene_counts_df.index), f"Unmatching between the spot barcodes in the positions file and the gene counts file for HER2ST dataset {project_dataset}"
    
    spot_barcode_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    spot_barcode_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    spot_barcode_df_copy.loc[:, "organ_type"] = "breast"
    image_extension = os.path.splitext(raw_image_path)[1]
    spot_barcode_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in spot_barcode_df_copy.index]
    spot_barcode_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in spot_barcode_df_copy.index]
    spot_barcode_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in spot_barcode_df_copy.index]
    spot_barcode_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in spot_barcode_df_copy.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in spot_barcode_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for HER2ST dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in spot_barcode_df_copy.columns]}"
    spot_barcode_df_copy = spot_barcode_df_copy.reindex(columns=obs_columns_order)
    total_obs_df = spot_barcode_df_copy
    
    gene_feature_df = pd.DataFrame(gene_counts_df.columns)
    gene_feature_df.columns = ["gene_names"]
    gene_feature_df = gene_feature_df.set_index("gene_names")
    
    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (total_obs_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)

    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)
        
    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path)
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image = Image.open(raw_image_path)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = row["pxl_row_in_fullres"]
            y_pixel = row["pxl_col_in_fullres"]
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))

    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for HER2ST dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def HCA001_dataset_list(
    main_data_storage: str,
    project_data_folder_name: str,
) -> List:
    dataset_name = ["Human_cell_atlas", "Human_cell_atlas001"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    return project_datasets

def HCA001_single_helper(
    main_data_storage: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["Human_cell_atlas", "Human_cell_atlas001"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 2, f"Unmatching between the number of files and the expected number of files for HCA001 dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)
    
    image, h5ad = False, False
    image_file = [content for content in project_dataset_raw_data_content_list if (".jpg" in content) or (".tif" in content)]
    assert len(image_file) == 1, f"More than one image file for HCA001 dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_data_path, image_file)
    image = True
    h5ad_file = [content for content in project_dataset_raw_data_content_list if ".h5ad" in content]
    assert len(h5ad_file) == 1, f"More than one h5ad file for HCA001 dataset {project_dataset}"
    h5ad_file = h5ad_file[0]
    raw_h5ad_path = os.path.join(project_dataset_raw_data_path, h5ad_file)
    h5ad = True
    assert (image & h5ad) == True, f"Missing filetype for HCA001 dataset {project_dataset}: {'image' if not image else ''} {'h5ad' if not h5ad else ''} is missing." 
    
    project_dataset_raw_stdata = anndata.read_h5ad(raw_h5ad_path)
    assert project_dataset_raw_stdata.obs["in_tissue"].unique() == [1], f"HCA001 dataset {project_dataset} has spots that do not fall inside tissue."
    spot_barcode_df = (project_dataset_raw_stdata.obs)[["in_tissue", "array_row", "array_col"]]
    spot_barcode_df.index.name = "spot_barcode"
    spot_barcode_df_copy = spot_barcode_df.copy()
    spot_barcode_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    spot_barcode_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    spot_barcode_df_copy.loc[:, "organ_type"] = "lung"
    image_extension = os.path.splitext(raw_image_path)[1]
    spot_barcode_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in spot_barcode_df.index]
    spot_barcode_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in spot_barcode_df.index]
    spot_barcode_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in spot_barcode_df.index]
    spot_barcode_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in spot_barcode_df.index]
    
    assert np.all(project_dataset_raw_stdata.obsm["X_spatial"] == project_dataset_raw_stdata.obsm["spatial"]), f"Unmatching between the spatial coordinates in the obsm and the spatial coordinates in the obs for HCA001 dataset {project_dataset}"
    positions_df = pd.DataFrame(project_dataset_raw_stdata.obsm["spatial"], columns=["pxl_row_in_fullres", "pxl_col_in_fullres"], index=spot_barcode_df.index)
    if "6332STDY102895" in project_dataset:
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] + 8560
    elif "6332STDY94791" in project_dataset:
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] + 7200
    else:
        raise ValueError(f"Unknown project dataset for HCA001 {project_dataset}")
    
    assert len(spot_barcode_df_copy) == len(positions_df), f"Unmatching between the number of spots in the barcode dataframe and the number of spots in the positions dataframe."
    total_obs_df = pd.concat([spot_barcode_df_copy, positions_df], axis=1)

    gene_feature_df = (project_dataset_raw_stdata.var)[["gene_ids", "feature_types"]]
    gene_feature_df = gene_feature_df.rename(columns={"gene_ids": "gene_barcode", "feature_types": "gene_expression"})
    gene_feature_df.index.name = "gene_name"

    project_dataset_stdata = anndata.AnnData(project_dataset_raw_stdata.X)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (positions_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)
    
    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)
        
    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path)
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image = Image.open(raw_image_path)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = row["pxl_row_in_fullres"]
            y_pixel = row["pxl_col_in_fullres"]
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))

    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for HCA001 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def Mendeley_data001_Patient_1_1k_dataset_list(
    main_data_storage: str,
    project_data_folder_name: str,
) -> List:
    dataset_name = ["Mendeley_data", "Mendeley_data001"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    project_datasets_patient_1_1k = [dataset for dataset in project_datasets if "Patient_1_1k" in dataset]
    project_datasets_patient_1_Visium = [dataset for dataset in project_datasets if "Patient_1_Visium" in dataset]
    project_datasets_patient_2 = [dataset for dataset in project_datasets if "Patient_2" in dataset]
    assert len(project_datasets_patient_1_1k) + len(project_datasets_patient_1_Visium) + len(project_datasets_patient_2) == len(project_datasets), f"Unmatching between the number of datasets and the expected number of datasets for Mendeley_data001 dataset."
    return project_datasets_patient_1_1k

def Mendeley_data001_Patient_1_1k_single_helper(
    main_data_storage: str,
    project_data_folder_name: str,
    gene_barcode_name_mapping_dict: dict,
    project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["Mendeley_data", "Mendeley_data001"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for Mendeley_data001 dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)
    
    image, stdata, positions = False, False, False
    image_file = [content for content in project_dataset_raw_data_content_list if (".jpg" in content) or (".tif" in content)]
    assert len(image_file) == 1, f"More than one image file for Mendeley_data001 dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_data_path, image_file)
    image = True
    stdata_file = [content for content in project_dataset_raw_data_content_list if ("stdata.tsv" in content)]
    assert len(stdata_file) == 1, f"More than one stdata file for Mendeley_data001 dataset {project_dataset}"
    stdata_file = stdata_file[0]
    raw_stdata_path = os.path.join(project_dataset_raw_data_path, stdata_file)
    stdata = True
    positions_file = [content for content in project_dataset_raw_data_content_list if (".csv" in content)]
    assert len(positions_file) == 1, f"More than one positions file for Mendeley_data001 dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(project_dataset_raw_data_path, positions_file)
    positions = True
    assert (image & stdata & positions) == True, f"Missing filetype for Mendeley_data001 dataset {project_dataset}: {'image' if not image else ''} {'stdata' if not stdata else ''} {'positions' if not positions else ''} is missing."
    
    gene_counts_df = pd.read_csv(raw_stdata_path, sep="\t")
    gene_counts_df = gene_counts_df.rename(columns={"Unnamed: 0": "spot_barcode"})
    gene_counts_df = gene_counts_df.set_index("spot_barcode")
    gene_counts_df_copy = gene_counts_df.copy()
    columns_without_gene_name = [column for column in gene_counts_df_copy.columns if column.split(".")[0] not in gene_barcode_name_mapping_dict.keys()]
    gene_counts_df_copy = gene_counts_df_copy.drop(columns=columns_without_gene_name)
    gene_counts_df_copy.columns = [gene_barcode_name_mapping_dict[column.split(".")[0]] for column in gene_counts_df_copy.columns]
    gene_counts_matrix = csr_matrix(gene_counts_df_copy.values)
    
    spot_barcode_df = pd.read_csv(raw_positions_path)
    spot_barcode_df = spot_barcode_df.rename(columns={"coordinate": "spot_barcode", "pixel_x": "pxl_row_in_fullres", "pixel_y": "pxl_col_in_fullres"})
    spot_barcode_df = spot_barcode_df.set_index("spot_barcode")
    spot_barcode_df_copy = spot_barcode_df.copy()

    spot_barcode_intersection = list(set(spot_barcode_df_copy.index).intersection(gene_counts_df.index))
    row_mask = np.isin(gene_counts_df.index, spot_barcode_intersection)
    gene_counts_matrix = gene_counts_matrix[row_mask, :]
    spot_barcode_df_copy = spot_barcode_df_copy.loc[spot_barcode_intersection]
    gene_counts_df = gene_counts_df.loc[spot_barcode_intersection]
    assert set(spot_barcode_df_copy.index) == set(gene_counts_df.index), f"Unmatching between the spot barcodes in the positions file and the gene counts file for Mendeley_data001 dataset {project_dataset}"
    
    spot_barcode_df_copy.loc[:, "in_tissue"] = 1
    spot_barcode_df_copy.loc[:, "array_row"] = [index.split("x")[0] for index in spot_barcode_df_copy.index]
    spot_barcode_df_copy.loc[:, "array_col"] = [index.split("x")[1] for index in spot_barcode_df_copy.index]
    spot_barcode_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    spot_barcode_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    spot_barcode_df_copy.loc[:, "organ_type"] = "prostate"
    image_extension = os.path.splitext(raw_image_path)[1]
    spot_barcode_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in spot_barcode_df_copy.index]
    spot_barcode_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in spot_barcode_df_copy.index]
    spot_barcode_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in spot_barcode_df_copy.index]
    spot_barcode_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in spot_barcode_df_copy.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in spot_barcode_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for Mendeley_data001 dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in spot_barcode_df_copy.columns]}"
    spot_barcode_df_copy = spot_barcode_df_copy.reindex(columns=obs_columns_order)
    total_obs_df = spot_barcode_df_copy

    gene_feature_df = pd.DataFrame(gene_counts_df_copy.columns)
    gene_feature_df.columns = ["gene_names"]
    gene_feature_df = gene_feature_df.set_index("gene_names")

    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (total_obs_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)

    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)
        
    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path)
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image = Image.open(raw_image_path)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = row["pxl_row_in_fullres"]
            y_pixel = row["pxl_col_in_fullres"]
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))

    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for Mendeley_data001 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def Mendeley_data001_Patient_1_Visium_dataset_list(
    main_data_storage: str,
    project_data_folder_name: str,
) -> List:
    dataset_name = ["Mendeley_data", "Mendeley_data001"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    project_datasets_patient_1_1k = [dataset for dataset in project_datasets if "Patient_1_1k" in dataset]
    project_datasets_patient_1_Visium = [dataset for dataset in project_datasets if "Patient_1_Visium" in dataset]
    project_datasets_patient_2 = [dataset for dataset in project_datasets if "Patient_2" in dataset]
    assert len(project_datasets_patient_1_1k) + len(project_datasets_patient_1_Visium) + len(project_datasets_patient_2) == len(project_datasets), f"Unmatching between the number of datasets and the expected number of datasets for Mendeley_data001 dataset."
    return project_datasets_patient_1_Visium

def Mendeley_data001_Patient_1_Visium_single_helper(
    main_data_storage: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["Mendeley_data", "Mendeley_data001"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 5, f"Unmatching between the number of files and the expected number of files for Mendeley_data001 dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)
    
    image, stdata, positions = False, False, False
    image_file = [content for content in project_dataset_raw_data_content_list if (".jpg" in content) or (".tif" in content)]
    assert len(image_file) == 1, f"More than one image file for Mendeley_data001 dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_data_path, image_file)
    image = True
    stdata_file = [content for content in project_dataset_raw_data_content_list if (".h5" in content)]
    assert len(stdata_file) == 1, f"More than one stdata file for Mendeley_data001 dataset {project_dataset}"
    stdata_file = stdata_file[0]
    raw_stdata_path = os.path.join(project_dataset_raw_data_path, stdata_file)
    stdata = True
    positions_file = [content for content in project_dataset_raw_data_content_list if ("tissue_positions_list.csv" in content)]
    assert len(positions_file) == 1, f"More than one positions file for Mendeley_data001 dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(project_dataset_raw_data_path, positions_file)
    positions = True
    assert (image & stdata & positions) == True, f"Missing filetype for Mendeley_data001 dataset {project_dataset}: {'image' if not image else ''} {'stdata' if not stdata else ''} {'positions' if not positions else ''} is missing."
    
    project_dataset_raw_stdata = h5py.File(raw_stdata_path, 'r')
    gene_counts_matrix = csc_matrix((project_dataset_raw_stdata['matrix/data'][:], project_dataset_raw_stdata['matrix/indices'][:], project_dataset_raw_stdata['matrix/indptr'][:]), shape=project_dataset_raw_stdata['matrix/shape'][:]).tocsr().transpose()
    spot_barcode_list = [item.decode() for item in project_dataset_raw_stdata['matrix/barcodes'][:]]
    spot_barcode_df = pd.DataFrame(spot_barcode_list, columns=["spot_barcode"])
    spot_barcode_df = spot_barcode_df.set_index("spot_barcode")
    positions_df = pd.read_csv(raw_positions_path, header=None)
    positions_df = positions_df.rename(columns={0:'spot_barcode', 1:'in_tissue', 2:'array_row', 3:'array_col', 4:'pxl_row_in_fullres', 5:'pxl_col_in_fullres'})
    positions_df = positions_df.set_index('spot_barcode')
    positions_df = positions_df[positions_df['in_tissue'] == 1]
    assert set(spot_barcode_df.index) == set(positions_df.index), f"Unmatching between the spot barcodes in barcodes.tsv and tissue_positions_list.csv, and the difference is {set(spot_barcode_df.index) - set(positions_df.index)}."
    concat_obs_df = pd.concat([spot_barcode_df, positions_df], axis=1)
    concat_obs_df_copy = concat_obs_df.copy()
    concat_obs_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    concat_obs_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    concat_obs_df_copy.loc[:, "organ_type"] = "prostate"
    image_extension = os.path.splitext(raw_image_path)[1]
    concat_obs_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in concat_obs_df.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in concat_obs_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for Mendeley_data001 dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in concat_obs_df_copy.columns]}"
    concat_obs_df_copy = concat_obs_df_copy.reindex(columns=obs_columns_order)
    total_obs_df = concat_obs_df_copy
    
    gene_barcode_list = [item.decode() for item in project_dataset_raw_stdata['matrix/features/id'][:]]
    gene_name_list = [item.decode() for item in project_dataset_raw_stdata['matrix/features/name'][:]]
    # barcode_without_name = [barcode for barcode in gene_barcode_list if barcode not in gene_barcode_name_mapping_dict.keys()]
    # gene_name_list = [gene_barcode_name_mapping_dict[barcode] for barcode in gene_barcode_list if barcode in gene_barcode_name_mapping_dict.keys()]
    gene_feature_df = pd.DataFrame(gene_name_list, columns=["gene_names"])
    gene_feature_df = gene_feature_df.set_index("gene_names")
    
    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (total_obs_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)

    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)
        
    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path)
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image = Image.open(raw_image_path)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = row["pxl_row_in_fullres"]
            y_pixel = row["pxl_col_in_fullres"]
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))

    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for Mendeley_data001 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def Mendeley_data001_Patient_2_dataset_list(
    main_data_storage: str,
    project_data_folder_name: str,
) -> List:
    dataset_name = ["Mendeley_data", "Mendeley_data001"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    project_datasets_patient_1_1k = [dataset for dataset in project_datasets if "Patient_1_1k" in dataset]
    project_datasets_patient_1_Visium = [dataset for dataset in project_datasets if "Patient_1_Visium" in dataset]
    project_datasets_patient_2 = [dataset for dataset in project_datasets if "Patient_2" in dataset]
    assert len(project_datasets_patient_1_1k) + len(project_datasets_patient_1_Visium) + len(project_datasets_patient_2) == len(project_datasets), f"Unmatching between the number of datasets and the expected number of datasets for Mendeley_data001 dataset."
    return project_datasets_patient_2

def Mendeley_data001_Patient_2_single_helper(
    main_data_storage: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["Mendeley_data", "Mendeley_data001"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 4, f"Unmatching between the number of files and the expected number of files for Mendeley_data001 dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)
    
    image, stdata, positions = False, False, False
    image_file = [content for content in project_dataset_raw_data_content_list if (".jpg" in content) or (".tif" in content)]
    assert len(image_file) == 1, f"More than one image file for Mendeley_data001 dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_data_path, image_file)
    image = True
    stdata_file = [content for content in project_dataset_raw_data_content_list if (".h5" in content)]
    assert len(stdata_file) == 1, f"More than one stdata file for Mendeley_data001 dataset {project_dataset}"
    stdata_file = stdata_file[0]
    raw_stdata_path = os.path.join(project_dataset_raw_data_path, stdata_file)
    stdata = True
    positions_file = [content for content in project_dataset_raw_data_content_list if ("tissue_positions_list.csv" in content)]
    assert len(positions_file) == 1, f"More than one positions file for Mendeley_data001 dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(project_dataset_raw_data_path, positions_file)
    positions = True
    assert (image & stdata & positions) == True, f"Missing filetype for Mendeley_data001 dataset {project_dataset}: {'image' if not image else ''} {'stdata' if not stdata else ''} {'positions' if not positions else ''} is missing."
    
    project_dataset_raw_stdata = h5py.File(raw_stdata_path, 'r')
    gene_counts_matrix = csc_matrix((project_dataset_raw_stdata['matrix/data'][:], project_dataset_raw_stdata['matrix/indices'][:], project_dataset_raw_stdata['matrix/indptr'][:]), shape=project_dataset_raw_stdata['matrix/shape'][:]).tocsr().transpose()
    spot_barcode_list = [item.decode() for item in project_dataset_raw_stdata['matrix/barcodes'][:]]
    spot_barcode_df = pd.DataFrame(spot_barcode_list, columns=["spot_barcode"])
    spot_barcode_df = spot_barcode_df.set_index("spot_barcode")
    positions_df = pd.read_csv(raw_positions_path, header=None)
    positions_df = positions_df.rename(columns={0:'spot_barcode', 1:'in_tissue', 2:'array_row', 3:'array_col', 4:'pxl_col_in_fullres', 5:'pxl_row_in_fullres'})
    positions_df = positions_df.set_index('spot_barcode')
    positions_df = positions_df[positions_df['in_tissue'] == 1]
    assert set(spot_barcode_df.index) == set(positions_df.index), f"Unmatching between the spot barcodes in barcodes.tsv and tissue_positions_list.csv, and the difference is {set(spot_barcode_df.index) - set(positions_df.index)}."
    positions_df_copy = positions_df.copy()
    scale_x = 0.5
    scale_y = 0.5
    x_positions = [int(pos * scale_x) for pos in positions_df['pxl_row_in_fullres'].tolist()]
    y_positions = [int(pos * scale_y) for pos in positions_df['pxl_col_in_fullres'].tolist()]
    coordinates = list(zip(x_positions, y_positions))
    matched_x_positions, matched_y_positions = list(), list()
    
    if project_dataset == "Patient_2_H2_1":
        x1, y1, x2, y2 = 72, 7385, 19936, 17583
        for x, y in coordinates:
            if x < x1 or x > x2 or y < y1 or y > y2:
                matched_x_positions.append(np.nan)
                matched_y_positions.append(np.nan)
            else:
                matched_x_positions.append(x - abs(x2 - max(x_positions)))
                matched_y_positions.append(y - abs(y2 - max(y_positions)))
    elif project_dataset == "Patient_2_H2_2":
        x1, y1, x2, y2 = 4200, 3100, 20800, 20610
        for x, y in coordinates:
            if x < x1 or x > x2 or y < y1 or y > y2:
                matched_x_positions.append(np.nan)
                matched_y_positions.append(np.nan)
            else:
                matched_x_positions.append(x - abs(x2 - max(x_positions)))
                matched_y_positions.append(y - abs(y2 - max(y_positions)))
    elif project_dataset == "Patient_2_H3_1":
        x1, y1, x2, y2 = 4100, 2580, 20800, 19070
        for x, y in coordinates:
            if x < x1 or x > x2 or y < y1 or y > y2:
                matched_x_positions.append(np.nan)
                matched_y_positions.append(np.nan)
            else:
                matched_x_positions.append(x - abs(x2 - max(x_positions)))
                matched_y_positions.append(y - abs(y2 - max(y_positions)))
    elif project_dataset == "Patient_2_H3_2":
        x1, y1, x2, y2 = 1900, 10330, 20710, 21000
        for x, y in coordinates:
            if x < x1 or x > x2 or y < y1 or y > y2:
                matched_x_positions.append(np.nan)
                matched_y_positions.append(np.nan)
            else:
                matched_x_positions.append(x - abs(x2 - max(x_positions)))
                matched_y_positions.append(y - abs(y2 - max(y_positions)))
    elif project_dataset == "Patient_2_H3_4":
        x1, y1, x2, y2 = 1900, 4400, 20710, 18200
        for x, y in coordinates:
            if x < x1 or x > x2 or y < y1 or y > y2:
                matched_x_positions.append(np.nan)
                matched_y_positions.append(np.nan)
            else:
                matched_x_positions.append(x - abs(x2 - max(x_positions)))
                matched_y_positions.append(y - abs(y2 - max(y_positions)))      
    elif project_dataset == "Patient_2_H3_5":
        x_positions = [int(pos * 0.495) for pos in positions_df['pxl_row_in_fullres'].tolist()]
        y_positions = [int(pos * 0.495) for pos in positions_df['pxl_col_in_fullres'].tolist()]
        coordinates = list(zip(x_positions, y_positions))
        x1, y1, x2, y2 = 8650, 3900, 21000, 19155
        for x, y in coordinates:
            if x < x1 or x > x2 or y < y1 or y > y2:
                matched_x_positions.append(np.nan)
                matched_y_positions.append(np.nan)
            else:
                matched_x_positions.append(x)
                matched_y_positions.append(y)
    elif project_dataset == "Patient_2_H3_6":
        x1, y1, x2, y2 = 2570, 3170, 17230, 21000
        for x, y in coordinates:
            if x < x1 or x > x2 or y < y1 or y > y2:
                matched_x_positions.append(np.nan)
                matched_y_positions.append(np.nan)
            else:
                matched_x_positions.append(x - abs(x2 - max(x_positions)))
                matched_y_positions.append(y - abs(y2 - max(y_positions))) 
    elif project_dataset == "Patient_2_V1_1":
        x_positions = [int(pos * 0.51) for pos in positions_df['pxl_row_in_fullres'].tolist()]
        y_positions = [int(pos * 0.51) for pos in positions_df['pxl_col_in_fullres'].tolist()]
        coordinates = list(zip(x_positions, y_positions))
        x1, y1, x2, y2 = 7410, 3200, 19420, 21510
        for x, y in coordinates:
            if x < x1 or x > x2 or y < y1 or y > y2:
                matched_x_positions.append(np.nan)
                matched_y_positions.append(np.nan)
            else:
                matched_x_positions.append(x - abs(x2 - max(x_positions)))
                matched_y_positions.append(y - abs(y2 - max(y_positions)))
    elif project_dataset == "Patient_2_V1_2":
        x_positions = [int(pos * 0.51) for pos in positions_df['pxl_row_in_fullres'].tolist()]
        y_positions = [int(pos * 0.51) for pos in positions_df['pxl_col_in_fullres'].tolist()]
        coordinates = list(zip(x_positions, y_positions))
        x1, y1, x2, y2 = 2720, 3640, 21270, 22190
        for x, y in coordinates:
            if x < x1 or x > x2 or y < y1 or y > y2:
                matched_x_positions.append(np.nan)
                matched_y_positions.append(np.nan)
            else:
                matched_x_positions.append(x - abs(x2 - max(x_positions)))
                matched_y_positions.append(y - abs(y2 - max(y_positions)))
    elif project_dataset == "Patient_2_V1_3":
        x_positions = [int(pos * 0.5) for pos in positions_df['pxl_row_in_fullres'].tolist()]
        y_positions = [int(pos * 0.51) for pos in positions_df['pxl_col_in_fullres'].tolist()]
        coordinates = list(zip(x_positions, y_positions))
        x1, y1, x2, y2 = 3000, 1935, 20930, 21960
        for x, y in coordinates:
            if x < x1 or x > x2 or y < y1 or y > y2:
                matched_x_positions.append(np.nan)
                matched_y_positions.append(np.nan)
            else:
                matched_x_positions.append(x)
                matched_y_positions.append(y - abs(y2 - max(y_positions)))
    elif project_dataset == "Patient_2_V1_4":
        x_positions = [int(pos * 0.51) for pos in positions_df['pxl_row_in_fullres'].tolist()]
        y_positions = [int(pos * 0.51) for pos in positions_df['pxl_col_in_fullres'].tolist()]
        coordinates = list(zip(x_positions, y_positions))
        x1, y1, x2, y2 = 1900, 5100, 21100, 22000
        for x, y in coordinates:
            if x < x1 or x > x2 or y < y1 or y > y2:
                matched_x_positions.append(np.nan)
                matched_y_positions.append(np.nan)
            else:
                matched_x_positions.append(x)
                matched_y_positions.append(y)
    elif project_dataset == "Patient_2_V1_5":
        x_positions = [int(pos * 0.51) for pos in positions_df['pxl_row_in_fullres'].tolist()]
        y_positions = [int(pos * 0.51) for pos in positions_df['pxl_col_in_fullres'].tolist()]
        coordinates = list(zip(x_positions, y_positions))
        x1, y1, x2, y2 = 2970, 2240, 21190, 22300
        for x, y in coordinates:
            if x < x1 or x > x2 or y < y1 or y > y2:
                matched_x_positions.append(np.nan)
                matched_y_positions.append(np.nan)
            else:
                matched_x_positions.append(x - abs(x2 - max(x_positions)))
                matched_y_positions.append(y - abs(y2 - max(y_positions)))
    elif project_dataset == "Patient_2_V1_6":
        x_positions = [int(pos * 0.51) for pos in positions_df['pxl_row_in_fullres'].tolist()]
        y_positions = [int(pos * 0.51) for pos in positions_df['pxl_col_in_fullres'].tolist()]
        coordinates = list(zip(x_positions, y_positions))
        x1, y1, x2, y2 = 3900, 8200, 20100, 22455
        for x, y in coordinates:
            if x < x1 or x > x2 or y < y1 or y > y2:
                matched_x_positions.append(np.nan)
                matched_y_positions.append(np.nan)
            else:
                matched_x_positions.append(x - abs(x2 - max(x_positions)))
                matched_y_positions.append(y - abs(y2 - max(y_positions)))
    elif project_dataset == "Patient_2_V2_1":
        x_positions = [int(pos * 0.51) for pos in positions_df['pxl_row_in_fullres'].tolist()]
        y_positions = [int(pos * 0.51) for pos in positions_df['pxl_col_in_fullres'].tolist()]
        coordinates = list(zip(x_positions, y_positions))
        x1, y1, x2, y2 = 2200, 6100, 19950, 21525
        for x, y in coordinates:
            if x < x1 or x > x2 or y < y1 or y > y2:
                matched_x_positions.append(np.nan)
                matched_y_positions.append(np.nan)
            else:
                matched_x_positions.append(x - abs(x2 - max(x_positions)))
                matched_y_positions.append(y - abs(y2 - max(y_positions)))
    elif project_dataset == "Patient_2_V2_2":
        x_positions = [int(pos * 0.51) for pos in positions_df['pxl_row_in_fullres'].tolist()]
        y_positions = [int(pos * 0.51) for pos in positions_df['pxl_col_in_fullres'].tolist()]
        coordinates = list(zip(x_positions, y_positions))
        x1, y1, x2, y2 = 2430, 2770, 21160, 21760
        for x, y in coordinates:
            if x < x1 or x > x2 or y < y1 or y > y2:
                matched_x_positions.append(np.nan)
                matched_y_positions.append(np.nan)
            else:
                matched_x_positions.append(x - abs(x2 - max(x_positions)))
                matched_y_positions.append(y - abs(y2 - max(y_positions)))
    else:
        raise ValueError(f"Unknown Mendeley_data001 Patient 2 dataset {project_dataset}")
    
    positions_df_copy["pxl_row_in_fullres"] = matched_x_positions
    positions_df_copy["pxl_col_in_fullres"] = matched_y_positions
    positions_df_copy = positions_df_copy.dropna()
    
    spot_barcode_intersection = list(set(spot_barcode_df.index).intersection(positions_df_copy.index))
    row_mask = np.isin(spot_barcode_df.index, spot_barcode_intersection)
    gene_counts_matrix = gene_counts_matrix[row_mask, :]
    spot_barcode_df = spot_barcode_df.loc[spot_barcode_intersection]
    positions_df_copy = positions_df_copy.loc[spot_barcode_intersection]
    assert set(spot_barcode_df.index) == set(positions_df_copy.index), f"Unmatching between the spot barcodes in barcodes.tsv and tissue_positions_list.csv, and the difference is {set(spot_barcode_df.index) - set(positions_df.index)}."
    
    concat_obs_df = pd.concat([spot_barcode_df, positions_df_copy], axis=1)
    concat_obs_df_copy = concat_obs_df.copy()
    concat_obs_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    concat_obs_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    concat_obs_df_copy.loc[:, "organ_type"] = "prostate"
    image_extension = os.path.splitext(raw_image_path)[1]
    concat_obs_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in concat_obs_df.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in concat_obs_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for Mendeley_data001 dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in concat_obs_df_copy.columns]}"
    concat_obs_df_copy = concat_obs_df_copy.reindex(columns=obs_columns_order)
    total_obs_df = concat_obs_df_copy
    
    spot_barcode_list = [item.decode() for item in project_dataset_raw_stdata['matrix/barcodes'][:]]
    spot_barcode_df = pd.DataFrame(spot_barcode_list, columns=["spot_barcode"])
    spot_barcode_df = spot_barcode_df.set_index("spot_barcode")
    
    # gene_barcode_list = [item.decode() for item in project_dataset_raw_stdata['matrix/features/id'][:]]
    gene_name_list = [item.decode() for item in project_dataset_raw_stdata['matrix/features/name'][:]]
    # barcode_without_name = [barcode for barcode in gene_barcode_list if barcode not in gene_barcode_name_mapping_dict.keys()]
    # gene_name_list = [gene_barcode_name_mapping_dict[barcode] for barcode in gene_barcode_list if barcode in gene_barcode_name_mapping_dict.keys()]
    gene_feature_df = pd.DataFrame(gene_name_list, columns=["gene_names"])
    gene_feature_df = gene_feature_df.set_index("gene_names")
    
    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (positions_df_copy[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)
    
    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)
        
    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path)
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image = Image.open(raw_image_path)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = row["pxl_row_in_fullres"]
            y_pixel = row["pxl_col_in_fullres"]
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))

    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for Mendeley_data001 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def Mendeley_data002_dataset_list(
    main_data_storage: str,
    project_data_folder_name: str,
) -> List:
    dataset_name = ["Mendeley_data", "Mendeley_data002"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    return project_datasets

def Mendeley_data002_single_helper(
    main_data_storage: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["Mendeley_data", "Mendeley_data002"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for Mendeley_data002 dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)
    
    image, positions, stdata = False, False, False
    image_file = [content for content in project_dataset_raw_data_content_list if (".jpg" in content) or (".tif" in content) or (".tiff" in content) or (".btf" in content)]
    assert len(image_file) == 1, f"More than one image file for Mendeley_data002 dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_data_path, image_file)
    image = True
    
    positions_folder = [content for content in project_dataset_raw_data_content_list if ("spatial" in content)]
    assert len(positions_folder) == 1, f"More than one positions folder for Mendeley_data002 dataset {project_dataset}"
    positions_folder = positions_folder[0]
    raw_positions_folder_path = os.path.join(project_dataset_raw_data_path, positions_folder)
    raw_positions_folder_content_list = os.listdir(raw_positions_folder_path)
    positions_file = [content for content in raw_positions_folder_content_list if ("tissue_positions_list" in content) or ("tissue_positions" in content)]
    assert len(positions_file) == 1, f"More than one positions file for Mendeley_data002 dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(raw_positions_folder_path, positions_file)
    positions = True

    stdata_file = [content for content in project_dataset_raw_data_content_list if (".h5" in content)]
    assert len(stdata_file) == 1, f"More than one stdata file for Mendeley_data002 dataset {project_dataset}"
    stdata_file = stdata_file[0]
    raw_stdata_path = os.path.join(project_dataset_raw_data_path, stdata_file)
    stdata = True
    assert (image & stdata & positions) == True, f"Missing filetype for Mendeley_data002 dataset {project_dataset}: {'image' if not image else ''} {'stdata' if not stdata else ''} {'positions' if not positions else ''} is missing."
    
    project_dataset_raw_stdata = h5py.File(raw_stdata_path, 'r')
    gene_counts_matrix = csc_matrix((project_dataset_raw_stdata['matrix/data'][:], project_dataset_raw_stdata['matrix/indices'][:], project_dataset_raw_stdata['matrix/indptr'][:]), shape=project_dataset_raw_stdata['matrix/shape'][:]).tocsr().transpose()
    spot_barcode_list = [item.decode() for item in project_dataset_raw_stdata['matrix/barcodes'][:]]
    spot_barcode_df = pd.DataFrame(spot_barcode_list, columns=["spot_barcode"])
    spot_barcode_df = spot_barcode_df.set_index("spot_barcode")
    positions_df = pd.read_csv(raw_positions_path, header=None)
    positions_df = positions_df.rename(columns={0:'spot_barcode', 1:'in_tissue', 2:'array_row', 3:'array_col', 4:'pxl_col_in_fullres', 5:'pxl_row_in_fullres'})
    positions_df = positions_df.set_index('spot_barcode')
    positions_df = positions_df[positions_df['in_tissue'] == 1]
    assert set(spot_barcode_df.index) == set(positions_df.index), f"Unmatching between the spot barcodes in barcodes.tsv and tissue_positions_list.csv, and the difference is {set(spot_barcode_df.index) - set(positions_df.index)}."
    concat_obs_df = pd.concat([spot_barcode_df, positions_df], axis=1)
    concat_obs_df_copy = concat_obs_df.copy()
    concat_obs_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    concat_obs_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    concat_obs_df_copy.loc[:, "organ_type"] = "small_and_large_intestine"
    image_extension = os.path.splitext(raw_image_path)[1]
    concat_obs_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in concat_obs_df.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in concat_obs_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for Mendeley_data002 dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in concat_obs_df_copy.columns]}"
    concat_obs_df_copy = concat_obs_df_copy.reindex(columns=obs_columns_order)
    total_obs_df = concat_obs_df_copy
    
    gene_name_list = [item.decode() for item in project_dataset_raw_stdata['matrix/features/name'][:]]
    gene_feature_df = pd.DataFrame(gene_name_list, columns=["gene_names"])
    gene_feature_df = gene_feature_df.set_index("gene_names")
    
    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (total_obs_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)

    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)
        
    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path)
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image = Image.open(raw_image_path)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = row["pxl_row_in_fullres"]
            y_pixel = row["pxl_col_in_fullres"]
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))

    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for Mendeley_data002 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")
    
def Mendeley_data003_dataset_list(
    main_data_storage: str,
    project_data_folder_name: str,
) -> List:
    dataset_name = ["Mendeley_data", "Mendeley_data003"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    return project_datasets

def Mendeley_data003_single_helper(
    main_data_storage: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["Mendeley_data", "Mendeley_data003"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for Mendeley_data003 dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)
    
    image, positions, stdata = False, False, False
    image_file = [content for content in project_dataset_raw_data_content_list if (".jpg" in content) or (".tif" in content) or (".tiff" in content) or (".btf" in content)]
    assert len(image_file) == 1, f"More than one image file for Mendeley_data003 dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_data_path, image_file)
    image = True
    
    positions_folder = [content for content in project_dataset_raw_data_content_list if ("spatial" in content)]
    assert len(positions_folder) == 1, f"More than one positions folder for Mendeley_data003 dataset {project_dataset}"
    positions_folder = positions_folder[0]
    raw_positions_folder_path = os.path.join(project_dataset_raw_data_path, positions_folder)
    raw_positions_folder_content_list = os.listdir(raw_positions_folder_path)
    positions_file = [content for content in raw_positions_folder_content_list if ("tissue_positions_list" in content) or ("tissue_positions" in content)]
    assert len(positions_file) == 1, f"More than one positions file for Mendeley_data003 dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(raw_positions_folder_path, positions_file)
    positions = True

    stdata_file = [content for content in project_dataset_raw_data_content_list if (".h5" in content)]
    assert len(stdata_file) == 1, f"More than one stdata file for Mendeley_data003 dataset {project_dataset}"
    stdata_file = stdata_file[0]
    raw_stdata_path = os.path.join(project_dataset_raw_data_path, stdata_file)
    stdata = True
    assert (image & stdata & positions) == True, f"Missing filetype for Mendeley_data003 dataset {project_dataset}: {'image' if not image else ''} {'stdata' if not stdata else ''} {'positions' if not positions else ''} is missing."
    
    project_dataset_raw_stdata = h5py.File(raw_stdata_path, 'r')
    gene_counts_matrix = csc_matrix((project_dataset_raw_stdata['matrix/data'][:], project_dataset_raw_stdata['matrix/indices'][:], project_dataset_raw_stdata['matrix/indptr'][:]), shape=project_dataset_raw_stdata['matrix/shape'][:]).tocsr().transpose()
    spot_barcode_list = [item.decode() for item in project_dataset_raw_stdata['matrix/barcodes'][:]]
    spot_barcode_df = pd.DataFrame(spot_barcode_list, columns=["spot_barcode"])
    spot_barcode_df = spot_barcode_df.set_index("spot_barcode")
    positions_df = pd.read_csv(raw_positions_path, header=None)
    positions_df = positions_df.rename(columns={0:'spot_barcode', 1:'in_tissue', 2:'array_row', 3:'array_col', 4:'pxl_col_in_fullres', 5:'pxl_row_in_fullres'})
    positions_df = positions_df.set_index('spot_barcode')
    positions_df = positions_df[positions_df['in_tissue'] == 1]
    assert set(spot_barcode_df.index) == set(positions_df.index), f"Unmatching between the spot barcodes in barcodes.tsv and tissue_positions_list.csv, and the difference is {set(spot_barcode_df.index) - set(positions_df.index)}."
    concat_obs_df = pd.concat([spot_barcode_df, positions_df], axis=1)
    concat_obs_df_copy = concat_obs_df.copy()
    concat_obs_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    concat_obs_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    concat_obs_df_copy.loc[:, "organ_type"] = "skin"
    image_extension = os.path.splitext(raw_image_path)[1]
    concat_obs_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in concat_obs_df.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in concat_obs_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for Mendeley_data003 dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in concat_obs_df_copy.columns]}"
    concat_obs_df_copy = concat_obs_df_copy.reindex(columns=obs_columns_order)
    total_obs_df = concat_obs_df_copy
    
    gene_name_list = [item.decode() for item in project_dataset_raw_stdata['matrix/features/name'][:]]
    gene_feature_df = pd.DataFrame(gene_name_list, columns=["gene_names"])
    gene_feature_df = gene_feature_df.set_index("gene_names")
    
    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (total_obs_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)

    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)
        
    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path)
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image = Image.open(raw_image_path)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = row["pxl_row_in_fullres"]
            y_pixel = row["pxl_col_in_fullres"]
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))

    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for Mendeley_data003 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def NCBI001_dataset_list(
    main_data_storage: str,
    project_data_folder_name: str,
) -> List:
    dataset_name = ["NCBI", "NCBI001"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    return project_datasets

def NCBI001_single_helper(
    main_data_storage: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["NCBI", "NCBI001"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for NCBI001 dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    project_dataset_raw_unzipped_data_content_list = os.listdir(project_dataset_raw_unzipped_data_path)
    assert len(project_dataset_raw_unzipped_data_content_list) == 1, f"Unmatching between the number of files and the expected number of files for NCBI001 dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)
    
    image, positions, stdata = False, False, False
    image_file = [content for content in project_dataset_raw_data_content_list if (".jpg" in content) or (".tif" in content) or (".tiff" in content) or (".btf" in content)]
    assert len(image_file) == 1, f"More than one image file for NCBI001 dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_data_path, image_file)
    image = True
    
    positions_folder = [content for content in project_dataset_raw_unzipped_data_content_list if ("spatial" in content)]
    assert len(positions_folder) == 1, f"More than one positions folder for NCBI001 dataset {project_dataset}"
    positions_folder = positions_folder[0]
    raw_positions_folder_path = os.path.join(project_dataset_raw_unzipped_data_path, positions_folder)
    raw_positions_folder_content_list = os.listdir(raw_positions_folder_path)
    positions_file = [content for content in raw_positions_folder_content_list if ("tissue_positions_list" in content) or ("tissue_positions" in content)]
    assert len(positions_file) == 1, f"More than one positions file for NCBI001 dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(raw_positions_folder_path, positions_file)
    positions = True

    stdata_file = [content for content in project_dataset_raw_data_content_list if (".h5" in content)]
    assert len(stdata_file) == 1, f"More than one stdata file for NCBI001 dataset {project_dataset}"
    stdata_file = stdata_file[0]
    raw_stdata_path = os.path.join(project_dataset_raw_data_path, stdata_file)
    stdata = True
    assert (image & stdata & positions) == True, f"Missing filetype for NCBI001 dataset {project_dataset}: {'image' if not image else ''} {'stdata' if not stdata else ''} {'positions' if not positions else ''} is missing."
    
    project_dataset_raw_stdata = h5py.File(raw_stdata_path, 'r')
    gene_counts_matrix = csc_matrix((project_dataset_raw_stdata['matrix/data'][:], project_dataset_raw_stdata['matrix/indices'][:], project_dataset_raw_stdata['matrix/indptr'][:]), shape=project_dataset_raw_stdata['matrix/shape'][:]).tocsr().transpose()
    spot_barcode_list = [item.decode() for item in project_dataset_raw_stdata['matrix/barcodes'][:]]
    spot_barcode_df = pd.DataFrame(spot_barcode_list, columns=["spot_barcode"])
    spot_barcode_df = spot_barcode_df.set_index("spot_barcode")
    positions_df = pd.read_csv(raw_positions_path, header=None)
    positions_df = positions_df.rename(columns={0:'spot_barcode', 1:'in_tissue', 2:'array_row', 3:'array_col', 4:'pxl_col_in_fullres', 5:'pxl_row_in_fullres'})
    positions_df = positions_df.set_index('spot_barcode')
    positions_df = positions_df[positions_df['in_tissue'] == 1]
    assert set(spot_barcode_df.index) == set(positions_df.index), f"Unmatching between the spot barcodes in barcodes.tsv and tissue_positions_list.csv, and the difference is {set(spot_barcode_df.index) - set(positions_df.index)}."
    concat_obs_df = pd.concat([spot_barcode_df, positions_df], axis=1)
    concat_obs_df_copy = concat_obs_df.copy()
    concat_obs_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    concat_obs_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    concat_obs_df_copy.loc[:, "organ_type"] = "nose"
    image_extension = os.path.splitext(raw_image_path)[1]
    concat_obs_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in concat_obs_df.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in concat_obs_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for NCBI001 dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in concat_obs_df_copy.columns]}"
    concat_obs_df_copy = concat_obs_df_copy.reindex(columns=obs_columns_order)
    total_obs_df = concat_obs_df_copy
    
    gene_name_list = [item.decode() for item in project_dataset_raw_stdata['matrix/features/name'][:]]
    gene_feature_df = pd.DataFrame(gene_name_list, columns=["gene_names"])
    gene_feature_df = gene_feature_df.set_index("gene_names")
    
    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (total_obs_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)

    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)
        
    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path)
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image = Image.open(raw_image_path)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = row["pxl_row_in_fullres"]
            y_pixel = row["pxl_col_in_fullres"]
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))

    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for NCBI001 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def NCBI002_dataset_list(
    main_data_storage: str,
    project_data_folder_name: str,
) -> List:
    dataset_name = ["NCBI", "NCBI002"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    return project_datasets

def NCBI002_single_helper(
    main_data_storage: str,
    project_data_folder_name: str,
    project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["NCBI", "NCBI002"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for NCBI002 dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    project_dataset_raw_unzipped_data_content_list = os.listdir(project_dataset_raw_unzipped_data_path)
    assert len(project_dataset_raw_unzipped_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for NCBI002 dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)
    
    image, positions, stdata = False, False, False
    image_file = [content for content in project_dataset_raw_unzipped_data_content_list if (".jpg" in content) or (".tif" in content) or (".tiff" in content) or (".btf" in content)]
    assert len(image_file) == 1, f"More than one image file for NCBI002 dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_unzipped_data_path, image_file)
    image = True
    positions_file = [content for content in project_dataset_raw_unzipped_data_content_list if ("spot_data" in content)]
    assert len(positions_file) == 1, f"More than one positions file for NCBI002 dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(project_dataset_raw_unzipped_data_path, positions_file)
    positions = True
    stdata_file = [content for content in project_dataset_raw_unzipped_data_content_list if ("stdata" in content)]
    assert len(stdata_file) == 1, f"More than one stdata file for NCBI002 dataset {project_dataset}"
    stdata_file = stdata_file[0]
    raw_stdata_path = os.path.join(project_dataset_raw_unzipped_data_path, stdata_file)
    stdata = True
    assert (image & stdata & positions) == True, f"Missing filetype for NCBI002 dataset {project_dataset}: {'image' if not image else ''} {'stdata' if not stdata else ''} {'positions' if not positions else ''} is missing."
    
    gene_counts_df = pd.read_csv(raw_stdata_path, sep="\t")
    gene_counts_df = gene_counts_df.rename(columns={"Unnamed: 0": "spot_barcode"})
    gene_counts_df = gene_counts_df.set_index("spot_barcode")
    gene_counts_matrix = csr_matrix(gene_counts_df.values)
    
    spot_barcode_df = pd.read_csv(raw_positions_path, sep="\t")
    spot_barcode_df_copy = spot_barcode_df.copy()
    spot_barcode_df_copy.loc[:, "spot_barcode"] = spot_barcode_df_copy.apply(lambda row: str(int(row['x'])) + "x" + str(int(row['y'])), axis=1)
    spot_barcode_df_copy = spot_barcode_df_copy.set_index("spot_barcode")
    spot_barcode_df_copy = spot_barcode_df_copy.loc[:, ["new_x", "new_y", "pixel_x", "pixel_y"]]
    spot_barcode_df_copy = spot_barcode_df_copy.rename(columns={"new_x": "array_row", "new_y": "array_col", "pixel_x": "pxl_row_in_fullres", "pixel_y": "pxl_col_in_fullres"})
    
    spot_barcode_intersection = list(set(spot_barcode_df_copy.index).intersection(gene_counts_df.index))
    row_mask = np.isin(gene_counts_df.index, spot_barcode_intersection)
    gene_counts_matrix = gene_counts_matrix[row_mask, :]
    spot_barcode_df_copy = spot_barcode_df_copy.loc[spot_barcode_intersection]
    gene_counts_df = gene_counts_df.loc[spot_barcode_intersection]
    assert set(spot_barcode_df_copy.index) == set(gene_counts_df.index), f"Unmatching between the spot barcodes in the positions file and the gene counts file for NCBI002 dataset {project_dataset}"
    
    spot_barcode_df_copy.loc[:, "in_tissue"] = 1
    spot_barcode_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    spot_barcode_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    spot_barcode_df_copy.loc[:, "organ_type"] = "skin"
    image_extension = os.path.splitext(raw_image_path)[1]
    spot_barcode_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in spot_barcode_df_copy.index]
    spot_barcode_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in spot_barcode_df_copy.index]
    spot_barcode_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in spot_barcode_df_copy.index]
    spot_barcode_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in spot_barcode_df_copy.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in spot_barcode_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for NCBI002 dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in spot_barcode_df_copy.columns]}"
    spot_barcode_df_copy = spot_barcode_df_copy.reindex(columns=obs_columns_order)
    total_obs_df = spot_barcode_df_copy
    
    gene_feature_df = pd.DataFrame(gene_counts_df.columns)
    gene_feature_df.columns = ["gene_names"]
    gene_feature_df = gene_feature_df.set_index("gene_names")
    
    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (total_obs_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)

    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)
        
    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path)
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image = Image.open(raw_image_path)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = row["pxl_row_in_fullres"]
            y_pixel = row["pxl_col_in_fullres"]
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))

    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for NCBI002 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def NCBI003_dataset_list(
        main_data_storage: str,
        project_data_folder_name: str,
) -> List:
    dataset_name = ["NCBI", "NCBI003"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    return project_datasets

def NCBI003_single_helper(
        main_data_storage: str,
        project_data_folder_name: str,
        project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["NCBI", "NCBI003"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for NCBI003 dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    project_dataset_raw_unzipped_data_content_list = os.listdir(project_dataset_raw_unzipped_data_path)
    assert len(project_dataset_raw_unzipped_data_content_list) == 2, f"Unmatching between the number of files and the expected number of files for NCBI003 dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)
    
    image, positions, stdata = False, False, False
    image_file = [content for content in project_dataset_raw_unzipped_data_content_list if (".jpg" in content) or (".tif" in content) or (".tiff" in content) or (".btf" in content)]
    assert len(image_file) == 1, f"More than one image file for NCBI003 dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_unzipped_data_path, image_file)
    image = True
    positions_folder = [content for content in project_dataset_raw_data_content_list if ("spatial" in content)]
    assert len(positions_folder) == 1, f"More than one positions folder for NCBI003 dataset {project_dataset}"
    positions_folder = positions_folder[0]
    raw_positions_folder_path = os.path.join(project_dataset_raw_data_path, positions_folder)
    raw_positions_folder_content_list = os.listdir(raw_positions_folder_path)
    positions_file = [content for content in raw_positions_folder_content_list if ("tissue_positions_list" in content)]
    assert len(positions_file) == 1, f"More than one positions file for NCBI003 dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(raw_positions_folder_path, positions_file)
    positions = True
    stdata_file = [content for content in project_dataset_raw_unzipped_data_content_list if (".csv" in content)]
    assert len(stdata_file) == 1, f"More than one stdata file for NCBI003 dataset {project_dataset}"
    stdata_file = stdata_file[0]
    raw_stdata_path = os.path.join(project_dataset_raw_unzipped_data_path, stdata_file)
    stdata = True
    assert (image & stdata & positions) == True, f"Missing filetype for NCBI003 dataset {project_dataset}: {'image' if not image else ''} {'stdata' if not stdata else ''} {'positions' if not positions else ''} is missing."
    
    gene_counts_df = pl.read_csv(raw_stdata_path).to_pandas()
    gene_counts_df_values = (gene_counts_df.iloc[2:, 1:]).values.T
    gene_counts_matrix = csc_matrix(gene_counts_df_values)
    
    spot_barcode_list = gene_counts_df.columns.tolist()[1:]
    spot_barcode_df = pd.DataFrame(spot_barcode_list, columns=["spot_barcode"])
    spot_barcode_df = spot_barcode_df.set_index("spot_barcode")
    positions_df = pd.read_csv(raw_positions_path, header=None, names=["spot_barcode", "in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"])
    positions_df = positions_df.set_index("spot_barcode")
    positions_df = positions_df[positions_df["in_tissue"].isin(["1", 1])]
    assert set(spot_barcode_df.index) == set(positions_df.index), f"Unmatching between the spot barcodes in barcodes.tsv and tissue_positions_list.csv, and the difference is {set(spot_barcode_df.index) - set(positions_df.index)}."
    
    concat_obs_df = pd.concat([spot_barcode_df, positions_df], axis=1)
    concat_obs_df_copy = concat_obs_df.copy()
    concat_obs_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    concat_obs_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    concat_obs_df_copy.loc[:, "organ_type"] = "kidney"
    image_extension = os.path.splitext(raw_image_path)[1]
    concat_obs_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in concat_obs_df.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in concat_obs_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for NCBI003 dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in concat_obs_df_copy.columns]}"
    concat_obs_df_copy = concat_obs_df_copy.reindex(columns=obs_columns_order)
    total_obs_df = concat_obs_df_copy
    
    gene_names_list = gene_counts_df[""].tolist()[2:]
    gene_feature_df = pd.DataFrame(gene_names_list, columns=["gene_names"])
    gene_feature_df = gene_feature_df.set_index("gene_names")
    
    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (total_obs_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)

    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)
        
    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path)
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image = Image.open(raw_image_path)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = row["pxl_row_in_fullres"]
            y_pixel = row["pxl_col_in_fullres"]
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))

    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for NCBI003 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def NCBI004_dataset_list(
        main_data_storage: str,
        project_data_folder_name: str,
) -> List:
    dataset_name = ["NCBI", "NCBI004"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    return project_datasets

def NCBI004_single_helper(
        main_data_storage: str,
        project_data_folder_name: str,
        project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["NCBI", "NCBI004"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for NCBI004 dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    project_dataset_raw_unzipped_data_content_list = os.listdir(project_dataset_raw_unzipped_data_path)
    assert len(project_dataset_raw_unzipped_data_content_list) == 2, f"Unmatching between the number of files and the expected number of files for NCBI004 dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)

    image, positions, barcodes, features, matrix = False, False, False, False, False
    image_file = [content for content in project_dataset_raw_data_content_list if (".jpg" in content) or (".tif" in content) or (".tiff" in content) or (".btf" in content)]
    assert len(image_file) == 1, f"More than one image file for NCBI004 dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_data_path, image_file)
    image = True
    
    positions_folder = [content for content in project_dataset_raw_unzipped_data_content_list if ("spatial" in content)]
    assert len(positions_folder) == 1, f"More than one positions folder for NCBI004 dataset {project_dataset}"
    positions_folder = positions_folder[0]
    raw_positions_folder_path = os.path.join(project_dataset_raw_unzipped_data_path, positions_folder)
    raw_positions_folder_content_list = os.listdir(raw_positions_folder_path)
    positions_file = [content for content in raw_positions_folder_content_list if ("tissue_positions_list" in content) or ("tissue_positions" in content)]
    assert len(positions_file) == 1, f"More than one positions file for NCBI004 dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(raw_positions_folder_path, positions_file)
    positions = True

    filtered_feature_bc_matrix_all_unzipped_folder = [content for content in project_dataset_raw_unzipped_data_content_list if "filtered_feature_bc_matrix_all_unzipped" in content]
    assert len(filtered_feature_bc_matrix_all_unzipped_folder) == 1, f"More than one filtered_feature_bc_matrix_all_unzipped folder for NCBI004 dataset {project_dataset}"
    filtered_feature_bc_matrix_all_unzipped_folder = filtered_feature_bc_matrix_all_unzipped_folder[0]
    filtered_feature_bc_matrix_all_unzipped_folder_path = os.path.join(project_dataset_raw_unzipped_data_path, filtered_feature_bc_matrix_all_unzipped_folder)
    filtered_feature_bc_matrix_all_unzipped_folder_content_list = os.listdir(filtered_feature_bc_matrix_all_unzipped_folder_path)
    
    barcodes_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "barcodes.tsv" in content]
    assert len(barcodes_file) == 1, f"More than one barcodes file for NCBI004 dataset {project_dataset}"
    barcodes_file = barcodes_file[0]
    raw_barcodes_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, barcodes_file)
    barcodes = True
    
    features_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "features.tsv" in content]
    assert len(features_file) == 1, f"More than one features file for NCBI004 dataset {project_dataset}"
    features_file = features_file[0]
    raw_features_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, features_file)
    features = True
    
    matrix_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "matrix.mtx" in content]
    assert len(matrix_file) == 1, f"More than one matrix file for NCBI004 dataset {project_dataset}"
    matrix_file = matrix_file[0]
    raw_matrix_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, matrix_file)
    matrix = True
    assert (image & positions & barcodes & features & matrix) == True, f"Missing filetype for NCBI004 dataset {project_dataset}: {'image' if not image else ''} {'positions' if not positions else ''} {'barcodes' if not barcodes else ''} {'features' if not features else ''} {'matrix' if not matrix else ''} is missing."

    gene_counts_matrix = csr_matrix(mmread(raw_matrix_path).T)
    spot_barcode_df = pd.read_csv(raw_barcodes_path, header=None, names=["spot_barcode"])
    spot_barcode_df = spot_barcode_df.set_index("spot_barcode")
    positions_df = pd.read_csv(raw_positions_path, header=None, names=["spot_barcode", "in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"])
    positions_df = positions_df.set_index("spot_barcode")
    positions_df = positions_df[positions_df["in_tissue"].isin(["1", 1])]
    assert set(spot_barcode_df.index) == set(positions_df.index), f"Unmatching between the spot barcodes in barcodes.tsv and tissue_positions_list.csv, and the difference is {set(spot_barcode_df.index) - set(positions_df.index)}."
    if project_dataset == "pt15":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] - 450
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] + 450
    elif project_dataset == "pt16":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] - 700
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] + 700
    elif project_dataset == "pt19":
        pass
    elif project_dataset == "pt24":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] + 450
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] - 450
    elif project_dataset == "pt26":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] + 450
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] - 450
    elif project_dataset == "pt27":
        positions_df["pxl_row_in_fullres"] = positions_df["pxl_row_in_fullres"] + 500
        positions_df["pxl_col_in_fullres"] = positions_df["pxl_col_in_fullres"] - 500
    else:
        raise ValueError(f"Unknown project dataset {project_dataset}.")

    concat_obs_df = pd.concat([spot_barcode_df, positions_df], axis=1)
    concat_obs_df_copy = concat_obs_df.copy()
    concat_obs_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    concat_obs_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    concat_obs_df_copy.loc[:, "organ_type"] = "brain"
    image_extension = os.path.splitext(raw_image_path)[1]
    concat_obs_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in concat_obs_df.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in concat_obs_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for NCBI004 dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in concat_obs_df_copy.columns]}"
    concat_obs_df_copy = concat_obs_df_copy.reindex(columns=obs_columns_order)
    total_obs_df = concat_obs_df_copy
    gene_feature_df =  pd.read_csv(raw_features_path, header=None, delimiter='\t', names=["gene_barcode", "gene_name", "gene_expression"])
    gene_feature_df = gene_feature_df.set_index("gene_name")

    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (positions_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)

    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)

    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path).convert("RGB")
        if project_dataset == "pt15":
            image = image.rotate(90) # 90
        elif project_dataset == "pt16":
            image = image.rotate(90) # 90
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif project_dataset == "pt19":
            pass
        elif project_dataset == "pt24":
            image = image.rotate(90) # 90
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif project_dataset == "pt26":
            image = image.rotate(90) # 90
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif project_dataset == "pt27":
            image = image.rotate(90) # 90
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            raise ValueError(f"Unknown project dataset {project_dataset}.")
        draw = ImageDraw.Draw(image)
        dot_size = 64
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image = Image.open(raw_image_path)
        if project_dataset == "pt15":
            image = image.rotate(90) # 90
        elif project_dataset == "pt16":
            image = image.rotate(90) # 90
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif project_dataset == "pt19":
            pass
        elif project_dataset == "pt24":
            image = image.rotate(90) # 90
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif project_dataset == "pt26":
            image = image.rotate(90) # 90
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif project_dataset == "pt27":
            image = image.rotate(90) # 90
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            raise ValueError(f"Unknown project dataset {project_dataset}.")
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = row["pxl_row_in_fullres"]
            y_pixel = row["pxl_col_in_fullres"]
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))

    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for NCBI004 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def NCBI005_dataset_list(
        main_data_storage: str,
        project_data_folder_name: str,
) -> List:
    dataset_name = ["NCBI", "NCBI005"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    return project_datasets

def NCBI005_single_helper(
        main_data_storage: str,
        project_data_folder_name: str,
        project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["NCBI", "NCBI005"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for NCBI005 dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    project_dataset_raw_unzipped_data_content_list = os.listdir(project_dataset_raw_unzipped_data_path)
    assert len(project_dataset_raw_unzipped_data_content_list) == 2, f"Unmatching between the number of files and the expected number of files for NCBI005 dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)

    image, positions, stdata = False, False, False
    image_file = [content for content in project_dataset_raw_unzipped_data_content_list if (".jpg" in content) or (".tif" in content) or (".tiff" in content) or (".btf" in content)]
    assert len(image_file) == 1, f"More than one image file for NCBI005 dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_unzipped_data_path, image_file)
    image = True
    positions_file = [content for content in project_dataset_raw_unzipped_data_content_list if ("tissue_positions_list" in content) or ("tissue_positions" in content)]
    assert len(positions_file) == 1, f"More than one positions file for NCBI005 dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(project_dataset_raw_unzipped_data_path, positions_file)
    positions = True
    stdata_file = [content for content in project_dataset_raw_data_content_list if (".h5" in content)]
    assert len(stdata_file) == 1, f"More than one stdata file for NCBI005 dataset {project_dataset}"
    stdata_file = stdata_file[0]
    raw_stdata_path = os.path.join(project_dataset_raw_data_path, stdata_file)
    stdata = True
    assert (image & stdata & positions) == True, f"Missing filetype for NCBI005 dataset {project_dataset}: {'image' if not image else ''} {'stdata' if not stdata else ''} {'positions' if not positions else ''} is missing."

    project_dataset_raw_stdata = h5py.File(raw_stdata_path, 'r')
    gene_counts_matrix = csc_matrix((project_dataset_raw_stdata['matrix/data'][:], project_dataset_raw_stdata['matrix/indices'][:], project_dataset_raw_stdata['matrix/indptr'][:]), shape=project_dataset_raw_stdata['matrix/shape'][:]).tocsr().transpose()
    spot_barcode_list = [item.decode() for item in project_dataset_raw_stdata['matrix/barcodes'][:]]
    spot_barcode_df = pd.DataFrame(spot_barcode_list, columns=["spot_barcode"])
    spot_barcode_df = spot_barcode_df.set_index("spot_barcode")
    positions_df = pd.read_csv(raw_positions_path, header=None)
    positions_df = positions_df.rename(columns={0:'spot_barcode', 1:'in_tissue', 2:'array_row', 3:'array_col', 4:'pxl_col_in_fullres', 5:'pxl_row_in_fullres'})
    positions_df = positions_df.set_index('spot_barcode')
    positions_df = positions_df[positions_df['in_tissue'] == 1]
    assert set(spot_barcode_df.index) == set(positions_df.index), f"Unmatching between the spot barcodes in barcodes.tsv and tissue_positions_list.csv, and the difference is {set(spot_barcode_df.index) - set(positions_df.index)}."
    concat_obs_df = pd.concat([spot_barcode_df, positions_df], axis=1)
    concat_obs_df_copy = concat_obs_df.copy()
    concat_obs_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    concat_obs_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    if project_dataset in ["GSM5621965", "GSM5621966", "GSM5621967"]:
        concat_obs_df_copy.loc[:, "organ_type"] = "kidney"
    elif project_dataset in ["GSM5621968", "GSM5621969", "GSM5621970", "GSM5621971"]:
        concat_obs_df_copy.loc[:, "organ_type"] = "lung"
    else:
        raise ValueError(f"Unknown project dataset {project_dataset}.")
    image_extension = os.path.splitext(raw_image_path)[1]
    concat_obs_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in concat_obs_df.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in concat_obs_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for NCBI005 dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in concat_obs_df_copy.columns]}"
    concat_obs_df_copy = concat_obs_df_copy.reindex(columns=obs_columns_order)
    total_obs_df = concat_obs_df_copy
    
    gene_name_list = [item.decode() for item in project_dataset_raw_stdata['matrix/features/name'][:]]
    gene_feature_df = pd.DataFrame(gene_name_list, columns=["gene_names"])
    gene_feature_df = gene_feature_df.set_index("gene_names")

    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (total_obs_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)

    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)
        
    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path)
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image = Image.open(raw_image_path)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = row["pxl_row_in_fullres"]
            y_pixel = row["pxl_col_in_fullres"]
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))

    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for NCBI005 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def NCBI007_dataset_list(
        main_data_storage: str,
        project_data_folder_name: str,
) -> List:
    dataset_name = ["NCBI", "NCBI007"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    return project_datasets

def NCBI007_single_helper(
        main_data_storage: str,
        project_data_folder_name: str,
        project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["NCBI", "NCBI007"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for NCBI007 dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    project_dataset_raw_unzipped_data_content_list = os.listdir(project_dataset_raw_unzipped_data_path)
    assert len(project_dataset_raw_unzipped_data_content_list) == 1, f"Unmatching between the number of files and the expected number of files for NCBI007 dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)

    image, positions, barcodes, features, matrix = False, False, False, False, False
    image_file = [content for content in project_dataset_raw_data_content_list if (".svs" in content)]
    assert len(image_file) == 1, f"More than one image file for NCBI007 dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_data_path, image_file)
    image = True
    
    positions_folder = [content for content in project_dataset_raw_data_content_list if ("spatial" in content)]
    assert len(positions_folder) == 1, f"More than one positions folder for NCBI007 dataset {project_dataset}"
    positions_folder = positions_folder[0]
    raw_positions_folder_path = os.path.join(project_dataset_raw_data_path, positions_folder)
    raw_positions_folder_content_list = os.listdir(raw_positions_folder_path)
    positions_file = [content for content in raw_positions_folder_content_list if ("tissue_positions_list" in content) or ("tissue_positions" in content)]
    assert len(positions_file) == 1, f"More than one positions file for NCBI007 dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(raw_positions_folder_path, positions_file)
    positions = True

    filtered_feature_bc_matrix_all_unzipped_folder = [content for content in project_dataset_raw_unzipped_data_content_list if "filtered_feature_bc_matrix_all_unzipped" in content]
    assert len(filtered_feature_bc_matrix_all_unzipped_folder) == 1, f"More than one filtered_feature_bc_matrix_all_unzipped folder for 10xGenomics dataset {project_dataset}"
    filtered_feature_bc_matrix_all_unzipped_folder = filtered_feature_bc_matrix_all_unzipped_folder[0]
    filtered_feature_bc_matrix_all_unzipped_folder_path = os.path.join(project_dataset_raw_unzipped_data_path, filtered_feature_bc_matrix_all_unzipped_folder)
    filtered_feature_bc_matrix_all_unzipped_folder_content_list = os.listdir(filtered_feature_bc_matrix_all_unzipped_folder_path)
    
    barcodes_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "barcodes.tsv" in content]
    assert len(barcodes_file) == 1, f"More than one barcodes file for NCBI007 dataset {project_dataset}"
    barcodes_file = barcodes_file[0]
    raw_barcodes_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, barcodes_file)
    barcodes = True
    
    features_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "features.tsv" in content]
    assert len(features_file) == 1, f"More than one features file for NCBI007 dataset {project_dataset}"
    features_file = features_file[0]
    raw_features_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, features_file)
    features = True
    
    matrix_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "matrix.mtx" in content]
    assert len(matrix_file) == 1, f"More than one matrix file for NCBI007 dataset {project_dataset}"
    matrix_file = matrix_file[0]
    raw_matrix_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, matrix_file)
    matrix = True
    assert (image & positions & barcodes & features & matrix) == True, f"Missing filetype for NCBI007 dataset {project_dataset}: {'image' if not image else ''} {'positions' if not positions else ''} {'barcodes' if not barcodes else ''} {'features' if not features else ''} {'matrix' if not matrix else ''} is missing."

    gene_counts_matrix = csr_matrix(mmread(raw_matrix_path).T)
    spot_barcode_df = pd.read_csv(raw_barcodes_path, header=None, names=["spot_barcode"])
    spot_barcode_df = spot_barcode_df.set_index("spot_barcode")
    positions_df = pd.read_csv(raw_positions_path, header=None, names=["spot_barcode", "in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"])
    positions_df = positions_df.set_index("spot_barcode")
    positions_df = positions_df[positions_df["in_tissue"].isin(["1", 1])]
    if project_dataset == "A1":
        x1, y1, x2, y2 = 8560, 3180 + 1000, 29620, 27365 + 1000
        scale_x = (x2 - x1) / (max(positions_df["pxl_row_in_fullres"]) - min(positions_df["pxl_row_in_fullres"]))
        scale_y = (y2 - y1) / (max(positions_df["pxl_col_in_fullres"]) - min(positions_df["pxl_col_in_fullres"]))
        scaled_x_positions = [int(scale_x * x) for x in positions_df["pxl_row_in_fullres"]]
        scaled_y_positions = [int(scale_y * y) for y in positions_df["pxl_col_in_fullres"]]
        matched_x_posiitons = [(x - abs(x2 - max(scaled_x_positions))) for x in scaled_x_positions]
        matched_y_positions = [(y - abs(y2 - max(scaled_y_positions))) for y in scaled_y_positions]
        positions_df["pxl_row_in_fullres"] = matched_x_posiitons
        positions_df["pxl_col_in_fullres"] = matched_y_positions
    elif project_dataset == "A2":
        x1, y1, x2, y2 = 9160, 3330, 30600, 27900
        scale_x = (x2 - x1) / (max(positions_df["pxl_row_in_fullres"]) - min(positions_df["pxl_row_in_fullres"]))
        scale_y = (y2 - y1) / (max(positions_df["pxl_col_in_fullres"]) - min(positions_df["pxl_col_in_fullres"]))
        scaled_x_positions = [int(scale_x * x) for x in positions_df["pxl_row_in_fullres"]]
        scaled_y_positions = [int(scale_y * y) for y in positions_df["pxl_col_in_fullres"]]
        matched_x_posiitons = [(x - abs(x2 - max(scaled_x_positions))) for x in scaled_x_positions]
        matched_y_positions = [(y - abs(y2 - max(scaled_y_positions))) for y in scaled_y_positions]
        positions_df["pxl_row_in_fullres"] = matched_x_posiitons
        positions_df["pxl_col_in_fullres"] = matched_y_positions
    elif project_dataset == "A3":
        x1, y1, x2, y2 = 9900, 7010, 29840, 30570
        # scale_x = (x2 - x1) / (max(positions_df["pxl_row_in_fullres"]) - min(positions_df["pxl_row_in_fullres"]))
        scale_x = 5
        scale_y = (y2 - y1) / (max(positions_df["pxl_col_in_fullres"]) - min(positions_df["pxl_col_in_fullres"]))
        scaled_x_positions = [int(scale_x * x) for x in positions_df["pxl_row_in_fullres"]]
        scaled_y_positions = [int(scale_y * y) for y in positions_df["pxl_col_in_fullres"]]
        matched_x_posiitons = [x for x in scaled_x_positions]
        matched_y_positions = [(y - abs(y2 - max(scaled_y_positions))) for y in scaled_y_positions]
        positions_df["pxl_row_in_fullres"] = matched_x_posiitons
        positions_df["pxl_col_in_fullres"] = matched_y_positions
    elif project_dataset == "A4":
        x1, y1, x2, y2 = 6860, 2600, 30570, 29000
        scale_x = (x2 - x1) / (max(positions_df["pxl_row_in_fullres"]) - min(positions_df["pxl_row_in_fullres"]))
        scale_y = (y2 - y1) / (max(positions_df["pxl_col_in_fullres"]) - min(positions_df["pxl_col_in_fullres"]))
        scaled_x_positions = [int(scale_x * x) for x in positions_df["pxl_row_in_fullres"]]
        scaled_y_positions = [int(scale_y * y) for y in positions_df["pxl_col_in_fullres"]]
        matched_x_posiitons = [x for x in scaled_x_positions]
        matched_y_positions = [(y - abs(y2 - max(scaled_y_positions))) for y in scaled_y_positions]
        positions_df["pxl_row_in_fullres"] = matched_x_posiitons
        positions_df["pxl_col_in_fullres"] = matched_y_positions
    elif project_dataset == "A6":
        x1, y1, x2, y2 = 6500, 8700, 30265, 26425
        scale_x = (x2 - x1) / (max(positions_df["pxl_row_in_fullres"]) - min(positions_df["pxl_row_in_fullres"]))
        scale_y = (y2 - y1) / (max(positions_df["pxl_col_in_fullres"]) - min(positions_df["pxl_col_in_fullres"]))
        scale_y = 3.5
        scaled_x_positions = [int(scale_x * x) for x in positions_df["pxl_row_in_fullres"]]
        scaled_y_positions = [int(scale_y * y) for y in positions_df["pxl_col_in_fullres"]]
        matched_x_posiitons = [(x - abs(x2 - max(scaled_x_positions))) for x in scaled_x_positions]
        matched_y_positions = [y for y in scaled_y_positions]
        positions_df["pxl_row_in_fullres"] = matched_x_posiitons
        positions_df["pxl_col_in_fullres"] = matched_y_positions
    elif project_dataset == "A7":
        x1, y1, x2, y2 = 7160, 7460, 30730, 26110
        scale_x = (x2 - x1) / (max(positions_df["pxl_row_in_fullres"]) - min(positions_df["pxl_row_in_fullres"]))
        scale_y = (y2 - y1) / (max(positions_df["pxl_col_in_fullres"]) - min(positions_df["pxl_col_in_fullres"]))
        scale_y = 3.5
        scaled_x_positions = [int(scale_x * x) for x in positions_df["pxl_row_in_fullres"]]
        scaled_y_positions = [int(scale_y * y) for y in positions_df["pxl_col_in_fullres"]]
        matched_x_posiitons = [(x - abs(x2 - max(scaled_x_positions))) for x in scaled_x_positions]
        matched_y_positions = [y for y in scaled_y_positions]
        positions_df["pxl_row_in_fullres"] = matched_x_posiitons
        positions_df["pxl_col_in_fullres"] = matched_y_positions
    elif project_dataset == "A8":
        x1, y1, x2, y2 = 6300, 3235, 29800, 27020
        scale_x = (x2 - x1) / (max(positions_df["pxl_row_in_fullres"]) - min(positions_df["pxl_row_in_fullres"]))
        scale_y = (y2 - y1) / (max(positions_df["pxl_col_in_fullres"]) - min(positions_df["pxl_col_in_fullres"]))
        scaled_x_positions = [int(scale_x * x) for x in positions_df["pxl_row_in_fullres"]]
        scaled_y_positions = [int(scale_y * y) for y in positions_df["pxl_col_in_fullres"]]
        matched_x_posiitons = [(x - abs(x2 - max(scaled_x_positions))) for x in scaled_x_positions]
        matched_y_positions = [y for y in scaled_y_positions]
        positions_df["pxl_row_in_fullres"] = matched_x_posiitons
        positions_df["pxl_col_in_fullres"] = matched_y_positions
    elif project_dataset == "A9":
        x1, y1, x2, y2 = 9700, 2900, 31100, 25330
        scale_x = (x2 - x1) / (max(positions_df["pxl_row_in_fullres"]) - min(positions_df["pxl_row_in_fullres"]))
        scale_y = (y2 - y1) / (max(positions_df["pxl_col_in_fullres"]) - min(positions_df["pxl_col_in_fullres"]))
        scaled_x_positions = [int(scale_x * x) for x in positions_df["pxl_row_in_fullres"]]
        scaled_y_positions = [int(scale_y * y) for y in positions_df["pxl_col_in_fullres"]]
        matched_x_posiitons = [x for x in scaled_x_positions]
        matched_y_positions = [(y - abs(y2 - max(scaled_y_positions))) for y in scaled_y_positions]
        positions_df["pxl_row_in_fullres"] = matched_x_posiitons
        positions_df["pxl_col_in_fullres"] = matched_y_positions
    else:
        raise ValueError(f"Unknown project dataset {project_dataset}.")
    assert set(spot_barcode_df.index) == set(positions_df.index), f"Unmatching between the spot barcodes in barcodes.tsv and tissue_positions_list.csv, and the difference is {set(spot_barcode_df.index) - set(positions_df.index)}."

    concat_obs_df = pd.concat([spot_barcode_df, positions_df], axis=1)
    concat_obs_df_copy = concat_obs_df.copy()
    concat_obs_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    concat_obs_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    concat_obs_df_copy.loc[:, "organ_type"] = "small_and_large_intestine"
    # image_extension = os.path.splitext(raw_image_path)[1]
    image_extension = ".tif"
    concat_obs_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in concat_obs_df.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in concat_obs_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for NCBI007 dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in concat_obs_df_copy.columns]}"
    concat_obs_df_copy = concat_obs_df_copy.reindex(columns=obs_columns_order)
    total_obs_df = concat_obs_df_copy
    gene_feature_df =  pd.read_csv(raw_features_path, header=None, delimiter='\t', names=["gene_barcode", "gene_name", "gene_expression"])
    gene_feature_df = gene_feature_df.set_index("gene_name")

    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (positions_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)

    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)
    
    # While the svs image is super large and slow to open, we only read it once.
    slide = openslide.open_slide(raw_image_path)
    image_width = slide.dimensions[0]
    image_height = slide.dimensions[1]
    num_levels = slide.level_count
    level = 0  # Level 0 represents highest resolution
    region = (0, 0, image_width, image_height)
    svs_image = slide.read_region(region, level, slide.level_dimensions[level])

    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        # image = Image.open(raw_image_path).convert("RGB")
        image = svs_image.convert("RGB")
        # image = Image.fromarray(image) # Optional?
        image = image.rotate(-90)
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        # image = Image.open(raw_image_path)
        image = svs_image.convert("RGB")
        # image = Image.fromarray(image) # Optional?
        image = image.rotate(-90)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = row["pxl_row_in_fullres"]
            y_pixel = row["pxl_col_in_fullres"]
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))

    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for NCBI007 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def NCBI008_dataset_list(
        main_data_storage: str,
        project_data_folder_name: str,
) -> List:
    dataset_name = ["NCBI", "NCBI008"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    return project_datasets

def NCBI008_single_helper(
        main_data_storage: str,
        project_data_folder_name: str,
        project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["NCBI", "NCBI008"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for NCBI008 dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    project_dataset_raw_unzipped_data_content_list = os.listdir(project_dataset_raw_unzipped_data_path)
    assert len(project_dataset_raw_unzipped_data_content_list) == 1, f"Unmatching between the number of files and the expected number of files for NCBI008 dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)

    image, positions, stdata = False, False, False
    image_file = [content for content in project_dataset_raw_data_content_list if (".jpg" in content) or (".tif" in content) or (".tiff" in content) or (".btf" in content)]
    assert len(image_file) == 1, f"More than one image file for NCBI008 dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_data_path, image_file)
    image = True
    positions_file = [content for content in project_dataset_raw_unzipped_data_content_list if ("tissue_positions_list" in content) or ("tissue_positions" in content)]
    assert len(positions_file) == 1, f"More than one positions file for NCBI008 dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(project_dataset_raw_unzipped_data_path, positions_file)
    positions = True
    stdata_file = [content for content in project_dataset_raw_data_content_list if (".h5" in content)]
    assert len(stdata_file) == 1, f"More than one stdata file for NCBI008 dataset {project_dataset}"
    stdata_file = stdata_file[0]
    raw_stdata_path = os.path.join(project_dataset_raw_data_path, stdata_file)
    stdata = True
    assert (image & stdata & positions) == True, f"Missing filetype for NCBI008 dataset {project_dataset}: {'image' if not image else ''} {'stdata' if not stdata else ''} {'positions' if not positions else ''} is missing."

    project_dataset_raw_stdata = h5py.File(raw_stdata_path, 'r')
    gene_counts_matrix = csc_matrix((project_dataset_raw_stdata['matrix/data'][:], project_dataset_raw_stdata['matrix/indices'][:], project_dataset_raw_stdata['matrix/indptr'][:]), shape=project_dataset_raw_stdata['matrix/shape'][:]).tocsr().transpose()
    spot_barcode_list = [item.decode() for item in project_dataset_raw_stdata['matrix/barcodes'][:]]
    spot_barcode_df = pd.DataFrame(spot_barcode_list, columns=["spot_barcode"])
    spot_barcode_df = spot_barcode_df.set_index("spot_barcode")
    positions_df = pd.read_csv(raw_positions_path, header=None)
    positions_df = positions_df.rename(columns={0:'spot_barcode', 1:'in_tissue', 2:'array_row', 3:'array_col', 4:'pxl_col_in_fullres', 5:'pxl_row_in_fullres'})
    positions_df = positions_df.set_index('spot_barcode')
    positions_df = positions_df[positions_df['in_tissue'] == 1]
    assert set(spot_barcode_df.index) == set(positions_df.index), f"Unmatching between the spot barcodes in barcodes.tsv and tissue_positions_list.csv, and the difference is {set(spot_barcode_df.index) - set(positions_df.index)}."
    concat_obs_df = pd.concat([spot_barcode_df, positions_df], axis=1)
    concat_obs_df_copy = concat_obs_df.copy()
    concat_obs_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    concat_obs_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    concat_obs_df_copy.loc[:, "organ_type"] = "liver"
    image_extension = os.path.splitext(raw_image_path)[1]
    concat_obs_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in concat_obs_df.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in concat_obs_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for NCBI008 dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in concat_obs_df_copy.columns]}"
    concat_obs_df_copy = concat_obs_df_copy.reindex(columns=obs_columns_order)
    total_obs_df = concat_obs_df_copy
    
    gene_name_list = [item.decode() for item in project_dataset_raw_stdata['matrix/features/name'][:]]
    gene_feature_df = pd.DataFrame(gene_name_list, columns=["gene_names"])
    gene_feature_df = gene_feature_df.set_index("gene_names")

    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (total_obs_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)

    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)
        
    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path)
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image = Image.open(raw_image_path)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = row["pxl_row_in_fullres"]
            y_pixel = row["pxl_col_in_fullres"]
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))

    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for NCBI008 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def NCBI009_dataset_list(
        main_data_storage: str,
        project_data_folder_name: str,
) -> None:
    dataset_name = ["NCBI", "NCBI009"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    return project_datasets

def NCBI009_single_helper(
        main_data_storage: str,
        project_data_folder_name: str,
        project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["NCBI", "NCBI009"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for NCBI009 dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)

    image, positions, stdata = False, False, False
    image_file = [content for content in project_dataset_raw_data_content_list if (".jpg" in content) or (".tif" in content) or (".tiff" in content) or (".btf" in content)]
    assert len(image_file) == 1, f"More than one image file for NCBI009 dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_data_path, image_file)
    image = True
    positions_folder = [content for content in project_dataset_raw_data_content_list if ("spatial" in content)]
    assert len(positions_folder) == 1, f"More than one positions folder for NCBI009 dataset {project_dataset}"
    positions_folder = positions_folder[0]
    raw_positions_folder_path = os.path.join(project_dataset_raw_data_path, positions_folder)
    raw_positions_folder_content_list = os.listdir(raw_positions_folder_path)
    positions_file = [content for content in raw_positions_folder_content_list if ("tissue_positions_list" in content) or ("tissue_positions" in content)]
    assert len(positions_file) == 1, f"More than one positions file for NCBI009 dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(raw_positions_folder_path, positions_file)
    positions = True
    stdata_file = [content for content in project_dataset_raw_data_content_list if (".h5" in content)]
    assert len(stdata_file) == 1, f"More than one stdata file for NCBI009 dataset {project_dataset}"
    stdata_file = stdata_file[0]
    raw_stdata_path = os.path.join(project_dataset_raw_data_path, stdata_file)
    stdata = True
    assert (image & stdata & positions) == True, f"Missing filetype for NCBI009 dataset {project_dataset}: {'image' if not image else ''} {'stdata' if not stdata else ''} {'positions' if not positions else ''} is missing."

    project_dataset_raw_stdata = h5py.File(raw_stdata_path, 'r')
    gene_counts_matrix = csc_matrix((project_dataset_raw_stdata['matrix/data'][:], project_dataset_raw_stdata['matrix/indices'][:], project_dataset_raw_stdata['matrix/indptr'][:]), shape=project_dataset_raw_stdata['matrix/shape'][:]).tocsr().transpose()
    spot_barcode_list = [item.decode() for item in project_dataset_raw_stdata['matrix/barcodes'][:]]
    spot_barcode_df = pd.DataFrame(spot_barcode_list, columns=["spot_barcode"])
    spot_barcode_df = spot_barcode_df.set_index("spot_barcode")
    positions_df = pd.read_csv(raw_positions_path, header=None)
    positions_df = positions_df.rename(columns={0:'spot_barcode', 1:'in_tissue', 2:'array_row', 3:'array_col', 4:'pxl_col_in_fullres', 5:'pxl_row_in_fullres'})
    positions_df = positions_df.set_index('spot_barcode')
    positions_df = positions_df[positions_df['in_tissue'] == 1]
    assert set(spot_barcode_df.index) == set(positions_df.index), f"Unmatching between the spot barcodes in barcodes.tsv and tissue_positions_list.csv, and the difference is {set(spot_barcode_df.index) - set(positions_df.index)}."
    concat_obs_df = pd.concat([spot_barcode_df, positions_df], axis=1)
    concat_obs_df_copy = concat_obs_df.copy()
    concat_obs_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    concat_obs_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    concat_obs_df_copy.loc[:, "organ_type"] = "lung"
    image_extension = os.path.splitext(raw_image_path)[1]
    concat_obs_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in concat_obs_df.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in concat_obs_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for NCBI009 dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in concat_obs_df_copy.columns]}"
    concat_obs_df_copy = concat_obs_df_copy.reindex(columns=obs_columns_order)
    total_obs_df = concat_obs_df_copy

    gene_name_list = [item.decode() for item in project_dataset_raw_stdata['matrix/features/name'][:]]
    gene_feature_df = pd.DataFrame(gene_name_list, columns=["gene_names"])
    gene_feature_df = gene_feature_df.set_index("gene_names")

    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (total_obs_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)

    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)
        
    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path)
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image = Image.open(raw_image_path)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = row["pxl_row_in_fullres"]
            y_pixel = row["pxl_col_in_fullres"]
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))

    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for NCBI009 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def STNet_dataset_list(
        main_data_storage: str,
        project_data_folder_name: str,
) -> List:
    dataset_name = ["STNet"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    return project_datasets

def STNet_single_helper(
        main_data_storage: str,
        project_data_folder_name: str,
        gene_barcode_name_mapping_dict: dict,
        project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["STNet"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 4, f"Unmatching between the number of files and the expected number of files for STNet dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    project_dataset_raw_unzipped_data_content_list = os.listdir(project_dataset_raw_unzipped_data_path)
    assert len(project_dataset_raw_unzipped_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for STNet dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)

    image, positions, stdata = False, False, False
    image_file = [content for content in project_dataset_raw_data_content_list if (".jpg" in content) or (".tif" in content) or (".tiff" in content) or (".btf" in content)]
    assert len(image_file) == 1, f"More than one image file for STNet dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_data_path, image_file)
    image = True
    positions_file = [content for content in project_dataset_raw_unzipped_data_content_list if ("spots" in content)]
    assert len(positions_file) == 1, f"More than one positions file for STNet dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(project_dataset_raw_unzipped_data_path, positions_file)
    positions = True
    stdata_file = [content for content in project_dataset_raw_data_content_list if ("stdata" in content)]
    assert len(stdata_file) == 1, f"More than one stdata file for STNet dataset {project_dataset}"
    stdata_file = stdata_file[0]
    raw_stdata_path = os.path.join(project_dataset_raw_data_path, stdata_file)
    stdata = True
    assert (image & stdata & positions) == True, f"Missing filetype for STNet dataset {project_dataset}: {'image' if not image else ''} {'stdata' if not stdata else ''} {'positions' if not positions else ''} is missing."

    gene_counts_df = pd.read_csv(raw_stdata_path, sep="\t")
    gene_counts_df = gene_counts_df.rename(columns={"Unnamed: 0": "spot_barcode"})
    gene_counts_df = gene_counts_df.set_index("spot_barcode")
    gene_counts_df_copy = gene_counts_df.copy()
    columns_without_gene_name = [column for column in gene_counts_df_copy.columns if column not in gene_barcode_name_mapping_dict.keys()]
    gene_counts_df_copy = gene_counts_df_copy.drop(columns=columns_without_gene_name)
    gene_counts_df_copy.columns = [gene_barcode_name_mapping_dict[column] for column in gene_counts_df_copy.columns]
    gene_counts_matrix = csr_matrix(gene_counts_df_copy.values)

    spot_barcode_df = pd.read_csv(raw_positions_path)
    spot_barcode_df = spot_barcode_df.rename(columns={"Unnamed: 0": "spot_barcode", "X": "pxl_row_in_fullres", "Y": "pxl_col_in_fullres"})
    spot_barcode_df = spot_barcode_df.set_index("spot_barcode")
    spot_barcode_df_copy = spot_barcode_df.copy()

    spot_barcode_intersection = list(set(spot_barcode_df_copy.index).intersection(gene_counts_df.index))
    row_mask = np.isin(gene_counts_df.index, spot_barcode_intersection)
    gene_counts_matrix = gene_counts_matrix[row_mask, :]
    spot_barcode_df_copy = spot_barcode_df_copy.loc[spot_barcode_intersection]
    gene_counts_df = gene_counts_df.loc[spot_barcode_intersection]
    assert set(spot_barcode_df_copy.index) == set(gene_counts_df.index), f"Unmatching between the spot barcodes in the positions file and the gene counts file for STNet dataset {project_dataset}"

    spot_barcode_df_copy.loc[:, "in_tissue"] = 1
    spot_barcode_df_copy.loc[:, "array_row"] = [index.split("x")[0] for index in spot_barcode_df_copy.index]
    spot_barcode_df_copy.loc[:, "array_col"] = [index.split("x")[1] for index in spot_barcode_df_copy.index]
    spot_barcode_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    spot_barcode_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    spot_barcode_df_copy.loc[:, "organ_type"] = "breast"
    image_extension = os.path.splitext(raw_image_path)[1]
    spot_barcode_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in spot_barcode_df_copy.index]
    spot_barcode_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in spot_barcode_df_copy.index]
    spot_barcode_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in spot_barcode_df_copy.index]
    spot_barcode_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in spot_barcode_df_copy.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in spot_barcode_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for STNet dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in spot_barcode_df_copy.columns]}"
    spot_barcode_df_copy = spot_barcode_df_copy.reindex(columns=obs_columns_order)
    total_obs_df = spot_barcode_df_copy

    gene_feature_df = pd.DataFrame(gene_counts_df_copy.columns)
    gene_feature_df.columns = ["gene_names"]
    gene_feature_df = gene_feature_df.set_index("gene_names")

    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (total_obs_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)

    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)
        
    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path)
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image = Image.open(raw_image_path)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = row["pxl_row_in_fullres"]
            y_pixel = row["pxl_col_in_fullres"]
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))

    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for STNet dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def Zenodo001_dataset_list(
        main_data_storage: str,
        project_data_folder_name: str,
) -> List:
    dataset_name = ["Zenodo", "Zenodo001"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    return project_datasets

def Zenodo001_single_helper(
        main_data_storage: str,
        project_data_folder_name: str,
        project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["Zenodo", "Zenodo001"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for Zenodo001 dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    project_dataset_raw_unzipped_data_content_list = os.listdir(project_dataset_raw_unzipped_data_path)
    assert len(project_dataset_raw_unzipped_data_content_list) == 1, f"Unmatching between the number of files and the expected number of files for Zenodo001 dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)

    image, positions, barcodes, features, matrix = False, False, False, False, False
    image_file = [content for content in project_dataset_raw_data_content_list if (".svs" in content)]
    assert len(image_file) == 1, f"More than one image file for Zenodo001 dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_data_path, image_file)
    image = True

    positions_file = [content for content in project_dataset_raw_data_content_list if ("tissue_positions_list" in content) or ("tissue_positions" in content)]
    assert len(positions_file) == 1, f"More than one positions file for Zenodo001 dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(project_dataset_raw_data_path, positions_file)
    positions = True

    filtered_feature_bc_matrix_all_unzipped_folder = [content for content in project_dataset_raw_unzipped_data_content_list if "filtered_feature_bc_matrix_all_unzipped" in content]
    assert len(filtered_feature_bc_matrix_all_unzipped_folder) == 1, f"More than one filtered_feature_bc_matrix_all_unzipped folder for 10xGenomics dataset {project_dataset}"
    filtered_feature_bc_matrix_all_unzipped_folder = filtered_feature_bc_matrix_all_unzipped_folder[0]
    filtered_feature_bc_matrix_all_unzipped_folder_path = os.path.join(project_dataset_raw_unzipped_data_path, filtered_feature_bc_matrix_all_unzipped_folder)
    filtered_feature_bc_matrix_all_unzipped_folder_content_list = os.listdir(filtered_feature_bc_matrix_all_unzipped_folder_path)
    
    barcodes_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "barcodes.tsv" in content]
    assert len(barcodes_file) == 1, f"More than one barcodes file for Zenodo001 dataset {project_dataset}"
    barcodes_file = barcodes_file[0]
    raw_barcodes_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, barcodes_file)
    barcodes = True
    
    features_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "features.tsv" in content]
    assert len(features_file) == 1, f"More than one features file for Zenodo001 dataset {project_dataset}"
    features_file = features_file[0]
    raw_features_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, features_file)
    features = True
    
    matrix_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "matrix.mtx" in content]
    assert len(matrix_file) == 1, f"More than one matrix file for Zenodo001 dataset {project_dataset}"
    matrix_file = matrix_file[0]
    raw_matrix_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, matrix_file)
    matrix = True
    assert (image & positions & barcodes & features & matrix) == True, f"Missing filetype for Zenodo001 dataset {project_dataset}: {'image' if not image else ''} {'positions' if not positions else ''} {'barcodes' if not barcodes else ''} {'features' if not features else ''} {'matrix' if not matrix else ''} is missing."

    gene_counts_matrix = csr_matrix(mmread(raw_matrix_path).T)
    spot_barcode_df = pd.read_csv(raw_barcodes_path, header=None, names=["spot_barcode"])
    spot_barcode_df = spot_barcode_df.set_index("spot_barcode")
    positions_df = pd.read_csv(raw_positions_path, header=None, names=["spot_barcode", "in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"])
    positions_df = positions_df.set_index("spot_barcode")
    positions_df = positions_df[positions_df["in_tissue"].isin(["1", 1])]
    if project_dataset == "Control1":
        x1, y1, x2, y2 = 1620, 4350, 12950, 14850
        scale_x = (x2 - x1) / (max(positions_df["pxl_row_in_fullres"]) - min(positions_df["pxl_row_in_fullres"]))
        scale_y = (y2 - y1) / (max(positions_df["pxl_col_in_fullres"]) - min(positions_df["pxl_col_in_fullres"]))
        scaled_x_positions = [int(scale_x * x) for x in positions_df["pxl_row_in_fullres"]]
        scaled_y_positions = [int(scale_y * y) for y in positions_df["pxl_col_in_fullres"]]
        matched_x_posiitons = [(x - abs(x2 - max(scaled_x_positions))) for x in scaled_x_positions]
        matched_y_positions = [(y + 0.5 * abs(y2 - max(scaled_y_positions))) for y in scaled_y_positions]
        positions_df["pxl_row_in_fullres"] = matched_x_posiitons
        positions_df["pxl_col_in_fullres"] = matched_y_positions
    elif project_dataset == "Control2":
        x1, y1, x2, y2 = 4850, 5930, 12570, 13900
        scale_x = (x2 - x1) / (max(positions_df["pxl_row_in_fullres"]) - min(positions_df["pxl_row_in_fullres"]))
        scale_y = (y2 - y1) / (max(positions_df["pxl_col_in_fullres"]) - min(positions_df["pxl_col_in_fullres"]))
        scaled_x_positions = [int(scale_x * x) for x in positions_df["pxl_row_in_fullres"]]
        scaled_y_positions = [int(scale_y * y) for y in positions_df["pxl_col_in_fullres"]]
        matched_x_posiitons = [(x - abs(x2 - max(scaled_x_positions))) for x in scaled_x_positions]
        matched_y_positions = [(y - abs(y2 - max(scaled_y_positions))) for y in scaled_y_positions]
        positions_df["pxl_row_in_fullres"] = matched_x_posiitons
        positions_df["pxl_col_in_fullres"] = matched_y_positions
    elif project_dataset == "Tumor1":
        x1, y1, x2, y2 = 4570, 5150, 11780, 13650
        scale_x = (x2 - x1) / (max(positions_df["pxl_row_in_fullres"]) - min(positions_df["pxl_row_in_fullres"]))
        scale_y = (y2 - y1) / (max(positions_df["pxl_col_in_fullres"]) - min(positions_df["pxl_col_in_fullres"]))
        scale_x, scale_y = 6, 6
        scaled_x_positions = [int(scale_x * x) for x in positions_df["pxl_row_in_fullres"]]
        scaled_y_positions = [int(scale_y * y) for y in positions_df["pxl_col_in_fullres"]]
        matched_x_posiitons = [(x - 0.5 * abs(x2 - max(scaled_x_positions))) for x in scaled_x_positions]
        matched_y_positions = [y for y in scaled_y_positions]
        positions_df["pxl_row_in_fullres"] = matched_x_posiitons
        positions_df["pxl_col_in_fullres"] = matched_y_positions
    elif project_dataset == "Tumor2":
        x1, y1, x2, y2 = 2730, 6270, 11660, 15910
        scale_x = (x2 - x1) / (max(positions_df["pxl_row_in_fullres"]) - min(positions_df["pxl_row_in_fullres"]))
        scale_y = (y2 - y1) / (max(positions_df["pxl_col_in_fullres"]) - min(positions_df["pxl_col_in_fullres"]))
        scaled_x_positions = [int(scale_x * x) for x in positions_df["pxl_row_in_fullres"]]
        scaled_y_positions = [int(scale_y * y) for y in positions_df["pxl_col_in_fullres"]]
        matched_x_posiitons = [(x - abs(x2 - max(scaled_x_positions))) for x in scaled_x_positions]
        matched_y_positions = [(y - abs(y2 - max(scaled_y_positions))) for y in scaled_y_positions]
        positions_df["pxl_row_in_fullres"] = matched_x_posiitons
        positions_df["pxl_col_in_fullres"] = matched_y_positions
    else:
        raise ValueError(f"Unknown project dataset {project_dataset} for Zenodo001 dataset.")
    assert set(spot_barcode_df.index) == set(positions_df.index), f"Unmatching between the spot barcodes in barcodes.tsv and tissue_positions_list.csv, and the difference is {set(spot_barcode_df.index) - set(positions_df.index)}."

    concat_obs_df = pd.concat([spot_barcode_df, positions_df], axis=1)
    concat_obs_df_copy = concat_obs_df.copy()
    concat_obs_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    concat_obs_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    concat_obs_df_copy.loc[:, "organ_type"] = "liver"
    # image_extension = os.path.splitext(raw_image_path)[1]
    image_extension = ".tif"
    concat_obs_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in concat_obs_df.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in concat_obs_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for Zenodo001 dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in concat_obs_df_copy.columns]}"
    concat_obs_df_copy = concat_obs_df_copy.reindex(columns=obs_columns_order)
    total_obs_df = concat_obs_df_copy
    gene_feature_df =  pd.read_csv(raw_features_path, header=None, delimiter='\t', names=["gene_barcode", "gene_name", "gene_expression"])
    gene_feature_df = gene_feature_df.set_index("gene_name")

    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (positions_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)

    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)
    
    # While the svs image is super large and slow to open, we only read it once.
    slide = openslide.open_slide(raw_image_path)
    image_width = slide.dimensions[0]
    image_height = slide.dimensions[1]
    num_levels = slide.level_count
    level = 0  # Level 0 represents highest resolution
    region = (0, 0, image_width, image_height)
    svs_image = slide.read_region(region, level, slide.level_dimensions[level])

    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        # image = Image.open(raw_image_path).convert("RGB")
        image = svs_image.convert("RGB")
        # image = Image.fromarray(image) # Optional?
        image = image.rotate(-90)
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        # image = Image.open(raw_image_path)
        image = svs_image.convert("RGB")
        # image = Image.fromarray(image) # Optional?
        image = image.rotate(-90)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = row["pxl_row_in_fullres"]
            y_pixel = row["pxl_col_in_fullres"]
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))

    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for Zenodo001 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")

def Zenodo002_dataset_list(
        main_data_storage: str,
        project_data_folder_name: str,
) -> List:
    dataset_name = ["Zenodo", "Zenodo002"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_datasets = os.listdir(project_data_folder_path)
    return project_datasets

def Zenodo002_single_helper(
        main_data_storage: str,
        project_data_folder_name: str,
        project_dataset: str,
) -> None:
    start_time = time.time()
    dataset_name = ["Zenodo", "Zenodo002"]
    project_data_folder_path = os.path.join(main_data_storage, project_data_folder_name, *dataset_name)
    project_dataset_raw_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_data")
    project_dataset_raw_data_content_list = os.listdir(project_dataset_raw_data_path)
    assert len(project_dataset_raw_data_content_list) == 3, f"Unmatching between the number of files and the expected number of files for Zenodo002 dataset {project_dataset}"
    project_dataset_raw_unzipped_data_path = os.path.join(project_data_folder_path, project_dataset, "raw_unzipped_data")
    project_dataset_raw_unzipped_data_content_list = os.listdir(project_dataset_raw_unzipped_data_path)
    assert len(project_dataset_raw_unzipped_data_content_list) == 1, f"Unmatching between the number of files and the expected number of files for Zenodo002 dataset {project_dataset}"
    project_dataset_preprocessed_data_path = os.path.join(project_data_folder_path, project_dataset, "preprocessed_data")
    if not os.path.exists(project_dataset_preprocessed_data_path):
        os.makedirs(project_dataset_preprocessed_data_path)

    image, positions, barcodes, features, matrix = False, False, False, False, False
    image_file = [content for content in project_dataset_raw_data_content_list if (".jpg" in content) or (".tif" in content) or (".tiff" in content) or (".btf" in content)]
    assert len(image_file) == 1, f"More than one image file for Zenodo002 dataset {project_dataset}"
    image_file = image_file[0]
    raw_image_path = os.path.join(project_dataset_raw_data_path, image_file)
    image = True

    positions_folder = [content for content in project_dataset_raw_data_content_list if ("spatial" in content)]
    assert len(positions_folder) == 1, f"More than one positions folder for Zenodo002 dataset {project_dataset}"
    positions_folder = positions_folder[0]
    raw_positions_folder_path = os.path.join(project_dataset_raw_data_path, positions_folder)
    raw_positions_folder_content_list = os.listdir(raw_positions_folder_path)
    positions_file = [content for content in raw_positions_folder_content_list if ("tissue_positions_list" in content) or ("tissue_positions" in content)]
    assert len(positions_file) == 1, f"More than one positions file for Zenodo002 dataset {project_dataset}"
    positions_file = positions_file[0]
    raw_positions_path = os.path.join(raw_positions_folder_path, positions_file)
    positions = True

    filtered_feature_bc_matrix_all_unzipped_folder = [content for content in project_dataset_raw_unzipped_data_content_list if "filtered_feature_bc_matrix_all_unzipped" in content]
    assert len(filtered_feature_bc_matrix_all_unzipped_folder) == 1, f"More than one filtered_feature_bc_matrix_all_unzipped folder for Zenodo002 dataset {project_dataset}"
    filtered_feature_bc_matrix_all_unzipped_folder = filtered_feature_bc_matrix_all_unzipped_folder[0]
    filtered_feature_bc_matrix_all_unzipped_folder_path = os.path.join(project_dataset_raw_unzipped_data_path, filtered_feature_bc_matrix_all_unzipped_folder)
    filtered_feature_bc_matrix_all_unzipped_folder_content_list = os.listdir(filtered_feature_bc_matrix_all_unzipped_folder_path)
    
    barcodes_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "barcodes.tsv" in content]
    assert len(barcodes_file) == 1, f"More than one barcodes file for Zenodo002 dataset {project_dataset}"
    barcodes_file = barcodes_file[0]
    raw_barcodes_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, barcodes_file)
    barcodes = True
    
    features_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "features.tsv" in content]
    assert len(features_file) == 1, f"More than one features file for Zenodo002 dataset {project_dataset}"
    features_file = features_file[0]
    raw_features_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, features_file)
    features = True
    
    matrix_file = [content for content in filtered_feature_bc_matrix_all_unzipped_folder_content_list if "matrix.mtx" in content]
    assert len(matrix_file) == 1, f"More than one matrix file for Zenodo002 dataset {project_dataset}"
    matrix_file = matrix_file[0]
    raw_matrix_path = os.path.join(filtered_feature_bc_matrix_all_unzipped_folder_path, matrix_file)
    matrix = True
    assert (image & positions & barcodes & features & matrix) == True, f"Missing filetype for Zenodo002 dataset {project_dataset}: {'image' if not image else ''} {'positions' if not positions else ''} {'barcodes' if not barcodes else ''} {'features' if not features else ''} {'matrix' if not matrix else ''} is missing."

    gene_counts_matrix = csr_matrix(mmread(raw_matrix_path).T)
    spot_barcode_df = pd.read_csv(raw_barcodes_path, header=None, names=["spot_barcode"])
    spot_barcode_df = spot_barcode_df.set_index("spot_barcode")
    positions_df = pd.read_csv(raw_positions_path, header=None, names=["spot_barcode", "in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"])
    positions_df = positions_df.set_index("spot_barcode")
    positions_df = positions_df[positions_df["in_tissue"].isin(["1", 1])]
    assert set(spot_barcode_df.index) == set(positions_df.index), f"Unmatching between the spot barcodes in barcodes.tsv and tissue_positions_list.csv, and the difference is {set(spot_barcode_df.index) - set(positions_df.index)}."
    concat_obs_df = pd.concat([spot_barcode_df, positions_df], axis=1)
    concat_obs_df_copy = concat_obs_df.copy()
    concat_obs_df_copy.loc[:, "dataset_name"] = os.path.join(*dataset_name)
    concat_obs_df_copy.loc[:, "patient_dir"] = os.path.join(*dataset_name, project_dataset)
    concat_obs_df_copy.loc[:, "organ_type"] = "heart"
    image_extension = os.path.splitext(raw_image_path)[1]
    concat_obs_df_copy.loc[:, "patch_name"] = [f"{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_name"] = [f"normed_{row_index + image_extension}" for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "patches", f"{row_index + image_extension}") for row_index in concat_obs_df.index]
    concat_obs_df_copy.loc[:, "normed_patch_path"] = [os.path.join(*dataset_name, project_dataset, "preprocessed_data", "normed_patches", f"normed_{row_index + image_extension}") for row_index in concat_obs_df.index]
    obs_columns_order = [
        'in_tissue',
        'array_row',
        'array_col',
        'dataset_name',
        'patient_dir',
        'organ_type',
        'patch_name',
        'normed_patch_name',
        'patch_path',
        'normed_patch_path',
        'pxl_row_in_fullres',
        'pxl_col_in_fullres',
    ]
    check_obs_columns_all_exists = all(column_name in concat_obs_df_copy.columns for column_name in obs_columns_order)
    assert check_obs_columns_all_exists, f"Missing columns in the concatenated obs dataframe for Zenodo002 dataset {project_dataset}: {[column_name for column_name in obs_columns_order if column_name not in concat_obs_df_copy.columns]}"
    concat_obs_df_copy = concat_obs_df_copy.reindex(columns=obs_columns_order)
    total_obs_df = concat_obs_df_copy
    gene_feature_df =  pd.read_csv(raw_features_path, header=None, delimiter='\t', names=["gene_barcode", "gene_name", "gene_expression"])
    gene_feature_df = gene_feature_df.set_index("gene_name")
    
    project_dataset_stdata = anndata.AnnData(gene_counts_matrix)
    project_dataset_stdata.obs = total_obs_df
    project_dataset_stdata.var = gene_feature_df
    project_dataset_stdata.obsm["spatial"] = (positions_df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values).astype(int)

    project_dataset_stdata_path = os.path.join(project_dataset_preprocessed_data_path, "stdata.h5ad")
    if not os.path.exists(project_dataset_stdata_path):
        project_dataset_stdata.write(project_dataset_stdata_path)

    project_dataset_coordinate_mapping_path = os.path.join(project_dataset_preprocessed_data_path, f"{project_dataset}_coordinate_mapping.jpg")
    if not os.path.exists(project_dataset_coordinate_mapping_path):
        image = Image.open(raw_image_path)
        draw = ImageDraw.Draw(image)
        dot_size = 224
        dot_color = (0, 255, 0) # green
        for x, y in project_dataset_stdata.obsm["spatial"]:
            draw.ellipse((x - dot_size/2, y - dot_size/2, x + dot_size/2, y + dot_size/2), fill=dot_color)
        image.save(project_dataset_coordinate_mapping_path)
    
    project_dataset_patches_folder_path = os.path.join(project_dataset_preprocessed_data_path, "patches")
    if not os.path.exists(project_dataset_patches_folder_path):
        os.makedirs(project_dataset_patches_folder_path)
        image_extension = os.path.splitext(raw_image_path)[1]
        image = Image.open(raw_image_path)
        patch_size = 224
        for barcode, row in (project_dataset_stdata.obs).iterrows():
            x_pixel = int(row["pxl_row_in_fullres"])
            y_pixel = int(row["pxl_col_in_fullres"])
            left = x_pixel - patch_size // 2
            right = x_pixel + patch_size // 2
            top = y_pixel - patch_size // 2
            bottom = y_pixel + patch_size // 2
            if left < 0 or top < 0 or right > image.width or bottom > image.height:
                continue
            patch = image.crop((left, top, right, bottom))
            patch.save(os.path.join(project_dataset_patches_folder_path, f"{barcode + image_extension}"))
    
    end_time = time.time()
    print(f"Coordinate and gene barcode mapping for Zenodo002 dataset {project_dataset} is finished. Time used: {running_time_display(end_time - start_time)}.")



























