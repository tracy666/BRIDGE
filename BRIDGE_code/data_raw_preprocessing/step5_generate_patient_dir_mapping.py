import argparse
import pandas as pd
import os
from datetime import datetime
import random

from step0_preprocess_helper_functions import find_all_stdata_h5ad, Genomics_project_dataset_to_organ
from step0_2_coordinate_barcode_mapping_and_patch_cutting_helper_functions import (
    locations_lung_non_FFPE_dataset_list,
    locations_lung_FFPE_dataset_list,
    Genomics_dataset_list,
    BLEEP_dataset_list,
    DLPFC_dataset_list,
    DRYAD001_dataset_list,
    HER2ST_dataset_list,
    HCA001_dataset_list,
    Mendeley_data001_Patient_1_1k_dataset_list,
    Mendeley_data001_Patient_1_Visium_dataset_list,
    Mendeley_data001_Patient_2_dataset_list,
    Mendeley_data002_dataset_list,
    Mendeley_data003_dataset_list,
    NCBI001_dataset_list,
    NCBI002_dataset_list,
    NCBI003_dataset_list,
    NCBI004_dataset_list,
    NCBI005_dataset_list,
    NCBI007_dataset_list,
    NCBI008_dataset_list,
    NCBI009_dataset_list,
    STNet_dataset_list,
    Zenodo001_dataset_list,
    Zenodo002_dataset_list,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_data_storage", type=str, default="/data1/zliang")
    parser.add_argument("--project_data_folder_name", type=str, default="BIG_600K")
    parser.add_argument("--working_codespace", type=str, default="/home/zliang/BRIDGE/BRIDGE_code")
    parser.add_argument("--generation_date", type=str, default="20240601")
    args = parser.parse_args()

    patient_dir_organ_mapping_dataframe = pd.DataFrame({
        "patient_dir": [],
        "organ_type": [],
    })

    # 5 locations lung
    dataset_name = ["5_locations_lung"]
    dataset_list = locations_lung_non_FFPE_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "lung",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)
    dataset_list = locations_lung_FFPE_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "lung",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)
    
    # 10xGenomics
    dataset_name = ["10xGenomics"]
    dataset_list = Genomics_dataset_list(args.main_data_storage, args.project_data_folder_name)
    Genomics_project_dataset_to_organ_mapping_dict = Genomics_project_dataset_to_organ()
    for dataset in dataset_list:
        dataset_number = dataset[len("10xGenomics"):]
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": Genomics_project_dataset_to_organ_mapping_dict[dataset_number],
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)
    
    # BLEEP
    dataset_name = ["BLEEP"]
    dataset_list = BLEEP_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "liver",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)  

    # downstream_task_data/DLPFC
    dataset_name = ["downstream_task_data", "DLPFC"]
    dataset_list = DLPFC_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "brain",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)          
            
    # DRYAD/DRYAD001
    dataset_name = ["DRYAD", "DRYAD001"]
    dataset_list = DRYAD001_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "brain",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)
        
    # HER2ST
    dataset_name = ["HER2ST"]
    dataset_list = HER2ST_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "breast",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)
    
    # Human_cell_atlas/Human_cell_atlas001
    dataset_name = ["Human_cell_atlas", "Human_cell_atlas001"]
    dataset_list = HCA001_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "lung",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)
    
    # Mendeley_data/Mendeley_data001
    dataset_name = ["Mendeley_data", "Mendeley_data001"]
    dataset_list = Mendeley_data001_Patient_1_1k_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "prostate",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)
    dataset_list = Mendeley_data001_Patient_1_Visium_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "prostate",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)
    dataset_list = Mendeley_data001_Patient_2_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "prostate",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)
    
    # Mendeley_data/Mendeley_data002
    dataset_name = ["Mendeley_data", "Mendeley_data002"]
    dataset_list = Mendeley_data002_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "small_and_large_intestine",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)

    # Mendeley_data/Mendeley_data003
    dataset_name = ["Mendeley_data", "Mendeley_data003"]
    dataset_list = Mendeley_data003_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "skin",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)

    # NCBI/NCBI001
    dataset_name = ["NCBI", "NCBI001"]
    dataset_list = NCBI001_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "nose",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)
    
    # NCBI/NCBI002
    dataset_name = ["NCBI", "NCBI002"]
    dataset_list = NCBI002_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "skin",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)
    
    # NCBI/NCBI003
    dataset_name = ["NCBI", "NCBI003"]
    dataset_list = NCBI003_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "kidney",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)

    # NCBI/NCBI004
    dataset_name = ["NCBI", "NCBI004"]
    dataset_list = NCBI004_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "brain",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)

    # NCBI/NCBI005
    dataset_name = ["NCBI", "NCBI005"]
    dataset_list = NCBI005_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        if dataset in ["GSM5621965", "GSM5621966", "GSM5621967"]:
            row = {
                "patient_dir": [os.path.join(*dataset_name, dataset)],
                "organ_type": "kidney",
            }
        elif dataset in ["GSM5621968", "GSM5621969", "GSM5621970", "GSM5621971"]:
            row = {
                "patient_dir": [os.path.join(*dataset_name, dataset)],
                "organ_type": "lung",
            }
        else:
            raise ValueError(f"{dataset} in NCBI/NCBI005 is not implemented with organ mapping.")
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)                
    
    # NCBI/NCBI007
    dataset_name = ["NCBI", "NCBI007"]
    dataset_list = NCBI007_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "small_and_large_intestine",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)

    # NCBI/NCBI008
    dataset_name = ["NCBI", "NCBI008"]
    dataset_list = NCBI008_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "liver",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)
    
    # NCBI/NCBI009
    dataset_name = ["NCBI", "NCBI009"]
    dataset_list = NCBI009_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "lung",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)

    # STNet
    dataset_name = ["STNet"]
    dataset_list = STNet_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "breast",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)
    
    # Zenodo/Zenodo001
    dataset_name = ["Zenodo", "Zenodo001"]
    dataset_list = Zenodo001_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "liver",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)
    
    # Zenodo/Zenodo002
    dataset_name = ["Zenodo", "Zenodo002"]
    dataset_list = Zenodo002_dataset_list(args.main_data_storage, args.project_data_folder_name)
    for dataset in dataset_list:
        row = {
            "patient_dir": [os.path.join(*dataset_name, dataset)],
            "organ_type": "heart",
        }
        patient_dir_organ_mapping_dataframe = pd.concat([patient_dir_organ_mapping_dataframe, pd.DataFrame(row)], ignore_index=True)

    generated_files_folder_path = os.path.join(args.working_codespace, "generated_files")
    if not os.path.exists(generated_files_folder_path):
        os.makedirs(generated_files_folder_path)
    current_time_folder_path = os.path.join(generated_files_folder_path, args.generation_date)
    if not os.path.exists(current_time_folder_path):
        os.makedirs(current_time_folder_path)
    patient_dir_organ_mapping_dataframe_saving_path = os.path.join(current_time_folder_path, "patient_dir_organ_mapping_df.csv")
    if not os.path.exists(patient_dir_organ_mapping_dataframe_saving_path):
        patient_dir_organ_mapping_dataframe.to_csv(patient_dir_organ_mapping_dataframe_saving_path, index=False)

if __name__ == "__main__":
    main()