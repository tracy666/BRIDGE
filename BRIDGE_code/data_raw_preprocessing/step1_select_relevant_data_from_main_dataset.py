from multiprocessing import Pool
import argparse
from tqdm import tqdm
from functools import partial
import time
from step0_preprocess_helper_functions import running_time_display

from step0_1_select_relevant_data_from_main_dataset_helper_functions import (
    Genomics_project_list, Genomics_select_data_single_helper,
    DLPFC_project_list, DLPFC_select_data_single_helper,
    TCGA_BRCA_project_list, TCGA_BRCA_select_data_single_helper,
    DRYAD001_project_list, DRYAD001_select_data_single_helper,
    HCA001_project_list, HCA001_select_data_single_helper,
    Mendeley_data001_Patient_1_1k_project_list, Mendeley_data001_Patient_1_1k_single_helper,
    Mendeley_data001_Patient_1_Visium_project_list, Mendeley_data001_Patient_1_Visium_single_helper,
    Mendeley_data001_Patient_2_project_list, Mendeley_data001_Patient_2_single_helper,
    Mendeley_data002_project_list, Mendeley_data002_single_helper,
    Mendeley_data003_project_list, Mendeley_data003_single_helper,
    NCBI001_project_list, NCBI001_single_helper,
    NCBI002_project_list, NCBI002_single_helper,
    NCBI003_project_list, NCBI003_single_helper,
    NCBI004_project_list, NCBI004_single_helper,
    NCBI005_project_list, NCBI005_single_helper,
    NCBI007_project_list, NCBI007_single_helper,
    NCBI008_project_list, NCBI008_single_helper,
    NCBI009_project_list, NCBI009_single_helper,
    Specific_datasets_5_locations_lung_non_FFPE_project_list, Specific_datasets_5_locations_lung_non_FFPE_single_helper,
    Specific_datasets_5_locations_lung_FFPE_project_list, Specific_datasets_5_locations_lung_FFPE_single_helper,
    Specific_datasets_BLEEP_project_list, Specific_datasets_BLEEP_single_helper,
    Specific_datasets_HER2ST_project_list, Specific_datasets_HER2ST_single_helper,
    Specific_datasets_STNet_project_list, Specific_datasets_STNet_single_helper,
    Zenodo001_project_list, Zenodo001_single_helper,
    Zenodo002_project_list, Zenodo002_single_helper,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_data_storage", type=str, default="/data1/zliang")
    parser.add_argument("--raw_data_folder_name", type=str, default="Histo_ST_raw")
    parser.add_argument("--project_data_folder_name", type=str, default="BIG_600K")
    parser.add_argument("--multiprocessing_pool_num", type=int, default=5)
    args = parser.parse_args()

    pool_parameters = [
        (
            "10xGenomics",
            len(Genomics_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(Genomics_select_data_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            Genomics_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)
        ),
        (
            "DLPFC",
            len(DLPFC_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(DLPFC_select_data_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            DLPFC_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "TCGA_BRCA",
            len(TCGA_BRCA_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(TCGA_BRCA_select_data_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            TCGA_BRCA_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "DRYAD001",
            len(DRYAD001_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(DRYAD001_select_data_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            DRYAD001_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "HCA001",
            len(HCA001_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(HCA001_select_data_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            HCA001_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "Mendeley_data001_Patient_1_1k",
            len(Mendeley_data001_Patient_1_1k_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(Mendeley_data001_Patient_1_1k_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            Mendeley_data001_Patient_1_1k_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "Mendeley_data001_Patient_1_Visium",
            len(Mendeley_data001_Patient_1_Visium_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(Mendeley_data001_Patient_1_Visium_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            Mendeley_data001_Patient_1_Visium_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "Mendeley_data001_Patient_2",
            len(Mendeley_data001_Patient_2_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(Mendeley_data001_Patient_2_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            Mendeley_data001_Patient_2_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "Mendeley_data002",
            len(Mendeley_data002_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(Mendeley_data002_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            Mendeley_data002_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "Mendeley_data003",
            len(Mendeley_data003_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(Mendeley_data003_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            Mendeley_data003_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "NCBI001",
            len(NCBI001_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(NCBI001_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            NCBI001_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "NCBI002",
            len(NCBI002_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(NCBI002_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            NCBI002_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "NCBI003",
            len(NCBI003_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(NCBI003_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            NCBI003_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "NCBI004",
            len(NCBI004_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(NCBI004_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            NCBI004_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "NCBI005",
            len(NCBI005_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(NCBI005_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            NCBI005_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "NCBI007",
            len(NCBI007_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(NCBI007_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            NCBI007_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "NCBI008",
            len(NCBI008_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(NCBI008_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            NCBI008_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "NCBI009",
            len(NCBI009_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(NCBI009_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            NCBI009_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "Specific_datasets_5_locations_lung_non_FFPE",
            len(Specific_datasets_5_locations_lung_non_FFPE_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(Specific_datasets_5_locations_lung_non_FFPE_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            Specific_datasets_5_locations_lung_non_FFPE_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "Specific_datasets_5_locations_lung_FFPE",
            len(Specific_datasets_5_locations_lung_FFPE_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(Specific_datasets_5_locations_lung_FFPE_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            Specific_datasets_5_locations_lung_FFPE_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "Specific_datasets_BLEEP",
            len(Specific_datasets_BLEEP_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(Specific_datasets_BLEEP_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            Specific_datasets_BLEEP_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "Specific_datasets_HER2ST",
            len(Specific_datasets_HER2ST_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(Specific_datasets_HER2ST_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            Specific_datasets_HER2ST_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "Specific_datasets_STNet",
            len(Specific_datasets_STNet_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(Specific_datasets_STNet_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            Specific_datasets_STNet_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "Zenodo001",
            len(Zenodo001_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(Zenodo001_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            Zenodo001_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
        (
            "Zenodo002",
            len(Zenodo002_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name)),
            partial(Zenodo002_single_helper, args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
            Zenodo002_project_list(args.main_data_storage, args.raw_data_folder_name, args.project_data_folder_name),
        ),
    ]

    for dataset_name, dataset_length, single_helper_function, project_list in pool_parameters:
        start_time = time.time()
        with Pool(args.multiprocessing_pool_num) as p:
            list(tqdm(p.imap_unordered(single_helper_function, project_list), total=dataset_length))
        end_time = time.time()
        print(f"Finished processing {dataset_name} in {running_time_display(end_time - start_time)}.")

if __name__ == "__main__":
    main()