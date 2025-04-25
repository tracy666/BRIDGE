from multiprocessing import Pool
import argparse
from tqdm import tqdm
from functools import partial
import time

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='anndata')

from step0_preprocess_helper_functions import running_time_display, gene_barcode_name_mapping
from step0_2_coordinate_barcode_mapping_and_patch_cutting_helper_functions import (
    locations_lung_non_FFPE_dataset_list, locations_lung_non_FFPE_single_helper,
    locations_lung_FFPE_dataset_list, locations_lung_FFPE_single_helper,
    Genomics_dataset_list, Genomics_single_helper,
    BLEEP_dataset_list, BLEEP_single_helper,
    DLPFC_dataset_list, DLPFC_single_helper,
    DRYAD001_dataset_list, DRYAD001_single_helper,
    HER2ST_dataset_list, HER2ST_single_helper,
    HCA001_dataset_list, HCA001_single_helper,
    Mendeley_data001_Patient_1_1k_dataset_list, Mendeley_data001_Patient_1_1k_single_helper,
    Mendeley_data001_Patient_1_Visium_dataset_list, Mendeley_data001_Patient_1_Visium_single_helper,
    Mendeley_data001_Patient_2_dataset_list, Mendeley_data001_Patient_2_single_helper,
    Mendeley_data002_dataset_list, Mendeley_data002_single_helper,
    Mendeley_data003_dataset_list, Mendeley_data003_single_helper,
    NCBI001_dataset_list, NCBI001_single_helper,
    NCBI002_dataset_list, NCBI002_single_helper,
    NCBI003_dataset_list, NCBI003_single_helper,
    NCBI004_dataset_list, NCBI004_single_helper,
    NCBI005_dataset_list, NCBI005_single_helper,
    NCBI007_dataset_list, NCBI007_single_helper,
    NCBI008_dataset_list, NCBI008_single_helper,
    NCBI009_dataset_list, NCBI009_single_helper,
    STNet_dataset_list, STNet_single_helper,
    Zenodo001_dataset_list, Zenodo001_single_helper,
    Zenodo002_dataset_list, Zenodo002_single_helper,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_data_storage", type=str, default="/data1/zliang")
    parser.add_argument("--project_data_folder_name", type=str, default="BIG_600K")
    parser.add_argument("--multiprocessing_pool_num", type=int, default=5)
    args = parser.parse_args()
    
    gene_barcode_name_mapping_dict = gene_barcode_name_mapping(
        main_data_storage=args.main_data_storage,
        project_data_folder_name=args.project_data_folder_name,
    )
    
    pool_parameters = [
        (
            "5 locations lung non-FFPE",
            len(locations_lung_non_FFPE_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(locations_lung_non_FFPE_single_helper, args.main_data_storage, args.project_data_folder_name),
            locations_lung_non_FFPE_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "5 locations lung FFPE",
            len(locations_lung_FFPE_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(locations_lung_FFPE_single_helper, args.main_data_storage, args.project_data_folder_name),
            locations_lung_FFPE_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "10xGenomics",
            len(Genomics_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(Genomics_single_helper, args.main_data_storage, args.project_data_folder_name),
            Genomics_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "BLEEP",
            len(BLEEP_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(BLEEP_single_helper, args.main_data_storage, args.project_data_folder_name),
            BLEEP_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "downstream_task_data/DLPFC",
            len(DLPFC_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(DLPFC_single_helper, args.main_data_storage, args.project_data_folder_name),
            DLPFC_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "DRYAD001/DRYAD001",
            len(DRYAD001_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(DRYAD001_single_helper, args.main_data_storage, args.project_data_folder_name),
            DRYAD001_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "HER2ST",
            len(HER2ST_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(HER2ST_single_helper, args.main_data_storage, args.project_data_folder_name),
            HER2ST_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "Human_cell_atlas/Human_cell_atlas001",
            len(HCA001_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(HCA001_single_helper, args.main_data_storage, args.project_data_folder_name),
            HCA001_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "Mendeley_data001/Patient_1_1k",
            len(Mendeley_data001_Patient_1_1k_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(Mendeley_data001_Patient_1_1k_single_helper, args.main_data_storage, args.project_data_folder_name, gene_barcode_name_mapping_dict),
            Mendeley_data001_Patient_1_1k_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "Mendeley_data001/Patient_1_Visium",
            len(Mendeley_data001_Patient_1_Visium_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(Mendeley_data001_Patient_1_Visium_single_helper, args.main_data_storage, args.project_data_folder_name),
            Mendeley_data001_Patient_1_Visium_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "Mendeley_data001/Patient_2",
            len(Mendeley_data001_Patient_2_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(Mendeley_data001_Patient_2_single_helper, args.main_data_storage, args.project_data_folder_name),
            Mendeley_data001_Patient_2_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "Mendeley_data002",
            len(Mendeley_data002_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(Mendeley_data002_single_helper, args.main_data_storage, args.project_data_folder_name),
            Mendeley_data002_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "Mendeley_data003",
            len(Mendeley_data003_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(Mendeley_data003_single_helper, args.main_data_storage, args.project_data_folder_name),
            Mendeley_data003_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "NCBI/NCBI001",
            len(NCBI001_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(NCBI001_single_helper, args.main_data_storage, args.project_data_folder_name),
            NCBI001_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "NCBI/NCBI002",
            len(NCBI002_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(NCBI002_single_helper, args.main_data_storage, args.project_data_folder_name),
            NCBI002_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "NCBI/NCBI003",
            len(NCBI003_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(NCBI003_single_helper, args.main_data_storage, args.project_data_folder_name),
            NCBI003_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "NCBI/NCBI004",
            len(NCBI004_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(NCBI004_single_helper, args.main_data_storage, args.project_data_folder_name),
            NCBI004_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "NCBI/NCBI005",
            len(NCBI005_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(NCBI005_single_helper, args.main_data_storage, args.project_data_folder_name),
            NCBI005_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "NCBI/NCBI007",
            len(NCBI007_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(NCBI007_single_helper, args.main_data_storage, args.project_data_folder_name),
            NCBI007_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "NCBI/NCBI008",
            len(NCBI008_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(NCBI008_single_helper, args.main_data_storage, args.project_data_folder_name),
            NCBI008_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "NCBI/NCBI009",
            len(NCBI009_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(NCBI009_single_helper, args.main_data_storage, args.project_data_folder_name),
            NCBI009_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "STNet",
            len(STNet_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(STNet_single_helper, args.main_data_storage, args.project_data_folder_name, gene_barcode_name_mapping_dict),
            STNet_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "Zenodo/Zenodo001",
            len(Zenodo001_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(Zenodo001_single_helper, args.main_data_storage, args.project_data_folder_name),
            Zenodo001_dataset_list(args.main_data_storage, args.project_data_folder_name),
        ),
        (
            "Zenodo/Zenodo002",
            len(Zenodo002_dataset_list(args.main_data_storage, args.project_data_folder_name)),
            partial(Zenodo002_single_helper, args.main_data_storage, args.project_data_folder_name),
            Zenodo002_dataset_list(args.main_data_storage, args.project_data_folder_name),
        )
    ]
    
    for dataset_name, dataset_length, single_helper_function, project_list in pool_parameters:
        start_time = time.time()
        with Pool(args.multiprocessing_pool_num) as p:
            list(tqdm(p.imap_unordered(single_helper_function, project_list), total=dataset_length))
        end_time = time.time()
        print(f"Finished processing {dataset_name} in {running_time_display(end_time - start_time)}.")

if __name__ == "__main__":
    main()