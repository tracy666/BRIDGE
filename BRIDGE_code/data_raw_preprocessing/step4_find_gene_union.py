import argparse
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd
from functools import partial
import os
import pandas as pd
import itertools
from step0_preprocess_helper_functions import find_all_stdata_h5ad, get_genes_of_each_slide,\
    get_spot_number_of_each_slide, get_mean_std_of_intersection_genes_of_each_slide, calculate_weighted_sum_std_of_intersection_genes,\
    get_top_k_genes, get_top_k_HEG_genes_per_slide, find_all_stdata_h5ad_200K

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='anndata')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_data_storage", type=str, default="/data1/zliang")
    parser.add_argument("--project_data_folder_name", type=str, default="BIG_600K")
    parser.add_argument("--working_codespace", type=str, default="/home/zliang/BRIDGE/BRIDGE_code")
    parser.add_argument("--multiprocessing_pool_num", type=int, default=5)
    args = parser.parse_args()

    project_stdata_list = find_all_stdata_h5ad(
        main_data_storage=args.main_data_storage,
        project_data_folder_name=args.project_data_folder_name,
        stdata_file_names=["stdata.h5ad"],
        include_downstream_task_data=True,
    )
    print(f"Length of project_stdata_list: {len(project_stdata_list)}")
    
    # Create a multiprocessing pool
    pool = Pool(processes=args.multiprocessing_pool_num)
    # Apply the function to the list using multiprocessing
    nested_gene_names_list = list()
    with tqdm(total=len(project_stdata_list)) as pbar:
        for single_slide_gene_names_list in pool.imap(get_genes_of_each_slide, project_stdata_list):
            nested_gene_names_list.append(single_slide_gene_names_list)
            pbar.update()
    # Close the multiprocessing pool
    pool.close()
    pool.join()

    nested_gene_names_sets = [set(single_slide_gene_names_list) for single_slide_gene_names_list in nested_gene_names_list]
    intersection_genes_set = set.intersection(*nested_gene_names_sets)
    intersection_genes_list = sorted(list(intersection_genes_set))
    # Current intersection genes number: 7730 (2024/04/16)

    pool = Pool(processes=args.multiprocessing_pool_num)
    spot_number_dict = dict()
    with tqdm(total=len(project_stdata_list)) as pbar:
        for stdata_path, single_slide_spot_number in pool.imap(get_spot_number_of_each_slide, project_stdata_list):
            spot_number_dict[stdata_path] = single_slide_spot_number
            pbar.update()
    pool.close()
    pool.join()

    pool = Pool(processes=args.multiprocessing_pool_num)
    total_slide_mean_for_each_gene_dict = dict()
    total_slide_std_for_each_gene_dict = dict()
    with tqdm(total=len(project_stdata_list)) as pbar:
        for stdata_path, (single_slide_mean_for_each_gene_list, single_slide_std_for_each_gene_list) in pool.imap(partial(get_mean_std_of_intersection_genes_of_each_slide, intersection_genes_list=intersection_genes_list), project_stdata_list):
            total_slide_mean_for_each_gene_dict[stdata_path] = single_slide_mean_for_each_gene_list
            total_slide_std_for_each_gene_dict[stdata_path] = single_slide_std_for_each_gene_list
            pbar.update()
    pool.close()
    pool.join()

    gene_index_list = list(range(len(list(total_slide_mean_for_each_gene_dict.values())[0])))
    pool = Pool(processes=args.multiprocessing_pool_num)
    weighted_mean_for_each_gene_dict = dict()
    weighted_std_for_each_gene_dict = dict()
    with tqdm(total=len(gene_index_list)) as pbar:
        for index, (index_gene_avg_mean, index_gene_avg_std) in pool.imap(partial(calculate_weighted_sum_std_of_intersection_genes, spot_number_dict=spot_number_dict, total_slide_mean_for_each_gene_dict=total_slide_mean_for_each_gene_dict, total_slide_std_for_each_gene_dict=total_slide_std_for_each_gene_dict,), gene_index_list):
            weighted_mean_for_each_gene_dict[index] = index_gene_avg_mean
            weighted_std_for_each_gene_dict[index] = index_gene_avg_std
            pbar.update()
    pool.close()
    pool.join()

    gene_names_csv_folder = os.path.join(args.working_codespace, "data_raw_preprocessing", "selected_genes")
    if not os.path.exists(gene_names_csv_folder):
        os.makedirs(gene_names_csv_folder)
    
    # Setting 1: All the intersection genes will be incorporated: 7730 genes
    setting_1_df = pd.DataFrame({
        "gene_names": intersection_genes_list,
    })
    setting_1_df_saving_path = os.path.join(gene_names_csv_folder, f"all_intersection_genes_number_{len(intersection_genes_list)}.csv")
    if not os.path.exists(setting_1_df_saving_path):
        setting_1_df.to_csv(setting_1_df_saving_path, index=False)
    
    # Setting 2: Each slide weighted mean HEG (weighted by spot number) from intersection genes: self-determined number, 50/100/250
    for top_k_value in [25, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000]:
        top_k_indices_list, top_k_genes_list = get_top_k_genes(
            weighted_value_for_each_gene_dict=weighted_mean_for_each_gene_dict,
            top_k_genes=top_k_value,
            intersection_genes_list=intersection_genes_list,
        )
        setting_2_indices_df = pd.DataFrame({
            "gene_indices": top_k_indices_list,
        })
        setting_2_df = pd.DataFrame({
            "gene_names": top_k_genes_list,
        })
        setting_2_indices_df_saving_path = os.path.join(gene_names_csv_folder, f"top_HEG_genes_indices_number_{len(top_k_genes_list)}.csv")
        if not os.path.exists(setting_2_indices_df_saving_path):
            setting_2_indices_df.to_csv(setting_2_indices_df_saving_path, index=False)
        setting_2_df_saving_path = os.path.join(gene_names_csv_folder, f"top_HEG_genes_number_{len(top_k_genes_list)}.csv")
        if not os.path.exists(setting_2_df_saving_path):
            setting_2_df.to_csv(setting_2_df_saving_path, index=False)
    
    # Setting 3: Each slide weighted mean HVG (weighted by spot number) from intersection genes: self-determined number, 50/100/250
    for top_k_value in [25, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000]:
        top_k_indices_list, top_k_genes_list = get_top_k_genes(
            weighted_value_for_each_gene_dict=weighted_std_for_each_gene_dict,
            top_k_genes=top_k_value,
            intersection_genes_list=intersection_genes_list,
        )
        setting_3_indices_df = pd.DataFrame({
            "gene_indices": top_k_indices_list,
        })
        setting_3_df = pd.DataFrame({
            "gene_names": top_k_genes_list,
        })
        setting_3_indices_df_saving_path = os.path.join(gene_names_csv_folder, f"top_HVG_genes_indices_number_{len(top_k_genes_list)}.csv")
        if not os.path.exists(setting_3_indices_df_saving_path):
            setting_3_indices_df.to_csv(setting_3_indices_df_saving_path, index=False)
        setting_3_df_saving_path = os.path.join(gene_names_csv_folder, f"top_HVG_genes_number_{len(top_k_genes_list)}.csv")
        if not os.path.exists(setting_3_df_saving_path):
            setting_3_df.to_csv(setting_3_df_saving_path, index=False)
    
    # Setting 4: Selected biomarkers
    biomarker_gene_list = [
        "ANAPC11", # Urothelial bladder cancer
        "ARF1", # Gastric cancer
        "ARPC3", # Hepatocellular carcinoma
        "ATF4", # Breast cancer
        "ATOX1", # Breast cancer
        "ATP1A1", # Gastric cancer, Melanoma, Renal cell carcinoma
        "ATP6V0B", # Glioma
        "C1orf43", # Lung adenocarcinoma
        "CANX", # Breast cancer, Colorectal cancer
        "CCT3", # Breast cancer, Cervical cancer, Hepatocellular carcinoma, Lung adenocarcinoma
        "CHCHD2", # Parkinson’s disease
        "CLIC1", # Epithelial ovarian cancer
        "CNBP", # Neuroblastoma
        "COX6A1", # Charcot-Marie-Tooth Disease, Oral leukoplakia
        "COX6B1", # Cytochrome c oxidase deficiency
        "COX7A2", # Glioma
        "COX7B", # Esophageal carcinoma
        "COX7C", # Alzheimer’s disease
        "COX8A", # Leigh-like syndrome and epilepsy
        "CSNK2B", # Colorectal cancer, Neurodevelopmental disability and epilepsy
        "CSTB", # Cervical cancer
        "DBI", # Alzheimer’s disease
        "DDT", # Melanoma
        "ECH1", # Alzheimer’s disease
        "EEF1B2", # Alzheimer’s disease
        "EIF3H", # Esophageal squamous cell carcinoma, Hepatocellular carcinoma
        "GPX4", # Diffuse large B lymphoma, Ischemic stroke, Sepsis
        "GUK1", # Sepsis
        "HDGF", # Cholangiocarcinoma, Gallbladder cancer, Gastric cancer, Gastrointestinal stromal tumors, Hepatocellular carcinoma, Lung cancer, Pancreatic cancer
        "HINT1", # Breast cancer, Hepatocellular carcinoma
        "HNRNPA2B1", # Breast cancer, Gastric adenocarcinoma, Lung cancer, Melanoma
        "HNRNPC", # Lung adenocarcinoma, Papillary renal cell carcinoma
        "HNRNPF", # Bladder cancer
        "HNRNPU", # Bladder cancer, Gastric cancer, Hepatocellular carcinoma
        "HSPD1", # Pituitary adenomas
        "HSPE1", # Lung adenocarcinoma, Prostate cancer
        "ILF2", # Breast cancer, Gastric cancer, Multiple myeloma
        "JTB", # Breast cancer
        "KDELR2", # Bladder urothelial carcinoma, Glioma
        "LAMTOR4", # Prostate cancer
        "LSM7", # Breast cancer, Lung adenocarcinoma
        "MDH2", # Lung cancer, Pediatric epileptic encephalopathy, Pheochromocytoma and paraganglioma
        "MLF2", # Dilated cardiomyopathy
        "NDUFA1", # Alzheimer’s disease
        "NDUFAB1", # Breast cancer
        "NDUFB10", # Isolated complex I deficiency
        "NDUFB11", # Mitochondrial complex I deficiency
        "NDUFB2", # Glioblastoma, Sepsis
        "NDUFB7", # Gastric cancer
        "NDUFB9", # Breast cancer, Medulloblastoma, Uveal melanoma
        "NDUFC1", # Gastric cancer, Hepatocellular carcinoma
        "NDUFS6", # Mitochondrial complex I deficiency
        "NDUFV2", # Bipolar disorder and schizophrenia, Progressive cavitating leukoencephalopathy
        "PDCD6", # Colorectal cancer, Gastric cancer
        "PDIA6", # Lung cancer, Pancreatic cancer
        "PHB2", # Blood cancer, Breast cancer, Colorectal cancer, Lung cancer
        "PRDX5", # Breast cancer, Lung cancer
        "PSMA7", # Cervical cancer, Colorectal cancer, Gastric cancer
        "PSMB1", # Breast cancer, Colorectal cancer
        "PSMB4", # Breast cancer
        "PSMB5", # Breast cancer, Diabetic retinopathy
        "PSME1", # Gastric cancer, Soft tissue leiomyosarcomas
        "RAC1", # Breast cancer, Lung adenocarcinoma, Melanoma
        "ROMO1", # Cervical cancer, Colorectal cancer, Gastric cancer, Endometrial and ovarian cancers, Gliomas, Hepatic tumor, Lung cancer
        "SEC61G", # Breast cancer, Head and neck squamous cell carcinoma, Kidney cancer, Lung adenocarcinoma, Oral Squamous Cell Carcinoma
        "SERBP1", # Breast cancer, Glioblastoma
        "SERP1", # Pancreatic ductal adenocarcinoma, Skin cutaneous melanoma
        "SLIRP", # Mitochondrial complex I and IV deficiency
        "SRSF2", # Colorectal carcinoma, Myeloproliferative neoplasm
        "TALDO1", # Hepatocellular carcinoma
        "TIMM13", # Osteosarcoma, Skin cutaneous melanoma
        "TMED9", # Breast cancer, Ovarian Cancer
        "TMEM59", # Colorectal cancer
        "TOMM7", # Growth retardation
        "TRMT112", # Pancreatic cancer
        "TUFM", # Colorectal carcinoma, Lung cancer
        "TXNL4A", # Burn-McKeown syndrome, Hepatocellular carcinoma
        "UQCRQ", # Gastirc cancer, Kidney transplant rejection
        "VDAC1", # Breast cancer, Cervical cancer, Lung cancer
        "WDR83OS", # Pediatric cholestatic liver disease
    ]
    for biomarker in biomarker_gene_list:
        assert biomarker in intersection_genes_list, f"Biomarker gene {biomarker} not found in intersection genes list"
    setting_4_df = pd.DataFrame({
        "gene_names": biomarker_gene_list,
    })
    setting_4_df_saving_path = os.path.join(gene_names_csv_folder, f"selected_biomarkers_number_{len(biomarker_gene_list)}.csv")
    if not os.path.exists(setting_4_df_saving_path):
        setting_4_df.to_csv(setting_4_df_saving_path, index=False)
    
if __name__ == "__main__":
    main()
