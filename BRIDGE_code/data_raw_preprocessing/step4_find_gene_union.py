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
    parser.add_argument("--working_codespace", type=str, default="/home/zliang/BRIDGE_BIG_600K")
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
    
if __name__ == "__main__":
    main()
    
    
    
    









