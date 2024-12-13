from glob import glob
import os


datasets = [
    "BLCA",
    "BRCA",
    "ESCA",
    "LUAD",
    "STAD",
]

for dataset in datasets:

    print(f"Processing {dataset} dataset")

    # After running step1_feature_extractor.py, we have the patch-level gene prediction files stored in the following path
    gene_pred_path_list = glob(f"/disk2/wqzhao/TCGA/TCGA_processed_data/{dataset}_BRIDGE_Genepred/*.pt")

    import torch
    from tqdm import tqdm

    bulk_list = []
    slide_list = []

    for slide_pt in tqdm(gene_pred_path_list):
        slide_id = slide_pt.split("/")[-1][:13]+"01"
        slide_expression = torch.load(slide_pt)
        slide_bulk_pred = slide_expression.mean(dim=0).numpy()
        bulk_list.append(slide_bulk_pred)
        slide_list.append(slide_id)
        
    import numpy as np

    bulk_matrix = np.stack(bulk_list)

    import pandas as pd

    gene_names = pd.read_csv("/home/zliang/BRIDGE_code/downstream_tasks/Part4_Survival_Analysis/core/top_HVG_genes_indices_number_1000.csv")

    import pandas as pd

    bulk_df = pd.DataFrame(data=bulk_matrix, index=slide_list, columns=gene_names["gene_names"].to_list())
    bulk_df_mean = bulk_df.groupby(bulk_df.index).mean()
    bulk_df_mean.to_csv(f"/disk2/wqzhao/TCGA/TCGA_processed_clinical_and_rna/{dataset}/top1000_bulk_pred.csv")