# Here we take BLCA as an example

# BRIDGE bulk prediction
python step3_survival.py --modality rna --dataset_csv /home/zliang/BRIDGE_code/downstream_tasks/Part4_Survival_Analysis/dataset_csv/Survival/BLCA_Survival.csv --RNA_csv /disk2/wqzhao/TCGA/TCGA_processed_clinical_and_rna/BRCA/top1000_bulk_pred.csv --epochs 200 --num_fold 10 --rna_model MLP --seed 0 --gpu_devices 1
# BRIDGE patch-level prediction
python step3_survival.py --dataset_csv /home/zliang/BRIDGE_code/downstream_tasks/Part4_Survival_Analysis/dataset_csv/Survival/BLCA_Survival.csv --features_folder /disk2/wqzhao/TCGA/TCGA_processed_data/BRCA_BRIDGE_Genepred --epochs 200 --num_fold 10 --top_gene_pred 1000
# Bulk RNA
python step3_survival.py --dataset_csv /home/zliang/BRIDGE_code/downstream_tasks/Part4_Survival_Analysis/dataset_csv/Survival/BLCA_Survival.csv --RNA_csv /disk2/wqzhao/TCGA/TCGA_processed_clinical_and_rna/BLCA/data_mrna_seq_v2_rsem_log2p_hvg1000.csv --modality rna --rna_model MLP --epochs 200 --num_fold 10
# CONCH
python step3_survival.py --dataset_csv /home/zliang/BRIDGE_code/downstream_tasks/Part4_Survival_Analysis/dataset_csv/Survival/BLCA_Survival.csv --features_folder /disk2/wqzhao/TCGA/TCGA_processed_data/BLCA_CONCH --epochs 200 --num_fold 10
# CTransPath
python step3_survival.py --dataset_csv /home/zliang/BRIDGE_code/downstream_tasks/Part4_Survival_Analysis/dataset_csv/Survival/BLCA_Survival.csv --features_folder /disk2/wqzhao/TCGA/TCGA_processed_data/BLCA_CTransPath --epochs 200 --num_fold 10
# DINOv2
python step3_survival.py --dataset_csv /home/zliang/BRIDGE_code/downstream_tasks/Part4_Survival_Analysis/dataset_csv/Survival/BLCA_Survival.csv --features_folder /disk2/wqzhao/TCGA/TCGA_processed_data/BLCA_dinov2_vits14 --epochs 200 --num_fold 10
# HIPT
python step3_survival.py --dataset_csv /home/zliang/BRIDGE_code/downstream_tasks/Part4_Survival_Analysis/dataset_csv/Survival/BLCA_Survival.csv --features_folder /disk2/wqzhao/TCGA/TCGA_processed_data/BLCA_HIPT --epochs 200 --num_fold 10
# PLIP
python step3_survival.py --dataset_csv /home/zliang/BRIDGE_code/downstream_tasks/Part4_Survival_Analysis/dataset_csv/Survival/BLCA_Survival.csv --features_folder /disk2/wqzhao/TCGA/TCGA_processed_data/BLCA_PLIP --epochs 200 --num_fold 10
# Quilt1M
python step3_survival.py --dataset_csv /home/zliang/BRIDGE_code/downstream_tasks/Part4_Survival_Analysis/dataset_csv/Survival/BLCA_Survival.csv --features_folder /disk2/wqzhao/TCGA/TCGA_processed_data/BLCA_Quilt1M --epochs 200 --num_fold 10
# UNI
python step3_survival.py --dataset_csv /home/zliang/BRIDGE_code/downstream_tasks/Part4_Survival_Analysis/dataset_csv/Survival/BLCA_Survival.csv --features_folder /disk2/wqzhao/TCGA/TCGA_processed_data/BLCA_UNI --epochs 200 --num_fold 10

# Again, you can replace BLCA with other TCGA datasets to get the results
# dataset_csv should be replaced by the path of the dataset csv file
# features_folder should be replaced by the path of the extracted features in step1_feature_extractor.py