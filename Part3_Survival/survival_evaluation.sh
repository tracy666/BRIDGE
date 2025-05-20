# Evaluation of BRIDGE-predicted bulk RNA
python survival.py --modality rna --dataset_csv ./dataset_csv/Survival/BRCA_HER2Plus_Survival.csv --RNA_csv ./TCGA_processed_clinical_and_rna/BRCA/BRIDGE_pred_top1000_bulk_hvg.csv --epochs 200 --num_fold 10 --rna_model MLP --seed 0 --gpu_devices 1 --wandb_dir ./wandb --model_weight_dir ./model_weights

python survival.py --modality rna --dataset_csv ./dataset_csv/Survival/BRCA_TNBC_Survival.csv --RNA_csv ./TCGA_processed_clinical_and_rna/BRCA/BRIDGE_pred_top1000_bulk_hvg.csv --epochs 200 --num_fold 10 --rna_model MLP --seed 0 --gpu_devices 1 --wandb_dir ./wandb --model_weight_dir ./model_weights

python survival.py --modality rna --dataset_csv ./dataset_csv/Survival/LUAD_Survival.csv --RNA_csv ./TCGA_processed_clinical_and_rna/NSCLC/BRIDGE_pred_top1000_bulk_hvg.csv --epochs 200 --num_fold 10 --rna_model MLP --seed 0 --gpu_devices 1 --wandb_dir ./wandb --model_weight_dir ./model_weights

python survival.py --modality rna --dataset_csv ./dataset_csv/Survival/BLCA_Survival.csv --RNA_csv ./TCGA_processed_clinical_and_rna/BLCA/BRIDGE_pred_top1000_bulk_hvg.csv --epochs 200 --num_fold 10 --rna_model MLP --seed 0 --gpu_devices 1 --wandb_dir ./wandb --model_weight_dir ./model_weights

python survival.py --modality rna --dataset_csv ./dataset_csv/Survival/STAD_Survival.csv --RNA_csv ./TCGA_processed_clinical_and_rna/STAD/BRIDGE_pred_top1000_bulk_hvg.csv --epochs 200 --num_fold 10 --rna_model MLP --seed 0 --gpu_devices 1 --wandb_dir ./wandb --model_weight_dir ./model_weights

python survival.py --modality rna --dataset_csv ./dataset_csv/Survival/ESCA_Survival.csv --RNA_csv ./TCGA_processed_clinical_and_rna/ESCA/BRIDGE_pred_top1000_bulk_hvg.csv --epochs 200 --num_fold 10 --rna_model MLP --seed 0 --gpu_devices 1 --wandb_dir ./wandb --model_weight_dir ./model_weights

# Evaluation of True bulk RNA
python survival.py --modality rna --dataset_csv ./dataset_csv/Survival/BRCA_HER2Plus_Survival.csv --RNA_csv ./TCGA_processed_clinical_and_rna/BRCA/data_mrna_seq_v2_rsem_log2p_hvg1000.csv --epochs 200 --num_fold 10 --rna_model MLP --seed 0 --gpu_devices 1 --wandb_dir ./wandb --model_weight_dir ./model_weights

python survival.py --modality rna --dataset_csv ./dataset_csv/Survival/BRCA_TNBC_Survival.csv --RNA_csv ./TCGA_processed_clinical_and_rna/BRCA/data_mrna_seq_v2_rsem_log2p_hvg1000.csv --epochs 200 --num_fold 10 --rna_model MLP --seed 0 --gpu_devices 1 --wandb_dir ./wandb --model_weight_dir ./model_weights

python survival.py --modality rna --dataset_csv ./dataset_csv/Survival/LUAD_Survival.csv --RNA_csv ./TCGA_processed_clinical_and_rna/NSCLC/data_mrna_seq_v2_rsem_log2p_hvg1000.csv --epochs 200 --num_fold 10 --rna_model MLP --seed 0 --gpu_devices 1 --wandb_dir ./wandb --model_weight_dir ./model_weights

python survival.py --modality rna --dataset_csv ./dataset_csv/Survival/BLCA_Survival.csv --RNA_csv ./TCGA_processed_clinical_and_rna/BLCA/data_mrna_seq_v2_rsem_log2p_hvg1000.csv --epochs 200 --num_fold 10 --rna_model MLP --seed 0 --gpu_devices 1 --wandb_dir ./wandb --model_weight_dir ./model_weights

python survival.py --modality rna --dataset_csv ./dataset_csv/Survival/STAD_Survival.csv --RNA_csv ./TCGA_processed_clinical_and_rna/STAD/data_mrna_seq_v2_rsem_log2p_hvg1000.csv --epochs 200 --num_fold 10 --rna_model MLP --seed 0 --gpu_devices 1 --wandb_dir ./wandb --model_weight_dir ./model_weights

python survival.py --modality rna --dataset_csv ./dataset_csv/Survival/ESCA_Survival.csv --RNA_csv ./TCGA_processed_clinical_and_rna/ESCA/data_mrna_seq_v2_rsem_log2p_hvg1000.csv --epochs 200 --num_fold 10 --rna_model MLP --seed 0 --gpu_devices 1 --wandb_dir ./wandb --model_weight_dir ./model_weights