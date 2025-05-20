## Simple Evaluation Run
To evaluate the survival performance of `BRIDGE` model and `bulk rna seq` quickly, run following command:

```
bash survival_evaluation.sh
```

### Experiment Logs and Results
After running the evaluation script:

1. **Experiment logs** are stored in the `wandb/` directory and can be viewed in your Weights & Biases dashboard. Each experiment is logged under the project name `Bridge-Benchmarking-{cancer_type}` with run names formatted as `{feature_extractor}-{timestamp}`.

2. **Model weights** are saved in the `model_weights/` directory, organized in folders named `{cancer_type}-{feature_extractor}-{timestamp}`.

3. **Performance metrics** such as C-index values are displayed in the terminal output and also logged to wandb for visualization.

You can modify the logging behavior using the following arguments:
- `--no_log true`: Disable logging to wandb
- `--wandb_dir`: Change the wandb logging directory
- `--model_weight_dir`: Change the model weights directory

---


## Evaluation Data
The survival evaluation is performed on multiple TCGA cancer datasets including:
- BRCA (Breast Cancer) - HER2+ and TNBC subtypes
- LUAD (Lung Adenocarcinoma)
- BLCA (Bladder Cancer)
- STAD (Stomach Adenocarcinoma)
- ESCA (Esophageal Carcinoma)

Our evaluation uses both predicted bulk RNA from the BRIDGE model and true bulk RNA sequencing data.

### Data Structure
The data files are organized as follows:

```
Part3_Survival/
├── dataset_csv/
│   └── Survival/
│       ├── BRCA_HER2Plus_Survival.csv
│       ├── BRCA_TNBC_Survival.csv
│       ├── LUAD_Survival.csv
│       ├── BLCA_Survival.csv
│       ├── STAD_Survival.csv
│       └── ESCA_Survival.csv
└── TCGA_processed_clinical_and_rna/
    ├── BRCA/
    │   ├── BRIDGE_pred_top1000_bulk_hvg.csv
    │   ├── data_mrna_seq_v2_rsem_log2p_hvg1000.csv
    │   └── clinical.csv
    ├── NSCLC/
    ├── BLCA/
    ├── STAD/
    └── ESCA/
```

#### Dataset CSV Files
Each dataset CSV file contains patient information with the following columns:
- `Patient_ID`: Unique identifier for each patient
- `Censor`: Binary indicator (1 for event observed, 0 for censored data)
- `Event_Time`: Time to event or censoring (in months)
- `Survival_Interval`: Binned survival interval (typically 0-3 representing quartiles)
- `Set`: Training/testing split indicator
- `Slide Feats File`: Path to the slide features file (for imaging modality)
- `Bulk_RNA_idx`: RNA identifier for matching with bulk RNA data

#### RNA Data Files
For each cancer type, two RNA data files are provided:
- `BRIDGE_pred_top1000_bulk_hvg.csv`: Predicted bulk RNA expression from the BRIDGE model for the top 1000 highly variable genes
- `data_mrna_seq_v2_rsem_log2p_hvg1000.csv`: True bulk RNA sequencing data (log2-transformed RSEM values) for the top 1000 highly variable genes

Both files are structured as matrices where rows represent patients (matching the `Bulk_RNA_idx` in the dataset CSV) and columns represent genes.

### Clinical Data
Clinical information for each patient is stored in `clinical.csv` files within each cancer type directory.

---

## On in-house Data
If you have your own in-house data to run the survival experiments, you need to prepare the following:

### Preprocessing Whole Slide Images
Before performing survival analysis with your own WSI data, you need to follow these preprocessing steps:

1. **Patch extraction**: Use `deepzoom_tiler.py` to extract patches from your whole slide images:
   ```
   python deepzoom_tiler.py --source /path/to/your/slides --dest /path/to/output/patches
   ```

2. **Spatial gene expression prediction**: Use `feature_extractor.py` to predict spatial gene expression for each patch and generate BRIDGE predicted bulk RNA sequence:
   ```
   python feature_extractor.py --slides_dir /path/to/output/patches --output_dir /path/to/predicted/rna
   ```

Note: Both `deepzoom_tiler.py` and `feature_extractor.py` are implemented based on the DS-MIL framework. For detailed usage and parameters, refer to the respective files.

### Survival Analysis Setup
After preprocessing your WSI data or if you already have RNA-seq data ready:

1. Create a dataset CSV file with the following columns:
   - `Patient_ID`: Unique identifier for each patient
   - `Censor`: Binary indicator (1 for event observed, 0 for censored data)
   - `Event_Time`: Time to event or censoring (in months)
   - `Survival_Interval`: Binned survival interval (typically 0-3 representing quartiles)
   - `Set`: Training/testing split indicator
   - `Slide Feats File`: Path to the slide features file (if using imaging modality)
   - `Bulk_RNA_idx`: RNA identifier for matching with bulk RNA data

2. Prepare your RNA data in CSV format with patients as rows and genes as columns.

3. Run the survival analysis with your data:
   ```
   python survival.py --modality rna --dataset_csv /path/to/your/dataset.csv \
                      --RNA_csv /path/to/your/rna_data.csv \
                      --epochs 200 --num_fold 10 --rna_model MLP \
                      --seed 0 --gpu_devices 1
   ```

### Command Arguments
- `--modality`: Data modality, either "rna" or "imaging"
- `--dataset_csv`: Path to your dataset CSV
- `--RNA_csv`: Path to your RNA CSV file (for RNA modality)
- `--features_folder`: Path to extracted slide features (for imaging modality)
- `--epochs`: Number of training epochs
- `--num_fold`: Number of Monte Carlo Cross-Validation folds
- `--rna_model`: Model to use for RNA data (MLP or SNN)
- `--mil_model`: Model to use for imaging data (ABMIL or MLPMIL)
- `--batch_size`: Training batch size
- `--seed`: Random seed for reproducibility

---

## Acknowledgments
This project was built on the top of amazing works, including [DS-MIL](https://github.com/binli123/dsmil-wsi), [TANGLE](https://github.com/Richarizardd/Self-Supervised-ViT-Path). We thank the authors for their great works.

---

## Contacts
If you encounter any problems with the code, have questions about implementation, or need clarification on the survival analysis procedure, please feel free to contact the author:

email: wqzhao98@connect.hku.hk