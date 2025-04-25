# BRIDGE
Official code repository for "BRIDGE: A Multi-organ Histo-ST Foundation Model for Virtual Spatial Transcriptomics to Enhance Few-shot Cancer Diagnosis"

### Our motivation and curated dataset BIG-600K summary

<div align="center">
  <img width="100%" alt="Description of your image" src="https://github.com/tracy666/BRIDGE/blob/06c85e7ea8cce07f6343eb3dfc66745449936983/BRIDGE_figure/motivation.png">
  <p><strong>Figure 1:</strong> Rationale for integrating histological imaging with spatial transcriptomics (ST) and overview of the BIG-600K dataset.</p>
</div>

### The framework of BRIDGE

<div align="center">
  <img width="100%" alt="Description of your image" src="https://github.com/tracy666/BRIDGE/blob/68c13cd61c26cc93ad905670d8f31d366f9de241/BRIDGE_figure/framework.png">
  <p><strong>Figure 2:</strong> Overview of the BRIDGE framework.</p>
</div>

## Environment Installation

To clone all files:
```bash
git clone https://github.com/tracy666/BRIDGE.git
```

To install Python dependencies:
```bash
conda install --file requirements.txt
```

## Downloading BIG-600K and data utilized in this study
We uploaded all our self-curated BIG-600K preprocessed data to Mendeley Data. Due to the space limitation of the platform, we separate the dataset into several parts and zipped the files. After downloading all the zipped folders, you could use the code to unzip them and organize them in the following directories:
```python
import zipfile

zip_data_file_path = "" # The path leading to the zip file
unzipped_folder_path = "" # The path you want to save your unzipped folder

with zipfile.ZipFile(zip_data_file_path, 'r') as zip_ref:
    zip_ref.extractall(unzipped_folder_path)
```
<details>
<summary>
Example Data Directory
</summary>

```bash
ROOT_DATA_DIR/
    └──BIG_600K/
        └── 5_locations_lung/
            └── WSA_LngSP8759311/
                └── preprocessed_data/
                    └── normed_patches/
                    └── patches/
                    └── stdata.h5ad
                    └── coordinate_mapping.jpg
            └── WSA_LngSP8759312/
            └── .../
        └── 10xGenomics/
            └── 10xGenomics001/
                └── preprocessed_data/
                    ├── ...
            └── 10xGenomics002/
            └── .../
        └── BLEEP/
            └── GSM7697868/
                └── preprocessed_data/
                    ├── ...
            └── GSM7697869/
            └── .../
        └── DRYAD/
            └── DRYAD001/
                └── #UKF242_T_ST/
                    └── preprocessed_data/
                        ├── ...
                └── #UKF243_T_ST/
                └── .../
        └── HER2ST/
            └── A1/
                └── preprocessed_data/
                    ├── ...
            └── A2/
            └── .../
        └── Human_cell_atlas/
            └── Human_cell_atlas001/
                └── 6332STDY9479166/
                    └── preprocessed_data/
                        ├── ...
                └── 6332STDY9479167/
                └── .../
        └── Mendeley_data/
            └── Mendeley_data001/
                └── Patient_1_1k_array_H1_1/
                    └── preprocessed_data/
                        ├── ...
                └── Patient_1_1k_array_H1_2/
                └── .../
            └── Mendeley_data002/
                └── V19T26-028_A1/
                    └── preprocessed_data/
                        ├── ...
                └── V19T26-028_B1/
                └── .../
            └── Mendeley_data003/
                └── V10F24-015_A1/
                    └── preprocessed_data/
                        ├── ...
                └── V10F24-015_B1/
                └── .../
        └── NCBI/
            └── NCBI001/
                └── AH4199551/
                    └── preprocessed_data/
                        ├── ...
                └── AJ3037946/
            └── NCBI002/
                └── GSM4284316/
                    └── preprocessed_data/
                        ├── ...
                └── GSM4284317/
                └── .../
            └── NCBI003/
                └── A1/
                    └── preprocessed_data/
                        ├── ...
            └── NCBI004/
                └── pt15/
                    └── preprocessed_data/
                        ├── ...
                └── pt16/
                └── .../
            └── NCBI005/
                └── GSM5621965/
                    └── preprocessed_data/
                        ├── ...
                └── GSM5621966/
                └── .../
            └── NCBI007/
                └── A1/
                    └── preprocessed_data/
                        ├── ...
                └── A2/
                └── .../
            └── NCBI008/
                └── JBO014/
                    └── preprocessed_data/
                        ├── ...
                └── JBO015/
                └── .../
            └── NCBI009/
                └── A1/
                    └── preprocessed_data/
                        ├── ...
                └── D1/
        └── STNet/
            └── 23209_C1/
                └── preprocessed_data/
                    ├── ...
            └── 23209_C2/
            └── .../
        └── Zenodo/
            └── Zenodo001/
                └── Control1/
                    └── preprocessed_data/
                        ├── ...
                └── Control2/
                └── .../
            └── Zenodo002/
                └── 10X001/
                    └── preprocessed_data/
                        ├── ...
                └── 10X009/
                └── .../
```
</details>

The links to download BIG-600K:
1. [BIG-600K (Part 1)](https://data.mendeley.com/datasets/fzzns2n7yg/1)
2. [BIG-600K (Part 2)](https://data.mendeley.com/datasets/dww54yfmtt/1)
3. [BIG-600K (Part 3)](https://data.mendeley.com/datasets/3w327r5br8/1)
4. [BIG-600K (Part 4)](https://data.mendeley.com/datasets/z8xxgmw2rh/1)
5. [BIG-600K (Part 5)](https://data.mendeley.com/datasets/f97r9vhpst/1)
6. [BIG-600K (Part 6)](https://data.mendeley.com/datasets/vvk6sbztt8/1)
7. [BIG-600K (Part 7)](https://data.mendeley.com/datasets/nzh9dv2dzd/1)
8. [BIG-600K (Part 8)](https://data.mendeley.com/datasets/d3rdrh65ys/1)

To download the single-cell datasets for retrieval, you could use the following commands to download them:
1. Brain
   ```bash
   wget https://datasets.cellxgene.cziscience.com/99dae17b-f5d6-4ba8-a5ee-f1ffac6b4b87.h5ad
   ```
2. Breast
   ```bash
   wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE234nnn/GSE234814/suppl/GSE234814_RAW.tar
   ```
3. Heart
   ```bash
   wget https://datasets.cellxgene.cziscience.com/9a433dac-a8ec-431b-b2a1-db0d67abe2ed.h5ad
   ```
4. Liver
   ```bash
   wget https://datasets.cellxgene.cziscience.com/1f42a859-0064-4db0-82b2-633b4a69c558.h5ad
   ```
5. Lung
   ```bash
   wget "https://cellgeni.cog.sanger.ac.uk/5-locations-lung/lung_5loc_sc_sn_raw_counts_cellxgene.h5ad"
   ```
6. Nasopharynx
   ```bash
   wget https://datasets.cellxgene.cziscience.com/b53b3bcd-3485-4562-a543-e473dfef3b27.h5ad
   ```
7. Ovary
   ```bash
   wget https://datasets.cellxgene.cziscience.com/de2a800c-249f-4072-8454-cde3d6bfb5b4.h5ad
   ```
8. Prostate
   ```bash
   wget https://datasets.cellxgene.cziscience.com/8a967e24-2297-43e5-b8b5-905c021470a6.h5ad
   ```
9. Skin
    ```bash
    wget https://datasets.cellxgene.cziscience.com/44f41e41-89a1-4a99-b30a-ed7925476946.h5ad
    ```
10. Small and large intestine
    ```bash
    wget https://datasets.cellxgene.cziscience.com/4b53ea1c-8f2d-44fe-ab20-6416d6e9c212.h5ad
    ```

## BRIDGE Walkthrough

## BRIDGE checkpoints
You can download all the checkpoints through shared Google Drive folder [BRIDGE_checkpoints](https://drive.google.com/drive/folders/1OIilkWFCcZjAdkcyJHAVaQfhTAC7_5jo?usp=sharing).

## Summary of Using the Pretrained BRIDGE for direct gene prediction
We first provide a sample script showing how to directly apply the pretrained BRIDGE for gene prediction on new unseen patches (assume the WSIs are already cropped).

```python
from BRIDGE_code.models.BRIDGE_model import BRIDGEModel
from sklearn.neighbors import KDTree
import scipy

sample_pretrained_weights_path = ".../weight.ckpt" # Please modify the path for the pretrained weights here
sample_patch_image_path = ".../13x15.jpg" # Please modify the path for the query patch image here
sample_gene_expression_pool_path = ".../gene_pool.pt" # Please modify the path for the sample retrieval reference pool here

pretrained_BRIDGE_model = BRIDGEModel.load_from_checkpoint(
    sample_pretrained_weights_path,
    strict=False,
)
pretrained_BRIDGE_model.eval()

# For prediction
patch_image = Image.open(sample_patch_image_path)
patch_feature = pretrained_BRIDGE_model.image_encoder(patch_image)
predicted_gene_expression = pretrained_BRIDGE_model.image_to_gene_decoder(patch_feature)

# For retrieval
retrieval_size = 256 # the k value for finding the k matching pairs with the highest similarity

patch_image = Image.open(sample_patch_image_path)
patch_feature = pretrained_BRIDGE_model.image_encoder(patch_image)
patch_embedding = pretrained_BRIDGE_model.image_encoder.forward_head(patch_feature)
patch_embedding = F.normalize(patch_embedding, p=2, dim=1)

gene_expression = torch.load(sample_gene_expression_pool_path)
gene_feature = pretrained_BRIDGE_model.gene_encoder(gene_expression)
gene_embedding = pretrained_BRIDGE_model.gene_encoder.forward_head(gene_feature)
gene_embedding = F.normalize(gene_embedding, p=2, dim=1)
# We utilize KDTree to get the similarity weights
kdtree = KDTree(gene_embedding)
distance, index = kdtree.query(patch_embedding, k=retrieval_size)
retrieved_expressions = gene_expression[index]
predicted_gene_expression = (scipy.special.softmax(1 - distance).reshape(-1, 1) * retrieved_expressions).sum(axis=0)
```

## Training BRIDGE from scratch using BIG-600K or self-defined data
To follow the training procedure we employed in this work, you can simply run
```bash
# multi-organ BRIDGE
python BRIDGE_code/runs/BRIDGE_train.py --gpu_cards 0,1,2,3,4,5,6,7 --num_devices 8 --organ_selected all

# single-organ BRIDGE(s)
python BRIDGE_code/runs/BRIDGE_train.py --gpu_cards 0,1,2,3,4,5,6,7 --num_devices 8 --organ_selected brain
python BRIDGE_code/runs/BRIDGE_train.py --gpu_cards 0,1,2,3,4,5,6,7 --num_devices 8 --organ_selected breast
python BRIDGE_code/runs/BRIDGE_train.py --gpu_cards 0,1,2,3,4,5,6,7 --num_devices 8 --organ_selected heart
python BRIDGE_code/runs/BRIDGE_train.py --gpu_cards 0,1,2,3,4,5,6,7 --num_devices 8 --organ_selected liver
python BRIDGE_code/runs/BRIDGE_train.py --gpu_cards 0,1,2,3,4,5,6,7 --num_devices 8 --organ_selected lung
python BRIDGE_code/runs/BRIDGE_train.py --gpu_cards 0,1,2,3,4,5,6,7 --num_devices 8 --organ_selected nose
python BRIDGE_code/runs/BRIDGE_train.py --gpu_cards 0,1,2,3,4,5,6,7 --num_devices 8 --organ_selected ovary
python BRIDGE_code/runs/BRIDGE_train.py --gpu_cards 0,1,2,3,4,5,6,7 --num_devices 8 --organ_selected prostate
python BRIDGE_code/runs/BRIDGE_train.py --gpu_cards 0,1,2,3,4,5,6,7 --num_devices 8 --organ_selected skin
python BRIDGE_code/runs/BRIDGE_train.py --gpu_cards 0,1,2,3,4,5,6,7 --num_devices 8 --organ_selected small_and_large_intestine
```
to get the checkpoints we used for downstream tasks. You can train `BLEEP`, `DeepSpaCE`, `ST-Net` in a highly similar manner.

Additionally, BRIDGE is a flexible framework that can incorporate a variety of choices for encoders and decoders that are implemented in `BRIDGE_code/backbones/encoders.py`. If you prefer specific model, you could enrich `ImageEncoder` and `GeneEncoder` in the file.

To use self-defined dataset to train BRIDGE, you could modify files `BRIDGE_code/dataset/image_gene_dataset.py` and `BRIDGE_code/dataset/image_gene_data_module.py` and then conduct training with the above commands.

## Gene prediction with BRIDGE
Run `BRIDGE_code/downstream_tasks/Part1_Prediction/1_1_direct_prediction_without_finetune/BRIDGE/step0_BRIDGE_direct_prediction.py` with modified paths to the checkpoints can generate a logger recording the numerical results. The predicted gene expression for the slide and gene-PCC dictionary will also be saved for further analysis.

Similarly, to access the performance of `DeepSpaCE` and `ST-Net`, you can run `BRIDGE_code/downstream_tasks/Part1_Prediction/1_1_direct_prediction_without_finetune/DeepSpaCE/step0_DeepSpaCE_direct_prediction.py` and `BRIDGE_code/downstream_tasks/Part1_Prediction/1_1_direct_prediction_without_finetune/STNet/step0_STNet_direct_prediction.py` accordingly.

## Gene retrieval with BRIDGE
We first save the extracted image and gene embedding as `pt` files so that we can conduct retrieval process smoothly and speedily later. The saving process is implemented in `BRIDGE_code/downstream_tasks/Part2_Retrieval/2_0_save_single_cell_dataset_embedding/BRIDGE/step0_save_BRIDGE_sc_embedding.py` and `BRIDGE_code/downstream_tasks/Part2_Retrieval/2_0_save_whole_st_training_data_embedding/BRIDGE/step0_save_BRIDGE_st_embedding.py`.

After the extraction, we design a two-stage retrieval procedure: (1) save the similarity distance and the retrieved indices; (2) get the retrieved gene expression. This design could greatly save time compared with a single file conducting two steps at the same time. For example, if you want to know the performance of multi-organ BRIDGE with multi-organ mixed-patient ST-seq Pool, run `BRIDGE_code/downstream_tasks/Part2_Retrieval/2_1_retrieval_from_whole_st_training_data/multi_organ_from_multi_st/BRIDGE/step0_BRIDGE_select_similar_indexes.py` first, and then run `BRIDGE_code/downstream_tasks/Part2_Retrieval/2_1_retrieval_from_whole_st_training_data/multi_organ_from_multi_st/BRIDGE/step1_BRIDGE_save_prediction_and_pcc.py`. Similar to prediction, we generate logger file, retrieval results, and gene-PCC dictionary for other downstream analysis.

## Cell clustering with BRIDGE
You can run `BRIDGE_code/downstream_tasks/Part3_Cell_clustering/BRIDGE_HER2ST_image_feature.py` to get the logger containing metrics performance (ARI, FMI, etc.) and the cluster assignment. Since we are adopting K-Means algorithm, the generated assignment is not order. To align the result with ground truth, you could use the following code snippet
```python
from itertools import permutations
import numpy as np

def generate_permutations_no_repeats(numbers):
    perm_list = list(permutations(numbers))
    result = [list(perm) for perm in perm_list]
    return result

def create_dictionary_from_lists(list1, list2):
    result_dict = {key: value for key, value in zip(list1, list2)}
    return result_dict

ground_truth_cluster_npy_path = "" # where the ground truth label is stored
prediction_cluster_npy_path = "" # where our predicted label is stored
ground_truth = np.load(ground_truth_cluster_npy_path, allow_pickle=True)
prediction = np.load(prediction_cluster_npy_path, allow_pickle=True)

labels = [0,1,2,3,4,5,6] # Subject to change. For the breast cancer slide we utilized here, the spots are categorized into seven labels.
permutations = generate_permutations_no_repeats(labels)

total_list_of_dictionary = []
for permutation in permutations:
    mapping_dict = create_dictionary_from_lists(labels, permutation)
    total_list_of_dictionary.append(mapping_dict)

for mapping in total_list_of_dictionary:
    mapped_prediction = apply_mapping(mapping, ground_truth, prediction)
    accuracy = calculate_accuracy(ground_truth, mapped_prediction)
    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        best_mapping = mapping
```
to get the best alignment mapping.

## Survival Analysis with BRIDGE
BRIDGE could perform survival analysis on external TCGA cohorts. Firstly, we extract patch-level visual features by image foundation models and generate spot-wise gene prediction by BRIDGE. The code is implemented in `BRIDGE_code/downstream_tasks/Part4_Survival_Analysis/step1_feature_extractor.py`, and we provide sample usage on TCGA-BLCA dataset in `BRIDGE_code/downstream_tasks/Part4_Survival_Analysis/step1_feature_extractor.sh`.

Besides utilizing the spot-level gene predictions as the input for survival analysis, we may choose to create a slide-level view by taking the mean value of each gene as the pseudo bulk RNA-seq. The idea is implemented in `BRIDGE_code/downstream_tasks/Part4_Survival_Analysis/step2_generate_bulkrna_pred.py`.

We then run `BRIDGE_code/downstream_tasks/Part4_Survival_Analysis/step3_survival.py` to train the survival model. Sample usages are listed in `BRIDGE_code/downstream_tasks/Part4_Survival_Analysis/step3_survival.sh` for different image models, BRIDGE gene predictions and ground truth bulk RNA-seq.
