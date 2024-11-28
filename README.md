# BRIDGE
Official code repository for "BRIDGE: A Cross-organ Foundation Model for Bridging Histology Imaging and Spatial Transcriptomics"

<div align="center">
  <img width="100%" alt="BIG-600K Data Distribution" src="https://github.com/tracy666/BRIDGE/blob/ef0a0cfb7c5a1cb7464b986376b3118600cb350e/BIG_600K.png">
</div>

## Updates / TODOs
Please follow this GitHub for more updates.
- [ ] Provide complete Mendeley links for downloading BIG-600K preprocessed data. (Uploading)

## Environment Installation

To clone all files:
```bash
git clone https://github.com/tracy666/BRIDGE.git
```

To install Python dependencies:
```bash
conda install --file requirements.txt
```
## BRIDGE Walkthrough

## Using the Pretrained BRIDGE
You can use the BRIDGE model out-of-the-box, and use it to plug-and-play into any of your downstream tasks (example below).

```python
from BRIDGE_code.models.BRIDGE_model import BRIDGEModel
from sklearn.neighbors import KDTree

sample_pretrained_weights_path = ".../weight.ckpt" # Please modify the path for the pretrained weights here
sample_patch_image_path = ".../13x15.jpg" # Please modify the path for the query patch image here
sample_gene_expression_pool_path = ".../gene_pool.pt" # Please modify the path for the sample gene counts here

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
distance, indices = kdtree.query(patch_embedding, k=retrieval_size)
```

## Downloading BIG-600K
We uploaded all our preprocessed data to Mendeley Data. Due to the space limitation of the platform, we separate the dataset into several parts and zipped the files. After downloading all the zipped folders, you could use the code to unzip them and organize them in the following directories:
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
