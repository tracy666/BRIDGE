# BRIDGE
Official code repository for "BRIDGE: A Cross-organ Foundation Model for Bridging Histology Imaging and Spatial Transcriptomics"

<div align="center">
  <img width="100%" alt="BIG-600K Data Distribution" src="https://github.com/tracy666/BRIDGE/blob/ef0a0cfb7c5a1cb7464b986376b3118600cb350e/BIG_600K.png">
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

## Downloading + Preprocessing BIG-600K
We uploaded all our preprocessed data to Mendeley Data (Link will be updated later). Due to the space limitation of the platform, we separate the dataset into several parts and zipped the files. After downloading all the zipped folders, you could use the code to unzip them and organize them in the following directories:
<details>
<summary>
Example Directory
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
