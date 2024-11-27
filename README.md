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
