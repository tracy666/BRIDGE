# Below is the sample for dataset TCGA-BLCA

python step1_feature_extractor.py --batch_size 128 --feature_extractor dinov2_vits14 --dataset_path /disk2/wqzhao/TCGA/WSI/BLCA/single/BLCA --save_path /disk2/wqzhao/TCGA/TCGA_processed_data/BLCA_dinov2_vits14 --gpu_index 0
python step1_feature_extractor.py --batch_size 128 --feature_extractor HIPT --dataset_path /disk2/wqzhao/TCGA/WSI/BLCA/single/BLCA --save_path /disk2/wqzhao/TCGA/TCGA_processed_data/BLCA_HIPT --gpu_index 0
python step1_feature_extractor.py --batch_size 128 --feature_extractor PLIP --dataset_path /disk2/wqzhao/TCGA/WSI/BLCA/single/BLCA --save_path /disk2/wqzhao/TCGA/TCGA_processed_data/BLCA_PLIP --gpu_index 0
python step1_feature_extractor.py --batch_size 128 --feature_extractor Quilt1M --dataset_path /disk2/wqzhao/TCGA/WSI/BLCA/single/BLCA --save_path /disk2/wqzhao/TCGA/TCGA_processed_data/BLCA_Quilt1M --gpu_index 0
python step1_feature_extractor.py --batch_size 128 --feature_extractor CONCH --dataset_path /disk2/wqzhao/TCGA/WSI/BLCA/single/BLCA --save_path /disk2/wqzhao/TCGA/TCGA_processed_data/BLCA_CONCH --gpu_index 0
python step1_feature_extractor.py --batch_size 128 --feature_extractor UNI --dataset_path /disk2/wqzhao/TCGA/WSI/BLCA/single/BLCA --save_path /disk2/wqzhao/TCGA/TCGA_processed_data/BLCA_UNI --gpu_index 0
python step1_feature_extractor.py --batch_size 128 --feature_extractor CTranPath --dataset_path /disk2/wqzhao/TCGA/WSI/BLCA/single/BLCA --save_path /disk2/wqzhao/TCGA/TCGA_processed_data/BLCA_CTranPath --gpu_index 0

python step1_feature_extractor.py --batch_size 128 --feature_extractor BRIDGE_Genepred --dataset_path /disk2/wqzhao/TCGA/WSI/BLCA/single/BLCA --save_path /disk2/wqzhao/TCGA/TCGA_processed_data/BLCA_BRIDGE_Genepred --gpu_index 0

# You may replace BLCA with other TCGA datasets, especially BRCA, ESCA, LUAD, STAD.
# dataset path should be replaced by where you store the TCGA dataset patches.
# save path should be replaced by where you want to save the extracted features / BRIDGE gene prediction.