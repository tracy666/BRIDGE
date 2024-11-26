import torch
import argparse
import logging
from datetime import datetime
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Union, Literal, Tuple
import anndata
import numpy as np
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score , adjusted_mutual_info_score, mutual_info_score

import warnings
warnings.filterwarnings('ignore')

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
current_workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))))
dataset_dir = os.path.join(current_workspace, "dataset")
sys.path.append(dataset_dir)
from image_gene_dataset import get_image_transforms

backbones_dir = os.path.join(current_workspace, "backbones")
sys.path.append(backbones_dir)
from encoders import ImageEncoder

models_dir = os.path.join(current_workspace, "models")
sys.path.append(models_dir)
from BLEEP_model import BLEEPModel

clustering_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
clustering_dataset_dir = os.path.join(clustering_dir, "3_0_datasets")
sys.path.append(clustering_dataset_dir)
from HER2ST import SingleSlideEvalImageGeneDataset_HER2ST

def get_natural_image_pretrained_model_image_feature(
        image_model,
        dataloader,
        device,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        total_image_feature_list = list()
        total_slide_label_numerical_list = list()
        for _, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Get natural image pretrained model image feature of the evaluation patient dirs"):
            original_patch_view = batch["all_original_normed_patch"].to(device)
            original_image_feature = image_model(original_patch_view)
            total_image_feature_list.append(original_image_feature.detach().cpu().numpy())
            patient_label_numerical = batch["all_patient_label_numerical"]
            total_slide_label_numerical_list.append(patient_label_numerical.detach().cpu().numpy())
        total_image_feature = np.concatenate(total_image_feature_list, axis=0)
        total_slide_label_numerical = np.concatenate(total_slide_label_numerical_list, axis=0)
    return total_image_feature, total_slide_label_numerical

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_data_storage", type=str, default="/data1/zliang")
    parser.add_argument("--raw_data_folder_name", type=str, default="Histo_ST_raw")
    parser.add_argument('--project_data_folder_name', type=str, default="BIG_600K")
    parser.add_argument("--working_codespace", type=str, default="/home/zliang/BRIDGE_BIG_600K")

    parser.add_argument("--generation_date", type=str, default="20240601")
    parser.add_argument('--gene_csv_name', type=str, default="all_intersection_genes_number_7730.csv")

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--gpu_cards', type=str, default='', help='Comma-separated list of GPU card numbers')
    args = parser.parse_args()

    dataset_name = "HER2ST"

    gpu_cards = args.gpu_cards.split(",") if args.gpu_cards else []
    gpu_cards_str = ",".join(gpu_cards)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_cards_str
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    downstream_tasks_saving_dir = current_dir.replace(
        args.working_codespace,
        args.main_data_storage,
    )
    os.makedirs(downstream_tasks_saving_dir, exist_ok=True)

    logger_dir = os.path.join(current_dir, "logger")
    os.makedirs(logger_dir, exist_ok=True)
    logger_name = datetime.now().strftime('%Y%m%d%H%M%S') + f"_natural_image_pretraining_single_organ_BLEEP_image_feature_for_dataset_{dataset_name}"
    # Set up logger
    logger = logging.getLogger(__name__) # Create a custom logger
    logger.setLevel(logging.INFO)
    # Create a file handler and set its level to INFO
    file_handler = logging.FileHandler(os.path.join(logger_dir, logger_name), 'w')
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout) # Create a stream handler and set its level to DEBUG
    stream_handler.setLevel(logging.DEBUG)
    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler) # Add the handlers to the logger
    logger.addHandler(stream_handler)
    logger = logging.getLogger(__name__)

    # HER2ST: breast
    single_organ_BLEEP_checkpoint = "/data1/zliang/0000_0901_finalized_checkpoints/BLEEP/single_organ/BLEEP_2024_09_01_19_27_51_breast/19.ckpt"
    pretrained_BLEEP_model = BLEEPModel.load_from_checkpoint(
        single_organ_BLEEP_checkpoint,
        strict=False,
        main_data_storage=args.main_data_storage,
        working_codespace=args.working_codespace,
        weight_decay=1e-3,
    )
    pretrained_BLEEP_model.eval()
    single_organ_BLEEP_image_model = pretrained_BLEEP_model.image_encoder

    test_df_path = os.path.join(args.working_codespace, "generated_files", args.generation_date, f"test_df.csv")
    test_df = pd.read_csv(test_df_path)
    test_patient_dir_list = test_df["patient_dir"].tolist()
    HER2ST_patient_dir_list = [patient_dir for patient_dir in test_patient_dir_list if "HER2ST" in patient_dir]
    assert len(HER2ST_patient_dir_list) == 8, f"len(HER2ST_patient_dir_list)={len(HER2ST_patient_dir_list)} != 8"
    selected_patient_dir_list = HER2ST_patient_dir_list

    for patient_dir in selected_patient_dir_list:
        logger.info(f"Single image modality - natural image pretraining - single_organ_BLEEP image feature - dataset {dataset_name} - patient {patient_dir}")

        dataset = SingleSlideEvalImageGeneDataset_HER2ST(
            main_data_storage=args.main_data_storage,
            raw_data_folder_name=args.raw_data_folder_name,
            project_data_folder_name=args.project_data_folder_name,
            working_codespace=args.working_codespace,
            patient_dir=patient_dir,
            gene_csv_name=args.gene_csv_name,
            generation_date=args.generation_date,
            train_or_test="test",
            scGPT_gene_mask_ratio=0.25,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )
        single_slide_image_feature, single_slide_label_numerical = get_natural_image_pretrained_model_image_feature(
            image_model=single_organ_BLEEP_image_model,
            dataloader=dataloader,
            device=device,
        )

        n_clusters = len(np.unique(single_slide_label_numerical))
        kmeans_prediction = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(single_slide_image_feature)
        assert len(single_slide_label_numerical) == len(kmeans_prediction.labels_) == single_slide_image_feature.shape[0]

        ARI_value = adjusted_rand_score(single_slide_label_numerical, kmeans_prediction.labels_)
        logger.info(f"ARI value for cluster number {n_clusters} in patient slide {patient_dir}: {ARI_value}")
        MI_value = mutual_info_score(single_slide_label_numerical, kmeans_prediction.labels_)
        logger.info(f"MI value for cluster number {n_clusters} in patient slide {patient_dir}: {MI_value}")
        AMI_value = adjusted_mutual_info_score(single_slide_label_numerical, kmeans_prediction.labels_)
        logger.info(f"AMI value for cluster number {n_clusters} in patient slide {patient_dir}: {AMI_value}")
        NMI_value = normalized_mutual_info_score(single_slide_label_numerical, kmeans_prediction.labels_)
        logger.info(f"NMI value for cluster number {n_clusters} in patient slide {patient_dir}: {NMI_value}")
        FMI_value = fowlkes_mallows_score(single_slide_label_numerical, kmeans_prediction.labels_)
        logger.info(f"FMI value for cluster number {n_clusters} in patient slide {patient_dir}: {FMI_value}")
        logger.info(f"##########################################################################################")

        patient_dir_without_slash = (patient_dir).replace("/", "_")
        ground_truth_label_saving_path = os.path.join(downstream_tasks_saving_dir, f"{patient_dir_without_slash}_ground_truth_label.npy")
        prediction_label_saving_path = os.path.join(downstream_tasks_saving_dir, f"{patient_dir_without_slash}_prediction_label.npy")
        if not os.path.exists(ground_truth_label_saving_path):
            np.save(ground_truth_label_saving_path, single_slide_label_numerical)
            print(f"Save ground truth label of patient {patient_dir} in {ground_truth_label_saving_path}")
        if not os.path.exists(prediction_label_saving_path):
            np.save(prediction_label_saving_path, kmeans_prediction.labels_)
            print(f"Save prediction label of patient {patient_dir} in {prediction_label_saving_path}")

if __name__ == "__main__":
    main()