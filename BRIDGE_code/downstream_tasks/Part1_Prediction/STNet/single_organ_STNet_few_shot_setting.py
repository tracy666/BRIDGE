import argparse
import logging
from datetime import datetime
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from typing import List, Optional, Union, Literal, Tuple
import scipy
import pickle
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import time
from sklearn.neighbors import KDTree
import h5py

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
current_workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
dataset_dir = os.path.join(current_workspace, "dataset")
sys.path.append(dataset_dir)
# from image_gene_dataset import SingleSlideEvalImageGeneDataset
from image_gene_dataset import get_image_transforms

models_dir = os.path.join(current_workspace, "models")
sys.path.append(models_dir)
from STNet_model import STNetModel

class SingleSlideEvalImageGeneDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            main_data_storage: str,
            project_data_folder_name: str,
            working_codespace: str,
            patient_dir: str,
            gene_csv_name: str,
            generation_date: str,
            train_or_test: Literal["train", "test"],
            scGPT_gene_mask_ratio: float,
    ):
        super().__init__()
        self.main_data_storage = main_data_storage
        self.project_data_folder_name = project_data_folder_name
        self.working_codespace = working_codespace
        self.patient_dir = patient_dir
        self.generation_date = generation_date
        self.train_or_test = train_or_test
        self.gene_mask_ratio_str = str(scGPT_gene_mask_ratio).split(".")[1]

        gene_csv_path = os.path.join(working_codespace, "data_raw_preprocessing", "selected_genes", gene_csv_name)
        selected_genes_list = pd.read_csv(gene_csv_path)["gene_names"].values.tolist()
        self.selected_genes_list = selected_genes_list
        self.gene_csv_name_without_extension = os.path.splitext(gene_csv_name)[0]
        assert os.path.exists(os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_masked_tokenized_gene_values_with_mask_ratio_{self.gene_mask_ratio_str}")), f"Haven't generate masked tokenized values for mask ratio {scGPT_gene_mask_ratio}"

        patient_dir_numerical_pt_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_patient_dir_numerical.pt")
        self.all_patient_dir_numerical = torch.load(patient_dir_numerical_pt_path)
        patch_path_npy_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_patch_path.npy")
        self.all_patch_path = np.load(patch_path_npy_path, allow_pickle=True)
        normed_patch_path_npy_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_normed_patch_path.npy")
        self.all_normed_patch_path = np.load(normed_patch_path_npy_path, allow_pickle=True)
        spot_barcode_npy_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_spot_barcode.npy")
        self.all_spot_barcode = np.load(spot_barcode_npy_path, allow_pickle=True)

        patient_dir_organ_mapping_df_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, "patient_dir_organ_mapping_df.csv")
        patient_dir_organ_mapping_df = pd.read_csv(patient_dir_organ_mapping_df_path)
        patient_dir_num_mapping_dict = dict(zip(patient_dir_organ_mapping_df["patient_dir"], patient_dir_organ_mapping_df.index))
        single_patient_dir_numerical = patient_dir_num_mapping_dict[self.patient_dir]
        mask = np.isin(self.all_patient_dir_numerical, single_patient_dir_numerical)

        index_list = list()
        for index, boolean_value in enumerate(mask):
            if boolean_value:
                index_list.append(index)
        self.selected_index_list = index_list
        
        if self.train_or_test == "train":
            self.train_image_transform1, self.train_image_transform2 = [get_image_transforms(train_or_test=self.train_or_test, image_size=224, crop_size=224)] * 2
        elif self.train_or_test == "test":
            self.test_image_transform = get_image_transforms(train_or_test=self.train_or_test, image_size=224, crop_size=224)
        else:
            raise ValueError(f"The train_or_test should be either 'train' or 'test', but now the value is {self.train_or_test}.")
    
    def __getitem__(
        self,
        index
    ):
        selected_index = self.selected_index_list[index]
        patient_dir_numerical = self.all_patient_dir_numerical[selected_index]
        patch_path = os.path.join(self.main_data_storage, self.project_data_folder_name, self.all_patch_path[selected_index])
        normed_patch_path = os.path.join(self.main_data_storage, self.project_data_folder_name, self.all_normed_patch_path[selected_index])
        spot_barcode = self.all_spot_barcode[selected_index]

        # 1. full_dataset_masked_tokenized_gene_values_with_mask_ratio_xx
        scgpt_masked_tokenized_gene_values_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_masked_tokenized_gene_values_with_mask_ratio_{self.gene_mask_ratio_str}", f"index_{selected_index}.pt")
        scgpt_masked_tokenized_gene_values = torch.load(scgpt_masked_tokenized_gene_values_path)
        # 2. full_dataset_tokenized_gene_ids
        scgpt_tokenized_gene_ids_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_tokenized_gene_ids", f"index_{selected_index}.pt")
        scgpt_tokenized_gene_ids = torch.load(scgpt_tokenized_gene_ids_path)
        # 3. full_dataset_tokenized_gene_values
        scgpt_tokenized_gene_values_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_tokenized_gene_values", f"index_{selected_index}.pt")
        scgpt_tokenized_gene_values = torch.load(scgpt_tokenized_gene_values_path)
        # 4. full_dataset_X_binned
        binned_gene_expression_path = os.path.join(self.working_codespace, "generated_files", self.generation_date, self.gene_csv_name_without_extension, f"full_dataset_X_binned", f"index_{selected_index}.pt")
        binned_gene_expression = torch.load(binned_gene_expression_path)

        raw_patch = Image.open(patch_path).convert('RGB')
        raw_normed_patch = Image.open(normed_patch_path).convert('RGB')
        if self.train_or_test == "train":
            original_patch = self.train_image_transform1(raw_patch)
            augmented_patch = self.train_image_transform2(raw_patch)
            original_normed_patch = self.train_image_transform1(raw_normed_patch)
            augmented_normed_patch = self.train_image_transform2(raw_normed_patch)
        elif self.train_or_test == "test":
            original_patch = self.test_image_transform(raw_patch)
            augmented_patch = original_patch
            original_normed_patch = self.test_image_transform(raw_normed_patch)
            augmented_normed_patch = original_normed_patch

        return {
            "all_patient_dir_numerical": patient_dir_numerical,
            "all_spot_barcode": spot_barcode,
            "all_binned_gene_expression": binned_gene_expression,
            "all_scgpt_tokenized_gene_ids": scgpt_tokenized_gene_ids,
            "all_scgpt_tokenized_gene_values": scgpt_tokenized_gene_values,
            "all_scgpt_masked_tokenized_gene_values": scgpt_masked_tokenized_gene_values,
            "all_original_patch": original_patch,
            "all_augmented_patch": augmented_patch,
            "all_original_normed_patch": original_normed_patch,
            "all_augmented_normed_patch": augmented_normed_patch,
        }
    
    def __len__(
            self
    ):
        return len(self.selected_index_list)

def get_ground_truth_and_prediction_gene_expressions(
        image_encoder,
        image_to_gene_decoder,
        dataloader,
        device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        total_slide_gene_ground_truth = list()
        total_slide_gene_prediction = list()
        for _, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Get gene prediction of the evaluation patient dirs"):
            original_patch_view = batch["all_original_normed_patch"].to(device)
            input_gene_expression = batch["all_binned_gene_expression"]
            total_slide_gene_ground_truth.append(input_gene_expression.cpu().numpy())
            original_image_feature = image_encoder(original_patch_view)
            original_image_embedding = image_encoder.forward_head(original_image_feature)
            original_image_embedding = F.normalize(original_image_embedding, p=2, dim=1)
            image_to_gene_prediction = image_to_gene_decoder(original_image_feature)
            total_slide_gene_prediction.append(image_to_gene_prediction.detach().cpu().numpy())
    total_slide_gene_ground_truth = np.concatenate(total_slide_gene_ground_truth, axis=0)
    total_slide_gene_prediction = np.concatenate(total_slide_gene_prediction, axis=0)
    return total_slide_gene_ground_truth, total_slide_gene_prediction

def compute_avg_pcc_given_genes(
        ground_truth_gene_expression: np.ndarray,
        prediction_gene_expression: np.ndarray,
        full_gene_names_list: List[str],
        eval_selected_gene_names_list: List[str],
) -> float:
    gene_names_indices_mapping = {gene_name: index for index, gene_name in enumerate(full_gene_names_list)}
    column_numbers_selected = [gene_names_indices_mapping[gene_name] for gene_name in full_gene_names_list if gene_name in eval_selected_gene_names_list]
    sorted_column_numbers_selected = sorted(column_numbers_selected, key=lambda x: eval_selected_gene_names_list.index(full_gene_names_list[x]))
    
    pcc_values_list = list()
    pcc_values_dict = dict()
    for column_number in sorted_column_numbers_selected:
        specific_gene_ground_truth = ground_truth_gene_expression[:, column_number]
        specific_gene_prediction = prediction_gene_expression[:, column_number]
        pcc_results = scipy.stats.pearsonr(specific_gene_ground_truth, specific_gene_prediction)
        pcc_statistic = pcc_results[0]
        pcc_pvalue = pcc_results[1]
        if np.isnan(pcc_statistic):
            pcc_statistic = 0
        pcc_values_list.append(pcc_statistic)
        pcc_values_dict[full_gene_names_list[column_number]] = pcc_statistic
    pcc_mean = np.mean(pcc_values_list)
    return pcc_mean, pcc_values_dict



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--main_data_storage', type=str, default="/data1/zliang")
    parser.add_argument('--project_data_folder_name', type=str, default="BIG_600K")
    parser.add_argument('--working_codespace', type=str, default="/home/zliang/BRIDGE/BRIDGE_code")

    parser.add_argument("--generation_date", type=str, default="20240601")
    parser.add_argument('--gene_csv_name', type=str, default="all_intersection_genes_number_7730.csv")
    parser.add_argument('--num_of_eval_genes', type=int, default=1000)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--gpu_cards', type=str, default='', help='Comma-separated list of GPU card numbers')
    parser.add_argument('--model_settings', type=str, default='few_shot_setting')
    args = parser.parse_args()
    
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
    
    logger_name = datetime.now().strftime('%Y%m%d%H%M%S') + f"_STNet_single_organ_for_{args.model_settings}"
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
    
    test_df_path = os.path.join(args.working_codespace, "generated_files", args.generation_date, f"few_shot_df.csv")
    test_df = pd.read_csv(test_df_path)
    
    full_genes_csv_path = os.path.join(args.working_codespace, "data_raw_preprocessing", "selected_genes", args.gene_csv_name)
    full_genes_list = pd.read_csv(full_genes_csv_path)["gene_names"].values.tolist()
    
    selected_genes_HEG_csv_path = os.path.join(args.working_codespace, "data_raw_preprocessing", "selected_genes", f"top_HEG_genes_number_{args.num_of_eval_genes}.csv")
    selected_genes_HEG_list = pd.read_csv(selected_genes_HEG_csv_path)["gene_names"].values.tolist()
    selected_genes_HVG_csv_path = os.path.join(args.working_codespace, "data_raw_preprocessing", "selected_genes", f"top_HVG_genes_number_{args.num_of_eval_genes}.csv")
    selected_genes_HVG_list = pd.read_csv(selected_genes_HVG_csv_path)["gene_names"].values.tolist()
    selected_genes_list = selected_genes_HEG_list + selected_genes_HVG_list
    selected_genes_list = list(set(selected_genes_list))
    
    biomarker_genes_csv_path = os.path.join(args.working_codespace, "data_raw_preprocessing", "selected_genes", f"selected_biomarkers_number_80.csv")
    biomarker_genes_list = pd.read_csv(biomarker_genes_csv_path)["gene_names"].values.tolist()
    logger.info(f"Biomarker genes setting is of {len(biomarker_genes_list)} involved.")
    
    full_gene_slide_pcc_list = list()
    selected_gene_slide_pcc_list = list()
    biomarker_gene_slide_pcc_list = list()
    
    for row in tqdm(test_df.itertuples(), total=len(test_df)):
        print(f"Processing {row.patient_dir} with {row.organ_type}")
        checkpoint = os.path.join(args.main_data_storage, "finalized_checkpoints", "STNet", "single_organ", f"STNet_{row.organ_type}", "19.ckpt")
        assert os.path.exists(checkpoint)
        
        logger.info(f"single-organ checkpoint path: {checkpoint} for {row.patient_dir} with {row.organ_type}")
        
        try:
            epoch_number = int(checkpoint.split("/")[-1].split(".")[0]) + 1
        except ValueError:
            epoch_number = int((checkpoint.split("/")[-1].split(".")[0]).split("_")[-1]) + 1
    
        pretrained_STNet_model = STNetModel.load_from_checkpoint(
            checkpoint,
            strict=False,
            main_data_storage=args.main_data_storage,
            working_codespace=args.working_codespace,
            weight_decay=1e-3,
        )
        pretrained_STNet_model.eval()
        
        single_slide_eval_dataset = SingleSlideEvalImageGeneDataset(
            main_data_storage=args.main_data_storage,
            project_data_folder_name=args.project_data_folder_name,
            working_codespace=args.working_codespace,
            patient_dir=row.patient_dir,
            gene_csv_name=args.gene_csv_name,
            generation_date=args.generation_date,
            train_or_test="test",
            scGPT_gene_mask_ratio=0.25,
        )
        single_slide_eval_dataloader = torch.utils.data.DataLoader(
            single_slide_eval_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )
        single_slide_gene_ground_truth, single_slide_gene_prediction = get_ground_truth_and_prediction_gene_expressions(
            image_encoder=pretrained_STNet_model.image_encoder,
            image_to_gene_decoder=pretrained_STNet_model.image_to_gene_decoder,
            dataloader=single_slide_eval_dataloader,
            device=device,
        )
        full_gene_slide_pcc, full_gene_slide_genes_pcc_dict = compute_avg_pcc_given_genes(
            ground_truth_gene_expression=single_slide_gene_ground_truth,
            prediction_gene_expression=single_slide_gene_prediction,
            full_gene_names_list=full_genes_list,
            eval_selected_gene_names_list=full_genes_list,
        )
        selected_gene_slide_pcc, selected_gene_slide_genes_pcc_dict = compute_avg_pcc_given_genes(
            ground_truth_gene_expression=single_slide_gene_ground_truth,
            prediction_gene_expression=single_slide_gene_prediction,
            full_gene_names_list=full_genes_list,
            eval_selected_gene_names_list=selected_genes_list,
        )
        biomarker_gene_slide_pcc, biomarker_gene_slide_genes_pcc_dict = compute_avg_pcc_given_genes(
            ground_truth_gene_expression=single_slide_gene_ground_truth,
            prediction_gene_expression=single_slide_gene_prediction,
            full_gene_names_list=full_genes_list,
            eval_selected_gene_names_list=biomarker_genes_list,
        )
        selected_gene_slide_pcc_list.append(selected_gene_slide_pcc)
        biomarker_gene_slide_pcc_list.append(biomarker_gene_slide_pcc)
        logger.info(f"For {row.patient_dir} with {row.organ_type}, single-organ STNet biomarker gene performance: {biomarker_gene_slide_pcc}")

        print(f"Start saving single-organ STNet data")
        saving_start_time = time.time()
        full_gene_slide_genes_pcc_dict = {k: v for k, v in sorted(full_gene_slide_genes_pcc_dict.items(), key=lambda item: item[1], reverse = True)}
        patient_dir_without_slash = (row.patient_dir).replace("/", "_")
        
        pred_data_folder_saving_path = os.path.join(
            downstream_tasks_saving_dir,
            f"{args.model_settings}_at_epoch_{epoch_number}",
            patient_dir_without_slash,
        )
        os.makedirs(pred_data_folder_saving_path, exist_ok=True)
        ground_truth_saving_path = os.path.join(pred_data_folder_saving_path, f"{patient_dir_without_slash}_ground_truth.npy")
        prediction_saving_path = os.path.join(pred_data_folder_saving_path, f"{patient_dir_without_slash}_prediction.npy")
        pcc_dict_saving_path = os.path.join(pred_data_folder_saving_path, f"{patient_dir_without_slash}_pcc_dict.npy")
        if not os.path.exists(ground_truth_saving_path):
            np.save(ground_truth_saving_path, single_slide_gene_ground_truth)
            print(f"{row.patient_dir} single-organ STNet ground truth saved")
        if not os.path.exists(prediction_saving_path):
            np.save(prediction_saving_path, single_slide_gene_prediction)
            print(f"{row.patient_dir} single-organ STNet prediction saved")
        if not os.path.exists(pcc_dict_saving_path):
            with open(pcc_dict_saving_path, "wb") as f:
                pickle.dump(full_gene_slide_genes_pcc_dict, f)
            print(f"{row.patient_dir} single-organ STNet pcc dict saved")
        saving_end_time = time.time()
        print(f"Finished saving single-organ STNet data, takes {saving_end_time - saving_start_time} seconds")
        
    assert len(selected_gene_slide_pcc_list) == len(biomarker_gene_slide_pcc_list)  == len(test_df)
    biomarker_gene_pcc_avg = sum(biomarker_gene_slide_pcc_list)/len(test_df)
    logger.info(f"For single-organ STNet under {args.model_settings}, Average biomarker gene performance: {biomarker_gene_pcc_avg}")
    
if __name__ == "__main__":
    main()