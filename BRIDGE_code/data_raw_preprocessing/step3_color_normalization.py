import os
os.environ['OMP_NUM_THREADS'] = '1' # OpenBLAS warning: precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata.
os.environ['OPENBLAS_NUM_THREADS'] = '1' # OpenBLAS warning: precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata.
import argparse
import time
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
from joblib import Parallel, delayed

import sys
# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the StainTools folder
staintools_path = os.path.join(current_dir, "StainTools")

# Append the relative path to sys.path
sys.path.append(staintools_path)
import stainNorm_Vahadane
import stain_utils
from step0_preprocess_helper_functions import find_all_patches, running_time_display

def color_normalization(
        single_patch_path: str,
        normalizer,
) -> None:
    try:
        # define the normed patch path
        normed_patch_name_path = single_patch_path.replace(single_patch_path.split("/")[-1], f"normed_{single_patch_path.split('/')[-1]}")
        normed_patch_path = normed_patch_name_path.replace(single_patch_path.split("/")[-2], f"normed_{single_patch_path.split('/')[-2]}")
        # read the patch
        original_patch = stain_utils.read_image(single_patch_path)
        normed_patch = normalizer.transform(original_patch)
        # save the normed patch
        normed_image = Image.fromarray(normed_patch)
        if not os.path.exists(normed_patch_path):
            normed_image.save(normed_patch_path)
    except Exception as e:
        print(f"Error in normalizing {single_patch_path}. Error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_data_storage", type=str, default="/data1/zliang")
    parser.add_argument("--raw_data_folder_name", type=str, default="Histo_ST_raw")
    parser.add_argument("--project_data_folder_name", type=str, default="BIG_600K")
    parser.add_argument("--multiprocessing_pool_num", type=int, default=5)
    args = parser.parse_args()

    patches_start_time = time.time()
    project_patches_list = find_all_patches(
        main_data_storage=args.main_data_storage,
        project_data_folder_name=args.project_data_folder_name,
        patches_folder_names=["patches", "patches_mag10_thre25"]
    )
    patches_end_time = time.time()
    print(f"The patches list is finished. Time used: {running_time_display(patches_end_time - patches_start_time)}. The total length of patches is {len(project_patches_list)}")
    
    reference_start_time = time.time()

    read_image_start_time = time.time()
    reference_image_path = os.path.join(args.main_data_storage, args.raw_data_folder_name, "Specific_datasets", "HER2ST_Version_3_0", "Spatial_deconvolution_of_HER2-positive_Breast_cancer_delineates_tumor-associated_cell_type_interactions", "images", "HE", "G1.jpg")
    # reference_image_path = "/data1/zliang/Histo_ST_raw/Specific_datasets/HER2ST_Version_3_0/Spatial_deconvolution_of_HER2-positive_Breast_cancer_delineates_tumor-associated_cell_type_interactions/images/HE/G1.jpg" # 5.6M
    # The reason to choose this image: HER2ST original paper chose this image as illustration; complex annotation
    reference_image = stain_utils.read_image(reference_image_path)
    read_image_end_time = time.time()
    print(f"The reference image is read. Time used: {running_time_display(read_image_end_time - read_image_start_time)}.")

    normalizer_initialization_start_time = time.time()
    normalizer = stainNorm_Vahadane.Normalizer()
    normalizer_initialization_end_time = time.time()
    print(f"The normalizer is initialized. Time used: {running_time_display(normalizer_initialization_end_time - normalizer_initialization_start_time)}.")

    normalizer_fit_start_time = time.time()
    normalizer.fit(reference_image)
    normalizer_fit_end_time = time.time()
    print(f"The normalizer is fitted. Time used: {running_time_display(normalizer_fit_end_time - normalizer_fit_start_time)}.")

    reference_end_time = time.time()
    print(f"The reference image is finished. Time used: {running_time_display(reference_end_time - reference_start_time)}.")

    transform_start_time = time.time()
    pbar = tqdm(project_patches_list)
    Parallel(n_jobs=args.multiprocessing_pool_num, max_nbytes=5000)(delayed(color_normalization)(single_patch_path, normalizer) for single_patch_path in pbar)
    transform_end_time = time.time()
    print(f"The transformation is finished. Time used: {running_time_display(transform_end_time - transform_start_time)}.")

if __name__ == "__main__":
    main()