# --> General imports
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import torch
import random
import pickle
from sklearn.model_selection import StratifiedShuffleSplit

# --> Torch imports 
import torch


def set_seed(SEED, disable_cudnn=False):
    torch.manual_seed(SEED)  # Seed the RNG for all devices (both CPU and CUDA).
    random.seed(SEED)        # Set python seed for custom operators.
    rs = RandomState(MT19937(SeedSequence(SEED)))  # If any of the libraries or code rely on NumPy seed the global NumPy RNG.
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # If you are using multi-GPU. In case of one GPU, you can use # torch.cuda.manual_seed(SEED).

    if not disable_cudnn:
        torch.backends.cudnn.benchmark = False 
        torch.backends.cudnn.deterministic = True  
    else:
        torch.backends.cudnn.enabled = False 
        
        
def Monte_Carlo_CrossValidation(args, dataset_df):
    # setup the parateters
    num_fold = args["num_fold"]
    test_size = args["test_size"]
    seed = args["seed"]
    
    # setup the patient-level dataframe
    patient_df = dataset_df.drop_duplicates()
    
    # setup the monte carlo cross-validation
    stratified_split = StratifiedShuffleSplit(n_splits=num_fold, test_size=test_size, random_state=seed)
    if 'Class' in patient_df.columns:
        split_key = 'Class'
    else:
        split_key = 'Survival_Interval'
    monte_carlo_cv_patient = stratified_split.split(patient_df, patient_df[split_key])
    
    # setup the slide-level monte carlo cross-validation
    monte_carlo_cv_dfs = []
    for train_index, test_index in monte_carlo_cv_patient:
        train_patients = patient_df.iloc[train_index]
        test_patients = patient_df.iloc[test_index]
        
        valid_size = args["valid_size"]  
        stratified_split_train = StratifiedShuffleSplit(n_splits=1, test_size=valid_size, random_state=42)

        for train_sub_index, valid_index in stratified_split_train.split(train_patients, train_patients[split_key]):
            train_sub_patients = train_patients.iloc[train_sub_index]
            valid_patients = train_patients.iloc[valid_index]
        
        train_slides = dataset_df[dataset_df['Patient_ID'].isin(train_sub_patients['Patient_ID'])]
        valid_slides = dataset_df[dataset_df['Patient_ID'].isin(valid_patients['Patient_ID'])]
        test_slides = dataset_df[dataset_df['Patient_ID'].isin(test_patients['Patient_ID'])]
        
        monte_carlo_cv_dfs.append([train_slides, valid_slides, test_slides])
    
    # return the monte carlo cross-validation
    return monte_carlo_cv_dfs


def make_metrics_dict(train_loss, train_auc, train_acc, train_f1,
                      valid_loss, valid_auc, valid_acc, valid_f1,
                      test_loss, test_auc, test_acc, test_f1,
                      current_fold):
    metrics_dict = {
        f"Fold_{current_fold}/Train_Loss": train_loss,
        f"Fold_{current_fold}/Train_AUC": train_auc,
        f"Fold_{current_fold}/Train_Accuracy": train_acc,
        f"Fold_{current_fold}/Train_F1": train_f1,
        f"Fold_{current_fold}/Valid_Loss": valid_loss,
        f"Fold_{current_fold}/Valid_AUC": valid_auc,
        f"Fold_{current_fold}/Valid_Accuracy": valid_acc,
        f"Fold_{current_fold}/Valid_F1": valid_f1,
        f"Fold_{current_fold}/Test_Loss": test_loss,
        f"Fold_{current_fold}/Test_AUC": test_auc,
        f"Fold_{current_fold}/Test_Accuracy": test_acc,
        f"Fold_{current_fold}/Test_F1": test_f1,
    }
    
    return metrics_dict