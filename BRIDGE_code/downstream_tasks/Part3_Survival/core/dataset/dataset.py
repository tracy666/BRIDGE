import os
import torch
from torch.utils.data import Dataset
import pandas as pd


class SlideDataset_Classification(Dataset):
    def __init__(self, features_file, features_folder, classes, n_tokens=512, selected_indices=None):
        self.features_file = features_file
        self.features_folder = features_folder
        self.slide_classes = classes
        self.n_tokens = n_tokens
        self.n_slides = len(features_file)
        self.selected_indices = selected_indices

    def __len__(self):
        return self.n_slides

    def __getitem__(self, index):
        slide_class = self.slide_classes[index]
        slide_id = self.features_file[index].split('_')[0]
        
        slide_features = torch.load(os.path.join(self.features_folder, self.features_file[index]))
        
        patch_indices = torch.randint(0, slide_features.shape[0], (self.n_tokens,))
        sampled_features = slide_features[patch_indices]
        
        if self.selected_indices is not None:
            sampled_features = sampled_features[:, 0:len(self.selected_indices)]
        
        return slide_id, sampled_features, slide_class


class RNADataset_Classification(Dataset):
    def __init__(self, RNA_df, slide_list, classes):
        self.RNA_df = RNA_df
        self.slide_list = slide_list
        self.slide_classes = classes
        
    def __len__(self):
        return len(self.slide_list)
    
    def __getitem__(self, index):
        slide_id = self.slide_list[index]
        slide_class = self.slide_classes[index]
        slide_RNA = torch.tensor(self.RNA_df.loc[slide_id[:12]+"-01"].values, dtype=torch.float32)
        return slide_id, slide_RNA, slide_class
    

class SlideDataset_Survival(Dataset):
    def __init__(self, features_file, features_folder, censor, survival_interval, event_time, n_tokens=512, selected_indices=None):
        self.features_file = features_file
        self.features_folder = features_folder
        self.censor = censor
        self.event_time = event_time
        self.survival_interval = survival_interval
        self.n_tokens = n_tokens
        self.n_slides = len(features_file)
        self.selected_indices = selected_indices

    def __len__(self):
        return self.n_slides

    def __getitem__(self, index):
        slide_censor = self.censor[index]
        slide_event_time = self.event_time[index]
        slide_survival_interval = self.survival_interval[index]
        slide_id = self.features_file[index].split('_')[0]
        slide_features = torch.load(os.path.join(self.features_folder, self.features_file[index]))
        patch_indices = torch.randint(0, slide_features.shape[0], (self.n_tokens,))
        sampled_features = slide_features[patch_indices]
        
        if self.selected_indices is not None:
            sampled_features = sampled_features[:, 0:len(self.selected_indices)]
        
        return slide_id, sampled_features, slide_censor, slide_event_time, slide_survival_interval 


class RNADataset_Survival(Dataset):
    def __init__(self, RNA_df, slide_list, censor, survival_interval, event_time):
        self.RNA_df = RNA_df
        self.slide_list = slide_list
        self.censor = censor
        self.survival_interval = survival_interval
        self.event_time = event_time
    
    def __len__(self):
        return len(self.slide_list)
    
    def __getitem__(self, index):
        slide_id = self.slide_list[index]
        slide_censor = self.censor[index]
        slide_event_time = self.event_time[index]
        slide_survival_interval = self.survival_interval[index]
        # if slide_id[:12]+"-01" not in self.RNA_df.index:
        #     slide_RNA = torch.tensor(self.RNA_df.loc[slide_id[:12]+"-02"].values[:250], dtype=torch.float32)
        # else:
        #     slide_RNA = torch.tensor(self.RNA_df.loc[slide_id[:12]+"-01"].values[:250], dtype=torch.float32)
        if slide_id[:12]+"-01" not in self.RNA_df.index:
            slide_RNA = torch.tensor(self.RNA_df.loc[slide_id[:12]+"-02"].values, dtype=torch.float32)
        else:
            slide_RNA = torch.tensor(self.RNA_df.loc[slide_id[:12]+"-01"].values, dtype=torch.float32)
        return slide_id, slide_RNA, slide_censor, slide_event_time, slide_survival_interval

    
def setup_dataloader(args, dataset_df, shuffle=True, drop_last=False):
    # setup dataset and dataloader
    if args["top_gene_pred"] is not None:
        selected_indices = list(range(args["top_gene_pred"]))
    else:
        selected_indices = None
    
    if 'Class' in dataset_df.columns:
        if args["modality"] == "imaging":
            dataset = SlideDataset_Classification(
                features_file=dataset_df['Slide Feats File'].values,
                features_folder=args["features_folder"],
                classes=dataset_df['Class'].values,
                n_tokens=args["n_tokens"],
                selected_indices=selected_indices
            )
        elif args["modality"] == "rna":
            RNA_df = pd.read_csv(args["RNA_csv"], index_col=0)
            dataset = RNADataset_Classification(
                RNA_df=RNA_df,
                slide_list=dataset_df['Slide Feats File'].values,
                classes=dataset_df['Class'].values
            )
        elif args["modality"] == "_Bridge_enhanced_imaging":
            print("a")
        else:
            raise NotImplementedError("Only imaging/rna datasets are supported.")
    else:
        if args["modality"] == "imaging":
            dataset = SlideDataset_Survival(
                features_file=dataset_df['Slide Feats File'].values,
                features_folder=args["features_folder"],
                censor=dataset_df['Censor'].values, 
                survival_interval=dataset_df['Survival_Interval'].values, 
                event_time=dataset_df['Event_Time'].values,
                n_tokens=args["n_tokens"],
                selected_indices=selected_indices
            )
        elif args["modality"] == "rna":
            RNA_df = pd.read_csv(args["RNA_csv"], index_col=0)
            dataset = RNADataset_Survival(
                RNA_df=RNA_df,
                slide_list=dataset_df['Slide Feats File'].values,
                censor=dataset_df['Censor'].values, 
                survival_interval=dataset_df['Survival_Interval'].values, 
                event_time=dataset_df['Event_Time'].values
            )
        else:
            raise NotImplementedError("Only imaging/rna datasets are supported.")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args["batch_size"], shuffle=shuffle, num_workers=args["num_workers"], pin_memory=True, prefetch_factor=2, drop_last=drop_last)
    return dataloader


def setup_train_valid_test_loader(args, train_df, valid_df, test_df=None):
    if test_df is None:
        train_loader = setup_dataloader(args, train_df, shuffle=True, drop_last=True)
        valid_loader = setup_dataloader(args, valid_df, shuffle=False)
        return train_loader, valid_loader
    else:
        train_loader = setup_dataloader(args, train_df, shuffle=True, drop_last=True)
        valid_loader = setup_dataloader(args, valid_df, shuffle=False)
        test_loader = setup_dataloader(args, test_df, shuffle=False)
        return train_loader, valid_loader, test_loader