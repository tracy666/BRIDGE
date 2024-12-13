import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms
from glob import glob
import sys, argparse, os, glob, copy
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from sklearn.utils import shuffle
from tqdm import tqdm
import json
import os
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download
import torch.nn.functional as F



class BagDataset():
    def __init__(self, imgs_list, transform=None):
        self.imgs_list = imgs_list
        self.transform = transform
        
    def __len__(self):
        return len(self.imgs_list)
    
    def __getitem__(self, idx):
        img_path = self.imgs_list[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img_path.split(os.path.sep)[-1].split(".")[0], img 
    

def bag_dataset(args, imgs_list):
    transformed_dataset = BagDataset(imgs_list=imgs_list,
                                     transform=transforms.Compose([
                                            transforms.Resize(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                            ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)


def compute_feats(args, slide_list, feature_extractor, save_path=None):
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    
    with torch.no_grad():
        for slide in tqdm(slide_list):
            imgs_list = glob.glob(os.path.join(slide, '*.jpeg'))
            if len(imgs_list) == 0:
                print('No valid patch extracted from: ' + slide.split(os.path.sep)[-1])
                continue
            
            dataloader, bag_size = bag_dataset(args, imgs_list)
            feats_list = []
            patch_id_list = []
            for img_path, img in dataloader:
                img = img.cuda()
                feats = feature_extractor(img)
                feats = feats.cpu()
                feats_list.extend(feats)
                patch_id_list.extend(img_path)
            
            feats_all = torch.stack(feats_list, dim=0)
            
            torch.save(feats_all, os.path.join(save_path, slide.split(os.path.sep)[-1] + '_feats.pt'))
            json.dump(patch_id_list, open(os.path.join(save_path, slide.split(os.path.sep)[-1] + '_patch_id.json'), 'w'))


def main():
    parser = argparse.ArgumentParser(description='Compute TCGA features from different feature extractor')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--batch_size', default=512, type=int, help='Batch size of dataloader [128]')
    parser.add_argument('--num_workers', default=16, type=int, help='Number of threads for datalodaer')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(3,), help='GPU ID(s) [1]')
    parser.add_argument('--dataset_path', default="/disk2/wqzhao/TCGA/WSI/BRCA/single/BRCA", 
                        type=str, help='The path of extracted dataset, contains patches for different slides')
    parser.add_argument('--save_path', default="/disk2/wqzhao/TCGA/TCGA_processed_data/BRCA_dinov2_vits14", 
                        type=str, help='The path to save the extracted features')
    parser.add_argument('--feature_extractor', default='clip-vit-base-patch32', type=str,
                        choices=['dinov2_vits14', 'CTransPath', 'UNI', 'HIPT', 'CONCH', 'PLIP', 
                                 'Quilt1M', 'BRIDGE_Imgfeat_40ckpt', 'BRIDGE_Genepred_40ckpt', 'BRIDGE_ImgandGene_40ckpt',
                                 'BRIDGE_Genepred_49ckpt', 'BRIDGE_Genepred_49ckpt_6loss',
                                 'QuiltNet-B-16-PMB', 'QuiltNet-B-32', 'PathClip', 'clip-vit-base-patch32'], 
                        help='The feature extractor to use [dinov2_vits14]')
    args = parser.parse_args()
    
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    slide_list = glob.glob(os.path.join(args.dataset_path, '*'))
    
    if args.feature_extractor == 'dinov2_vits14':
        feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    
    elif args.feature_extractor == 'UNI':
        login("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")  # login with your User Access Token, found at https://huggingface.co/settings/tokens
        local_dir = "./assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
        os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
        hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
        feature_extractor = model
    
    # To use CTransPath, please download the Github repository from https://github.com/Xiyue-Wang/TransPath
    elif args.feature_extractor == 'CTransPath':
        from TransPath.ctran import ctranspath
        model = ctranspath()
        model.head = nn.Identity()
        td = torch.load(r'./TransPath/model_weight/ctranspath.pth')
        model.load_state_dict(td['model'], strict=True)
        feature_extractor = model
    
    # To use CONCH, please download the Github repository from https://github.com/mahmoodlab/CONCH
    elif args.feature_extractor == 'CONCH':
        from conch.open_clip_custom import create_model_from_pretrained
        model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token="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")
        class FeatureExtractor(nn.Module):
            def __init__(self, model):
                super(FeatureExtractor, self).__init__()
                self.model = model
            def forward(self, image):
                return self.model.encode_image(image, proj_contrast=False, normalize=False)
        feature_extractor = FeatureExtractor(model)
    
    # To use HIPT, please download the Github repository from https://github.com/mahmoodlab/HIPT
    elif args.feature_extractor == 'HIPT':
        from HIPT.HIPT_4K.hipt_model_utils import get_vit256
        model256 = get_vit256(pretrained_weights='./HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth')
        feature_extractor = model256
    
    # To use PLIP, please download the Github repository from https://github.com/PathologyFoundation/plip
    elif args.feature_extractor == 'PLIP':
        from plip.plip import PLIP
        model = PLIP('vinid/plip')
        class FeatureExtractor(nn.Module):
            def __init__(self, model):
                super(FeatureExtractor, self).__init__()
                self.model = model
            def forward(self, image):
                return self.model.model.get_image_features(image)
        feature_extractor = FeatureExtractor(model)
    
    elif args.feature_extractor == 'Quilt1M':
        import open_clip
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-16')
        class FeatureExtractor(nn.Module):
            def __init__(self, model):
                super(FeatureExtractor, self).__init__()
                self.model = model
            def forward(self, image):
                return self.model.encode_image(image)
        feature_extractor = FeatureExtractor(model)
    
    elif args.feature_extractor == "QuiltNet-B-16-PMB":
        import open_clip
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-16-PMB')
        class FeatureExtractor(nn.Module):
            def __init__(self, model):
                super(FeatureExtractor, self).__init__()
                self.model = model
            def forward(self, image):
                return self.model.encode_image(image)
        feature_extractor = FeatureExtractor(model)
    
    elif args.feature_extractor == "QuiltNet-B-32":
        import open_clip
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
        class FeatureExtractor(nn.Module):
            def __init__(self, model):
                super(FeatureExtractor, self).__init__()
                self.model = model
            def forward(self, image):
                return self.model.encode_image(image)
        feature_extractor = FeatureExtractor(model)
    
    elif args.feature_extractor == "PathClip":
        import open_clip
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-B-16', 
        pretrained='/disk2/wqzhao/Benchmarking_Path_CLIP/pathclip/pathclip-base.pt',
        cache_dir='/disk2/wqzhao/Benchmarking_Path_CLIP/pathclip/cache_dir', 
        force_quick_gelu=True)
        class FeatureExtractor(nn.Module):
            def __init__(self, model):
                super(FeatureExtractor, self).__init__()
                self.model = model
            def forward(self, image):
                return self.model.encode_image(image)
        feature_extractor = FeatureExtractor(model)
    
    elif args.feature_extractor == 'clip-vit-base-patch32':
        from transformers import AutoTokenizer, CLIPModel, AutoProcessor
        model = CLIPModel.from_pretrained(f"openai/{args.feature_extractor}")
        class FeatureExtractor(nn.Module):
            def __init__(self, model):
                super(FeatureExtractor, self).__init__()
                self.model = model
            def forward(self, image):
                return self.model.get_image_features(image)
        feature_extractor = FeatureExtractor(model)
    
    elif args.feature_extractor == "BRIDGE_Genepred":
        sys.path.append("/home/zliang/BRIDGE_code/models")
        from BRIDGE_model import BRIDGEModel
        pretrained_BRIDGE_model = BRIDGEModel.load_from_checkpoint(
            "/data1/zliang/0000_0901_finalized_checkpoints/BRIDGE/multi_organ/fifth_version_6loss_DenseNet_TabNet_dim_128_BRIDGE_2024_08_29_07_40_36_all/49.ckpt",
            strict=False,
        )
        selected_indices = pd.read_csv("/home/zliang/BRIDGE_code/downstream_tasks/Part4_Survival_Analysis/core/top_HVG_genes_indices_number_1000.csv", index_col=0).index.tolist()
        class Gene_Predictor(nn.Module):
            def __init__(self, image_encoder, image_to_gene_decoder, selected_indices):
                super(Gene_Predictor, self).__init__()
                self.image_encoder = image_encoder
                self.image_to_gene_decoder = image_to_gene_decoder
                self.selected_indices = selected_indices
                
            def forward(self, image):
                original_image_feature = self.image_encoder(image)
                image_to_gene_prediction = self.image_to_gene_decoder(original_image_feature)
                image_to_gene_prediction = image_to_gene_prediction[:, self.selected_indices]
                return image_to_gene_prediction
        feature_extractor = Gene_Predictor(pretrained_BRIDGE_model.image_encoder, pretrained_BRIDGE_model.image_to_gene_decoder, selected_indices=selected_indices)

    
    feature_extractor.cuda()
    feature_extractor.eval()
    
    compute_feats(args, slide_list, feature_extractor, save_path=args.save_path)
    

if __name__ == '__main__':
    main()