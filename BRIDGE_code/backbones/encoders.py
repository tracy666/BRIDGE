import vision_transformer as vits
import torch
import os
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, densenet121, DenseNet121_Weights
import timm
import argparse
from pytorch_tabnet.tab_network import TabNet
from pathlib import Path
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
import json

import warnings
warnings.filterwarnings("ignore", category=ResourceWarning) # ResourceWarning: Implicitly cleaning up <TemporaryDirectory>
warnings.filterwarnings('ignore', message='torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.')

# direct copy from https://github.com/mahmoodlab/HIPT/blob/master/HIPT_4K/hipt_model_utils.py#L39
def get_vit256(pretrained_weights, arch='vit_small', device=torch.device('cuda:0')):
    r"""
    Builds ViT-256 Model.
    
    Args:
    - pretrained_weights (str): Path to ViT-256 Model Checkpoint.
    - arch (str): Which model architecture.
    - device (torch): Torch device to save model.
    
    Returns:
    - model256 (torch.nn): Initialized model.
    """
    
    checkpoint_key = 'teacher'
    device = torch.device("cpu")
    # checkpoint_key is set to the string 'teacher', and 
    # device is explicitly set to 'cpu' regardless of the input parameter.
    model256 = vits.__dict__[arch](patch_size=16, num_classes=0)
    # creates an instance of the Vision Transformer (ViT) model with a patch size of 16 and zero output classes
    # The vits.__dict__[arch] syntax dynamically selects the ViT model class based on the arch parameter
    """
    def vit_small(patch_size=16, **kwargs):
        model = VisionTransformer(
            patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        return model
    """

    # 这里注释了，需要他更新参数
    # for p in model256.parameters(): # iterates over all the parameters
    #     p.requires_grad = False # sets their requires_grad attribute to False
    # # This effectively freezes the parameters, preventing them from being updated during training
    # model256.eval()
    # model256.to(device) # move it to the specified device (GPU or CPU)

    if os.path.isfile(pretrained_weights): # checks if the pretrained_weights file exists
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        # loads the state dictionary from the file using torch.load()
        # The map_location="cpu" argument ensures that the weights are loaded onto the CPU

        if checkpoint_key is not None and checkpoint_key in state_dict:
            # print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # If the checkpoint_key is not None and exists in the loaded state dictionary, 
        # it selects only the portion of the state dictionary corresponding to that key
        # default是只用teacher model的参数
    
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # done to handle cases where the pretrained weights were saved with a different model structure that included these prefixes
        msg = model256.load_state_dict(state_dict, strict=False)
        # The strict=False argument allows for partial loading of the weights, skipping any missing keys
        # print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        
    return model256

# direct copy from https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/modules.py#L48
class Projection_Head(nn.Module):
    def __init__(self, embedding_dim, projection_dim):
        super().__init__()
        self.projector = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projection = self.projector(x)
        x = self.gelu(projection)
        x = self.fc(x)
        x = x + projection
        x = self.layer_norm(x)
        return x
    
class ImageEncoder(nn.Module):
    def __init__(
            self,
            model_name: str,
            latent_embedding_dimension: int,
            main_data_storage: str,
            raw_data_folder_name: str,
            device,
    ):
        super().__init__()
        if model_name == "HIPT":
            self.model = get_vit256(
                pretrained_weights=os.path.join(main_data_storage, raw_data_folder_name, "Supplementary_data", "pretrained", model_name, "vit256_small_dino.pth"),
                arch='vit_small',
                device=device,
            )
            self.projection_head = Projection_Head(384, latent_embedding_dimension)
            self.feature_size = 384
        elif model_name == "CTransPath":
            from ctran import ctranspath
            self.model = ctranspath()
            self.model.head = nn.Identity()
            model_weights = torch.load(
                os.path.join(main_data_storage, raw_data_folder_name, "Supplementary_data", "pretrained", model_name, "ctranspath.pth"),
            )
            self.model.load_state_dict(model_weights["model"], strict=True)
            self.projection_head = Projection_Head(768, latent_embedding_dimension)
            self.feature_size = 768
        elif model_name == "ResNet50":
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Identity()
            self.projection_head = Projection_Head(2048, latent_embedding_dimension)
            self.feature_size = 2048
        elif model_name == "DenseNet121":
            self.model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            self.model.classifier = nn.Identity()
            self.projection_head = Projection_Head(1024, latent_embedding_dimension)
            self.feature_size = 1024
        elif model_name == "DINOv1":
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
            self.projection_head = Projection_Head(2048, latent_embedding_dimension)
            self.feature_size = 2048
        elif model_name == "DINOv2":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.projection_head = Projection_Head(384, latent_embedding_dimension)
            self.feature_size = 384
        elif model_name == "UNI":
            self.model = timm.create_model(
                "vit_large_patch16_224",
                img_size=224,
                patch_size=16,
                init_values=1e-5,
                num_classes=0,
                dynamic_img_size=True,
            )
            model_weights_path = os.path.join(main_data_storage, raw_data_folder_name, "Supplementary_data", "pretrained", model_name, "pytorch_model.bin")
            self.model.load_state_dict(torch.load(model_weights_path, map_location="cpu"), strict=True)
            self.projection_head = Projection_Head(1024, latent_embedding_dimension)
            self.feature_size = 1024
        elif model_name == "CONCH":
            from conch.open_clip_custom import create_model_from_pretrained
            model_weights_path = os.path.join(main_data_storage, raw_data_folder_name, "Supplementary_data", "pretrained", model_name, "pytorch_model.bin")
            full_model, preprocess = create_model_from_pretrained('conch_ViT-B-16', model_weights_path)
            class FeatureExtractor(nn.Module):
                def __init__(self, model):
                    super(FeatureExtractor, self).__init__()
                    self.model = model
                def forward(self, image):
                    return self.model.encode_image(image, proj_contrast=False, normalize=False)
            self.model = FeatureExtractor(full_model)
            self.projection_head = Projection_Head(512, latent_embedding_dimension)
            self.feature_size = 512
        else:
            raise NotImplementedError(f"Model {model_name} is not implemented as an image encoder.")
    
    def forward(
            self,
            images_tensor: torch.Tensor,
            retrieval_parallel: bool = False,
    ):
        model_out = self.model(images_tensor)
        if retrieval_parallel:
            return model_out, self.projection_head(model_out)
        return model_out
    
    def forward_head(
            self,
            x,
    ):
        model_forward_out = self.projection_head(x)
        return model_forward_out

class GeneEncoder(nn.Module):
    def __init__(
            self,
            model_name: str,
            input_gene_dimension: int,
            output_dimension: int,
            latent_embedding_dimension: int,
            main_data_storage: str,
            raw_data_folder_name: str,
            device,
    ):
        super().__init__()
        self.model_name = model_name
        if self.model_name == "TabNet":
            self.model = TabNet(
                input_gene_dimension,
                output_dimension,
            )
            self.projection_head = Projection_Head(output_dimension, latent_embedding_dimension)
        elif self.model_name == "scGPT":
            model_pretrained_files_path = Path(os.path.join(main_data_storage, raw_data_folder_name, "Supplementary_data", "pretrained", model_name, f"{model_name}_human"))
            model_config_file_path = model_pretrained_files_path / "args.json"
            model_weights_file_path = model_pretrained_files_path / "best_model.pt"
            model_vocab_file_path = model_pretrained_files_path / "vocab.json"

            scgpt_vocab = GeneVocab.from_file(model_vocab_file_path)
            special_tokens_list = ["<pad>", "<cls>", "<eoc>"]
            for special_token in special_tokens_list:
                if special_token not in scgpt_vocab:
                    scgpt_vocab.append(special_token) # if any of the special_tokens are missing from the vocabulary, they are added using vocab.append_token(s)
            
            with open(model_config_file_path, "r") as f:
                scgpt_human_configs = json.load(f) # opens the args.json file and loads its contents into model_configs
            
            # The default values are from https://github.com/bowang-lab/scGPT/blob/706526a76d547de4ed711fa028c99be5bdf6ad8a/scgpt/model/model.py#L28
            self.model = TransformerModel(
                ntoken = len(scgpt_vocab),
                d_model = scgpt_human_configs["embsize"],
                nhead = scgpt_human_configs["nheads"],
                d_hid = scgpt_human_configs["d_hid"],
                nlayers = scgpt_human_configs["nlayers"],
                nlayers_cls = 3,
                n_cls = 1,
                vocab = scgpt_vocab,
                dropout = 0.2, # The default value is 0.5, we modified to be 0.2
                pad_token = "<pad>",
                pad_value = -2, # The default value is 0, we modified to be -2
                do_mvc = False,
                do_dab= False,
                use_batch_labels = False,
                num_batch_labels = None,
                domain_spec_batchnorm = False,
                input_emb_style = "continuous",
                n_input_bins = 51,
                cell_emb_style = "cls",
                mvc_decoder_style = "inner product",
                ecs_threshold = 0., # The default value is 0.3, we modified to be 0
                explicit_zero_prob = False,
                use_fast_transformer = True, # The default value is False, we modified to be True
                fast_transformer_backend = "flash",
                pre_norm = False,
            )

            try:
                self.model.load_state_dict(torch.load(model_weights_file_path))
                print(f"Loading scGPT model from {model_weights_file_path}")
            except:
                # only load params that are in the model and match the size
                model_weights_dict = self.model.state_dict()
                pretrained_model_weights_dict = torch.load(model_weights_file_path)
                selected_pretrained_model_weights_dict = {
                    key: value
                    for key, value in pretrained_model_weights_dict.items()
                    if key in model_weights_dict and value.shape == model_weights_dict[key].shape
                }
                model_weights_dict.update(selected_pretrained_model_weights_dict)
                self.model.load_state_dict(model_weights_dict)
            
            self.pad_token_id = scgpt_vocab["<pad>"]
            self.projection_head = Projection_Head(output_dimension, latent_embedding_dimension)
        elif model_name == "MLP":
            self.model = Projection_Head(
                input_gene_dimension,
                output_dimension,
            )
            self.projection_head = Projection_Head(output_dimension, latent_embedding_dimension)
        else:
            raise NotImplementedError(f"Model {model_name} is not implemented as a gene encoder.")
    
    def forward(
            self,
            x,
            retrieval_parallel: bool = False,
    ):
        if self.model_name == "TabNet":
            model_out = self.model(x)[0]
            if retrieval_parallel:
                return model_out, self.projection_head(model_out)
            return model_out
        elif self.model_name == "scGPT":
            input_gene_ids, input_gene_values = x
            src_key_padding_mask = input_gene_ids.eq(self.pad_token_id)
            output_dict = self.model(
                input_gene_ids,
                input_gene_values,
                src_key_padding_mask=src_key_padding_mask,
                CLS=False,
            )
            if retrieval_parallel:
                return output_dict["cell_emb"], output_dict["mlm_output"], self.projection_head(output_dict["cell_emb"])
            return output_dict["cell_emb"], output_dict["mlm_output"]
        elif self.model_name == "MLP":
            if retrieval_parallel:
                return self.model(x), self.projection_head(self.model(x))
            return self.model(x)
        else:
            raise NotImplementedError(f"Model {self.model.name} is not implemented as a gene encoder.")
    
    def forward_head(
            self,
            x
    ):
        return self.projection_head(x)

