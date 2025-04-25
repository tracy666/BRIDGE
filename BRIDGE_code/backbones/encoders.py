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
    model256 = vits.__dict__[arch](patch_size=16, num_classes=0)
    
    # for p in model256.parameters(): # iterates over all the parameters
    #     p.requires_grad = False # sets their requires_grad attribute to False
    # # This effectively freezes the parameters, preventing them from being updated during training
    # model256.eval()
    # model256.to(device) # move it to the specified device (GPU or CPU)

    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")

        if checkpoint_key is not None and checkpoint_key in state_dict:
            state_dict = state_dict[checkpoint_key]
    
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model256.load_state_dict(state_dict, strict=False)
        
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


def main_ctranspath():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_data_storage", type=str, default="/data1/zliang")
    parser.add_argument("--raw_data_folder_name", type=str, default="Histo_ST_raw")
    parser.add_argument('--gpu_cards', type=str, default='', help='Comma-separated list of GPU card numbers')
    args = parser.parse_args()

    gpu_cards = args.gpu_cards.split(",") if args.gpu_cards else []
    gpu_cards_str = ",".join(gpu_cards)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_cards_str
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    randomized_image_batch = torch.rand((2, 3, 224, 224))

    HIPT_image_encoder = ImageEncoder(
        model_name="HIPT",
        latent_embedding_dimension=256,
        main_data_storage=args.main_data_storage,
        raw_data_folder_name=args.raw_data_folder_name,
        device=device,
    )
    print(f"The feature size of HIPT: {HIPT_image_encoder.feature_size}")
    HIPT_out = HIPT_image_encoder(randomized_image_batch)
    print(f"HIPT_out size: {HIPT_out.size()}")
    HIPT_forward_out = HIPT_image_encoder.forward_head(HIPT_out)
    print(f"HIPT_forward_out size: {HIPT_forward_out.size()}")

    # We need to change to conda env bridge_ctranspath to run the code here
    CTransPath_image_encoder = ImageEncoder(
        model_name="CTransPath",
        latent_embedding_dimension=256,
        main_data_storage=args.main_data_storage,
        raw_data_folder_name=args.raw_data_folder_name,
        device=device,
    )
    print(f"The feature size of CTransPath: {CTransPath_image_encoder.feature_size}")
    CTransPath_out = CTransPath_image_encoder(randomized_image_batch)
    print(f"CTransPath_out size: {CTransPath_out.size()}")
    CTransPath_forward_out = CTransPath_image_encoder.forward_head(CTransPath_out)
    print(f"CTransPath_forward_out size: {CTransPath_forward_out.size()}")

    ResNet50_image_encoder = ImageEncoder(
        model_name="ResNet50",
        latent_embedding_dimension=256,
        main_data_storage=args.main_data_storage,
        raw_data_folder_name=args.raw_data_folder_name,
        device=device,
    )
    print(f"The feature size of ResNet50: {ResNet50_image_encoder.feature_size}")
    ResNet50_out = ResNet50_image_encoder(randomized_image_batch)
    print(f"ResNet50_out size: {ResNet50_out.size()}")
    ResNet50_forward_out = ResNet50_image_encoder.forward_head(ResNet50_out)
    print(f"ResNet50_forward_out size: {ResNet50_forward_out.size()}")

    DenseNet121_image_encoder = ImageEncoder(
        model_name="DenseNet121",
        latent_embedding_dimension=256,
        main_data_storage=args.main_data_storage,
        raw_data_folder_name=args.raw_data_folder_name,
        device=device,
    )
    print(f"The feature size of DenseNet121: {DenseNet121_image_encoder.feature_size}")
    DenseNet121_out = DenseNet121_image_encoder(randomized_image_batch)
    print(f"DenseNet121_out size: {DenseNet121_out.size()}")
    DenseNet121_forward_out = DenseNet121_image_encoder.forward_head(DenseNet121_out)
    print(f"DenseNet121_forward_out size: {DenseNet121_forward_out.size()}")
    print(DenseNet121_image_encoder)

    DINOv1_image_encoder = ImageEncoder(
        model_name="DINOv1",
        latent_embedding_dimension=256,
        main_data_storage=args.main_data_storage,
        raw_data_folder_name=args.raw_data_folder_name,
        device=device,
    )
    DINOv1_out = DINOv1_image_encoder(randomized_image_batch)
    print(f"DINOv1_out size: {DINOv1_out.size()}")
    DINOv1_forward_out = DINOv1_image_encoder.forward_head(DINOv1_out)
    print(f"DINOv1_forward_out size: {DINOv1_forward_out.size()}")

    DINOv2_image_encoder = ImageEncoder(
        model_name="DINOv2",
        latent_embedding_dimension=256,
        main_data_storage=args.main_data_storage,
        raw_data_folder_name=args.raw_data_folder_name,
        device=device,
    )
    DINOv2_out = DINOv2_image_encoder(randomized_image_batch)
    print(f"DINOv2_out size: {DINOv2_out.size()}")
    DINOv2_forward_out = DINOv2_image_encoder.forward_head(DINOv2_out)
    print(f"DINOv2_forward_out size: {DINOv2_forward_out.size()}")
    

def main_uni():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_data_storage", type=str, default="/data1/zliang")
    parser.add_argument("--raw_data_folder_name", type=str, default="Histo_ST_raw")
    parser.add_argument('--gpu_cards', type=str, default='', help='Comma-separated list of GPU card numbers')
    args = parser.parse_args()

    gpu_cards = args.gpu_cards.split(",") if args.gpu_cards else []
    gpu_cards_str = ",".join(gpu_cards)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_cards_str
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    randomized_image_batch = torch.rand((2, 3, 224, 224))
    
    # We need to change to conda env bridge_uni to run the code here
    UNI_image_encoder = ImageEncoder(
        model_name="UNI",
        latent_embedding_dimension=256,
        main_data_storage=args.main_data_storage,
        raw_data_folder_name=args.raw_data_folder_name,
        device=device,
    )
    UNI_out = UNI_image_encoder(randomized_image_batch)
    print(f"UNI_out size: {UNI_out.size()}")
    UNI_forward_out = UNI_image_encoder.forward_head(UNI_out)
    print(f"UNI_forward_out size: {UNI_forward_out.size()}")
    
    total_params = sum(p.numel() for p in UNI_image_encoder.parameters())
    print(f"Number of parameters: {total_params}")


def main_gene():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_data_storage", type=str, default="/data1/zliang")
    parser.add_argument("--raw_data_folder_name", type=str, default="Histo_ST_raw")
    parser.add_argument('--gpu_cards', type=str, default='', help='Comma-separated list of GPU card numbers')
    args = parser.parse_args()

    gpu_cards = args.gpu_cards.split(",") if args.gpu_cards else []
    gpu_cards_str = ",".join(gpu_cards)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_cards_str
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    randomized_gene_batch = torch.rand((2, 250))
    MLP_gene_encoder = GeneEncoder(
        model_name="MLP",
        input_gene_dimension=randomized_gene_batch.shape[1],
        output_dimension=512,
        latent_embedding_dimension=128,
        main_data_storage=args.main_data_storage,
        raw_data_folder_name=args.raw_data_folder_name,
        device=device,
    )
    MLP_out = MLP_gene_encoder(randomized_gene_batch)
    print(f"MLP_out size: {MLP_out.size()}")
    MLP_forward_out = MLP_gene_encoder.forward_head(MLP_out)
    print(f"MLP_forward_out size: {MLP_forward_out.size()}")

    TabNet_gene_encoder = GeneEncoder(
        model_name="TabNet",
        input_gene_dimension=randomized_gene_batch.shape[1],
        output_dimension=512,
        latent_embedding_dimension=128,
        main_data_storage=args.main_data_storage,
        raw_data_folder_name=args.raw_data_folder_name,
        device=device,
    )
    TabNet_out = TabNet_gene_encoder(randomized_gene_batch)
    print(f"TabNet_out size: {TabNet_out.size()}")
    TabNet_forward_out = TabNet_gene_encoder.forward_head(TabNet_out)
    print(f"TabNet_forward_out size: {TabNet_forward_out.size()}")

    scGPT_gene_encoder = GeneEncoder(
        model_name="scGPT",
        input_gene_dimension=randomized_gene_batch.shape[1],
        output_dimension=512,
        latent_embedding_dimension=128,
        main_data_storage=args.main_data_storage,
        raw_data_folder_name=args.raw_data_folder_name,
        device=device,
    ).to(device)
    scGPT_gene_encoder = scGPT_gene_encoder.half() # Convert model parameters to float16
    for name, parameters in scGPT_gene_encoder.named_parameters():
        print(f"Layer: {name}, Parameters Data Type: {parameters.dtype} for model scGPT.")
    gene_ids = torch.randint(low=0, high=100, size=(2, 100), dtype=torch.long).to(device)
    gene_values = torch.rand(2, 100, dtype=torch.float16).to(device)
    scGPT_out = scGPT_gene_encoder((gene_ids, gene_values))
    print(f"scGPT_out 0 size: {scGPT_out[0].size()}")
    print(f"scGPT_out 1 size: {scGPT_out[1].size()}")
    scGPT_forward_out = scGPT_gene_encoder.forward_head(scGPT_out[0])
    print(f"scGPT_forward_out size: {scGPT_forward_out.size()}")
    
    
if __name__ == "__main__":
    main_ctranspath()
    # main_uni()
    main_gene()