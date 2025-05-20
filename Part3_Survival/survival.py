# --> General imports
import pandas as pd
from tqdm import tqdm
import os
import wandb
import datetime
from dateutil import tz 
import json

# --> Torch imports
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
import torch.nn as nn
import torchmetrics
from torchmetrics import MetricCollection
import torch

# --> internal imports 
from core.utils.process_args import process_args
from core.utils.learning import set_seed, Monte_Carlo_CrossValidation, make_metrics_dict
from core.dataset.dataset import setup_train_valid_test_loader
from core.model.abmil import ABMIL_Classifier, MLPMIL_Classifier
from core.model.snn import SNN
from core.model.mlp import MLP 
from core.utils.survival_loss import NLLSurvLoss, CrossEntropySurvLoss
from core.utils.c_index_metrics import C_index_metric


def train_loop(args, model, train_loader, optimizer, scheduler, scheduler_warmup, loss_fn, metrics, epoch, num_epochs, current_fold):
    model.train()
    train_loss = 0
    
    with tqdm(total=len(train_loader), desc=f"Current Fold {current_fold} | Training Epoch {epoch+1}/{num_epochs}", leave=False) as pbar:
        for batch_idx, (slide_id, slide_feats, slide_censor, slide_event_time, slide_survival_interval) in enumerate(train_loader):
            optimizer.zero_grad()
            
            slide_feats = slide_feats.cuda()
            slide_censor, slide_event_time, slide_survival_interval = slide_censor.cuda(), slide_event_time.cuda(), slide_survival_interval.cuda()
            
            logits = model(slide_feats)

            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            loss = loss_fn(hazards=hazards, S=S, Y=slide_survival_interval, c=slide_censor, alpha=0)
            
            loss.backward()
            optimizer.step()
            
            if epoch <= args["warmup_epochs"]:
                scheduler_warmup.step()
            else:
                scheduler.step()
            
            train_loss = train_loss + loss.item()
            
            slide_risk = -torch.sum(S, dim=1)
            metrics.update(risk=slide_risk.detach().cpu(), censor=slide_censor.detach().cpu(), event_time=slide_event_time.detach().cpu())
            
            pbar.update(1)
        
        train_loss /= len(train_loader)
        train_c_index = metrics.compute()
        metrics.reset()
        
        pbar.close()  
            
    return train_loss, train_c_index


def val_loop(args, model, valid_loader, loss_fn, metrics, epoch, num_epochs, current_fold, desc="Validation"):
    model.eval()
    valid_loss = 0
    
    with tqdm(total=len(valid_loader), desc=f"Current Fold {current_fold} | {desc} Epoch {epoch+1}/{num_epochs}", leave=False) as pbar:
        for batch_idx, (slide_id, slide_feats, slide_censor, slide_event_time, slide_survival_interval) in enumerate(valid_loader):
            slide_feats = slide_feats.cuda()
            slide_censor, slide_event_time, slide_survival_interval = slide_censor.cuda(), slide_event_time.cuda(), slide_survival_interval.cuda()
            
            with torch.no_grad():
                logits = model(slide_feats)

                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)
                loss = loss_fn(hazards=hazards, S=S, Y=slide_survival_interval, c=slide_censor, alpha=0)
                
                valid_loss = valid_loss + loss.item()
                
                slide_risk = -torch.sum(S, dim=1)
                metrics.update(risk=slide_risk.detach().cpu(), censor=slide_censor.detach().cpu(), event_time=slide_event_time.detach().cpu())
                
                pbar.update(1)
        
        valid_loss /= len(valid_loader)
        valid_c_index = metrics.compute()
        metrics.reset()
        
        pbar.close()
        
    return valid_loss, valid_c_index


if __name__ == "__main__":
    # setup args and seed
    args = process_args()
    args = vars(args)
    set_seed(args["seed"])
    
    # print necessary information about the task and feature extractor
    if args["modality"] == "imaging":
        task, feature_extractor = args["dataset_csv"].split('/')[-1].split('.')[0], args["features_folder"].split('/')[-1]
        if args["top_gene_pred"] is not None:
            feature_extractor = f"{feature_extractor}_top_{args['top_gene_pred']}"
        if args["mil_model"] == "MLPMIL":
            feature_extractor = f"{feature_extractor}_MLPMIL"
    elif args["modality"] == "rna":
        TCGA_dataset = args["dataset_csv"].split('/')[-1].split('.')[0].split('_')[0]
        rna_file = args['RNA_csv'].split("/")[-1].split('.')[0]
        task, feature_extractor = args["dataset_csv"].split('/')[-1].split('.')[0], f"{TCGA_dataset}_{rna_file}_{args['rna_model']}"
    else:
        raise NotImplementedError("Only imaging/rna modalities are supported.")
    
    # setup the wandb logger
    if args["no_log"] is False:
        current_time = datetime.datetime.now(tz.tzlocal()).strftime("%Y_%m_%d_%H_%M_%S")
        os.environ['WANDB_DIR'] = os.path.join(args["wandb_dir"], "dir")
        os.environ['WANDB_CACHE_DIR'] = os.path.join(args["wandb_dir"], "cache")
        os.environ['WANDB_CONFIG_DIR'] = os.path.join(args["wandb_dir"], "config")
        run = wandb.init(project=f"Bridge-Benchmarking-{task}",
                        name=f"{feature_extractor}-{current_time}", 
                        config=args)
        model_weight_dir = os.path.join(args["model_weight_dir"], f"{task}-{feature_extractor}-{current_time}")
        os.mkdir(model_weight_dir)
        fold_test_performance_list = list()
    
    # setup the gpu to be used
    gpu_ids = tuple(args["gpu_devices"])
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    # load the whole dataset csv
    dataset_df = pd.read_csv(args["dataset_csv"]).drop(columns=['Set'])
    
    # setup the Monte Carlo Cross-Validation
    monte_carlo_cv_dfs = Monte_Carlo_CrossValidation(args, dataset_df)
    
    # begin the Monte Carlo Cross-Validation
    print(f"Evaluating task: {task}, Using features extracted from: {feature_extractor}")
    current_fold = 1
    for train_df, valid_df, test_df in monte_carlo_cv_dfs:
        # setup the dataloaders
        train_df = pd.concat([train_df, test_df])
        train_loader, valid_loader = setup_train_valid_test_loader(args, train_df=train_df, valid_df=valid_df, test_df=None)
        
        # obtain the feats_dim and num_classes
        feats_dim = next(iter(train_loader))[1].shape[-1]
        num_classes = len(train_df['Survival_Interval'].unique())
        
        # setup the model
        if args["modality"] == "imaging":
            if args["mil_model"] == "ABMIL":
                model = ABMIL_Classifier(config=args, input_dim=feats_dim, num_classes=num_classes)
            elif args["mil_model"] == "MLPMIL":
                model = MLPMIL_Classifier(input_dim=feats_dim, hidden_dim=args["hidden_dim"], output_dim=num_classes)
            else:
                raise NotImplementedError("Only ABMIL/MLPMIL models are supported for imaging modality.")
        elif args["modality"] == "rna":
            if args["rna_model"] == "SNN":
                model = SNN(in_dim=feats_dim, out_dim=num_classes, hidden_dim=args["hidden_dim"], n_layers=4, dropout_prob=0.0)
            elif args["rna_model"] == "MLP":
                model = MLP(input_dim=feats_dim, hidden_dim=args["hidden_dim"], output_dim=num_classes)
            else:
                raise NotImplementedError("Only SNN/MLP models are supported for rna modality.")
        else:
            raise NotImplementedError("Only imaging/rna models are supported.")
        model.cuda()
        
        # setup optimizer
        optimizer = optim.AdamW(model.parameters(), lr=args["learning_rate"])
        
        # set up schedulers
        T_max = (args["epochs"] - args["warmup_epochs"]) * len(train_loader) if args["warmup"] else args["epochs"] * len(train_loader)
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=T_max,
            eta_min=args["end_learning_rate"]
        )
        if args["warmup"]:
            scheduler_warmup = LinearLR(
                optimizer, 
                start_factor=0.00001,
                total_iters=args["warmup_epochs"] * len(train_loader)
        )
        else:
            scheduler_warmup = None
        
        # setup loss function
        loss_fn = NLLSurvLoss()
        
        # setup metric calculation class
        metrics = C_index_metric()
        
        # main training loop
        best_valid_c_index = 0.0
        for epoch in range(args["epochs"]):
            # training, validation and testing loop
            train_loss, train_c_index = train_loop(args, model, train_loader, optimizer, scheduler, scheduler_warmup, loss_fn, metrics, epoch, args["epochs"], current_fold)
            valid_loss, valid_c_index = val_loop(args, model, valid_loader, loss_fn, metrics, epoch, args["epochs"], current_fold, desc="Validation")
            # test_loss, test_c_index = val_loop(args, model, test_loader, loss_fn, metrics, epoch, args["epochs"], current_fold, desc="Testing")
            
            # setup the metrics dictionary
            metrics_dict = {
                f"Fold_{current_fold}/Train_Loss": train_loss, f"Fold_{current_fold}/Train_C_Index": train_c_index,
                f"Fold_{current_fold}/Valid_Loss": valid_loss, f"Fold_{current_fold}/Valid_C_Index": valid_c_index,
                # f"Fold_{current_fold}/Test_Loss": test_loss, f"Fold_{current_fold}/Test_C_Index": test_c_index
            }
            
            # save the best model up to now
            if args["no_log"] is False:
                wandb.log(metrics_dict)
                if valid_c_index > best_valid_c_index:
                    best_valid_c_index = valid_c_index
                    best_model_path = os.path.join(model_weight_dir, f'best_valid_c_index_model_fold_{current_fold}.pth')
                    torch.save(model.state_dict(), best_model_path)
            else:
                print(metrics_dict)
        
        if args["no_log"] is False:
            # evaluate on the saved best model        
            best_model = model.load_state_dict(torch.load(best_model_path))
            test_loss, test_c_index = val_loop(args, model, valid_loader, loss_fn, metrics, epoch, args["epochs"], current_fold, desc="Testing")
            fold_test_performance_list.append(
                {
                    "fold": float(current_fold),
                    "valid_loss": float(test_loss),
                    "valid_c_index": float(test_c_index),
                }
            )
            
            # save the valid and test dataframes
            train_df.to_csv(os.path.join(model_weight_dir, f'train_fold_{current_fold}.csv'), index=False)
            valid_df.to_csv(os.path.join(model_weight_dir, f'valid_fold_{current_fold}.csv'), index=False)
            # test_df.to_csv(os.path.join(model_weight_dir, f'test_fold_{current_fold}.csv'), index=False) 
        
        # update the current fold value
        current_fold = current_fold + 1
    
    # log the test performance of all the folds
    if args["no_log"] is False:
        fold_test_performance_df = pd.DataFrame(fold_test_performance_list)
        mean_row, std_row = fold_test_performance_df.mean(), fold_test_performance_df.std()
        fold_test_performance_df.loc['mean'], fold_test_performance_df.loc['std'] = mean_row, std_row
        fold_test_performance_df.to_csv(os.path.join(model_weight_dir, 'fold_valid_performance.csv'), index=False)
        run.finish()