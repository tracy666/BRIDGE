import argparse


def process_args():

    parser = argparse.ArgumentParser(description='Configurations for TANGLE pretraining')

    #----> use bulk seq or imaging features
    parser.add_argument('--modality', type=str, default='imaging', choices=["rna", "imaging"], help='Modality: use bulk seq or imaging features')

    #-----> model args 
    parser.add_argument('--hidden_dim', type=int, default=768, help='Internal dim of ABMIL.')
    parser.add_argument('--activation', type=str, default='softmax', help='Activation function used in ABMIL attention weight agg (sigmoid or softmax).')
    parser.add_argument('--rna_model', type=str, default='SNN', choices=["SNN", "MLP"], help='Model used for RNA modality.')
    parser.add_argument('--mil_model', type=str, default='ABMIL', choices=["ABMIL", "MLPMIL"], help='Model used for Imaging modality.')

    #----> training args
    parser.add_argument('--num_fold', type=int, default=10, help='Number of Monte Carlo Cross-Validation.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Size of test set during Monte Carlo Cross-Validation.')
    parser.add_argument('--valid_size', type=float, default=0.25, help='Size of validation set during Monte Carlo Cross-Validation.')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--n_tokens', type=int, default=512, help='Number of patches to sample during training.')
    parser.add_argument('--gpu_devices', type=list, default=[2], help='List of GPUs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--warmup', type=bool, default=True, help='If doing warmup.')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs.')
    parser.add_argument('--end_learning_rate', type=float, default=1e-8, help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=3407, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs to train (default: 2)')
    parser.add_argument('--num_workers', type=int, default=8, help='number of cpu workers')
    parser.add_argument('--wandb_dir', type=str, default='/disk2/wqzhao/TCGA/Downstream_benchmarking/wandb', help='Wandb directory')
    parser.add_argument('--model_weight_dir', type=str, default='/disk2/wqzhao/TCGA/Downstream_benchmarking/model_weights', help='Model weight directory')
    parser.add_argument('--no_log', type=bool, default=False, help='If no logging to wandb and save model weights.')


    parser.add_argument('--temperature', type=float, default=0.01, help='InfoNCE temperature.')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay.')
    
    #----> dataset args
    parser.add_argument('--dataset_csv', type=str, default='/disk2/wqzhao/TCGA/Downstream_benchmarking/dataset_csv/Classification/BRCA_IDC_ILC_Subtyping.csv', help='Dataset csv file path')
    parser.add_argument('--features_folder', type=str, default='/disk2/wqzhao/TCGA/TCGA_processed_data/BRCA_CTransPath', help='Extracted slide feature path')
    parser.add_argument('--RNA_csv', type=str, default='/disk2/wqzhao/TCGA/TCGA_processed_clinical_and_rna/BRCA/data_mrna_seq_v2_rsem_log2p_hvg1000.csv', help='Extracted RNA feature path')
    parser.add_argument('--top_gene_pred', type=int, default=None, help='Number of top genes predictions used.')
    
    #----> inference args
    parser.add_argument('--runs', type=str, default=None, help='Runs to evaluate.')

    args = parser.parse_args()

    return args