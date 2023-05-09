import torch
import os
import numpy as np
import pandas as pd
import argparse
from src.dataset import DatasetFor0D
from torch.utils.data import DataLoader, RandomSampler
from src.utils.sampler import ImbalancedDatasetSampler
from src.utils.utility import preparing_0D_dataset, plot_learning_curve, generate_prob_curve_from_0D, seed_everything
from src.models.BNN import BayesianTransformer, compute_ensemble_probability
from src.config import Config

config = Config()
ts_cols = config.input_features

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="training disruption prediction model with 0D data for Bayesian approach")
    
    # random seed
    parser.add_argument("--random_seed", type = int, default = 42)
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "BayesianTransformer")
    parser.add_argument("--save_dir", type = str, default = "./results")
    
    # test shot for disruption probability curve
    parser.add_argument("--test_shot_num", type = int, default = 21310)

    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)
    
    # scaler type
    parser.add_argument("--scaler", type = str, choices=['Robust', 'Standard', 'MinMax', 'None'], default = "None")
    
    # batch size / sequence length / epochs / distance / num workers / pin memory use
    parser.add_argument("--batch_size", type = int, default = 256)
    parser.add_argument("--num_epoch", type = int, default = 128)
    parser.add_argument("--seq_len", type = int, default = 16)
    parser.add_argument("--dist", type = int, default = 2)
    parser.add_argument("--num_workers", type = int, default = 4)
    parser.add_argument("--pin_memory", type = bool, default = True)
    
    # optimizer : SGD, RMSProps, Adam, AdamW
    parser.add_argument("--optimizer", type = str, default = "AdamW", choices=["SGD","RMSProps","Adam","AdamW"])
    
    # learning rate, step size and decay constant
    parser.add_argument("--lr", type = float, default = 2e-4)
    parser.add_argument("--use_scheduler", type = bool, default = True)
    parser.add_argument("--step_size", type = int, default = 4)
    parser.add_argument("--gamma", type = float, default = 0.95)
    
    # early stopping
    parser.add_argument('--early_stopping', type = bool, default = True)
    parser.add_argument("--early_stopping_patience", type = int, default = 32)
    parser.add_argument("--early_stopping_verbose", type = bool, default = True)
    parser.add_argument("--early_stopping_delta", type = float, default = 1e-3)

    # imbalanced dataset processing
    # Re-sampling
    parser.add_argument("--use_sampling", type = bool, default = False)
    
    # Re-weighting
    parser.add_argument("--use_weighting", type = bool, default = False)
    
    # loss type : CE, Focal, LDAM
    parser.add_argument("--loss_type", type = str, default = "Focal", choices = ['CE','Focal', 'LDAM'])
    
    # label smoothing
    parser.add_argument("--use_label_smoothing", type = bool, default = False)
    parser.add_argument("--smoothing", type = float, default = 0.05)
    parser.add_argument("--kl_weight", type = float, default = 0.2)
    
    # LDAM Loss parameter
    parser.add_argument("--max_m", type = float, default = 0.5)
    parser.add_argument("--s", type = float, default = 1.0)
    
    # Focal Loss parameter
    parser.add_argument("--focal_gamma", type = float, default = 2.0)
    
    # monitoring the training process
    parser.add_argument("--verbose", type = int, default = 4)
    
    # model setup : transformer
    parser.add_argument("--alpha", type = float, default = 0.01)
    parser.add_argument("--dropout", type = float, default = 0.1)
    parser.add_argument("--kernel_size", type = int, default = 5)
    parser.add_argument("--feature_dims", type = int, default = 128)
    parser.add_argument("--n_layers", type = int, default = 4)
    parser.add_argument("--n_heads", type = int, default = 8)
    parser.add_argument("--dim_feedforward", type = int, default = 512)
    parser.add_argument("--cls_dims", type = int, default = 128)
    
    # Bayesian classifier
    parser.add_argument("--prior_pi", type = float, default = 0.5)
    parser.add_argument("--prior_sigma1", type = float, default = 1.0)
    parser.add_argument("--prior_sigma2", type = float, default = 0.0025)
    
    args = vars(parser.parse_args())

    return args

# torch device state
print("================= device setup =================")
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()


if __name__ == "__main__":

    args = parsing()
    
    # seed initialize
    seed_everything(args['random_seed'], False)
    
    # save directory
    save_dir = args['save_dir']
    
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    if not os.path.isdir("./weights"):
        os.mkdir("./weights")
        
    if not os.path.isdir("./runs"):
        os.mkdir("./runs")
    
    # tag : {model_name}_clip_{seq_len}_dist_{pred_len}_{Loss-type}_{Boosting-type}
    loss_type = args['loss_type']
    
    if args['use_label_smoothing']:
        loss_type = "{}_smoothing".format(loss_type)
    
    if args['use_sampling'] and not args['use_weighting']:
        boost_type = "RS"
    elif args['use_sampling'] and args['use_weighting']:
        boost_type = "RS_RW"
    elif not args['use_sampling'] and args['use_weighting']:
        boost_type = "RW"
    elif not args['use_sampling'] and not args['use_weighting']:
        boost_type = "Normal"
        
    if args['scaler'] == 'None':
        scale_type = "no_scaling"
    else:
        scale_type = args['scaler']
    
    tag = "{}_clip_{}_dist_{}_{}_{}_{}_seed_{}".format(args["tag"], args["seq_len"], args["dist"], loss_type, boost_type, scale_type, args['random_seed'])
    print("================= Running code =================")
    print("Setting : {}".format(tag))
    
    save_best_dir = "./weights/{}_best.pt".format(tag)
    save_last_dir = "./weights/{}_last.pt".format(tag)
    exp_dir = os.path.join("./runs/", "tensorboard_{}".format(tag))
 
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(args["gpu_num"])
    else:
        device = 'cpu'
        
    # dataset setup
    ts_train, ts_valid, ts_test, ts_scaler = preparing_0D_dataset("./dataset/KSTAR_Disruption_ts_data_extend.csv", ts_cols = ts_cols, scaler = args['scaler'])
    kstar_shot_list = pd.read_csv('./dataset/KSTAR_Disruption_Shot_List_2022.csv', encoding = "euc-kr")

    train_data = DatasetFor0D(ts_train, kstar_shot_list, seq_len = args['seq_len'], cols = ts_cols, dist = args['dist'], dt = 0.02, scaler = ts_scaler)
    valid_data = DatasetFor0D(ts_valid, kstar_shot_list, seq_len = args['seq_len'], cols = ts_cols, dist = args['dist'], dt = 0.02, scaler = ts_scaler)
    test_data = DatasetFor0D(ts_test, kstar_shot_list, seq_len = args['seq_len'], cols = ts_cols, dist = args['dist'], dt = 0.02, scaler = ts_scaler)
    
    # define model
    model = BayesianTransformer(
        n_features=len(ts_cols),
        kernel_size = args['kernel_size'],
        feature_dims = args['feature_dims'],
        max_len = args['seq_len'],
        n_layers = args['n_layers'],
        n_heads = args['n_heads'],
        dim_feedforward=args['dim_feedforward'],
        dropout = args['dropout'],
        cls_dims = args['cls_dims'],
        n_classes = 2,
        prior_pi  = args['prior_pi'], 
        prior_sigma1 = args['prior_sigma1'], 
        prior_sigma2 = args['prior_sigma2']
    )
    
    print("\n==================== model summary ====================\n")
    model.summary()
    model.to(device)

    # Re-sampling
    if args["use_sampling"]:
        train_sampler = ImbalancedDatasetSampler(train_data)
        valid_sampler = RandomSampler(valid_data)
        test_sampler = RandomSampler(test_data)

    else:
        train_sampler = RandomSampler(train_data)
        valid_sampler = RandomSampler(valid_data)
        test_sampler = RandomSampler(test_data)
       
        
    model.load_state_dict(torch.load(save_best_dir))
    
    # experiment
    test_loader = DataLoader(test_data, batch_size = 1, sampler=test_sampler, num_workers = 1, pin_memory=args["pin_memory"])

    test_data = None
    test_label = None
    
    for idx, (data, label) in enumerate(test_loader):
        if label.numpy() == 0:
            test_data = data
            test_label = label
            break
    
    test_pred = compute_ensemble_probability(model, test_data, device, n_samples = 128)
    print("test_pred : ", test_pred)
    print("true label : ", test_label)
    
    preds_disrupt = test_pred[:,0]
    preds_normal = test_pred[:,1]
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2)
    axes[0].hist(preds_disrupt.reshape(-1,), bins = 32, color = 'lightgray')
    axes[1].hist(preds_normal.reshape(-1,), bins = 32, color = 'lightgray')
    
    axes[0].set_xlabel("Output(probs)")
    axes[0].set_xlim([0,1.0])
    axes[0].set_ylabel('n-samples')
    axes[0].set_title("Disruption")
    
    axes[1].set_xlabel("Output(probs)")
    axes[1].set_xlim([0,1.0])
    axes[1].set_ylabel('n-samples')
    axes[1].set_title('Normal')
    
    fig.tight_layout()
    plt.savefig("./test.png")
    
    for idx, (data, label) in enumerate(test_loader):
        if label.numpy() == 1:
            test_data = data
            test_label = label
            break