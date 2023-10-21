import torch
import os
import numpy as np
import pandas as pd
import argparse
from src.dataset import MultiSignalDataset
from torch.utils.data import DataLoader, RandomSampler
from src.utils.sampler import ImbalancedDatasetSampler
from src.utils.utility import preparing_0D_dataset, seed_everything
from src.loss import FocalLoss, LDAMLoss, CELoss, LabelSmoothingLoss
from src.models.predictor import Predictor
from src.models.tcn import calc_seq_length, calc_dilation
from src.config import Config
from typing import Dict

import logging
from functools import partial
from copy import deepcopy

# train and evaluation function
from src.hpo import train, train_DRW, evaluate

# hyperparameter tuning library
import ray
from ray import tune, air
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# ray-tune air : https://docs.ray.io/en/latest/tune/examples/tune-pytorch-cifar.html
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler

# hyperopt : bayesian method searching algorithm
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch
import warnings

# remove warning
warnings.filterwarnings("ignore")

config = Config()

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="training disruption prediction model with multi-signal data")
    
    # random seed
    parser.add_argument("--random_seed", type = int, default = 42)
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "TCN")
    parser.add_argument("--save_dir", type = str, default = "./results")
    
    # test shot for disruption probability curve
    parser.add_argument("--test_shot_num", type = int, default = 30312)

    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)
    parser.add_argument("--use_multi_gpu", type = bool, default = False)
    
    # mode : predicting thermal quench vs current quench
    parser.add_argument("--mode", type = str, default = 'TQ', choices=['TQ','CQ'])

    # batch size / sequence length / epochs / distance / num workers / pin memory use
    parser.add_argument("--batch_size", type = int, default = 100)
    parser.add_argument("--num_epoch", type = int, default = 32)
    parser.add_argument("--seq_len_efit", type = int, default = 10)
    parser.add_argument("--seq_len_ece", type = int, default = 50)
    parser.add_argument("--seq_len_diag", type = int, default = 50)
    parser.add_argument("--dist", type = int, default = 4)
    parser.add_argument("--num_workers", type = int, default = 4)
    parser.add_argument("--pin_memory", type = bool, default = True)
    
    # num samples
    parser.add_argument("--num_samples", type = int, default = 256)
    
    # scaler type
    parser.add_argument("--scaler", type = str, choices=['Robust', 'Standard', 'MinMax', 'None'], default = "Robust")
    
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
    parser.add_argument("--early_stopping_verbose", type = bool, default = False)
    parser.add_argument("--early_stopping_delta", type = float, default = 1e-3)

    # imbalanced dataset processing
    # Re-sampling
    parser.add_argument("--use_sampling", type = bool, default = False)
    
    # Re-weighting
    parser.add_argument("--use_weighting", type = bool, default = False)
    
    # Deffered Re-weighting
    parser.add_argument("--use_DRW", type = bool, default = False)
    parser.add_argument("--beta", type = float, default = 0.25)
    
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
    parser.add_argument("--verbose", type = int, default = 128)
    
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

args = parsing()
    
# seed initialize
seed_everything(args['random_seed'], False)

# device allocation
if (torch.cuda.device_count() >= 1):
    device = "cuda:" + str(args["gpu_num"])
else:
    device = 'cpu'

# save directory
save_dir = args['save_dir']

ray.shutdown()  # Restart Ray defensively in case the ray connection is lost. 
ray.init(log_to_driver=True, ignore_reinit_error=True)

# directory for checkpoint
if not os.path.isdir("./hpo_checkpoint"):
    os.mkdir("./hpo_checkpoint")
    
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

tag = "{}_dist_{}_{}_{}_{}_{}_seed_{}".format(args["tag"], args["dist"], loss_type, boost_type, scale_type, args['mode'], args['random_seed'])

print("================= Running code =================")
print("Hyper-parameter optimization setting : {}".format(tag))

checkpoint_dir = os.path.join("./hpo_checkpoint", tag)

# To restore a checkpoint, use `session.get_checkpoint()`.
loaded_checkpoint = session.get_checkpoint()
if loaded_checkpoint:
    checkpoint_dir = loaded_checkpoint.as_directory()
    
# dataset setup 
train_list, valid_list, test_list, scaler_list = preparing_0D_dataset(config.filepath, None, args['scaler'], args['test_shot_num'])

print("================= Dataset information =================")
train_data = MultiSignalDataset(train_list['disrupt'], train_list['efit'], train_list['ece'], train_list['diag'], args['seq_len_efit'], args['seq_len_ece'], args['seq_len_diag'], args['dist'], 0.01, scaler_list['efit'], scaler_list['ece'], scaler_list['diag'], args['mode'], 'train')
valid_data = MultiSignalDataset(valid_list['disrupt'], valid_list['efit'], valid_list['ece'], valid_list['diag'], args['seq_len_efit'], args['seq_len_ece'], args['seq_len_diag'], args['dist'], 0.01, scaler_list['efit'], scaler_list['ece'], scaler_list['diag'], args['mode'], 'valid')
test_data = MultiSignalDataset(test_list['disrupt'], test_list['efit'], test_list['ece'], test_list['diag'], args['seq_len_efit'], args['seq_len_ece'], args['seq_len_diag'], args['dist'], 0.01, scaler_list['efit'], scaler_list['ece'], scaler_list['diag'], args['mode'], 'test')

# label distribution for LDAM / Focal Loss
train_data.get_num_per_cls()
cls_num_list = train_data.get_cls_num_list()

# Re-sampling
if args["use_sampling"]:
    train_sampler = ImbalancedDatasetSampler(train_data)
    valid_sampler = RandomSampler(valid_data)
    test_sampler = RandomSampler(test_data)

else:
    train_sampler = RandomSampler(train_data)
    valid_sampler = RandomSampler(valid_data)
    test_sampler = RandomSampler(test_data)

train_loader = DataLoader(train_data, batch_size = args['batch_size'], sampler=train_sampler, num_workers = args["num_workers"], pin_memory=args["pin_memory"], drop_last=True)
valid_loader = DataLoader(valid_data, batch_size = args['batch_size'], sampler=valid_sampler, num_workers = args["num_workers"], pin_memory=args["pin_memory"], drop_last=True)
test_loader = DataLoader(test_data, batch_size = args['batch_size'], sampler=test_sampler, num_workers = args["num_workers"], pin_memory=args["pin_memory"], drop_last=True)

# Re-weighting
if args['use_weighting']:
    per_cls_weights = 1.0 / np.array(cls_num_list)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
else:
    per_cls_weights = np.array([1,1])
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    
# loss
if args['loss_type'] == "CE":
    loss_fn = CELoss(weight = per_cls_weights)
elif args['loss_type'] == 'LDAM':
    max_m = args['max_m']
    s = args['s']
    loss_fn = LDAMLoss(cls_num_list, max_m = max_m, s = s, weight = per_cls_weights)
elif args['loss_type'] == 'Focal':
    focal_gamma = args['focal_gamma']
    loss_fn = FocalLoss(weight = per_cls_weights, gamma = focal_gamma)
else:
    loss_fn = CELoss(weight = per_cls_weights)
    
if args['use_label_smoothing']:
    loss_fn = LabelSmoothingLoss(loss_fn, alpha = args['smoothing'], kl_weight = args['kl_weight'], classes = 2)
    
def load_model(configuration:Dict, device : str):
    
    header_config = {
        "efit":{
                "num_inputs":len(config.EFIT + config.EFIT_FE),
                "hidden_dim":configuration['efit-hidden_dim'],
                "num_channels":[configuration['efit-channels']] * configuration['efit-nlevel'],
                "kernel_size":configuration['efit-kernel_size'],
                "dropout":configuration['efit-dropout'],
                "dilation_size":calc_dilation(configuration['efit-kernel_size'],configuration['efit-dilation_size'], configuration['efit-nlevel'], configuration['efit-nrecept']),
                "seq_len":10
        },
        "ece":{
                "num_inputs":len(config.ECE),
                "hidden_dim":configuration['ece-hidden_dim'],
                "num_channels":[configuration['ece-channels']] * configuration['ece-nlevel'],
                "kernel_size":configuration['ece-kernel_size'],
                "dropout":configuration['ece-dropout'],
                "dilation_size":calc_dilation(configuration['ece-kernel_size'],configuration['ece-dilation_size'], configuration['ece-nlevel'], configuration['ece-nrecept']),
                "seq_len":50
        },
        "diag":{
                "num_inputs":len(config.DIAG),
                "hidden_dim":configuration['diag-hidden_dim'],
                "num_channels":[configuration['diag-channels']] * configuration['diag-nlevel'],
                "kernel_size":configuration['diag-kernel_size'],
                "dropout":configuration['diag-dropout'],
                "dilation_size":calc_dilation(configuration['diag-kernel_size'],configuration['diag-dilation_size'], configuration['diag-nlevel'], configuration['diag-nrecept']),
                "seq_len":50
        },
    }
    
    classifier_config = {
        "cls_dim" : configuration['cls-cls_dim'],
        "n_classes":2,
        "prior_pi":configuration['cls-prior_pi'],
        "prior_sigma1":configuration['cls-prior_sigma1'],
        "prior_sigma2":configuration['cls-prior_sigma2']
    }
    
    model = Predictor(header_config, classifier_config, device)
    return model

def train_for_hpo(
    configuration: Dict,
    checkpoint_dir : str,
    train_loader,
    valid_loader,
    ):
    
    if torch.cuda.is_available():
        device = "cuda:{}".format(args['gpu_num'])
        model = load_model(configuration, device)
        
        if torch.cuda.device_count() > 1 and args['use_multi_gpu']:
            device = "cuda:0"
            model = torch.nn.DataParallel(model)
            
        model.to(device)
    else:
        device = "cpu"    
        model = load_model(configuration)

    # optimizer
    if args["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = args['lr'])
    elif args["optimizer"] == "RMSProps":
        optimizer = torch.optim.RMSprop(model.parameters(), lr = args['lr'])
    elif args["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = args['lr'])
    elif args["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr = args['lr'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr = args['lr'])
    
    # scheduler
    if args["use_scheduler"] and not args["use_DRW"]:    
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args['step_size'], gamma=args['gamma'])
        
    elif args["use_DRW"]:
        scheduler = "DRW"
        
    else:
        scheduler = None
    
    # training process
    if args['use_DRW']:
        train_loss, train_f1, valid_loss, valid_f1 = train_DRW(
            train_loader,
            valid_loader,
            model,
            optimizer,
            loss_fn,
            device,
            args['num_epoch'],
            max_norm_grad = 1.0,
            cls_num_list=cls_num_list,
            betas = [0, args['beta'], args['beta'] * 2, args['beta']*3],
            checkpoint_dir=checkpoint_dir
        )
        
    else:
        train_loss, train_f1, valid_loss, valid_f1 = train(
            train_loader,
            valid_loader,
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            args['num_epoch'],
            max_norm_grad = 1.0,
            checkpoint_dir=checkpoint_dir
        )
    
    return

if __name__ == "__main__":
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    num_samples = args['num_samples']
    
    cpus_per_trial = 16
    gpus_per_trial = 4
    
    # define model
    model_argument = {
        # EFIT channels
        "efit-hidden_dim":tune.choice([32, 64, 128, 256]),
        "efit-channels":tune.choice([32, 64, 128, 256]),
        "efit-kernel_size":tune.choice([2,3,4]),
        "efit-dropout":tune.loguniform(1e-2, 5e-1),
        "efit-dilation_size":tune.choice([2,3]),
        "efit-nlevel":tune.choice([2,3,4,5]),
        "efit-nrecept":tune.choice([1024, 1024 * 8, 1024 * 16]),

        # ECE channels
        "ece-hidden_dim":tune.choice([32, 64, 128, 256]),
        "ece-channels":tune.choice([32, 64, 128, 256]),
        "ece-kernel_size":tune.choice([2,3,4]),
        "ece-dropout":tune.loguniform(1e-2, 5e-1),
        "ece-dilation_size":tune.choice([2,3]),
        "ece-nlevel":tune.choice([2,3,4,5]),
        "ece-nrecept":tune.choice([1024, 1024 * 8, 1024 * 16]),

        # Diag channels
        "diag-hidden_dim":tune.choice([32, 64, 128, 256]),
        "diag-channels":tune.choice([32, 64, 128, 256]),
        "diag-kernel_size":tune.choice([2,3,4]),
        "diag-dropout":tune.loguniform(1e-2, 5e-1),
        "diag-dilation_size":tune.choice([2,3]),
        "diag-nlevel":tune.choice([2,3,4,5]),
        "diag-nrecept":tune.choice([1024, 1024 * 8, 1024 * 16]),
       
        # Classifier
        "cls-cls_dim" : tune.choice([32, 64, 128, 256]),
        "cls-prior_pi":tune.loguniform(0.2, 0.5),
        "cls-prior_sigma1":tune.loguniform(0.5, 5.0),
        "cls-prior_sigma2":tune.loguniform(0.001, 0.005)
    }
      
    tune_scheduler = ASHAScheduler(
        metric="f1_score",
        mode="max",
        max_t = args['num_epoch'],
        grace_period=1,
        reduction_factor=2
    )
    
    tune_reporter = CLIReporter(
        metric_columns=["loss", "f1_score", "training_iteration"],
        parameter_columns=[key for key in model_argument.keys()],
        sort_by_metric = True,
    )
    
    # the solution : https://stackoverflow.com/questions/69777578/the-actor-implicitfunc-is-too-large-error
    trainable = tune.with_parameters(
        train_for_hpo,
        checkpoint_dir = checkpoint_dir,
        train_loader = train_loader,
        valid_loader = valid_loader,
    )
        
    tune_result = tune.run(
        trainable,
        resources_per_trial={
            "cpu":cpus_per_trial, 
            "gpu":gpus_per_trial
        },
        local_dir = checkpoint_dir,
        config = model_argument,
        num_samples=num_samples,
        scheduler=tune_scheduler,
        progress_reporter=tune_reporter,
        checkpoint_at_end=False
    )
        
    best_trial = tune_result.get_best_trial("f1_score", "max", "last")
    print("Best trial final validation f1 score : {:.3f}".format(best_trial.last_result["f1_score"]))
    print("Best trial config: {}".format(best_trial.config))
    
    best_model = load_model(best_trial.config, device)
    best_model.to(device)
    
    best_checkpoint = best_trial.checkpoint
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint.dir_or_data, "checkpoint"))
    
    best_model.load_state_dict(model_state) 
    
    test_loss, test_f1 = evaluate(
        test_loader,
        best_model,
        loss_fn,
        device,
        0.5,
    )
    
    print("Best trial test f1-score:{:.3f}".format(test_f1))
    
    with open("./results/hpo_{}.txt".format(tag), 'w') as f:
        header = "="*20 +  "Hyperparameter optimization" + "="*20
        f.write(header)
        f.write("\ntag : {}".format(tag))
        
        best_config = best_trial.config
        
        for key in best_config.keys():
            f.write("\n{}:{}".format(key, best_config[key]))
            
        summary = "\nBest trial test f1-score:{:.3f}".format(test_f1)
        f.write(summary)
        
    ray.shutdown()