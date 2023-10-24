import torch
import os
import numpy as np
import pandas as pd
import argparse
from src.dataset import MultiSignalDataset
from torch.utils.data import DataLoader, RandomSampler
from src.utils.sampler import ImbalancedDatasetSampler
from src.utils.utility import preparing_0D_dataset, plot_learning_curve, generate_model_performance, seed_everything
from src.visualization.visualize_latent_space import visualize_2D_latent_space, visualize_2D_decision_boundary
from src.train import train
from src.evaluate import evaluate
from src.loss import FocalLoss, LDAMLoss, CELoss, LabelSmoothingLoss
from src.models.predictor import Predictor
from src.config import Config

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
    
    # mode : predicting thermal quench vs current quench
    parser.add_argument("--mode", type = str, default = 'CQ', choices=['TQ','CQ'])

    # batch size / sequence length / epochs / distance / num workers / pin memory use
    parser.add_argument("--batch_size", type = int, default = 100)
    parser.add_argument("--num_epoch", type = int, default = 128)
    parser.add_argument("--seq_len_efit", type = int, default = 10)
    parser.add_argument("--seq_len_ece", type = int, default = 50)
    parser.add_argument("--seq_len_diag", type = int, default = 50)
    parser.add_argument("--dist", type = int, default = 4)
    parser.add_argument("--num_workers", type = int, default = 4)
    parser.add_argument("--pin_memory", type = bool, default = True)
    
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
    parser.add_argument("--early_stopping_patience", type = int, default = 40)
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
    parser.add_argument("--verbose", type = int, default = 16)
    
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
    
    # tag initialize
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
    
    # save directory
    save_dir = os.path.join(args['save_dir'], tag)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if not os.path.isdir("./weights"):
        os.mkdir("./weights")
        
    if not os.path.isdir("./runs"):
        os.mkdir("./runs")
    
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
    train_list, valid_list, test_list, scaler_list = preparing_0D_dataset(config.filepath, None, args['scaler'], args['test_shot_num'])
    
    print("================= Dataset information =================")
    train_data = MultiSignalDataset(train_list['disrupt'], train_list['efit'], train_list['ece'], train_list['diag'], args['seq_len_efit'], args['seq_len_ece'], args['seq_len_diag'], args['dist'], 0.01, scaler_list['efit'], scaler_list['ece'], scaler_list['diag'], args['mode'], 'train')
    valid_data = MultiSignalDataset(valid_list['disrupt'], valid_list['efit'], valid_list['ece'], valid_list['diag'], args['seq_len_efit'], args['seq_len_ece'], args['seq_len_diag'], args['dist'], 0.01, scaler_list['efit'], scaler_list['ece'], scaler_list['diag'], args['mode'], 'valid')
    test_data = MultiSignalDataset(test_list['disrupt'], test_list['efit'], test_list['ece'], test_list['diag'], args['seq_len_efit'], args['seq_len_ece'], args['seq_len_diag'], args['dist'], 0.01, scaler_list['efit'], scaler_list['ece'], scaler_list['diag'], args['mode'], 'test')

    # label distribution for LDAM / Focal Loss
    train_data.get_num_per_cls()
    cls_num_list = train_data.get_cls_num_list()
    
    # define model
    model = Predictor(config.header_config, config.classifier_config, device)
    
    print("\n==================== model summary ====================\n")
    model.to(device)
    model.summary()

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
    if args["use_scheduler"]:    
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args['step_size'], gamma=args['gamma'])
    else:
        scheduler = None
        
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
    
    # training process
    print("\n======================= training process =======================\n")
    train_loss,  train_acc, train_f1, valid_loss, valid_acc, valid_f1 = train(
        train_loader,
        valid_loader,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device,
        args['num_epoch'],
        args['verbose'],
        save_best_dir = save_best_dir,
        save_last_dir = save_last_dir,
        exp_dir = exp_dir,
        max_norm_grad = 1.0,
        test_for_check_per_epoch=test_loader,
        is_early_stopping = args['early_stopping'],
        early_stopping_verbose = args['early_stopping_verbose'],
        early_stopping_patience = args['early_stopping_patience'],
        early_stopping_delta = args['early_stopping_delta'],
        scaler_type = args['scaler']
    )
    
    # plot the learning curve
    save_learning_curve = os.path.join(save_dir, "learning_curve.png")
    plot_learning_curve(train_loss, valid_loss, train_f1, valid_f1, figsize = (12,6), save_dir = save_learning_curve)
    
    # evaluation process
    print("\n====================== evaluation process ======================\n")
    model.load_state_dict(torch.load(save_best_dir))
    
    save_conf = os.path.join(save_dir, "test_confusion.png")
    save_txt = os.path.join(save_dir, "test_eval.txt")
    
    test_loss, test_acc, test_f1 = evaluate(
        test_loader,
        model,
        optimizer,
        loss_fn,
        device,
        save_conf = save_conf,
        save_txt = save_txt
    )
    
    # Additional analyzation
    print("\n====================== Visualization process ======================\n")
    if args['use_sampling']:
        train_loader = DataLoader(train_data, batch_size = args['batch_size'], sampler=None, num_workers = args["num_workers"], pin_memory=args["pin_memory"])
    
    try:
        visualize_2D_latent_space(
            model, 
            train_loader,
            device,
            os.path.join(save_dir, "latent_2D_train.png")
        )
        
        visualize_2D_latent_space(
            model, 
            test_loader,
            device,
            os.path.join(save_dir, "latent_2D_test.png")
        )
        
        visualize_2D_decision_boundary(
            model, 
            train_loader,
            device,
            os.path.join(save_dir, "decision_boundary_2D_train.png")
        )
        
        visualize_2D_decision_boundary(
            model, 
            test_loader,
            device,
            os.path.join(save_dir, "decision_boundary_2D_test.png")
        )
            
    except:
        print("{} : visualize 2D latent space doesn't work due to stability error".format(tag))
    
    # plot probability curve
    test_shot_num = args['test_shot_num']
    print("\n====================== Probability curve generation process ======================\n")
    
    generate_model_performance(
        filepath = config.filepath,
        model = model, 
        device = device,
        save_dir = save_dir,
        shot_num = test_shot_num,
        seq_len_efit = args['seq_len_efit'], 
        seq_len_ece = args['seq_len_ece'],
        seq_len_diag = args['seq_len_diag'], 
        dist = args['dist'],
        dt = 0.01,
        mode = args['mode'], 
        scaler_type = args['scaler'],
        is_plot_shot_info=True,
        is_plot_uncertainty=False,
        is_plot_feature_importance=True
    )