import torch
import os
import numpy as np
import pandas as pd
import argparse
from src.utils.utility import preparing_0D_dataset, seed_everything, plot_disrupt_prob, plot_disrupt_prob_causes, plot_disrupt_prob_uncertainty, moving_avarage_smoothing, MultiSignalDataset
from src.models.predictor import Predictor, BayesianPredictor
from src.models.BNN import compute_uncertainty_per_data
from src.feature_importance import compute_relative_importance
from src.config import Config
import warnings
from tqdm.auto import tqdm

warnings.filterwarnings(action = 'ignore')

config = Config()

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="Continuous disruption prediction with multi-signal data")
    
    # choose model type
    parser.add_argument("--model_type", type = str, default = 'Bayesian', choices=['Bayesian', 'Frequentist'])
    
    # choose evaluation processes
    parser.add_argument("--plot_shot_info", type = bool, default = False)
    parser.add_argument("--plot_uncertainty", type = bool, default = False)
    parser.add_argument("--plot_feature_importance", type = bool, default = False)
    
    # random seed
    parser.add_argument("--random_seed", type = int, default = 42)
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "TCN")
    parser.add_argument("--save_dir", type = str, default = "./results")
    
    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)
    
    # mode : predicting thermal quench vs current quench
    parser.add_argument("--mode", type = str, default = 'TQ', choices=['TQ','CQ'])

    # batch size / sequence length / epochs / distance / num workers / pin memory use
    parser.add_argument("--seq_len_efit", type = int, default = 100)
    parser.add_argument("--seq_len_ece", type = int, default = 1000)
    parser.add_argument("--seq_len_diag", type = int, default = 1000)
    parser.add_argument("--dist_warning", type = int, default = 400)
    parser.add_argument("--dist", type = int, default = 40)
    parser.add_argument("--dt", type = float, default = 0.001)
    
    # scaler type
    parser.add_argument("--scaler", type = str, choices=['Robust', 'Standard', 'MinMax', 'None'], default = "Robust")
    
    # imbalanced dataset processing
    # Re-sampling
    parser.add_argument("--use_sampling", type = bool, default = False)
    
    # Re-weighting
    parser.add_argument("--use_weighting", type = bool, default = False)
    
    # loss type : CE, Focal, LDAM
    parser.add_argument("--loss_type", type = str, default = "Focal", choices = ['CE','Focal', 'LDAM'])
    
    # label smoothing
    parser.add_argument("--use_label_smoothing", type = bool, default = False)

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
        
    if args['model_type'] == 'Bayesian':
        tag = "Bayes_{}_warning_{}_dist_{}_{}_{}_{}_{}_seed_{}".format(args["tag"], args['dist_warning'], args["dist"], loss_type, boost_type, scale_type, args['mode'], args['random_seed'])
    else:
        tag = "{}_warning_{}_dist_{}_{}_{}_{}_{}_seed_{}".format(args["tag"], args['dist_warning'], args["dist"], loss_type, boost_type, scale_type, args['mode'], args['random_seed'])
    
    # save directory
    save_dir = os.path.join(args['save_dir'], tag)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if not os.path.isdir("./weights"):
        os.mkdir("./weights")
        
    print("================= Running code =================")
    print("Setting : {}".format(tag))
    
    save_best_dir = "./weights/{}_best.pt".format(tag)
    save_last_dir = "./weights/{}_last.pt".format(tag)
 
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(args["gpu_num"])
    else:
        device = 'cpu'
    
    # dataset setup 
    train_list, valid_list, test_list, scaler_list = preparing_0D_dataset(config.filepath, None, args['scaler'], None)
    
    # define model
    if args['model_type'] == 'Bayesian':
        model = BayesianPredictor(config.header_config, config.classifier_config, device)
    else:
        model = Predictor(config.header_config, config.classifier_config, device)
    
    print("\n==================== model summary ====================\n")
    model.to(device)
    model.summary()

    # plot probability curve
    print("\n====================== Probability curve generation process ======================\n")
    model.load_state_dict(torch.load(save_best_dir))
        
    if not os.path.exists(os.path.join(save_dir, "prediction-test")):
        os.makedirs(os.path.join(save_dir, "prediction-test"))
        
    save_dir = os.path.join(save_dir, "prediction-test")
    
    ori_data_disrupt = test_list['disrupt']
    ori_data_efit = test_list['efit']
    ori_data_ece = test_list['ece']
    ori_data_diag = test_list['diag']
    
    test_shot_list = ori_data_disrupt.shot.unique()
    
    for shot_num in tqdm(test_shot_list):
    
        tTQend = ori_data_disrupt[ori_data_disrupt.shot == shot_num].t_tmq.values[0]
        tftsrt = ori_data_disrupt[ori_data_disrupt.shot == shot_num].t_flattop_start.values[0]
        tipminf = ori_data_disrupt[ori_data_disrupt.shot == shot_num].t_ip_min_fault.values[0]
    
        t_disrupt = tTQend
        t_current = tipminf
    
        data_efit = ori_data_efit[ori_data_efit.shot == shot_num]
        data_ece = ori_data_ece[ori_data_ece.shot == shot_num]
        data_diag = ori_data_diag[ori_data_diag.shot == shot_num]
        data_disrupt = ori_data_disrupt[ori_data_disrupt.shot == shot_num]
        
        dataset = MultiSignalDataset(
            data_disrupt,
            data_efit,
            data_ece,
            data_diag,
            args['seq_len_efit'],
            args['seq_len_ece'],
            args['seq_len_diag'],
            args['dist_warning'],
            args['dist'],
            args['dt'],
            scaler_list['efit'],
            scaler_list['ece'],
            scaler_list['diag'],
            args['mode']
        )

        prob_list = []
        is_disruption = []
        feature_dict = None
        aus = []
        eus = []

        model.to(device)
        model.eval()
        
        # computation of disruption probability
        for idx in range(dataset.__len__()):
            
            data = dataset.__getitem__(idx)
            
            for key in data.keys():
                data[key] = data[key].unsqueeze(0)
                
            if args['plot_feature_importance']:
                output = model(data)
                probs = torch.nn.functional.sigmoid(output)[0]
                probs = probs.cpu().detach().numpy().tolist()
            else:
                with torch.no_grad():
                    output = model(data)
                    probs = torch.nn.functional.sigmoid(output)[0]
                    probs = probs.cpu().detach().numpy().tolist()

            prob_list.extend(
                probs
            )
            
            is_disruption.extend(
                torch.where(torch.nn.functional.sigmoid(output) > 0.5, 1, 0).cpu().detach().numpy().tolist()
            )
            
            if args['plot_uncertainty']:
                au, eu = compute_uncertainty_per_data(model, data, device = device, n_samples = 16)
                aus.append(au[0])
                eus.append(eu[0])
                
            t = dataset.time_slice[idx]
            
            if t >= t_disrupt - args['dist'] * args['dt'] and t <= t_disrupt and args['plot_feature_importance']:
                if feature_dict is None:
                    feat = compute_relative_importance(data, model, 0, None, 16, device)
                    feature_dict = {}
                    feature_dict['time'] = [t_disrupt - t]
                    
                    for key in feat.keys():
                        feature_dict[key] = [feat[key]]
                else:
                    feat = compute_relative_importance(data, model, 0, None, 16, device)
                    feature_dict['time'].append(t_disrupt - t)
                    
                    for key in feat.keys():
                        feature_dict[key].append(feat[key])
        
        n_ftsrt = int(dataset.time_slice[0] // args['dt'])
        time_slice = np.linspace(0, dataset.time_slice[0] - args['dt'], n_ftsrt).tolist() + [v for v in dataset.time_slice] 
        probs = np.array([0] * n_ftsrt + prob_list)
        probs = moving_avarage_smoothing(probs, 12, 'center')
        
        if args['plot_uncertainty']:
            aus = np.array([0] * n_ftsrt + aus)
            eus = np.array([0] * n_ftsrt + eus)
        
        # plot disrupt prob
        plot_disrupt_prob(probs, time_slice, tftsrt, t_disrupt, t_current, save_dir, args['dt'] * args['dist'], args['dt'] * args['dist_warning'], shot_num)
        
        if args['plot_uncertainty']:
            aus = np.array(aus).reshape(-1)
            eus = np.array(eus).reshape(-1)
            plot_disrupt_prob_uncertainty(probs, time_slice, aus, eus, tftsrt, t_disrupt, t_current, save_dir, args['dt'] * args['dist'], args['dt'] * args['dist_warning'], shot_num)
     
        if args['plot_feature_importance']:
            plot_disrupt_prob_causes(feature_dict, shot_num, args['dt'], args['dist'], save_dir)
            
        del data_efit
        del data_ece
        del data_diag
        del data_disrupt
        
        del aus
        del eus
        del probs
        del time_slice
        del feature_dict