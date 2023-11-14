import torch
import os
import numpy as np
import pandas as pd
import argparse
from src.dataset import MultiSignalDataset
from torch.utils.data import DataLoader, RandomSampler
from src.utils.utility import preparing_0D_dataset, seed_everything
from src.models.predictor import BayesianPredictor
from src.models.BNN import compute_ensemble_probability, compute_uncertainty_per_data
from src.config import Config

config = Config()

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="testing bayesian disruption prediction model with multi-signal data")
    
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
    parser.add_argument("--mode", type = str, default = 'TQ', choices=['TQ','CQ'])

    # batch size / sequence length / epochs / distance / num workers / pin memory use
    parser.add_argument("--batch_size", type = int, default = 100)
    parser.add_argument("--num_epoch", type = int, default = 128)
    parser.add_argument("--seq_len_efit", type = int, default = 100)
    parser.add_argument("--seq_len_ece", type = int, default = 1000)
    parser.add_argument("--seq_len_diag", type = int, default = 1000)
    parser.add_argument("--dist_warning", type=int, default=400)
    parser.add_argument("--dist", type = int, default = 40)
    parser.add_argument("--dt", type = float, default = 0.001)
    parser.add_argument("--num_workers", type = int, default = 4)
    parser.add_argument("--pin_memory", type = bool, default = True)
    
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
    parser.add_argument("--smoothing", type = float, default = 0.05)
    parser.add_argument("--kl_weight", type = float, default = 0.2)
    
    # LDAM Loss parameter
    parser.add_argument("--max_m", type = float, default = 0.5)
    parser.add_argument("--s", type = float, default = 1.0)
    
    # Focal Loss parameter
    parser.add_argument("--focal_gamma", type = float, default = 2.0)
    
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
    
    tag = "Bayes_{}_warning_{}_dist_{}_{}_{}_{}_{}_seed_{}".format(args["tag"], args['dist_warning'], args["dist"], loss_type, boost_type, scale_type, args['mode'], args['random_seed'])
    
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
    test_data = MultiSignalDataset(test_list['disrupt'], test_list['efit'], test_list['ece'], test_list['diag'], args['seq_len_efit'], args['seq_len_ece'], args['seq_len_diag'], args['dist'], args['dt'], scaler_list['efit'], scaler_list['ece'], scaler_list['diag'], args['mode'], 'test', args['dist_warning'])
    test_data.get_shot_num = True
    
    # define model
    model = BayesianPredictor(config.header_config, config.classifier_config, device)
    model.to(device)

    test_sampler = RandomSampler(test_data)

    # evaluation process
    print("\n====================== Uncertainty computation ======================\n")
    model.load_state_dict(torch.load(save_best_dir))
   
    # experiment
    test_loader = DataLoader(test_data, batch_size = 1, sampler=test_sampler, num_workers = 1, pin_memory=args["pin_memory"])

    test_data = None
    test_label = None
    
    import matplotlib.pyplot as plt
    from typing import Optional
    def plot_output_distribution(pred : np.ndarray, title : Optional[str] = None, save_dir : Optional[str] = None):
    
        fig, ax = plt.subplots(1,1)
        counts, bins = np.histogram(pred.reshape(-1,))
        ax.hist(bins[:-1], bins = bins, weights = counts, color = 'gray')
        
        ax.set_xlabel("Output (prob)")
        ax.set_xlim([0, 1.0])
        ax.set_ylabel('n-samples')
        ax.set_title("Probability histogram")
   
        if title:
            plt.suptitle(title)
            
        fig.tight_layout()
        
        if save_dir:
            plt.savefig(save_dir)
            
        return fig, ax
    
    for idx, data in enumerate(test_loader):
        
        if data['label'].numpy() == 1:
            test_input = data
            test_shot = int(data['shot_num'].item())
            test_label = data['label']
            pred = torch.nn.functional.sigmoid(model(test_input)).detach().cpu().numpy()
            pred = np.where(pred > 0.5, 1, 0)
            
            if pred == 0:
                break
        
    test_pred = compute_ensemble_probability(model, test_input, device, n_samples = 128)
    au, eu = compute_uncertainty_per_data(model, test_input, device = device, n_samples = 128)
    
    print("\n====================== False Negative case ======================\n")
    print("true label : ", test_label)
    print("test shot : ", test_shot)
    print("aleatoric uncertainty: ", au[0])
    print("epistemic uncertainty: ", eu[0])
    
    fig, axes = plot_output_distribution(test_pred, "Disruptive phase - Missing alarm case, shot : {}".format(test_shot), "./results/analysis_file/test-FN.png")
    
    for idx, data in enumerate(test_loader):
        
        if data['label'].numpy() == 0:
            test_input = data
            test_shot = int(data['shot_num'].item())
            test_label = data['label']
            pred = torch.nn.functional.sigmoid(model(test_input)).detach().cpu().numpy()
            pred = np.where(pred > 0.5, 1, 0)
            
            if pred == 1:
                break
        
    test_pred = compute_ensemble_probability(model, test_input, device, n_samples = 128)
    au, eu = compute_uncertainty_per_data(model, test_input, device = device, n_samples = 128)
    
    print("\n====================== False Positive case ======================\n")
    print("true label : ", test_label)
    print("test shot : ", test_shot)
    print("aleatoric uncertainty: ", au[0])
    print("epistemic uncertainty: ", eu[0])
    
    fig, axes = plot_output_distribution(test_pred, "Disruptive phase - False alarm case, shot : {}".format(test_shot), "./results/analysis_file/test-FP.png")
    
    for idx, data in enumerate(test_loader):
        
        if data['label'].numpy() == 1:
            test_input = data
            test_shot = int(data['shot_num'].item())
            test_label = data['label']
            pred = torch.nn.functional.sigmoid(model(test_input)).detach().cpu().numpy()
            pred = np.where(pred > 0.5, 1, 0)
            
            if pred == 1:
                break
        
    test_pred = compute_ensemble_probability(model, test_input, device, n_samples = 128)
    au, eu = compute_uncertainty_per_data(model, test_input, device = device, n_samples = 128)
    
    print("\n====================== True Positive case ======================\n")
    print("true label : ", test_label)
    print("test shot : ", test_shot)
    print("aleatoric uncertainty: ", au[0])
    print("epistemic uncertainty: ", eu[0])
    
    fig, axes = plot_output_distribution(test_pred, "Disruptive phase - True Positive case, shot : {}".format(test_shot), "./results/analysis_file/test-TP.png")
    
    # uncertainty computation for test dataset
    aus = []
    eus = []
    preds = []
    shots = []
    cases = []
    dists = []
    
    test_data = MultiSignalDataset(test_list['disrupt'], test_list['efit'], test_list['ece'], test_list['diag'], args['seq_len_efit'], args['seq_len_ece'], args['seq_len_diag'], args['dist'], args['dt'], scaler_list['efit'], scaler_list['ece'], scaler_list['diag'], args['mode'], 'eval', args['dist_warning'])
    test_sampler = RandomSampler(test_data)
    test_loader = DataLoader(test_data, batch_size = 1, sampler=test_sampler, num_workers = 1, pin_memory=args["pin_memory"])

    from tqdm.auto import tqdm
    
    for idx, data in enumerate(tqdm(test_loader, 'uncertainty analysis results prepared...')):
        test_input = data
        test_shot = int(data['shot_num'].item())
        test_label = data['label'].numpy()
        pred = torch.nn.functional.sigmoid(model(test_input)).detach().cpu().numpy()
        pred = np.where(pred > 0.5, 1, 0)
        dist = data['dist'].item()
        
        test_pred = compute_ensemble_probability(model, test_input, device, n_samples = 32)
        au, eu = compute_uncertainty_per_data(model, test_input, device = device, n_samples = 32)
        aus.append(au[0])
        eus.append(eu[0])
        preds.append(test_pred)
        shots.append(test_shot)
        dists.append(dist)
        
        if test_label == 1:
            # TP
            if pred == 1:
                cases.append("TP")
            # FN
            else:
                cases.append("FN") 
        else:
            # FP
            if pred == 1:
                cases.append("FP")
            # TN
            else:
                cases.append("TN")
    
    results = {
        "au" : aus,
        "eu" : eus,
        "preds" : preds,
        "shot" : shots,
        "cases" : cases
    }
    results = pd.DataFrame(results)
    results.to_pickle("./results/analysis_file/analysis_uncertainty_test_{}.pkl".format(tag))    