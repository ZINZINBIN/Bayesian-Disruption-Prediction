''' 
Codes for Permutation importance and Perturbation importance
Permutation feature importance algorithm
    Input : trained model, feature matrix, target vector, error measure
    1. Estimate the original model error e_orig = L(y,f(x))
    2. For each feature j, do the processes below
        - Generate feature matrix X_perm by permuting feature j in the data X. 
        - Estimate error e_perm = L(y, f(X_perm)) based on the predictions of the permuted data
        - Calculate permutation feature matrix as quotient FI_j = e_perm / e_orig
    3. Sort features by descending FI
Reference
- https://towardsdatascience.com/neural-feature-importance-1c1868a4bf53
- https://github.com/pytorch/captum
- https://moondol-ai.tistory.com/401
'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from typing import Literal, List, Optional, Dict
from sklearn.metrics import f1_score
from src.config import Config

config = Config()

def compute_loss(
    dataloader : DataLoader, 
    model : nn.Module,
    loss_fn : nn.Module,
    device : str = "cpu",
    ):

    model.eval()
    model.to(device)

    total_loss = 0
    total_f1 = 0
    total_pred = np.array([])
    total_label = np.array([])
    total_size = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        with torch.no_grad():
            
            if isinstance(data, dict):
                output = model(data)
            else:
                output = model(data.to(device)) 
                
            loss = loss_fn(output, target.to(device))
    
            total_loss += loss.item()
            pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
            total_size += pred.size(0)

            total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
            total_label = np.concatenate((total_label, target.cpu().numpy().reshape(-1,)))

    total_f1 = f1_score(total_label, total_pred, average = "macro")

    return total_loss, total_f1


# permutation importance
def compute_permute_feature_importance(
    model : nn.Module,
    dataloader : DataLoader,
    features : List,
    loss_fn : nn.Module,
    device : str,
    criteria : Literal['loss','score'],
    save_dir : str,
    ):
    
    # convert get_shot_num variable true
    dataloader.dataset.get_shot_num = False

    n_features = len(features)
    data_orig = dataloader.dataset.ts_data.copy()
    
    results = []
    
    loss_orig, score_orig = compute_loss(dataloader, model, loss_fn, device)
    
    for k in tqdm(range(n_features), desc = "processing for feature importance"):
        
        # permutate the k-th features
        np.random.shuffle(dataloader.dataset.ts_data[features[k]].values)
        
        # compute the loss
        loss, score = compute_loss(dataloader, model, loss_fn, device)
        
        # return the order of the features
        dataloader.dataset.ts_data = data_orig
        
        if criteria == 'loss':
            fi = abs(abs(loss - loss_orig) / loss_orig)
        else:
            fi = abs(score - score_orig) / score_orig
        
        # update result
        results.append({"feature":features[k], "loss":loss, "score":score, "feature_importance":fi})
        
    df = pd.DataFrame(results)
    
    # feature name mapping
    feature_map = config.feature_map
    
    def _convert_str(x : str):
        return feature_map[x]
    
    df['feature'] = df['feature'].apply(lambda x : _convert_str(x))
    df = df.sort_values('feature_importance')
    
    plt.figure(figsize = (8,8))
    plt.barh(np.arange(n_features), df.feature_importance)
    plt.yticks(np.arange(n_features), df.feature.values)
    plt.title('0D data - feature importance')
    plt.ylim((-1,n_features + 1))
    
    # plt.xlim([0, 5.0])
    plt.xlabel('Permutation feature importance')
    plt.ylabel('Feature', size = 14)
    plt.savefig(save_dir)
    
# direct feature importance estimation : integrated-gradient method
# reference: https://github.com/TianhongDai/integrated-gradient-pytorch/blob/master/integrated_gradients.py
def calculate_outputs_and_gradients(inputs:torch.Tensor, model:nn.Module, target_label_idx:Optional[int], device:str = 'cpu'):
    
    gradients = []
    
    for input in inputs:
        output = model(input)
        output = torch.nn.functional.softmax(output, dim = 1)
        
        if target_label_idx is None:
            target_label_idx = torch.argmax(output, 1).item()
        
        index = torch.ones((output.size()[0], 1), dtype = torch.int64) * target_label_idx
        
        output = output.gather(1, index.to(device))
        
        model.zero_grad()
        output.backward()
        
        gradient = {}

        for key in [col for col in input.keys() if col not in ['shot_num', 'label']]:
            gradient[key] = input[key].grad.detach()
        
        gradients.append(gradient)
        
    return gradients

def integrated_gradients(inputs:Dict[str, torch.Tensor], model:nn.Module, target_label_idx:Optional[int], baseline = None, steps:int = 50, device : str = 'cpu'):
    
    if baseline is None:
        baseline = {}
        for key in inputs.keys():
            baseline[key] = torch.zeros_like(inputs[key])
            
    scaled_inputs = []
    
    for step in range(0, steps + 1):
        
        comp = {}
        
        for key in [col for col in inputs.keys() if col not in ['shot_num', 'label']]:
            comp[key] = baseline[key] + (float(step) / steps) * (inputs[key] - baseline[key])
            comp[key].requires_grad = True
            
        scaled_inputs.append(comp)
        
    grads = calculate_outputs_and_gradients(scaled_inputs, model, target_label_idx, device)
  
    avg_grads = {}
    
    for key in [col for col in inputs.keys() if col not in ['shot_num', 'label']]:
        avg_grads[key] = torch.mean(torch.stack([grad[key] for grad in grads]), dim = 0)
    
    integrated_grad = {}
        
    for key in [col for col in inputs.keys() if col not in ['shot_num', 'label']]:
        integrated_grad[key] = (inputs[key] - baseline[key]).detach() * avg_grads[key]
    
    return integrated_grad

def compute_relative_importance(inputs:Dict[str, torch.Tensor], model:nn.Module, target_label_idx:Optional[int], baseline = None, steps:int = 50, device : str = 'cpu'):
    
    model.train()
    IG = integrated_gradients(inputs, model, target_label_idx, baseline, steps, device)
    
    feat_imp = []
    
    for key in [col for col in inputs.keys() if col not in ['shot_num', 'label']]:
        feat_imp.extend(IG[key].squeeze(0).abs().sum(dim=1).detach().cpu().numpy().reshape(-1,).tolist())

    # Reconstruction of feature importance
    info_dict = {}
    idx_last = 0
    
    for idx, col in enumerate(config.COLUMN_NAME_SCALER['efit']):
        info_dict[col[1:]] = feat_imp[idx]
    
    idx_last += idx + 1
    
    imp_ece = 0
    for idx, col in enumerate(config.COLUMN_NAME_SCALER['ece']):
        imp_ece += feat_imp[idx + idx_last] / len(config.COLUMN_NAME_SCALER['ece'])
    
    idx_last += idx + 1
    info_dict['ECE'] = imp_ece
    
    imp_lm = 0
    for idx, col in enumerate(config.LM):
        imp_lm += feat_imp[idx + idx_last] / len(config.LM)
        
    idx_last += idx + 1 
    info_dict['LM'] = imp_lm
    
    imp_dl = 0
    for idx, col in enumerate(config.DL):
        imp_dl += feat_imp[idx + idx_last] / len(config.DL)
        
    idx_last += idx + 1 
    info_dict['DL'] = imp_dl
    
    imp_hcm = 0
    for idx, col in enumerate(config.HCM):
        imp_hcm += feat_imp[idx + idx_last] / len(config.HCM)
        
    idx_last += idx + 1 
    info_dict['HCM'] = imp_hcm
    
    imp_tci = 0
    for idx, col in enumerate(config.TCI):
        imp_tci += feat_imp[idx + idx_last] / len(config.TCI)
        
    idx_last += idx + 1 
    info_dict['TCI'] = imp_tci
    
    imp_lv = 0
    for idx, col in enumerate(config.LV):
        imp_lv += feat_imp[idx + idx_last] / len(config.LV)
        
    idx_last += idx + 1 
    info_dict['LV'] = imp_lv
    
    imp_rc = 0
    for idx, col in enumerate(config.RC):
        imp_rc += feat_imp[idx + idx_last] / len(config.RC)
        
    idx_last += idx + 1 
    info_dict['RC'] = imp_rc
    
    imp_ha = 0
    for idx, col in enumerate(config.HA):
        imp_ha += feat_imp[idx + idx_last] / len(config.HA)
        
    idx_last += idx + 1 
    info_dict['HA'] = imp_ha
    
    imp_heating = 0
    for idx, col in enumerate(config.ECH + config.NBH):
        imp_heating += feat_imp[idx + idx_last] / len(config.ECH + config.NBH)
        
    idx_last += idx + 1 
    info_dict['HEATING'] = imp_heating
    
    imp_bol = 0
    for idx, col in enumerate(config.BOL):
        imp_bol += feat_imp[idx + idx_last] / len(config.BOL)
        
    idx_last += idx + 1 
    info_dict['BOL'] = imp_bol
    
    # min max normalization
    for key in info_dict.keys():
        info_dict[key] = (info_dict[key] - min(info_dict.values())) / (max(info_dict.values())- min(info_dict.values()))

    return info_dict

def plot_feature_importance(
    inputs:Dict[str, torch.Tensor], 
    model:nn.Module, 
    target_label_idx:Optional[int], 
    baseline = None, 
    steps:int = 50, 
    device : str = 'cpu',
    title : str = "Relative importance",
    save_dir : Optional[str] = "./results/feature_importance.png"
    ):
    
    fig, ax = plt.subplots(1,1)
    feat_imp = compute_relative_importance(inputs, model, target_label_idx, baseline, steps, device)
    ax.barh(list(feat_imp.keys()), list(feat_imp.values()))
    ax.set_yticks([i for i in range(len(list(feat_imp.keys())))], labels = list(feat_imp.keys()))
    ax.invert_yaxis()
    ax.set_ylabel('Input features')
    ax.set_xlabel('Relative feature importance')
    
    plt.suptitle(title)
    fig.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir)