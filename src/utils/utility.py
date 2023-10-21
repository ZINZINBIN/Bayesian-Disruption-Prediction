import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import cv2, os, glob2, random
import pandas as pd
import numpy as np
from typing import Optional, List, Literal, Union, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator
from src.config import Config
from src.feature_importance import compute_relative_importance, plot_feature_importance
from src.models.BNN import compute_uncertainty_per_data
import time

config = Config()
STATE_FIXED = config.STATE_FIXED

# For reproduction
def seed_everything(seed : int = 42, deterministic : bool = False):
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
        
# function for deterministic train_test_split function
def deterministic_split(shot_list : List, test_size : float = 0.2):
    n_length = len(shot_list)
    n_test = int(test_size * n_length)
    divided = n_length // n_test

    total_indices = [idx for idx in range(n_length)]
    test_indices = [idx for idx in range(n_length) if idx % divided == 0]
    
    train_list = []
    test_list = []
    
    for idx in total_indices:
        if idx in test_indices:
            test_list.append(shot_list[idx])
        else:
            train_list.append(shot_list[idx])
    
    return train_list, test_list


# train-test split for 0D data models
def preparing_0D_dataset(
    filepath:Dict[str, str],
    random_state : Optional[int] = STATE_FIXED, 
    scaler : Literal['Robust', 'Standard', 'MinMax', 'None'] = 'Robust', 
    test_shot : Optional[int] = 21310
    ):
    
    data_disrupt = pd.read_csv(filepath['disrupt'], encoding = "euc-kr")
    data_efit = pd.read_csv(filepath['efit'])
    data_ece = pd.read_csv(filepath['ece'])
    data_diag = pd.read_csv(filepath['diag'])
    
    # train / valid / test data split
    shot_list = np.unique(data_disrupt.shot.values)
    
    if test_shot is not None:
        shot_list = np.array([shot for shot in shot_list if int(shot) != test_shot])

    # stochastic train_test_split
    if random_state is not None:
        shot_train, shot_test = train_test_split(shot_list, test_size = 0.2, random_state = random_state)
        shot_train, shot_valid = train_test_split(shot_train, test_size = 0.2, random_state = random_state)
    else:
    # deterministic train_test_split
        shot_train, shot_test = deterministic_split(shot_list, test_size = 0.2)
        shot_train, shot_valid = deterministic_split(shot_train, test_size = 0.2)
    
    train = {
        "efit": data_efit[data_efit.shot.isin(shot_train)],
        "ece":data_ece[data_ece.shot.isin(shot_train)],
        "diag":data_diag[data_diag.shot.isin(shot_train)],
        "disrupt":data_disrupt[data_disrupt.shot.isin(shot_train)],
    }
    
    valid = {
        "efit": data_efit[data_efit.shot.isin(shot_valid)],
        "ece":data_ece[data_ece.shot.isin(shot_valid)],
        "diag":data_diag[data_diag.shot.isin(shot_valid)],
        "disrupt":data_disrupt[data_disrupt.shot.isin(shot_valid)],
    }
    
    test = {
        "efit": data_efit[data_efit.shot.isin(shot_test)],
        "ece":data_ece[data_ece.shot.isin(shot_test)],
        "diag":data_diag[data_diag.shot.isin(shot_test)],
        "disrupt":data_disrupt[data_disrupt.shot.isin(shot_test)],
    }
        
    if scaler == 'Robust':
        scaler = {
            "efit":RobustScaler(),
            "ece":RobustScaler(),
            "diag":RobustScaler(),
        }
    elif scaler == 'Standard':
        scaler = {
            "efit":StandardScaler(),
            "ece":StandardScaler(),
            "diag":StandardScaler(),
        }
    elif scaler == 'MinMax':
        scaler = {
            "efit":MinMaxScaler(),
            "ece":MinMaxScaler(),
            "diag":MinMaxScaler(),
        }
    else:
        scaler = None
    
    if scaler is not None:
        for key in scaler.keys():
            scaler[key].fit(train[key][config.COLUMN_NAME_SCALER[key]])

    return train, valid, test, scaler
    
# Multi sginal dataset for generating probability curve
class MultiSignalDataset(Dataset):
    def __init__(
        self, 
        data_disrupt : pd.DataFrame, 
        data_efit:pd.DataFrame, 
        data_ece:pd.DataFrame, 
        data_diag:pd.DataFrame, 
        seq_len_efit:int = 20, 
        seq_len_ece:int = 100,
        seq_len_diag:int = 100, 
        dist:int = 4,
        dt:float = 0.01, 
        scaler_efit=None,
        scaler_ece=None,
        scaler_diag=None, 
        mode : Literal['TQ', 'CQ'] = 'CQ', 
        ):
        
        # KSTAR dataset
        self.data_disrupt = data_disrupt
        self.data_efit = data_efit
        self.data_ece = data_ece
        self.data_diag = data_diag
        
        # input data configuration
        self.seq_len_efit = seq_len_efit
        self.seq_len_ece = seq_len_ece
        self.seq_len_diag = seq_len_diag
        self.dt = dt
    
        # prediction task
        self.dist = dist
        self.mode = mode
        
        # scaler
        self.scaler_efit = scaler_efit
        self.scaler_ece = scaler_ece
        self.scaler_diag = scaler_diag
        
        # dataloader setup
        self.indices = []
        self.time_slice = []
        self.labels = []
        self.shot_num = []
        self.shot_list = []
        self.get_shot_num = False
        self.n_classes = 2
        
        # generate indices
        self.generate_indices()
        
        # preprocessing
        self.preprocessing()
            
    def generate_indices(self):
        tTQend = self.data_disrupt.t_tmq.values[0]
        tftsrt = self.data_disrupt.t_flattop_start.values[0]
        tipminf = self.data_disrupt.t_ip_min_fault.values[0]
            
        if self.mode == 'TQ':
            t_disrupt = tTQend
        elif self.mode =='CQ':
            t_disrupt = tipminf
            
        idx_efit = int(tftsrt / self.dt) // 5
        idx_ece = int(tftsrt / self.dt)
        idx_diag = int(tftsrt / self.dt)
            
        idx_efit_last = len(self.data_efit.index)
        idx_ece_last = len(self.data_ece.index)
        idx_diag_last = len(self.data_diag.index)
        
        t_max = self.data_ece['time'].values[-1]
        
        while(idx_ece < idx_ece_last or idx_diag < idx_diag_last):
                
            t_efit = self.data_efit.iloc[idx_efit]['time']
            t_ece = self.data_ece.iloc[idx_ece]['time']
            t_diag = self.data_diag.iloc[idx_diag]['time']
            
            assert abs(t_diag - t_ece) <= 0.02, "Dignostic data and ECE data are not consistent"
                
            # synchronize t_efit and t_ece
            if t_efit <= t_ece - 0.05:
                t_diff = t_ece - t_efit
                idx_efit += int(round(t_diff / self.dt / 5, 1))
                
            if t_ece >= tftsrt and t_ece <= t_max - self.dt * (self.seq_len_ece + self.dist) * 1.25:
                indx_ece = self.data_ece.index.values[idx_ece]
                indx_efit = self.data_efit.index.values[idx_efit]
                indx_diag = self.data_diag.index.values[idx_diag]
                
                self.time_slice.append(t_ece + self.dt * self.dist + self.seq_len_ece * self.dt)
                self.indices.append((indx_efit, indx_ece, indx_diag))
                idx_ece += 1
                idx_diag += 1
            
            elif t_ece < tftsrt:
                idx_ece += 5
                idx_diag += 5
                
            elif t_ece > t_max - self.dt * (self.seq_len_ece + self.dist) * 1.25:
                break
            
            else:
                idx_ece += 5
                idx_diag += 5
                
    def preprocessing(self):
        
        if self.scaler_efit is not None:
            self.data_efit[config.EFIT + config.EFIT_FE] = self.scaler_efit.transform(self.data_efit[config.EFIT + config.EFIT_FE])
            
        if self.scaler_ece is not None:
            self.data_ece[config.ECE] = self.scaler_ece.transform(self.data_ece[config.ECE])
        
        if self.scaler_diag is not None:
            self.data_diag[config.DIAG] = self.scaler_diag.transform(self.data_diag[config.DIAG])
            
        print("# Inference | preprocessing the data")
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx : int):
        indx_efit, indx_ece, indx_diag = self.indices[idx]
        data_efit = self.data_efit.loc[indx_efit + 1:indx_efit + self.seq_len_efit, config.EFIT + config.EFIT_FE].values.reshape(-1, self.seq_len_efit)
        data_ece = self.data_ece.loc[indx_ece + 1:indx_ece + self.seq_len_ece, config.ECE].values.reshape(-1, self.seq_len_ece)
        data_diag = self.data_diag.loc[indx_diag + 1:indx_diag + self.seq_len_diag, config.DIAG].values.reshape(-1, self.seq_len_diag)
 
        data_efit = torch.from_numpy(data_efit).float()
        data_ece = torch.from_numpy(data_ece).float()
        data_diag = torch.from_numpy(data_diag).float()
        
        data = {
            "efit":data_efit,
            "ece":data_ece,
            "diag":data_diag,
        }
    
        return data

def plot_shot_info(
        filepath : Dict[str, str],
        shot_num : Optional[int] = None,
    ):
    
    save_dir = "./results/shot_{}_info.png".format(shot_num)
    
    data_disrupt = pd.read_csv(filepath['disrupt'], encoding = "euc-kr")
    data_efit = pd.read_csv(filepath['efit'])
    data_ece = pd.read_csv(filepath['ece'])
    data_diag = pd.read_csv(filepath['diag'])
    
    tTQend = data_disrupt[data_disrupt.shot == shot_num].t_tmq.values[0]
    tftsrt = data_disrupt[data_disrupt.shot == shot_num].t_flattop_start.values[0]
    tipminf = data_disrupt[data_disrupt.shot == shot_num].t_ip_min_fault.values[0]
    
    t_disrupt = tTQend
    t_current = tipminf
    
    plot_efit = data_efit[data_efit.shot == shot_num]
    plot_ece = data_ece[data_ece.shot == shot_num]
    plot_diag = data_diag[data_diag.shot == shot_num]
    
    # plot the disruption probability with plasma status
    fig = plt.figure(figsize = (21, 8))
    fig.suptitle("Disruption prediction with shot : {}".format(shot_num))
    gs = GridSpec(nrows = 4, ncols = 3)
    
    # EFIT plot : betap, internal inductance, q95, plasma current
    # plasma current
    ax_ip = fig.add_subplot(gs[0,0])
    ax_ip.plot(plot_efit['time'], plot_efit['\\ipmhd'], label = 'Ip')
    ax_ip.text(0.85, 0.8, "Ip", transform = ax_ip.transAxes)
    ax_ip.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_ip.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")

    # betap
    ax_betap = fig.add_subplot(gs[1,0])
    ax_betap.plot(plot_efit['time'], plot_efit['\\betap'], label = 'betap')
    ax_betap.text(0.85, 0.8, "betap", transform = ax_betap.transAxes)
    ax_betap.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_betap.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    
    # li
    ax_li = fig.add_subplot(gs[2,0])
    ax_li.plot(plot_efit['time'], plot_efit['\\li'], label = 'li')
    ax_li.text(0.85, 0.8, "li", transform = ax_li.transAxes)
    ax_li.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_li.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    
    # q95
    ax_q95 = fig.add_subplot(gs[3,0])
    ax_q95.plot(plot_efit['time'], plot_efit['\\q95'], label = 'q95')
    ax_q95.text(0.85, 0.8, "q95", transform = ax_q95.transAxes)
    ax_q95.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_q95.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    ax_q95.set_ylim([0, 10.0])
    ax_q95.set_xlabel("time(unit:s)")
    
    # ECE part
    ax_ece = fig.add_subplot(gs[:,1])
    
    for name in config.ECE:
        ax_ece.plot(plot_ece['time'], plot_ece[name], label = name[1:])

    ax_ece.axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed")
    ax_ece.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_ece.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    ax_ece.set_ylabel("ECE(unit:KeV)")
    ax_ece.set_xlabel("time(unit:s)")
    ax_ece.legend(loc = 'upper right')
    
    # Diagnostic part
    # LM
    ax_lm = fig.add_subplot(gs[0,2])
    
    for col in config.LM:
        ax_lm.plot(plot_diag['time'], plot_diag[col] / plot_diag[col].abs().max(), label = col[1:])
        
    ax_lm.text(0.85, 0.8, "LM signals", transform = ax_lm.transAxes)
    ax_lm.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_lm.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    ax_lm.set_xlabel("time(unit:s)")
    ax_lm.legend(loc = 'upper right')
    
    # EC heating
    ax_ech = fig.add_subplot(gs[1,2])

    for col in config.ECH:
        ax_ech.plot(plot_diag['time'], plot_diag[col] / plot_diag[col].abs().max(), label = col[1:])
    
    ax_ech.text(0.85, 0.8, "EC heating", transform = ax_ech.transAxes)
    ax_ech.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_ech.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    # ax_ech.set_ylim([0, 10.0])
    ax_ech.set_xlabel("time(unit:s)")
    ax_ech.legend(loc = 'upper right')
    
    # NB heating
    ax_nbh = fig.add_subplot(gs[2,2])

    for col in config.NBH:
        ax_nbh.plot(plot_diag['time'], plot_diag[col] / plot_diag[col].abs().max(), label = col[1:])
    
    ax_nbh.text(0.85, 0.8, "NB heating", transform = ax_nbh.transAxes)
    ax_nbh.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_nbh.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    # ax_nbh.set_ylim([0, 10.0])
    ax_nbh.set_xlabel("time(unit:s)")
    ax_nbh.legend(loc = 'upper right')
    
    # DL
    ax_dl = fig.add_subplot(gs[3,2])

    for col in config.DL:
        ax_dl.plot(plot_diag['time'], plot_diag[col] / plot_diag[col].abs().max(), label = col[1:])
    
    ax_dl.text(0.85, 0.8, "Diamagnetic Loop", transform = ax_dl.transAxes)
    ax_dl.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_dl.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    # ax_dl.set_ylim([0, 10.0])
    ax_dl.set_xlabel("time(unit:s)")
    ax_dl.legend(loc = 'upper right')
    
    fig.tight_layout()
    plt.savefig(save_dir, facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)
    return 

def generate_prob_curve_from_0D(
        filepath : Dict[str, str],
        model : torch.nn.Module, 
        device : str = "cpu",
        save_dir : Optional[str] = "./results/disruption_probs_curve.png",
        shot_num : Optional[int] = None,
        seq_len_efit:int = 20, 
        seq_len_ece:int = 100,
        seq_len_diag:int = 100, 
        dist : Optional[int] = None,
        dt : Optional[float] = None,
        data_disrupt:Optional[pd.DataFrame] = None,
        data_efit:Optional[pd.DataFrame] = None,
        data_ece:Optional[pd.DataFrame] = None,
        data_diag:Optional[pd.DataFrame] = None, 
        mode : Literal['TQ', 'CQ'] = 'CQ', 
        scaler_type = None
    ):
     
    # obtain tTQend, tipmin and tftsrt
    _, _, _, scaler_list = preparing_0D_dataset(filepath, random_state = STATE_FIXED, scaler = scaler_type, test_shot = shot_num)

    data_disrupt = pd.read_csv(filepath['disrupt'], encoding = "euc-kr")
    data_efit = pd.read_csv(filepath['efit'])
    data_ece = pd.read_csv(filepath['ece'])
    data_diag = pd.read_csv(filepath['diag'])
    
    tTQend = data_disrupt[data_disrupt.shot == shot_num].t_tmq.values[0]
    tftsrt = data_disrupt[data_disrupt.shot == shot_num].t_flattop_start.values[0]
    tipminf = data_disrupt[data_disrupt.shot == shot_num].t_ip_min_fault.values[0]
    
    t_disrupt = tTQend
    t_current = tipminf
    
    data_efit = data_efit[data_efit.shot == shot_num]
    data_ece = data_ece[data_ece.shot == shot_num]
    data_diag = data_diag[data_diag.shot == shot_num]
    
    plot_efit = data_efit.copy(deep = True)
    plot_ece = data_ece.copy(deep = True)
    plot_diag = data_diag.copy(deep = True)
    
    dataset = MultiSignalDataset(
        data_disrupt,
        data_efit,
        data_ece,
        data_diag,
        seq_len_efit,
        seq_len_ece,
        seq_len_diag,
        dist,
        dt,
        scaler_list['efit'],
        scaler_list['ece'],
        scaler_list['diag'],
        mode
    )

    prob_list = []
    is_disruption = []
    feature_dict = None
    is_feat_saved = False
    data_for_feature = None

    model.to(device)
    model.eval()
    
    from tqdm.auto import tqdm
    
    for idx in tqdm(range(dataset.__len__())):
        with torch.no_grad():
            data = dataset.__getitem__(idx)
            
            for key in data.keys():
                data[key] = data[key].unsqueeze(0)
            
            output = model(data)
            probs = torch.nn.functional.softmax(output, dim = 1)[:,0]
            probs = probs.cpu().detach().numpy().tolist()

            prob_list.extend(
                probs
            )
            
            is_disruption.extend(
                torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1].cpu().detach().numpy().tolist()
            )
            
            t = dataset.time_slice[idx]
            
            if t >= t_disrupt and not is_feat_saved and np.where(probs[0] > 0.5,0,1) == 0:
                data_for_feature = data
                is_feat_saved = True
                
    if data_for_feature:
        feature_dict = compute_relative_importance(
            data_for_feature,
            model,
            0,
            None,
            8,
            device
        )
    
    n_ftsrt = int(dataset.time_slice[0] // dt)
    time_slice = [v for v in dataset.time_slice]
    time_slice = np.linspace(0, dataset.time_slice[0] - dt, n_ftsrt).tolist() + time_slice 
    prob_list = [0] * n_ftsrt + prob_list 
    
    print("time_slice : ", len(time_slice))
    print("prob_list : ", len(prob_list))

    print("\n(Info) flat-top : {:.3f}(s) | thermal quench : {:.3f}(s) | current quench : {:.3f}(s)\n".format(tftsrt, tTQend, tipminf))
    
    # plot the disruption probability with plasma status
    if feature_dict:
        fig = plt.figure(figsize = (28, 8))
        fig.suptitle("Disruption prediction with shot : {}".format(shot_num))
        gs = GridSpec(nrows = 4, ncols = 4)
    else:
        fig = plt.figure(figsize = (21, 8))
        fig.suptitle("Disruption prediction with shot : {}".format(shot_num))
        gs = GridSpec(nrows = 4, ncols = 3)
    
    # EFIT plot : betap, internal inductance, q95, plasma current
    # plasma current
    ax_ip = fig.add_subplot(gs[0,0])
    ax_ip.plot(plot_diag['time'], plot_diag['\\RC03'].apply(lambda x : x if x > 0 else abs(x)), label = 'Ip')
    ax_ip.text(0.85, 0.8, "Ip", transform = ax_ip.transAxes)
    ax_ip.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_ip.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")

    # betap
    ax_betap = fig.add_subplot(gs[1,0])
    ax_betap.plot(plot_efit['time'], plot_efit['\\betap'], label = 'betap')
    ax_betap.text(0.85, 0.8, "betap", transform = ax_betap.transAxes)
    ax_betap.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_betap.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    
    # li
    ax_li = fig.add_subplot(gs[2,0])
    ax_li.plot(plot_efit['time'], plot_efit['\\li'], label = 'li')
    ax_li.text(0.85, 0.8, "li", transform = ax_li.transAxes)
    ax_li.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_li.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    
    # q95
    ax_q95 = fig.add_subplot(gs[3,0])
    ax_q95.plot(plot_efit['time'], plot_efit['\\q95'], label = 'q95')
    ax_q95.text(0.85, 0.8, "q95", transform = ax_q95.transAxes)
    ax_q95.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_q95.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    ax_q95.set_ylim([0, 10.0])
    ax_q95.set_xlabel("time(unit:s)")
    
    # ECE part
    ax_ece = fig.add_subplot(gs[:,1])
    
    for name in config.ECE:
        ax_ece.plot(plot_ece['time'], plot_ece[name], label = name[1:])

    ax_ece.axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed")
    ax_ece.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_ece.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    
    ax_ece.set_ylabel("ECE(unit:KeV)")
    ax_ece.set_xlabel("time(unit:s)")
    ax_ece.legend(loc = 'upper right')
    
    # Disruption prediction
    threshold_line = [0.5] * len(time_slice)
    ax_prob = fig.add_subplot(gs[:,2])
    ax_prob.plot(time_slice, prob_list, 'b', label = 'disrupt prob')
    ax_prob.plot(time_slice, threshold_line, 'k', label = "threshold(p = 0.5)")
    ax_prob.axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed", label = "flattop (t={:.3f})".format(tftsrt))
    ax_prob.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed", label = "TQ (t={:.3f})".format(t_disrupt))
    ax_prob.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed", label = "CQ (t={:.3f})".format(t_current))
    ax_prob.set_ylabel("probability")
    ax_prob.set_xlabel("time(unit:s)")
    ax_prob.set_ylim([0,1])
    ax_prob.set_xlim([0, max(time_slice) + dt * 8])
    ax_prob.legend(loc = 'upper right')
    
    if feature_dict:
        ax_feat = fig.add_subplot(gs[:,3])
        ax_feat.barh(list(feature_dict.keys()), list(feature_dict.values()))
        ax_feat.set_yticks([i for i in range(len(list(feature_dict.keys())))], labels = list(feature_dict.keys()))
        ax_feat.invert_yaxis()
        ax_feat.set_ylabel('Input features')
        ax_feat.set_xlabel('Relative feature importance')
    
    fig.tight_layout()

    if save_dir:
        plt.savefig(save_dir, facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)

    return fig, time_slice, prob_list

def generate_3D_feature_importance(
        filepath : Dict[str, str],
        model : torch.nn.Module, 
        device : str = "cpu",
        save_dir : Optional[str] = "./results/disruption_feature_importance.png",
        shot_num : Optional[int] = None,
        seq_len_efit:int = 20, 
        seq_len_ece:int = 100,
        seq_len_diag:int = 100, 
        dist : Optional[int] = None,
        dt : Optional[float] = None,
        data_disrupt:Optional[pd.DataFrame] = None,
        data_efit:Optional[pd.DataFrame] = None,
        data_ece:Optional[pd.DataFrame] = None,
        data_diag:Optional[pd.DataFrame] = None, 
        mode : Literal['TQ', 'CQ'] = 'CQ', 
        scaler_type = None
    ):
     
    # obtain tTQend, tipmin and tftsrt
    _, _, _, scaler_list = preparing_0D_dataset(filepath, random_state = STATE_FIXED, scaler = scaler_type, test_shot = shot_num)

    data_disrupt = pd.read_csv(filepath['disrupt'], encoding = "euc-kr")
    data_efit = pd.read_csv(filepath['efit'])
    data_ece = pd.read_csv(filepath['ece'])
    data_diag = pd.read_csv(filepath['diag'])
    
    tTQend = data_disrupt[data_disrupt.shot == shot_num].t_tmq.values[0]
    tftsrt = data_disrupt[data_disrupt.shot == shot_num].t_flattop_start.values[0]
    tipminf = data_disrupt[data_disrupt.shot == shot_num].t_ip_min_fault.values[0]
    
    t_disrupt = tTQend
    t_current = tipminf
    
    data_efit = data_efit[data_efit.shot == shot_num]
    data_ece = data_ece[data_ece.shot == shot_num]
    data_diag = data_diag[data_diag.shot == shot_num]
    
    plot_efit = data_efit.copy(deep = True)
    plot_ece = data_ece.copy(deep = True)
    plot_diag = data_diag.copy(deep = True)
    
    dataset = MultiSignalDataset(
        data_disrupt,
        data_efit,
        data_ece,
        data_diag,
        seq_len_efit,
        seq_len_ece,
        seq_len_diag,
        dist,
        dt,
        scaler_list['efit'],
        scaler_list['ece'],
        scaler_list['diag'],
        mode
    )

    prob_list = []
    is_disruption = []
    feature_dict = None

    model.to(device)
    model.eval()
    
    from tqdm.auto import tqdm
    
    for idx in tqdm(range(dataset.__len__())):
        data = dataset.__getitem__(idx)
        
        for key in data.keys():
            data[key] = data[key].unsqueeze(0)
        
        output = model(data)
        probs = torch.nn.functional.softmax(output, dim = 1)[:,0]
        probs = probs.cpu().detach().numpy().tolist()

        prob_list.extend(
            probs
        )
        
        is_disruption.extend(
            torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1].cpu().detach().numpy().tolist()
        )
        
        t = dataset.time_slice[idx]
        
        if t >= t_disrupt and t <= t_disrupt + dist * dt:
            if feature_dict is None:
                feat = compute_relative_importance(data, model, 0, None, 8, device)
                feature_dict = {}
                feature_dict['time'] = [t_disrupt - t + dist * dt]
                
                for key in feat.keys():
                    feature_dict[key] = [feat[key]]
            else:
                feat = compute_relative_importance(data, model, 0, None, 8, device)
                feature_dict['time'].append(t_disrupt - t + dist * dt)
                
                for key in feat.keys():
                    feature_dict[key].append(feat[key])
    
    n_ftsrt = int(dataset.time_slice[0] // dt)
    time_slice = [v for v in dataset.time_slice]
    time_slice = np.linspace(0, dataset.time_slice[0] - dt, n_ftsrt).tolist() + time_slice 
    prob_list = [0] * n_ftsrt + prob_list 
    
    print("time_slice : ", len(time_slice))
    print("prob_list : ", len(prob_list))

    print("\n(Info) flat-top : {:.3f}(s) | thermal quench : {:.3f}(s) | current quench : {:.3f}(s)\n".format(tftsrt, tTQend, tipminf))
    
    # plot the disruption probability with plasma status
    if feature_dict:
        fig = plt.figure(figsize = (28, 8))
        fig.suptitle("Disruption prediction with shot : {}".format(shot_num))
        gs = GridSpec(nrows = 4, ncols = 4)
    else:
        fig = plt.figure(figsize = (21, 8))
        fig.suptitle("Disruption prediction with shot : {}".format(shot_num))
        gs = GridSpec(nrows = 4, ncols = 3)
    
    # EFIT plot : betap, internal inductance, q95, plasma current
    # plasma current
    ax_ip = fig.add_subplot(gs[0,0])
    ax_ip.plot(plot_diag['time'], plot_diag['\\RC03'].apply(lambda x : x if x > 0 else abs(x)), label = 'Ip')
    ax_ip.text(0.85, 0.8, "Ip", transform = ax_ip.transAxes)
    ax_ip.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_ip.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")

    # betap
    ax_betap = fig.add_subplot(gs[1,0])
    ax_betap.plot(plot_efit['time'], plot_efit['\\betap'], label = 'betap')
    ax_betap.text(0.85, 0.8, "betap", transform = ax_betap.transAxes)
    ax_betap.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_betap.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    
    # li
    ax_li = fig.add_subplot(gs[2,0])
    ax_li.plot(plot_efit['time'], plot_efit['\\li'], label = 'li')
    ax_li.text(0.85, 0.8, "li", transform = ax_li.transAxes)
    ax_li.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_li.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    
    # q95
    ax_q95 = fig.add_subplot(gs[3,0])
    ax_q95.plot(plot_efit['time'], plot_efit['\\q95'], label = 'q95')
    ax_q95.text(0.85, 0.8, "q95", transform = ax_q95.transAxes)
    ax_q95.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_q95.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    ax_q95.set_ylim([0, 10.0])
    ax_q95.set_xlabel("time(unit:s)")
    
    # ECE part
    ax_ece = fig.add_subplot(gs[:,1])
    
    for name in config.ECE:
        ax_ece.plot(plot_ece['time'], plot_ece[name], label = name[1:])

    ax_ece.axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed")
    ax_ece.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_ece.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    
    ax_ece.set_ylabel("ECE(unit:KeV)")
    ax_ece.set_xlabel("time(unit:s)")
    ax_ece.legend(loc = 'upper right')
    
    # Disruption prediction
    threshold_line = [0.5] * len(time_slice)
    ax_prob = fig.add_subplot(gs[:,2])
    ax_prob.plot(time_slice, prob_list, 'b', label = 'disrupt prob')
    ax_prob.plot(time_slice, threshold_line, 'k', label = "threshold(p = 0.5)")
    ax_prob.axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed", label = "flattop (t={:.3f})".format(tftsrt))
    ax_prob.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed", label = "TQ (t={:.3f})".format(t_disrupt))
    ax_prob.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed", label = "CQ (t={:.3f})".format(t_current))
    ax_prob.set_ylabel("probability")
    ax_prob.set_xlabel("time(unit:s)")
    ax_prob.set_ylim([0,1])
    ax_prob.set_xlim([0, max(time_slice) + dt * 8])
    ax_prob.legend(loc = 'upper right')
    
    if feature_dict:
        ax_feat = fig.add_subplot(gs[:,3], projection = '3d')
        
        feature_importance = {}
        for key in feature_dict.keys():
            if key != 'time':
                feature_importance[key] = feature_dict[key]
        
        for idx, t in enumerate(feature_dict['time']):
            hist = [feature_importance[key][idx] for key in feature_importance.keys()]
            ax_feat.bar(list(feature_importance.keys()), hist, zs = t, zdir = 'y', alpha = 0.8, in_layout = True)
        
        ax_feat.set_xticks([i for i in range(len(list(feature_importance.keys())))], labels = list(feature_importance.keys()), rotation = 90, fontsize = 8)
        ax_feat.set_yticks(ticks = feature_dict['time'], labels = ["{:02d} ms".format(int(i * dt * 1000)) for i in reversed(range(1, dist + 1))])
        ax_feat.set_zlabel('Feature importance')
        ax_feat.set_ylabel('$t_{pred}$ before TQ (unit:ms)')
        ax_feat.set_zlim([0,1])
    
    fig.tight_layout()

    if save_dir:
        plt.savefig(save_dir, facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)
 
    return fig, time_slice, prob_list

def generate_bayes_prob_curve(
        filepath : Dict[str, str],
        model : torch.nn.Module, 
        device : str = "cpu",
        save_dir : Optional[str] = "./results/disruption_bayes_probs_curve.png",
        shot_num : Optional[int] = None,
        seq_len_efit:int = 20, 
        seq_len_ece:int = 100,
        seq_len_diag:int = 100, 
        dist : Optional[int] = None,
        dt : Optional[float] = None,
        data_disrupt:Optional[pd.DataFrame] = None,
        data_efit:Optional[pd.DataFrame] = None,
        data_ece:Optional[pd.DataFrame] = None,
        data_diag:Optional[pd.DataFrame] = None, 
        mode : Literal['TQ', 'CQ'] = 'CQ', 
        scaler_type = None,
    ):
     
    # obtain tTQend, tipmin and tftsrt
    _, _, _, scaler_list = preparing_0D_dataset(filepath, random_state = STATE_FIXED, scaler = scaler_type, test_shot = shot_num)

    data_disrupt = pd.read_csv(filepath['disrupt'], encoding = "euc-kr")
    data_efit = pd.read_csv(filepath['efit'])
    data_ece = pd.read_csv(filepath['ece'])
    data_diag = pd.read_csv(filepath['diag'])
    
    tTQend = data_disrupt[data_disrupt.shot == shot_num].t_tmq.values[0]
    tftsrt = data_disrupt[data_disrupt.shot == shot_num].t_flattop_start.values[0]
    tipminf = data_disrupt[data_disrupt.shot == shot_num].t_ip_min_fault.values[0]
    
    t_disrupt = tTQend
    t_current = tipminf
    
    data_efit = data_efit[data_efit.shot == shot_num]
    data_ece = data_ece[data_ece.shot == shot_num]
    data_diag = data_diag[data_diag.shot == shot_num]
    
    plot_efit = data_efit.copy(deep = True)
    plot_ece = data_ece.copy(deep = True)
    plot_diag = data_diag.copy(deep = True)
    
    dataset = MultiSignalDataset(
        data_disrupt,
        data_efit,
        data_ece,
        data_diag,
        seq_len_efit,
        seq_len_ece,
        seq_len_diag,
        dist,
        dt,
        scaler_list['efit'],
        scaler_list['ece'],
        scaler_list['diag'],
        mode
    )

    prob_list = []
    aus = []
    eus = []
    is_disruption = []
    feature_dict = None

    model.to(device)
    model.eval()
    
    from tqdm.auto import tqdm
    
    for idx in tqdm(range(dataset.__len__())):
        
        data = dataset.__getitem__(idx)
        
        for key in data.keys():
            data[key] = data[key].unsqueeze(0)
        
        output = model(data)
        probs = torch.nn.functional.softmax(output, dim = 1)[:,0]
        probs = probs.cpu().detach().numpy().tolist()
        
        au, eu = compute_uncertainty_per_data(model, data, device = device, n_samples = 128, normalized=False)
        aus.append(au)
        eus.append(eu)
        
        prob_list.extend(
            probs
        )
        
        is_disruption.extend(
            torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1].cpu().detach().numpy().tolist()
        )
        
        t = dataset.time_slice[idx]
        
        if t >= t_disrupt and t <= t_disrupt + dist * dt:
            if feature_dict is None:
                feat = compute_relative_importance(data, model, 0, None, 8, device)
                feature_dict = {}
                feature_dict['time'] = [t_disrupt - t + dist * dt]
                
                for key in feat.keys():
                    feature_dict[key] = [feat[key]]
            else:
                feat = compute_relative_importance(data, model, 0, None, 8, device)
                feature_dict['time'].append(t_disrupt - t + dist * dt)
                
                for key in feat.keys():
                    feature_dict[key].append(feat[key])
                
    time_slice = [v for v in dataset.time_slice]
    aus = np.array(aus).reshape(-1,2)
    eus = np.array(eus).reshape(-1,2)
    
    print("time_slice : ", len(time_slice))
    print("prob_list : ", len(prob_list))
    print("uncertainty : ", len(aus))

    print("\n(Info) flat-top : {:.3f}(s) | thermal quench : {:.3f}(s) | current quench : {:.3f}(s)\n".format(tftsrt, tTQend, tipminf))
    
    # plot the disruption probability with plasma status
    fig = plt.figure(figsize = (21, 8))
    fig.suptitle("Disruption prediction with shot : {}".format(shot_num))
    gs = GridSpec(nrows = 4, ncols = 3)
    
    # Disruption prediction
    threshold_line = [0.5] * len(time_slice)
    ax_prob = fig.add_subplot(gs[:,0])
    ax_prob.plot(time_slice, prob_list, 'b', label = 'disrupt prob')
    ax_prob.plot(time_slice, threshold_line, 'k', label = "threshold(p = 0.5)")
    ax_prob.axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed", label = "flattop (t={:.3f})".format(tftsrt))
    ax_prob.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed", label = "TQ (t={:.3f})".format(t_disrupt))
    ax_prob.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed", label = "CQ (t={:.3f})".format(t_current))
    ax_prob.set_ylabel("probability")
    ax_prob.set_xlabel("time(unit:s)")
    ax_prob.set_ylim([0,1])
    ax_prob.set_xlim([t_disrupt - 2.0, max(time_slice) + dt * 8])
    ax_prob.legend(loc = 'upper right')
    
    ax_au = fig.add_subplot(gs[:,1])
    ax_au.plot(time_slice, aus[:,0], 'b', label = 'aleatoric uncertainty')
    ax_au.axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed", label = "flattop (t={:.3f})".format(tftsrt))
    ax_au.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed", label = "TQ (t={:.3f})".format(t_disrupt))
    ax_au.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed", label = "CQ (t={:.3f})".format(t_current))
    ax_au.set_xlim([t_disrupt - 2.0, max(time_slice) + dt * 8])
    ax_au.tick_params(axis = 'y', labelcolor = 'b')
    ax_au.set_xlabel("time(unit:s)")
    ax_au.set_ylabel("Aleatoric uncertainty")
    
    ax_eu = ax_au.twinx()
    ax_eu.plot(time_slice, eus[:,0], 'r', label = 'epistemic uncertainty')
    ax_eu.tick_params(axis = 'y', labelcolor = 'r')
    ax_eu.set_ylabel("Epistemic uncertainty")

    ax_feat = fig.add_subplot(gs[:,2], projection = '3d')
    feature_importance = {}
    for key in feature_dict.keys():
        if key != 'time':
            feature_importance[key] = feature_dict[key]
    
    for idx, t in enumerate(feature_dict['time']):
        hist = [feature_importance[key][idx] for key in feature_importance.keys()]
        ax_feat.bar(list(feature_importance.keys()), hist, zs = t, zdir = 'y', alpha = 0.8, in_layout = True)
    
    ax_feat.set_xticks([i for i in range(len(list(feature_importance.keys())))], labels = list(feature_importance.keys()), rotation = 90, fontsize = 8)
    ax_feat.set_yticks(ticks = feature_dict['time'], labels = ["{:02d} ms".format(int(i * dt * 1000)) for i in reversed(range(1, dist + 1))])
    ax_feat.set_zlabel('Feature importance')
    ax_feat.set_ylabel('$t_{pred}$ before TQ (unit:ms)')
    ax_feat.set_zlim([0,1])

    fig.tight_layout()

    if save_dir:
        plt.savefig(save_dir, facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)

    return fig, time_slice, prob_list

def plot_learning_curve(train_loss, valid_loss, train_f1, valid_f1, figsize : Tuple[int,int] = (12,6), save_dir : str = "./results/learning_curve.png"):
    
    x_epochs = range(1, len(train_loss) + 1)
    
    if len(train_loss) == 0:
        print("The length of training loss is 0, skip plotting learning curves")
        return

    plt.figure(1, figsize=figsize, facecolor = 'white')
    plt.subplot(1,2,1)
    plt.plot(x_epochs, train_loss, 'ro-', label = "train loss")
    plt.plot(x_epochs, valid_loss, 'bo-', label = "valid loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("train and valid loss curve")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(x_epochs, train_f1, 'ro-', label = "train f1 score")
    plt.plot(x_epochs, valid_f1, 'bo-', label = "valid f1 score")
    plt.xlabel("epochs")
    plt.ylabel("f1 score")
    plt.title("train and valid f1 score curve")
    plt.legend()
    plt.savefig(save_dir)

def measure_computation_time(model : torch.nn.Module, input_shape : Tuple, n_samples : int = 1, device : str = "cpu"):
    
    model.to(device)
    model.eval()
    
    t_measures = []
    
    for n_iter in range(n_samples):
        
        torch.cuda.empty_cache()
        torch.cuda.init()
        
        with torch.no_grad():
            sample_data = torch.zeros(input_shape)
            t_start = time.time()
            sample_output = model(sample_data.to(device))
            t_end = time.time()
            dt = t_end - t_start
            t_measures.append(dt)
            
            sample_output.cpu()
            sample_data.cpu()
        
        del sample_data
        del sample_output
        
    # statistical summary
    dt_means = np.mean(t_measures)
    dt_std = np.std(t_measures)
    
    return dt_means, dt_std, t_measures    