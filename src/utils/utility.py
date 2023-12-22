import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import os, random, pickle
import pandas as pd
import numpy as np
from typing import Optional, List, Literal, Union, Tuple, Dict
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator
from src.config import Config
from src.feature_importance import compute_relative_importance
from src.models.BNN import compute_uncertainty_per_data, compute_ensemble_probability
from src.utils.compute import moving_avarage_smoothing
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
    test_shot : Optional[int] = 21310,
    is_split:bool = True,
    ):
    
    data_disrupt = pd.read_csv(filepath['disrupt'], encoding = "euc-kr")
    data_efit = pd.read_csv(filepath['efit']) if filepath['efit'].split(".")[-1] == "csv" else pd.read_pickle(filepath['efit'])
    data_ece = pd.read_csv(filepath['ece']) if filepath['ece'].split(".")[-1] == "csv" else pd.read_pickle(filepath['ece'])
    data_diag = pd.read_csv(filepath['diag']) if filepath['diag'].split(".")[-1] == "csv" else pd.read_pickle(filepath['diag'])
    
    # year selection: 2019, 2020, 2021, 2022
    data_disrupt_lm = data_disrupt[data_disrupt.shot.isin([20948,20951,20975])]
    data_disrupt = data_disrupt[data_disrupt.Year.astype(int).isin([2019,2020,2021,2022])]
    
    # train / valid / test data split
    ori_shot_list = np.unique(data_disrupt.shot.values)
    
    if test_shot is not None:
        shot_list = np.array([shot for shot in ori_shot_list if int(shot) != test_shot])
    else:
        shot_list = ori_shot_list

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
            
    if is_split:
        return train, valid, test, scaler
    else:
        # adding LM disruptive plasmas : 2023.11.23
        ori_shot_list = np.array(ori_shot_list.tolist() + [20948,20951,20975])
        ori_shot_list = np.unique(ori_shot_list)
        data_disrupt = pd.concat([data_disrupt, data_disrupt_lm])
        
        total = {
            "efit": data_efit[data_efit.shot.isin(ori_shot_list)],
            "ece":data_ece[data_ece.shot.isin(ori_shot_list)],
            "diag":data_diag[data_diag.shot.isin(ori_shot_list)],
            "disrupt":data_disrupt[data_disrupt.shot.isin(ori_shot_list)],
        }
        return total, scaler
    
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
        dist_warning:int = 400,
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
        self.dist_warning = dist_warning
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
            
        idx_efit = (self.data_efit['time'] - tftsrt).abs().argmin()
        idx_ece =(self.data_ece['time'] - tftsrt).abs().argmin()
        idx_diag =(self.data_diag['time'] - tftsrt).abs().argmin() 
            
        idx_efit_last = len(self.data_efit.index)
        idx_ece_last = len(self.data_ece.index)
        idx_diag_last = len(self.data_diag.index)
        
        t_max = self.data_ece['time'].values[-1]
        
        if self.dt == 0.01:
            dt_efit_ece = 0.045
            ratio = 5
        else:
            dt_efit_ece = 0.009
            ratio = 10
        
        while(idx_ece < idx_ece_last and idx_diag < idx_diag_last and idx_efit < idx_efit_last):
                
            t_efit = self.data_efit.iloc[idx_efit]['time']
            t_ece = self.data_ece.iloc[idx_ece]['time']
            t_diag = self.data_diag.iloc[idx_diag]['time']
            
            assert abs(t_diag - t_ece) <= 2 * self.dt, "Dignostic data and ECE data are not consistent"
            
            # synchronize t_efit and t_ece
            if t_efit <= t_ece - dt_efit_ece:
                t_diff = t_ece - t_efit
                idx_efit += int(round(t_diff / self.dt / ratio, 1))
                
            if t_ece >= tftsrt and t_ece <= t_disrupt - self.dt * (self.seq_len_ece + self.dist) * 2.0:
                indx_ece = self.data_ece.index.values[idx_ece]
                indx_efit = self.data_efit.index.values[idx_efit]
                indx_diag = self.data_diag.index.values[idx_diag]
                
                self.time_slice.append(t_ece + self.seq_len_ece * self.dt)
                self.indices.append((indx_efit, indx_ece, indx_diag))
                
                if self.dt == 0.01:
                    idx_ece += 1
                    idx_diag += 1
                        
                elif self.dt == 0.001:
                    idx_ece += 10
                    idx_diag += 10
                    
            elif t_ece >= t_disrupt - self.dt * (self.seq_len_ece + self.dist) * 2.0 and t_ece <= t_disrupt - self.dt * self.seq_len_ece:
                indx_ece = self.data_ece.index.values[idx_ece]
                indx_efit = self.data_efit.index.values[idx_efit]
                indx_diag = self.data_diag.index.values[idx_diag]
                
                self.time_slice.append(t_ece + self.seq_len_ece * self.dt)
                self.indices.append((indx_efit, indx_ece, indx_diag))
                
                if self.dt == 0.01:
                    idx_ece += 1
                    idx_diag += 1
                        
                elif self.dt == 0.001:
                    idx_ece += 1
                    idx_diag += 1
            
            elif t_ece >= t_disrupt - self.dt * self.seq_len_ece and t_ece <= t_max - self.dt * (self.seq_len_ece + self.dist) * 1.0:
                indx_ece = self.data_ece.index.values[idx_ece]
                indx_efit = self.data_efit.index.values[idx_efit]
                indx_diag = self.data_diag.index.values[idx_diag]
                
                self.time_slice.append(t_ece + self.seq_len_ece * self.dt)
                self.indices.append((indx_efit, indx_ece, indx_diag))
                
                if self.dt == 0.01:
                    idx_ece += 1
                    idx_diag += 1
                        
                elif self.dt == 0.001:
                    idx_ece += 5
                    idx_diag += 5

            elif t_ece < tftsrt:  
                        
                if self.dt == 0.01:
                    idx_ece += 1
                    idx_diag += 1
                        
                elif self.dt == 0.001:
                    idx_ece += 10
                    idx_diag += 10
                
            elif t_ece > t_max - self.dt * (self.seq_len_ece + self.dist) * 1.25:
                break
            
            else:
                if self.dt == 0.01:
                    idx_ece += 1
                    idx_diag += 1
                        
                elif self.dt == 0.001:
                    idx_ece += 10
                    idx_diag += 10
                
    def preprocessing(self):
        
        if self.scaler_efit is not None:
            self.data_efit.loc[:,config.EFIT + config.EFIT_FE] = self.scaler_efit.transform(self.data_efit[config.EFIT + config.EFIT_FE].values)
            
        if self.scaler_ece is not None:
            self.data_ece.loc[:,config.ECE] = self.scaler_ece.transform(self.data_ece[config.ECE].values)
        
        if self.scaler_diag is not None:
            self.data_diag.loc[:,config.DIAG + config.DIAG_FE] = self.scaler_diag.transform(self.data_diag[config.DIAG + config.DIAG_FE].values)
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx : int):
        indx_efit, indx_ece, indx_diag = self.indices[idx]
        data_efit = self.data_efit.loc[indx_efit + 1:indx_efit + self.seq_len_efit, config.EFIT + config.EFIT_FE].values.reshape(-1, self.seq_len_efit)
        data_ece = self.data_ece.loc[indx_ece + 1:indx_ece + self.seq_len_ece, config.ECE].values.reshape(-1, self.seq_len_ece)
        data_diag = self.data_diag.loc[indx_diag + 1:indx_diag + self.seq_len_diag, config.DIAG + config.DIAG_FE].values.reshape(-1, self.seq_len_diag)
 
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
        save_dir: Optional[str] = None,
    ):
    
    data_disrupt = pd.read_csv(filepath['disrupt'], encoding = "euc-kr")    
    data_efit = pd.read_csv(filepath['efit']) if filepath['efit'].split(".")[-1] == "csv" else pd.read_pickle(filepath['efit'])
    data_ece = pd.read_csv(filepath['ece']) if filepath['ece'].split(".")[-1] == "csv" else pd.read_pickle(filepath['ece'])
    data_diag = pd.read_csv(filepath['diag']) if filepath['diag'].split(".")[-1] == "csv" else pd.read_pickle(filepath['diag'])
    
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
    ln1 = ax_ip.plot(plot_diag['time'], plot_diag['\\RC03'].abs(), label = 'Ip')
    
    ax_iv = ax_ip.twinx()
    ax_ip.axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed")
    ax_ip.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_ip.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    
    ln2 = ax_iv.plot(plot_diag['time'], plot_diag['\\VCM03'].abs() - plot_diag['\\RC03'].abs(), c = 'r', label = 'Iv')
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax_ip.legend(lns, labs, loc = 'upper right')
    ax_ip.set_ylabel("Ip")

    # betap
    ax_betap = fig.add_subplot(gs[1,0])
    ax_betap.plot(plot_efit['time'], plot_efit['\\betap'], label = 'betap')
    ax_betap.axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed")
    ax_betap.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_betap.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    ax_betap.legend(loc = 'upper right')
    ax_betap.set_ylabel("betap")
    
    # li
    ax_li = fig.add_subplot(gs[2,0])
    ax_li.plot(plot_efit['time'], plot_efit['\\li'], label = 'li')
    ax_li.axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed")
    ax_li.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_li.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    ax_li.legend(loc = 'upper right')
    ax_li.set_ylabel("li")
    
    # q95
    ax_q95 = fig.add_subplot(gs[3,0])
    ax_q95.plot(plot_efit['time'], plot_efit['\\q95'], label = 'q95')
    ax_q95.axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed")
    ax_q95.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_q95.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    ax_q95.set_ylim([0, 10.0])
    ax_q95.set_ylabel("q95")
    ax_q95.set_xlabel("time(unit:s)")
    ax_q95.legend(loc = 'upper right')
    
    # ECE part
    ax_ece = fig.add_subplot(gs[0:2,1])
    ax_ece.plot(plot_ece['time'], plot_ece['\\ECE67'].ewm(com=0.5).mean(), label = 'R=1.808m (Core)')
    ax_ece.plot(plot_ece['time'], plot_ece['\\ECE64'].ewm(com=0.5).mean(), label = 'R=1.863m')
    ax_ece.plot(plot_ece['time'], plot_ece['\\ECE63'].ewm(com=0.5).mean(), label = 'R=1.883m')
    ax_ece.axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed")
    ax_ece.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_ece.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    ax_ece.set_ylabel("ECE(unit:KeV)")
    ax_ece.legend(loc = 'upper right')
    
    ax_tci = fig.add_subplot(gs[2:,1])
    
    if int(data_disrupt[data_disrupt.shot == shot_num].Year) == 2018:
        ax_tci.plot(plot_diag['time'], plot_diag['\\ne_tci01'], label = 'TCI01(R=1.34m)')
        ax_tci.plot(plot_diag['time'], plot_diag['\\ne_tci02'], label = 'TCI02(R=1.78m)')
        ax_tci.plot(plot_diag['time'], plot_diag['\\ne_tci05'], label = 'TCI05(R=2.16m)')
        
    elif int(data_disrupt[data_disrupt.shot == shot_num].Year) == 2019:
        ax_tci.plot(plot_diag['time'], plot_diag['\\ne_tci01'], label = 'TCI01(R=1.34m)')
        ax_tci.plot(plot_diag['time'], plot_diag['\\ne_tci02'], label = 'TCI02(R=1.78m)')
        ax_tci.plot(plot_diag['time'], plot_diag['\\ne_tci03'], label = 'TCI03(R=1.91m)')
        ax_tci.plot(plot_diag['time'], plot_diag['\\ne_tci04'], label = 'TCI04(R=2.04m)')   
        
    else:
        ax_tci.plot(plot_diag['time'], plot_diag['\\ne_tci01'], label = 'TCI01(R=1.34m)')
        ax_tci.plot(plot_diag['time'], plot_diag['\\ne_tci02'], label = 'TCI02(R=1.78m)')
        ax_tci.plot(plot_diag['time'], plot_diag['\\ne_tci03'], label = 'TCI03(R=1.91m)')
        ax_tci.plot(plot_diag['time'], plot_diag['\\ne_tci04'], label = 'TCI04(R=2.04m)')
        ax_tci.plot(plot_diag['time'], plot_diag['\\ne_tci05'], label = 'TCI05(R=2.16m)')
        
    ax_tci.axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed")
    ax_tci.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_tci.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    ax_tci.set_ylabel("ne-TCI(unit:$10^{19}$ $m^{-3}$)")
    ax_tci.set_xlabel("time(unit:s)")
    ax_tci.set_ylim([0, 6.0])
    ax_tci.legend(loc = 'upper right')
    
    # Diagnostic part
    # Stored energy
    ax_w = fig.add_subplot(gs[0,2])
    ax_w.plot(plot_diag['time'], plot_diag['\\WTOT_DLM03'], label = 'Stored energy')
    ax_w.axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed")
    ax_w.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_w.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    ax_w.legend(loc = 'upper right')
    
    # EC heating
    ax_ech = fig.add_subplot(gs[1,2])
    ax_ech.plot(plot_diag['time'], plot_diag[config.ECH].sum(axis = 1) / plot_diag[config.ECH].sum(axis = 1).abs().max(), label = "EC heating")
    ax_ech.axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed")
    ax_ech.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_ech.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    ax_ech.legend(loc = 'upper right')
    
    # NB heating
    ax_nbh = fig.add_subplot(gs[2,2])
    ax_nbh.plot(plot_diag['time'], plot_diag[config.NBH].sum(axis = 1) / plot_diag[config.NBH].sum(axis = 1).abs().max(), label = "NB heating")
    ax_nbh.axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed")
    ax_nbh.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_nbh.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    ax_nbh.legend(loc = 'upper right')
    
    # Bolometer
    ax_bol = fig.add_subplot(gs[3,2])
    ax_bol.plot(plot_diag['time'], plot_diag['\\ax3_bolo02:FOO'] / plot_diag['\\ax3_bolo02:FOO'].abs().max(), label = 'AXUV bolometer')
    ax_bol.axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed")
    ax_bol.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_bol.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    ax_bol.set_xlabel("time(unit:s)")
    ax_bol.legend(loc = 'upper right')
    
    fig.tight_layout()
    
    if save_dir:
        pathname = os.path.join(save_dir, "info_shot_{}.png".format(int(shot_num)))
        fig.patch.set_facecolor('white')
        plt.savefig(pathname, facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)
    return fig

def plot_disrupt_prob(probs:Union[np.array, List], time_slice:Union[np.array, List], tftsrt, t_tq, t_cq, save_dir:str, t_pred:Optional[float] = 0.04, dt_warning: Optional[float] = 0.4, shot_num : Optional[int] = None, error_bar:Optional[np.array]=None, threshold_probability : Optional[float] = None):
    
    
    if threshold_probability is None:
        threshold_line = [0.5] * len(time_slice)
    else:
        threshold_line = [threshold_probability] * len(time_slice)
    
    fig, axes = plt.subplots(1,2, figsize = (14, 5))
    if shot_num:
        fig.suptitle("Disruption prediction with shot : {}".format(int(shot_num)))
    axes = axes.ravel()
    
    # plot general case
    axes[0].plot(time_slice, probs, 'b', label = 'disrupt prob')
    axes[0].plot(time_slice, threshold_line, 'k', label = "threshold(p = 0.5)")
    axes[0].axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed", label = "Flattop (t={:.3f})".format(tftsrt))
    axes[0].axvline(x = t_tq, ymin = 0, ymax = 1, color = "red", linestyle = "dashed", label = "TQ (t={:.3f})".format(t_tq))
    axes[0].axvline(x = t_cq, ymin = 0, ymax = 1, color = "green", linestyle = "dashed", label = "CQ (t={:.3f})".format(t_cq))
    
    if t_pred:
        t_minimum = t_tq - t_pred
        t_warning = t_tq - dt_warning
        
        axes[0].axvline(x = t_minimum, ymin = 0, ymax = 1, color = "blue", linestyle = "dashed", label = "Disruptive (t={:.3f})".format(t_minimum))
        axes[0].axvline(x = t_warning, ymin = 0, ymax = 1, color = "darkorange", linestyle = "dashed", label = "Warning (t={:.3f})".format(t_warning))
        
    axes[0].set_ylabel("probability")
    axes[0].set_xlabel("time(unit:s)")   
    axes[0].legend(loc = 'upper left', facecolor = 'white', framealpha=1)
    axes[0].set_ylim([0,1])
    axes[0].set_xlim([0, min(max(time_slice) + 0.05, t_cq + 0.1)])
    
    # plot zoom-in case
    axes[1].plot(time_slice, probs, 'b', label = 'disrupt prob')
    axes[1].plot(time_slice, threshold_line, 'k', label = "threshold(p = 0.5)")
    axes[1].axvline(x = t_tq, ymin = 0, ymax = 1, color = "red", linestyle = "dashed", label = "TQ (t={:.3f})".format(t_tq))
    axes[1].axvline(x = t_cq, ymin = 0, ymax = 1, color = "green", linestyle = "dashed", label = "CQ (t={:.3f})".format(t_cq))
    
    if t_pred:
        t_minimum = t_tq - t_pred
        t_warning = t_tq - dt_warning
        
        axes[1].axvline(x = t_minimum, ymin = 0, ymax = 1, color = "blue", linestyle = "dashed", label = "TQ-{:02d}ms (t={:.3f})".format(int(t_pred * 1000),t_minimum))
        axes[1].axvline(x = t_warning, ymin = 0, ymax = 1, color = "darkorange", linestyle = "dashed", label = "Warning (t={:.3f})".format(t_warning))
     
    axes[1].set_ylabel("probability")
    axes[1].set_xlabel("time(unit:s)")   
    axes[1].legend(loc = 'upper left', facecolor = 'white', framealpha=1)
    axes[1].set_ylim([0,1])
    
    if t_pred:
        axes[1].set_xlim([t_warning - 0.25, t_cq + 0.05])
    else:
        axes[1].set_xlim([t_tq - 0.1, t_cq + 0.05])
        
    if error_bar is not None:
        upper = probs + error_bar
        lower = probs - error_bar
        upper = np.clip(upper, 0, 1)
        lower = np.clip(lower, 0 ,1)
        clr = plt.cm.Purples(0.9)
        axes[0].fill_between(time_slice, lower, upper, alpha = 0.25, edgecolor = clr)
        axes[1].fill_between(time_slice, lower, upper, alpha = 0.25, edgecolor = clr)
    
    fig.tight_layout()
    
    if save_dir:
        pathname = os.path.join(save_dir, "disruption_probability_curve_shot_{}.png".format(int(shot_num)))  
        plt.savefig(pathname, facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)
        
        # save as eps file
        plt.savefig(os.path.join(save_dir, "disruption_probability_curve_shot_{}.eps".format(int(shot_num))), facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)
    
    return fig

def plot_disrupt_prob_uncertainty(probs:Union[np.array, List], time_slice:Union[np.array, List], aus : np.ndarray, eus:np.ndarray, tftsrt, t_tq, t_cq, save_dir:str, t_pred:Optional[float] = 0.04, dt_warning:Optional[float] = 0.4, shot_num : Optional[int] = None, threshold_probability : Optional[float] = None, threshold_aleatoric : Optional[float] = None, threshold_epistemic : Optional[float] = None):
    
    if threshold_probability is None:
        threshold_line = [0.5] * len(time_slice)
    else:
        threshold_line = [threshold_probability] * len(time_slice)
        
    if threshold_aleatoric is not None:
        threshold_uncertainty = [threshold_aleatoric] * len(time_slice)
    elif threshold_epistemic is not None:
        threshold_uncertainty = [threshold_epistemic] * len(time_slice)
    else:
        threshold_uncertainty = None
    
    fig, axes = plt.subplots(1,2, figsize = (16, 5)) # (14,5)
    if shot_num:
        fig.suptitle("Disruption prediction with shot : {}".format(int(shot_num)))
    axes = axes.ravel()

    # plot zoom-out case
    axes[0].plot(time_slice, probs, 'b', label = 'disrupt prob')
    axes[0].plot(time_slice, threshold_line, 'k', label = "threshold(p = 0.5)")
    axes[0].axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed", label = "Flattop (t={:.3f})".format(tftsrt))
    axes[0].axvline(x = t_tq, ymin = 0, ymax = 1, color = "red", linestyle = "dashed", label = "TQ (t={:.3f})".format(t_tq))
    axes[0].axvline(x = t_cq, ymin = 0, ymax = 1, color = "green", linestyle = "dashed", label = "CQ (t={:.3f})".format(t_cq))
    
    if t_pred:
        t_minimum = t_tq - t_pred
        t_warning = t_tq - dt_warning
        
        axes[0].axvline(x = t_minimum, ymin = 0, ymax = 1, color = "blue", linestyle = "dashed", label = "TQ-{:02d}ms (t={:.3f})".format(int(t_pred * 1000),t_minimum))
        axes[0].axvline(x = t_warning, ymin = 0, ymax = 1, color = "darkorange", linestyle = "dashed", label = "Warning (t={:.3f})".format(t_warning))
            
    axes[0].set_ylabel("probability")
    axes[0].set_xlabel("time(unit:s)")   
    axes[0].legend(loc = 'upper left', facecolor = 'white', framealpha=1)
    axes[0].set_xlim([0, min(max(time_slice) + 0.05, t_cq + 0.1)])
    axes[0].set_ylim([0,1])
    
    # shot 28158
    axes[0].fill([2.2,2.2,3.0,3.0], [0,1,1,0], color = 'yellow', alpha = 0.5)
    axes[0].fill([7.2,7.2,7.92,7.92], [0,1,1,0], color = 'yellow', alpha = 0.5)

    # plot uncertainty
    axes[1].plot(time_slice, aus, 'b', label = 'aleatoric uncertainty')
    axes[1].axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed", label = "Flattop (t={:.3f})".format(tftsrt))
    axes[1].axvline(x = t_tq, ymin = 0, ymax = 1, color = "red", linestyle = "dashed", label = "TQ (t={:.3f})".format(t_tq))
    axes[1].axvline(x = t_cq, ymin = 0, ymax = 1, color = "green", linestyle = "dashed", label = "CQ (t={:.3f})".format(t_cq))
    
    if t_pred:
        axes[1].axvline(x = t_minimum, ymin = 0, ymax = 1, color = "blue", linestyle = "dashed", label = "TQ-{:02d}ms (t={:.3f})".format(int(t_pred * 1000),t_minimum))
        axes[1].axvline(x = t_warning, ymin = 0, ymax = 1, color = "darkorange", linestyle = "dashed", label = "Warning (t={:.3f})".format(t_warning))
        
    if threshold_aleatoric is not None:
        axes[1].plot(time_slice, threshold_uncertainty, 'k', label = 'Thres-aleatoric:{:.3f}'.format(threshold_aleatoric))
        axes[1].legend(loc = 'upper left', facecolor = 'white', framealpha=1)
        
    axes[1].tick_params(axis = 'y', labelcolor = 'b')
    axes[1].set_xlabel("time(unit:s)")
    axes[1].set_ylabel("Aleatoric uncertainty")
    axes[1].set_xlim([0, min(max(time_slice) + 0.05, t_cq + 0.1)])
    
    # shot 28158
    # axes[1].fill([2.2,2.2,3.0,3.0], [0,0.25,0.25,0], color = 'yellow', alpha = 0.5)
    # axes[1].fill([7.2,7.2,7.92,7.92], [0,0.25,0.25,0], color = 'yellow', alpha = 0.5)
    
    axes_ = axes[1].twinx()
    axes_.plot(time_slice, eus, 'r', label = 'epistemic uncertainty')
    axes_.tick_params(axis = 'y', labelcolor = 'r')
    axes_.set_ylabel("Epistemic uncertainty")
    
    if threshold_epistemic is not None:
        axes_.plot(time_slice, threshold_uncertainty, 'k', label = 'Thres-epistemic:{:.3f}'.format(threshold_epistemic))
        axes_.legend(loc = 'upper left', facecolor = 'white', framealpha=1)
        
    fig.tight_layout()
    
    if save_dir:
        pathname = os.path.join(save_dir, "disruption_prob_uncertainty_shot_{}.png".format(int(shot_num)))
        plt.savefig(pathname, facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)
        
        # save as eps file
        plt.savefig(os.path.join(save_dir, "disruption_prob_uncertainty_shot_{}.eps".format(int(shot_num))), facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)  
        
    # zoom in and save it again
    if t_pred:
        axes[0].set_xlim([t_warning - 0.25, t_cq + 0.05])
    else:
        axes[0].set_xlim([t_tq - 0.1, t_cq + 0.05])
    
    if t_pred:
        axes[1].set_xlim([t_warning - 0.25, t_cq + 0.05])
    else:
        axes[1].set_xlim([t_tq - 0.1, t_cq + 0.05])
        
    if save_dir:
        pathname = os.path.join(save_dir, "disruption_prob_uncertainty_shot_{}_zoom_in.png".format(int(shot_num)))
        plt.savefig(pathname, facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)  
        
        # save as eps file
        plt.savefig(os.path.join(save_dir, "disruption_prob_uncertainty_shot_{}_zoom_in.eps".format(int(shot_num))), facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)  
        
    return fig

def plot_disrupt_prob_causes(feature_dict : Dict, shot_num : int, dt : float, dist : int, save_dir:str):
    
    # plot feature importance with each time
    for idx in range(len(feature_dict['time'])):
        
        feature_importance = {}
        
        for key in feature_dict.keys():
            if key != 'time':
                feature_importance[key] = feature_dict[key][idx]
        
        fig,ax = plt.subplots(1,1,figsize = (8, 8))
        fig.suptitle("Feature importance for predicting disruptions at shot :{}".format(int(shot_num)))
        
        ax.barh(list(feature_importance.keys()), list(feature_importance.values()))
        ax.set_yticks([i for i in range(len(list(feature_importance.keys())))], labels = list(feature_importance.keys()))
        ax.invert_yaxis()
        ax.set_ylabel('Input features')
        ax.set_xlabel('Feature importance')
        
        fig.tight_layout()
        
        if save_dir:
            pred_time = "{:02d}ms".format(int(feature_dict['time'][idx] * 1000))
            
            if pred_time not in ['00ms','05ms','10ms','15ms','20ms','25ms','30ms','35ms','40ms']:
                continue
            
            pathname = os.path.join(save_dir, "feature_importance_TQ_{}_shot_{}.png".format(pred_time, int(shot_num)))
            plt.savefig(pathname, facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)
    
    # plot the feature importance along time
    fig = plt.figure(figsize = (10, 10))
    # fig.suptitle("Feature importance for predicting disruptions at shot :{}".format(shot_num))
    ax = fig.add_subplot(111, projection = '3d')
    
    feature_importance = {}
    for key in feature_dict.keys():
        if key != 'time':
            feature_importance[key] = feature_dict[key]
    
    time_slice = []
    
    for idx, t in enumerate(feature_dict['time']):
        
        if int(t * 1000) not in [0,10,20,30,40]: # [0,50,100,150,200,250,300]: # [0,5,10,15,20,25,30,35,40]:
            continue
        
        time_slice.append(t)
        
        hist = [feature_importance[key][idx] for key in feature_importance.keys()]
        ax.bar(list(feature_importance.keys()), hist, zs = t, zdir = 'y', alpha = 0.8, in_layout = True)
    
    ax.set_xticks([i for i in range(len(list(feature_importance.keys())))], labels = list(feature_importance.keys()), rotation = 90, fontsize = 11)
    
    if dt == 0.01:
        ax.set_yticks(ticks = time_slice, labels = ["{:02d} ms".format(int(t * 1000)) for i in time_slice])
        
    elif dt == 0.001:
        ax.set_yticks(ticks = time_slice, labels = ["{:02d} ms".format(int(t * 1000)) for t in time_slice])
        
    ax.set_zlabel('Feature importance', fontsize = 12)
    # ax.set_ylabel('$t_{pred}$ before TQ (unit:ms)', fontsize = 12)
    ax.set_zlim([0,1])
    fig.tight_layout()

    if save_dir:
        pathname = os.path.join(save_dir, "feature_importance_shot_{}.png".format(int(shot_num)))
        plt.savefig(pathname, facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)
        
    return fig

def generate_model_performance(
        filepath : Dict[str, str],
        model : torch.nn.Module, 
        device : str = "cpu",
        save_dir : Optional[str] = "./results/test",
        shot_num : Optional[int] = None,
        seq_len_efit:int = 20, 
        seq_len_ece:int = 100,
        seq_len_diag:int = 100, 
        dist_warning:Optional[int] = None,
        dist : Optional[int] = None,
        dt : Optional[float] = None,
        mode : Literal['TQ', 'CQ'] = 'CQ', 
        scaler_type = None,
        is_plot_shot_info:bool = False,
        is_plot_uncertainty:bool = False,
        is_plot_feature_importance:bool = False,
        is_plot_error_bar:bool = False,
        is_plot_temporal_feature_importance:bool = False,
    ):
    
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    # obtain tTQend, tipmin and tftsrt
    total, scaler_list = preparing_0D_dataset(filepath, random_state = STATE_FIXED, scaler = scaler_type, test_shot = shot_num, is_split = False)

    data_disrupt = total['disrupt']
    data_efit = total['efit']
    data_ece = total['ece']
    data_diag = total['diag']
    
    tTQend = data_disrupt[data_disrupt.shot == shot_num].t_tmq.values[0]
    tftsrt = data_disrupt[data_disrupt.shot == shot_num].t_flattop_start.values[0]
    tipminf = data_disrupt[data_disrupt.shot == shot_num].t_ip_min_fault.values[0]
    t_warning = data_disrupt[data_disrupt.shot == shot_num].t_warning.values[0]
    
    t_disrupt = tTQend
    t_current = tipminf
    t_warning = t_disrupt - t_warning
    
    data_efit = data_efit[data_efit.shot == shot_num]
    data_ece = data_ece[data_ece.shot == shot_num]
    data_diag = data_diag[data_diag.shot == shot_num]
    data_disrupt = data_disrupt[data_disrupt.shot == shot_num]
    
    dataset = MultiSignalDataset(
        data_disrupt,
        data_efit,
        data_ece,
        data_diag,
        seq_len_efit,
        seq_len_ece,
        seq_len_diag,
        dist_warning,
        dist,
        dt,
        scaler_list['efit'],
        scaler_list['ece'],
        scaler_list['diag'],
        mode
    )

    prob_list = []
    feature_dict = None
    aus = []
    eus = []
    error_bars = []
    temporal_feature_dict = None

    model.to(device)
    model.eval()
    
    # computation of disruption probability
    for idx in tqdm(range(dataset.__len__())):
        
        data = dataset.__getitem__(idx)
        
        for key in data.keys():
            data[key] = data[key].unsqueeze(0)
            
        if is_plot_feature_importance:
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
        
        if is_plot_uncertainty:
            au, eu = compute_uncertainty_per_data(model, data, device = device, n_samples = 16)
            aus.append(au[0])
            eus.append(eu[0])
            
        if is_plot_error_bar:
            ensemble_probs = compute_ensemble_probability(model, data, device, n_samples = 16)
            error_bars.append(np.std(ensemble_probs))
            
        t = dataset.time_slice[idx]
        
        if is_plot_temporal_feature_importance:
            if temporal_feature_dict is None:
                feat = compute_relative_importance(data, model, 0, None, 8, device)
                temporal_feature_dict = {}
                temporal_feature_dict['time'] = [t]
                
                for key in feat.keys():
                    temporal_feature_dict[key] = [feat[key]]
            else:
                feat = compute_relative_importance(data, model, 0, None, 8, device)
                temporal_feature_dict['time'].append(t)
                
                for key in feat.keys():
                    temporal_feature_dict[key].append(feat[key])
        
        if t >= t_disrupt - (dist+5) * dt and t <= t_disrupt and is_plot_feature_importance:
        # if t >= t_disrupt - t_warning and t <= t_disrupt and is_plot_feature_importance:
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
    
    n_ftsrt = int(dataset.time_slice[0] // dt)
    time_slice = np.linspace(0, dataset.time_slice[0] - dt, n_ftsrt).tolist() + [v for v in dataset.time_slice] 
    probs = np.array([0] * n_ftsrt + prob_list)
    probs = moving_avarage_smoothing(probs, 12, 'center')
    
    if is_plot_uncertainty:
        aus = np.array([0] * n_ftsrt + aus)
        eus = np.array([0] * n_ftsrt + eus)
        
    print("time_slice : ", len(time_slice))
    print("prob_list : ", len(probs))
    print("\n(Info) flat-top : {:.3f}(s) | thermal quench : {:.3f}(s) | current quench : {:.3f}(s)\n".format(tftsrt, tTQend, tipminf))
    
    # plot shot info
    if is_plot_shot_info:
        fig_shot = plot_shot_info(filepath, shot_num, save_dir)
    else:
        fig_shot = None
    
    # plot disrupt prob
    if is_plot_error_bar:
        error_bar = np.array([0] * n_ftsrt + error_bars)
    else:
        error_bar = None
    
    fig_dis = plot_disrupt_prob(probs, time_slice, tftsrt, t_disrupt, t_current, save_dir, dt * dist, t_warning, shot_num, error_bar)
    
    if is_plot_uncertainty:        
        aus = np.array(aus).reshape(-1,1)
        eus = np.array(eus).reshape(-1,1)
        fig_uc = plot_disrupt_prob_uncertainty(probs, time_slice, aus, eus, tftsrt, t_disrupt, t_current, save_dir, dt * dist, t_warning, shot_num)
    else:
        fig_uc = None
    
    if is_plot_feature_importance:
        fig_fi = plot_disrupt_prob_causes(feature_dict, shot_num, dt, dist, save_dir)
    else:
        fig_fi = None
        
    if is_plot_temporal_feature_importance:
        temporal_feature_dict = pd.DataFrame(temporal_feature_dict)
        temporal_feature_dict.to_csv(os.path.join(save_dir, "temporal_feature_importance_shot_{}.png".format(int(shot_num))), index = False)

    return time_slice, prob_list, fig_dis, fig_uc, fig_fi, fig_shot

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