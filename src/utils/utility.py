import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import cv2, os, glob2, random
import pandas as pd
import numpy as np
from typing import Optional, List, Literal, Union, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator
from src.config import Config
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
def preparing_0D_dataset(filepath : str = "./dataset/KSTAR_Disruption_ts_data_extend.csv", random_state : int = STATE_FIXED, ts_cols : Optional[List] = None, scaler : Literal['Robust', 'Standard', 'MinMax', 'None'] = 'Robust'):
    
    # preparing 0D data for use
    df = pd.read_csv(filepath).reset_index()

    # nan interpolation
    df.interpolate(method = 'linear', limit_direction = 'forward')

    # float type
    if ts_cols is None:
        for col in df.columns:
            df[col] = df[col].astype(np.float32)
    else:
        for col in ts_cols:
            df[col] = df[col].astype(np.float32)

    # train / valid / test data split
    shot_list = np.unique(df.shot.values)

    # stochastic train_test_split
    # shot_train, shot_test = train_test_split(shot_list, test_size = 0.2, random_state = random_state)
    # shot_train, shot_valid = train_test_split(shot_train, test_size = 0.2, random_state = random_state)
    
    # deterministic train_test_split
    shot_train, shot_test = deterministic_split(shot_list, test_size = 0.2)
    shot_train, shot_valid = deterministic_split(shot_train, test_size = 0.2)
    
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()

    for shot in shot_train:
        df_train = pd.concat([df_train, df[df.shot == shot]], axis = 0)

    for shot in shot_valid:
        df_valid = pd.concat([df_valid, df[df.shot == shot]], axis = 0)

    for shot in shot_test:
        df_test = pd.concat([df_test, df[df.shot == shot]], axis = 0)
        
    if scaler == 'Robust':
        scaler = RobustScaler()
    elif scaler == 'Standard':
        scaler = StandardScaler()
    elif scaler == 'MinMax':
        scaler = MinMaxScaler()
    else:
        scaler = None
    
    if scaler is not None:
        scaler.fit(df_train[ts_cols].values)

    return df_train, df_valid, df_test, scaler
    

TS_COLS = [
    '\\q95', '\\ipmhd', '\\kappa', 
    '\\tritop', '\\tribot','\\betap','\\betan',
    '\\li', '\\WTOT_DLM03', '\\ne_inter01', 
    '\\TS_NE_CORE_AVG', '\\TS_TE_CORE_AVG'
]

# 0D dataset for generating probability curve
class DatasetFor0D(Dataset):
    def __init__(
        self, 
        ts_data : pd.DataFrame, 
        cols : List, 
        seq_len : int = 21, 
        dist:int = 3, 
        dt : float = 1.0 / 210 * 4,
        scaler : Optional[BaseEstimator] = None,
        ):
        
        self.ts_data = ts_data
        self.seq_len = seq_len
        self.dt = dt
        self.cols = cols
        self.dist = dist
        self.indices = [idx for idx in range(0, len(self.ts_data) - seq_len - dist)]
        
        self.scaler = scaler
            
        if self.scaler is not None:
            self.ts_data[cols] = self.scaler.fit_transform(self.ts_data[cols].values)
   
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx : int):
        return self.get_data(idx)

    def get_data(self, idx : int):
        idx_srt = self.indices[idx]
        idx_end = idx_srt + self.seq_len
        
        data = self.ts_data[self.cols].iloc[idx_srt + 1: idx_end + 1].values
        data = torch.from_numpy(data)
        return data

def generate_prob_curve_from_0D(
    model : torch.nn.Module, 
    device : str = "cpu", 
    save_dir : Optional[str] = "./results/disruption_probs_curve.png",
    ts_data_dir : Optional[str] = "./dataset/KSTAR_Disruption_ts_data_extend.csv",
    ts_cols : Optional[List] = None,
    shot_list_dir : Optional[str] = './dataset/KSTAR_Disruption_Shot_List_extend.csv',
    shot_num : Optional[int] = None,
    seq_len : Optional[int] = None,
    dist : Optional[int] = None,
    dt : Optional[int] = None,
    scaler : Optional[BaseEstimator] = None,
    ):
    
    # obtain tTQend, tipmin and tftsrt
    shot_list_dir = pd.read_csv(shot_list_dir, encoding = "euc-kr")
    tTQend = shot_list_dir[shot_list_dir.shot == shot_num].t_tmq.values[0]
    tftsrt = shot_list_dir[shot_list_dir.shot == shot_num].t_flattop_start.values[0]
    tipminf = shot_list_dir[shot_list_dir.shot == shot_num].t_ip_min_fault.values[0]

    # input data generation
    ts_data = pd.read_csv(ts_data_dir).reset_index()

    for col in ts_cols:
        ts_data[col] = ts_data[col].astype(np.float32)

    ts_data.interpolate(method = 'linear', limit_direction = 'forward')
    
    ts_data_0D = ts_data[ts_data['shot'] == shot_num]
    
    t = ts_data_0D.time
    ip = ts_data_0D['\\ipmhd']
    kappa = ts_data_0D['\\kappa']
    betap = ts_data_0D['\\betap']
    betan = ts_data_0D['\\betan']
    li = ts_data_0D['\\li']
    q95 = ts_data_0D['\\q95']
    tritop = ts_data_0D['\\tritop']
    tribot = ts_data_0D['\\tribot']
    W_tot = ts_data_0D['\\WTOT_DLM03']
    ne = ts_data_0D['\\ne_inter01']
    te = ts_data_0D['\\TS_TE_CORE_AVG']
    
    # video data
    dataset = DatasetFor0D(ts_data_0D, ts_cols, seq_len, dist, dt, scaler)

    prob_list = []
    is_disruption = []

    model.to(device)
    model.eval()
    
    from tqdm.auto import tqdm
    
    for idx in tqdm(range(dataset.__len__())):
        with torch.no_grad():
            data = dataset.__getitem__(idx)
            data = data.to(device).unsqueeze(0)
            output = model(data)
            probs = torch.nn.functional.softmax(output, dim = 1)[:,0]
            probs = probs.cpu().detach().numpy().tolist()

            prob_list.extend(
                probs
            )
            
            is_disruption.extend(
                torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1].cpu().detach().numpy().tolist()
            )
            
    interval = 1
    fps = int(1 / 0.025)
    
    prob_list = [0] * seq_len + prob_list
    
    # correction for startup peaking effect : we will soon solve this problem
    for idx, prob in enumerate(prob_list):
        
        if idx < fps * 1 and prob >= 0.5:
            prob_list[idx] = 0
            
    from scipy.interpolate import interp1d
    prob_x = np.linspace(0, len(prob_list) * interval, num = len(prob_list), endpoint = True)
    prob_y = np.array(prob_list)
    f_prob = interp1d(prob_x, prob_y, kind = 'cubic')
    prob_list = f_prob(np.linspace(0, len(prob_list) * interval, num = len(prob_list) * interval, endpoint = False))
    
    time_x = np.arange(0, len(prob_list)) * (1/fps)
    
    print("time : ", len(time_x))
    print("prob : ", len(prob_list))
    print("flat-top : ", tftsrt)
    print("thermal quench : ", tTQend)
    print("current quench: ", tipminf)
    
    t_disrupt = tTQend
    t_current = tipminf
    
    # plot the disruption probability with plasma status
    fig = plt.figure(figsize = (16, 8))
    fig.suptitle("Disruption prediction with shot : {}".format(shot_num))
    gs = GridSpec(nrows = 4, ncols = 3)
    
    quantile = [0, 0.25, 0.5, 0.75, 1.0, 1.25]
    t_quantile = [q * t_current for q in quantile]
        
    # kappa
    ax_kappa = fig.add_subplot(gs[0,0])
    ax_kappa.plot(t, kappa, label = 'kappa')
    ax_kappa.text(0.85, 0.8, "kappa", transform = ax_kappa.transAxes)
    ax_kappa.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_kappa.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")

    # tri top
    ax_tri_top = fig.add_subplot(gs[1,0])
    ax_tri_top.plot(t, tritop, label = 'tri_top')
    ax_tri_top.text(0.85, 0.8, "tri_top", transform = ax_tri_top.transAxes)
    ax_tri_top.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_tri_top.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    
    # tri bot
    ax_tri_bot = fig.add_subplot(gs[2,0])
    ax_tri_bot.plot(t, tribot, label = 'tri_bot')
    ax_tri_bot.text(0.85, 0.8, "tri_bot", transform = ax_tri_bot.transAxes)
    ax_tri_bot.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_tri_bot.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    
    # plasma current
    ax_Ip = fig.add_subplot(gs[3,0])
    ax_Ip.plot(t, ip, label = 'Ip')
    ax_Ip.text(0.85, 0.8, "Ip", transform = ax_Ip.transAxes)
    ax_Ip.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_Ip.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    
    ax_Ip.set_xlabel("time(s)")
    ax_Ip.set_xticks(t_quantile)
    ax_Ip.set_xticklabels(["{:.1f}".format(t) for t in t_quantile])
    
    # line density
    ax_li = fig.add_subplot(gs[0,1])
    ax_li.plot(t, li, label = 'line density')
    ax_li.text(0.85, 0.8, "li", transform = ax_li.transAxes)
    ax_li.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_li.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    
    # q95
    ax_q95 = fig.add_subplot(gs[1,1])
    ax_q95.plot(t, q95, label = 'q95')
    ax_q95.text(0.85, 0.8, "q95", transform = ax_q95.transAxes)
    ax_q95.set_ylim([0,10])
    ax_q95.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_q95.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    
    # betap
    ax_bp = fig.add_subplot(gs[2,1])
    ax_bp.plot(t, betap, label = 'beta p')
    ax_bp.text(0.85, 0.8, "betap", transform = ax_bp.transAxes)
    ax_bp.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_bp.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")

    # electron density
    ax_ne = fig.add_subplot(gs[3,1])
    ax_ne.plot(t, ne, label = 'ne')
    ax_ne.text(0.85, 0.8, "ne", transform = ax_ne.transAxes)
    ax_ne.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed")
    ax_ne.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed")
    
    ax_ne.set_xlabel("time(s)")
    ax_ne.set_xticks(t_quantile)
    ax_ne.set_xticklabels(["{:.1f}".format(t) for t in t_quantile])
    
    # probability
    threshold_line = [0.5] * len(time_x)
    ax2 = fig.add_subplot(gs[:,2])
    ax2.plot(time_x, prob_list, 'b', label = 'disrupt prob')
    ax2.plot(time_x, threshold_line, 'k', label = "threshold(p = 0.5)")
    ax2.axvline(x = tftsrt, ymin = 0, ymax = 1, color = "black", linestyle = "dashed", label = "flattop")
    ax2.axvline(x = t_disrupt, ymin = 0, ymax = 1, color = "red", linestyle = "dashed", label = "TQ")
    ax2.axvline(x = t_current, ymin = 0, ymax = 1, color = "green", linestyle = "dashed", label = "CQ")
    ax2.set_ylabel("probability")
    ax2.set_xlabel("time(unit : s)")
    ax2.set_ylim([0,1])
    ax2.set_xlim([0, max(time_x)])
    ax2.legend(loc = 'upper right')
    
    fig.tight_layout()

    plt.savefig(save_dir, facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)

    return time_x, prob_list



def plot_learning_curve(train_loss, valid_loss, train_f1, valid_f1, figsize : Tuple[int,int] = (12,6), save_dir : str = "./results/learning_curve.png"):
    x_epochs = range(1, len(train_loss) + 1)

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