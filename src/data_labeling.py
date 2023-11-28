import torch
import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from src.config import Config
from sklearn.preprocessing import RobustScaler

config = Config()

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="Data labeling for identifying warning regime in KSTAR")
    
    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)
    parser.add_argument("--scaler", type = str, choices=['Robust', 'Standard', 'MinMax', 'None'], default = "Robust")
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

def compute_MMD(x:torch.Tensor, y:torch.Tensor, device : str):
    
    x = x.to(device)
    y = y.to(device)
    
    xx,yy,zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x, y.t())
    
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2.0 * xx
    dyy = ry.t() + ry - 2.0 * yy

    dxy = xx.diag().unsqueeze(0).t() + yy.diag().unsqueeze(0) - 2.0 * zz
    
    XX,YY,XY = torch.zeros_like(dxx).to(device), torch.zeros_like(dyy).to(device), torch.zeros_like(dxy).to(device) 
    
    gamma_range = [0.1, 1.0, 2.0, 4.0, 10.0]
    for a in gamma_range:
        XX += torch.exp(-0.5*dxx*a)
        YY += torch.exp(-0.5*dyy*a)
        XY += torch.exp(-0.5*dxy*a) 

    return torch.sqrt(XX.mean()+YY.mean()-2.0*XY.mean())

def compute_MMD_sequence(x : torch.Tensor, device : str, is_correct:bool = True):

    mmds = []
    coeffs = []
    
    for idx in range(len(x)-1):
        xl = x[:idx+1,:]
        xr = x[idx+1:,:]
        mmd = compute_MMD(xl,xr,device)
        mmds.append(mmd)
        coeffs.append(torch.Tensor([(len(x) - 1)/((idx + 1) * (len(x) - idx - 1))]))
        
    if len(mmds) != 0:
        mmds = torch.stack(mmds).view(-1)
        coeffs = torch.stack(coeffs).view(-1)
    else:
        mmds = None
        coeffs = None
        return mmds
    
    if is_correct:
        cmmds = mmds - coeffs.to(device) * mmds.max()
        return cmmds.cpu().numpy()
    else:
        return mmds.cpu().numpy()

if __name__ == "__main__":

    args = parsing()
    
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(args["gpu_num"])
    else:
        device = 'cpu'
    
    print("# Load files...")
    data_diag = pd.read_csv("./dataset/Bayesian_Disruption_diag.csv") 
    disrupt = pd.read_csv("./dataset/Bayesian_Disruption_Shot_List.csv") 
    
    cols_diag = config.COLUMN_NAME_SCALER["diag"]
    cols_diag = ['\\WTOT_DLM03', '\\RC03', '\\ax3_bolo02:FOO', '\\BETAP_DLM03', '\\DMF_DLM03']
    
    scaler = RobustScaler()
    data_diag.loc[:,cols_diag] = scaler.fit_transform(data_diag[cols_diag].values)
    
    if not os.path.exists("./results/analysis_MMD_warning"):
        os.makedirs("./results/analysis_MMD_warning")
    
    print("# Maximum mean discrepancy: change point detection for identifying warning regime...")
    valid_shot_list = []
    t_warning_list = []
    
    for shot in disrupt.shot.unique():
        
        diag = data_diag[data_diag.shot == shot]
        
        tftsrt = disrupt[disrupt.shot == shot].t_flattop_start.values[0]
        t_tq = disrupt[disrupt.shot == shot].t_tmq.values[0]
        t_cq = disrupt[disrupt.shot == shot].t_ip_min_fault.values[0]
        
        diag = diag[(diag.time >= tftsrt + 1.0)&(diag.time <= t_tq)]
        
        # if too short: ignore the shot
        if abs(diag.time.min() - diag.time.max()) < 4.0:
            continue
        
        val = diag[cols_diag].values
        val = torch.Tensor(val)
        cmmd = compute_MMD_sequence(val, device, True)
        
        if cmmd is None:
            print("Shot:{} | MMD error".format(int(shot)))
            continue
        
        idx_max = np.argmax(cmmd)
        t_warning = diag['time'].iloc[idx_max]
        time_interval = t_tq - t_warning
        
        fig, axes = plt.subplots(1,2,figsize = (12,5))
        axes[0].plot(diag.time.iloc[1:], cmmd, label = 'Corrected MMD')
        axes[0].axvline(x = tftsrt, ymin = 0, ymax = 1, c = 'b', label = 'Flattop')
        axes[0].axvline(x = t_tq, ymin = 0, ymax = 1, c = 'r', label = 'TQ')
        axes[0].axvline(x = t_cq, ymin = 0, ymax = 1, c = 'k', label = 'CQ')
        axes[0].set_xlabel("time(s)")
        axes[0].set_ylabel("CMMD")
        axes[0].legend()
        axes[0].set_xlim([tftsrt, t_cq+0.2])
        
        axes[1].plot(diag.time.iloc[1:], diag['\\RC03'].iloc[1:], label = 'RC03')
        axes[1].axvline(x = tftsrt, ymin = 0, ymax = 1, c = 'b', label = 'Flattop')
        axes[1].axvline(x = t_tq, ymin = 0, ymax = 1, c = 'r', label = 'TQ')
        axes[1].axvline(x = t_cq, ymin = 0, ymax = 1, c = 'k', label = 'CQ')
        
        axes[1].set_xlabel("time(s)")
        axes[1].set_ylabel("RC03")
        axes[1].legend()
        axes[1].set_xlim([tftsrt, t_cq+0.2])
        
        fig.tight_layout()
        plt.savefig("./results/analysis_MMD_warning/{}.png".format(int(shot)))
        
        # check the validity
        if time_interval < 0.04:
            print("Shot:{} | warning regime is shorter than 40ms".format(int(shot)))
            continue
        
        if time_interval > 2.0:
            print("Shot:{} | warning regime is larger than 2000ms".format(int(shot)))
            continue
        
        if t_warning - tftsrt - 1.0 < 2.0:
            print("Shot:{} | warning regime starts too early".format(int(shot)))
            continue
        
        t_warning_list.append(t_warning)
        valid_shot_list.append(int(shot))
        print("Shot:{} |t-warning:{:.3f} | Flattop:{:.3f} | TQ:{:.3f} | CQ:{:.3f}".format(int(shot), t_warning, tftsrt, t_tq, t_cq))
        
    print("Original shot:{}".format(len(disrupt)))
    print("Valid shot:{}".format(len(valid_shot_list)))
    
    disrupt['shot'] = disrupt['shot'].astype(int)
    disrupt = disrupt[disrupt.shot.isin(valid_shot_list)]
    disrupt['t_warning'] = t_warning_list
    disrupt.to_csv("./dataset/Bayesian_Disruption_Shot_List_revised.csv", index = False)
    
    disrupt.head()