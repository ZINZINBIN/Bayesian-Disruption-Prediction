import torch
from src.data_labeling import compute_MMD_sequence
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

cols_diag = ['\\WTOT_DLM03', '\\RC03', '\\ax3_bolo02:FOO', '\\BETAP_DLM03', '\\DMF_DLM03']
data_diag = pd.read_csv("./dataset/Bayesian_Disruption_diag.csv") 
disrupt = pd.read_csv("./dataset/Bayesian_Disruption_Shot_List.csv") 
    
scaler = RobustScaler()
data_diag.loc[:,cols_diag] = scaler.fit_transform(data_diag[cols_diag].values)

valid_shot_list = []
t_warning_list = []

shot_list = [23621, 26069, 26715, 28357, 32325, 32497]

for shot in shot_list:
    
    if shot not in data_diag.shot.unique():
        continue
    
    diag = data_diag[data_diag.shot == shot]
    
    tftsrt = disrupt[disrupt.shot == shot].t_flattop_start.values[0]
    t_tq = disrupt[disrupt.shot == shot].t_tmq.values[0]
    t_cq = disrupt[disrupt.shot == shot].t_ip_min_fault.values[0]
    
    diag = diag[(diag.time >= tftsrt + 1.0)&(diag.time <= t_tq)]
    
    val = diag[cols_diag].values
    val = torch.Tensor(val)
    cmmd = compute_MMD_sequence(val, 'cuda:0', True)
    
    if cmmd is None:
        print("Shot:{} | MMD error".format(int(shot)))
        continue
    
    idx_max = np.argmax(cmmd)
    t_warning = diag['time'].iloc[idx_max]
    time_interval = t_tq - t_warning
    
    fig, axes = plt.subplots(1,2,figsize = (12,5))
    axes[0].plot(diag.time.iloc[1:], cmmd, label = 'Corrected MMD')
    axes[0].axvline(x = tftsrt, ymin = 0, ymax = 1, c = 'b', label = 'Flattop')
    axes[0].axvline(x = t_warning, ymin = 0, ymax = 1, c = 'g', label = 'warning-MMD')
    axes[0].axvline(x = t_tq, ymin = 0, ymax = 1, c = 'r', label = 'TQ')
    axes[0].axvline(x = t_cq, ymin = 0, ymax = 1, c = 'k', label = 'CQ')
    axes[0].set_xlabel("time(s)")
    axes[0].set_ylabel("CMMD")
    axes[0].legend()
    axes[0].set_xlim([tftsrt, t_cq+0.2])
    
    axes[1].plot(diag.time.iloc[1:], diag['\\RC03'].iloc[1:], label = 'RC03')
    axes[1].axvline(x = tftsrt, ymin = 0, ymax = 1, c = 'b', label = 'Flattop')
    axes[1].axvline(x = t_warning, ymin = 0, ymax = 1, c = 'g', label = 'warning-MMD')
    axes[1].axvline(x = t_tq, ymin = 0, ymax = 1, c = 'r', label = 'TQ')
    axes[1].axvline(x = t_cq, ymin = 0, ymax = 1, c = 'k', label = 'CQ')
    
    axes[1].set_xlabel("time(s)")
    axes[1].set_ylabel("RC03")
    axes[1].legend()
    axes[1].set_xlim([tftsrt, t_cq+0.2])
    
    fig.tight_layout()
    plt.savefig("./results/analysis_MMD_warning_jwl/{}.png".format(int(shot)))
    
    t_warning_list.append(t_warning)
    valid_shot_list.append(int(shot))
    print("Shot:{} |t-warning:{:.3f} | Flattop:{:.3f} | TQ:{:.3f} | CQ:{:.3f}".format(int(shot), t_warning, tftsrt, t_tq, t_cq))
    
print("Original shot:{}".format(len(disrupt)))
print("Valid shot:{}".format(len(valid_shot_list)))

disrupt['shot'] = disrupt['shot'].astype(int)
disrupt = disrupt[disrupt.shot.isin(valid_shot_list)]
disrupt['t_warning'] = t_warning_list

print(disrupt)