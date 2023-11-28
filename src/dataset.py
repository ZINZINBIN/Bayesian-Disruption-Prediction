import numpy as np
import pandas as pd
import torch, os, math
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Literal, Optional
from src.config import Config

config = Config()

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
        dist:int = 40,
        dt:float = 0.01, 
        scaler_efit=None,
        scaler_ece=None,
        scaler_diag=None, 
        mode : Literal['TQ', 'CQ'] = 'CQ', 
        task : Literal['train','valid','test','eval'] = 'train',
        dist_warning:int = 400,
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
        self.dist_warning = dist_warning
        self.mode = mode
        self.task = task
        
        # scaler
        self.scaler_efit = scaler_efit
        self.scaler_ece = scaler_ece
        self.scaler_diag = scaler_diag
        
        # dataloader setup
        self.indices = []
        self.labels = []
        self.shot_num = []
        self.shot_list = []
        self.dist_list = []
        self.t_warning_list = []
        self.get_shot_num = False
        self.n_classes = 2
        
        # initialize experimental shot list
        self.init_shot_list()
        
        # preprocessing: scaler application
        self.preprocessing()
        
        if self.task != 'eval':
            # generate indice for matching labels
            self.generate_indices()
        else:
            self.generate_indices_eval()
            self.get_shot_num = True
        
    def eval(self):
        self.task = 'eval'
        self.generate_indices_eval()
        self.get_shot_num = True
        
    def init_shot_list(self):
        shot_list = self.data_ece.shot.unique()
        self.shot_list = [shot for shot in shot_list if shot in self.data_disrupt.shot.unique()]
        print("# {} | initialize experimental shot | # of shots : {}".format(self.task, len(self.shot_list)))
        
    def preprocessing(self):
    
        if self.scaler_efit is not None:
            self.data_efit.loc[:,config.EFIT + config.EFIT_FE] = self.scaler_efit.transform(self.data_efit[config.EFIT + config.EFIT_FE].values)
            
        if self.scaler_ece is not None:
            self.data_ece.loc[:,config.ECE] = self.scaler_ece.transform(self.data_ece[config.ECE].values)
        
        if self.scaler_diag is not None:
            self.data_diag.loc[:,config.DIAG + config.DIAG_FE] = self.scaler_diag.transform(self.data_diag[config.DIAG + config.DIAG_FE].values)
            
        print("# {} | preprocessing the data".format(self.task))
    
    def generate_indices(self):
        
        for shot in tqdm(self.shot_list, desc = "# {} | generate indice".format(self.task)):
            tTQend = self.data_disrupt[self.data_disrupt.shot == shot].t_tmq.values[0]
            tftsrt = self.data_disrupt[self.data_disrupt.shot == shot].t_flattop_start.values[0]
            tipminf = self.data_disrupt[self.data_disrupt.shot == shot].t_ip_min_fault.values[0]
            t_warning = self.data_disrupt[self.data_disrupt.shot == shot].t_warning.values[0]
            
            if self.mode == 'TQ':
                t_disrupt = tTQend
            elif self.mode =='CQ':
                t_disrupt = tipminf
            
            df_efit_shot = self.data_efit[self.data_efit.shot == shot]
            df_ece_shot = self.data_ece[self.data_ece.shot == shot]
            df_diag_shot = self.data_diag[self.data_diag.shot == shot]
            
            indices = []
            labels = []
            dists = []
            
            if self.dt == 0.01:
                idx_efit = int(tftsrt / self.dt) // 5
                ratio = 5
                dt_efit_ece = 0.045
            elif self.dt == 0.001:
                idx_efit = int(tftsrt / self.dt) // 10
                ratio = 10
                dt_efit_ece = 0.009
                
            idx_ece = int(tftsrt / self.dt)
            idx_diag = int(tftsrt / self.dt)
            
            idx_efit_last = len(df_efit_shot.index)
            idx_ece_last = len(df_ece_shot.index)
            idx_diag_last = len(df_diag_shot.index)
            
            while(idx_ece < idx_ece_last or idx_diag < idx_diag_last):
                
                t_efit = df_efit_shot.iloc[idx_efit]['time']
                t_ece = df_ece_shot.iloc[idx_ece]['time']
                t_diag = df_diag_shot.iloc[idx_diag]['time']
                
                assert abs(t_diag - t_ece) <= self.dt * 2, "Dignostic data and ECE data are not consistent"
                
                # end condition
                if idx_ece_last - idx_ece - self.seq_len_ece - self.dist < 0 or idx_efit_last - idx_efit - self.seq_len_efit - self.dist // ratio < 0:
                    break
                
                # synchronize t_efit and t_ece
                if t_efit <= t_ece - dt_efit_ece:
                    t_diff = t_ece - t_efit
                    idx_efit += int(round(t_diff / self.dt / ratio, 1))
                
                if t_ece >= tftsrt + 1.0 and t_ece < t_warning - self.dt * (self.seq_len_ece + self.dist_warning):
                    indx_ece = df_ece_shot.index.values[idx_ece]
                    indx_efit = df_efit_shot.index.values[idx_efit]
                    indx_diag = df_diag_shot.index.values[idx_diag]
                    
                    indices.append((indx_efit, indx_ece, indx_diag))
                    labels.append(0)
                    
                    pred_time = t_disrupt - t_ece - self.dt * self.seq_len_ece
                    dists.append(pred_time)
                    
                    if self.dt == 0.01:
                        idx_ece += self.seq_len_ece // 10
                        idx_diag += self.seq_len_diag // 10
                        
                    elif self.dt == 0.001:
                        idx_ece += self.seq_len_ece // 50
                        idx_diag += self.seq_len_diag // 50
                
                elif t_ece >= t_warning - self.dt * (self.seq_len_ece + self.dist_warning) and t_ece < t_warning - self.dt * self.seq_len_ece:
                    indx_ece = df_ece_shot.index.values[idx_ece]
                    indx_efit = df_efit_shot.index.values[idx_efit]
                    indx_diag = df_diag_shot.index.values[idx_diag]
                    
                    indices.append((indx_efit, indx_ece, indx_diag))
                    pred_time = t_disrupt - t_ece - self.dt * self.seq_len_ece
                    dists.append(pred_time)
                    
                    # exponantial smoothing
                    delta = self.dist_warning * self.dt
                    delta_t = t_warning - t_ece - self.dt * self.seq_len_ece
                    gamma = 2.0
                    alpha = (math.exp(gamma * (delta - delta_t) / delta) - 1) / (math.exp(gamma) - 1)
            
                    if alpha > 1:
                        alpha = 1
                        
                    elif alpha < 0:
                        alpha = 0
                    
                    if self.task == 'train':
                        labels.append(alpha)
                    else:
                        labels.append(0)
                    
                    if self.dt == 0.01:
                        idx_ece += self.seq_len_ece // 10
                        idx_diag += self.seq_len_diag // 10
                        
                    elif self.dt == 0.001:
                        idx_ece += self.seq_len_ece // 100
                        idx_diag += self.seq_len_diag // 100        
                
                elif t_ece >=  t_warning - self.dt * self.seq_len_ece and t_ece < t_disrupt - self.dt * (self.seq_len_ece + self.dist):
                    indx_ece = df_ece_shot.index.values[idx_ece]
                    indx_efit = df_efit_shot.index.values[idx_efit]
                    indx_diag = df_diag_shot.index.values[idx_diag]
                    
                    indices.append((indx_efit, indx_ece, indx_diag))
                    
                    pred_time = t_disrupt - t_ece - self.dt * self.seq_len_ece
                    dists.append(pred_time)
                    labels.append(1)
                    
                    if self.dt == 0.01:
                        idx_ece += 1
                        idx_diag += 1
                        
                    elif self.dt == 0.001:
                        idx_ece += self.seq_len_ece // 100
                        idx_diag += self.seq_len_diag // 100  
                
                elif t_ece >= t_disrupt - self.dt * (self.seq_len_ece + self.dist) and t_ece <= tipminf - self.dt * self.seq_len_ece:
                    indx_ece = df_ece_shot.index.values[idx_ece]
                    indx_efit = df_efit_shot.index.values[idx_efit]
                    indx_diag = df_diag_shot.index.values[idx_diag]
                    
                    indices.append((indx_efit, indx_ece, indx_diag))
                    labels.append(1)
                    
                    pred_time = t_disrupt - t_ece - self.dt * self.seq_len_ece
                    pred_time = pred_time if pred_time > 0 else 0
                    dists.append(pred_time)
                    
                    if self.dt == 0.01:
                        idx_ece += 1
                        idx_diag += 1
                        
                    elif self.dt == 0.001:
                        idx_ece += 1
                        idx_diag += 1
                
                elif t_ece < tftsrt + 1.0:                
                    if self.dt == 0.01:
                        idx_ece += self.seq_len_ece // 5
                        idx_diag += self.seq_len_diag // 5
                        
                    elif self.dt == 0.001:
                        idx_ece += self.seq_len_ece // 5
                        idx_diag += self.seq_len_diag // 5
                    
                elif t_ece > t_disrupt:
                    break
                
                else:
                    if self.dt == 0.01:
                        idx_ece += self.seq_len_ece // 5
                        idx_diag += self.seq_len_diag // 5
                        
                    elif self.dt == 0.001:
                        idx_ece += self.seq_len_ece // 5
                        idx_diag += self.seq_len_diag // 5
            
            self.shot_num.extend([shot for _ in range(len(indices))])
            self.t_warning_list.extend([t_disrupt - t_warning for _ in range(len(indices))])
            self.indices.extend(indices)
            self.labels.extend(labels)
            self.dist_list.extend(dists)
        
        self.n_disrupt = np.sum(np.array(self.labels) == 1)
        self.n_normal = np.sum(np.array(self.labels) == 0)
        
        print("# {} | disruptive : {} | normal : {}".format(self.task, self.n_disrupt, self.n_normal))     
    
    def generate_indices_eval(self):
        
        # initialize 
        self.shot_num.clear()
        self.indices.clear()
        self.labels.clear()
        self.t_warning_list.clear()
        
        self.n_disrupt = None
        self.n_normal = None
        
        for shot in tqdm(self.shot_list, desc = "# {} | generate indice".format(self.task)):
            
            tTQend = self.data_disrupt[self.data_disrupt.shot == shot].t_tmq.values[0]
            tftsrt = self.data_disrupt[self.data_disrupt.shot == shot].t_flattop_start.values[0]
            tipminf = self.data_disrupt[self.data_disrupt.shot == shot].t_ip_min_fault.values[0]
            t_warning = self.data_disrupt[self.data_disrupt.shot == shot].t_warning.values[0]
            
            if self.mode == 'TQ':
                t_disrupt = tTQend
                
            elif self.mode =='CQ':
                t_disrupt = tipminf
            
            df_efit_shot = self.data_efit[self.data_efit.shot == shot]
            df_ece_shot = self.data_ece[self.data_ece.shot == shot]
            df_diag_shot = self.data_diag[self.data_diag.shot == shot]
            
            indices = []
            labels = []
            dists = []
            
            if self.dt == 0.01:
                idx_efit = int(tftsrt / self.dt) // 5
                ratio = 5
                dt_efit_ece = 0.045
            elif self.dt == 0.001:
                idx_efit = int(tftsrt / self.dt) // 10
                ratio = 10
                dt_efit_ece = 0.009

            idx_ece = int(tftsrt / self.dt)
            idx_diag = int(tftsrt / self.dt)
            
            idx_efit_last = len(df_efit_shot.index)
            idx_ece_last = len(df_ece_shot.index)
            idx_diag_last = len(df_diag_shot.index)
            
            while(idx_ece < idx_ece_last or idx_diag < idx_diag_last):
                
                t_efit = df_efit_shot.iloc[idx_efit]['time']
                t_ece = df_ece_shot.iloc[idx_ece]['time']
                t_diag = df_diag_shot.iloc[idx_diag]['time']
                
                assert abs(t_diag - t_ece) <= self.dt * 2, "Dignostic data and ECE data are not consistent"
                
                # end condition
                if idx_ece_last - idx_ece - self.seq_len_ece - self.dist < 0 or idx_efit_last - idx_efit - self.seq_len_efit - self.dist // ratio < 0:
                    break
                
                # synchronize t_efit and t_ece
                if t_efit <= t_ece - dt_efit_ece:
                    t_diff = t_ece - t_efit
                    idx_efit += int(round(t_diff / self.dt / ratio, 1))
                
                if t_ece >= tftsrt + 1.0 and t_ece < t_warning - self.dt * (self.seq_len_ece + self.dist_warning):
                    indx_ece = df_ece_shot.index.values[idx_ece]
                    indx_efit = df_efit_shot.index.values[idx_efit]
                    indx_diag = df_diag_shot.index.values[idx_diag]
                    
                    indices.append((indx_efit, indx_ece, indx_diag))
                    labels.append(0)
                    
                    pred_time = t_disrupt - t_ece - self.dt * self.seq_len_ece
                    dists.append(pred_time)
        
                    if self.dt == 0.01:
                        idx_ece += self.seq_len_ece // 5
                        idx_diag += self.seq_len_diag // 5
                        
                    elif self.dt == 0.001:
                        idx_ece += self.seq_len_ece // 5
                        idx_diag += self.seq_len_diag // 5
                
                elif t_ece >=  t_warning - self.dt * (self.seq_len_ece + self.dist_warning) and t_ece < t_disrupt - self.dt * (self.seq_len_ece + self.dist):
                    indx_ece = df_ece_shot.index.values[idx_ece]
                    indx_efit = df_efit_shot.index.values[idx_efit]
                    indx_diag = df_diag_shot.index.values[idx_diag]
                    
                    if self.dt == 0.01:
                        idx_ece += self.seq_len_ece // 50
                        idx_diag += self.seq_len_diag // 50
                        
                    elif self.dt == 0.001:
                        idx_ece += self.seq_len_ece // 100
                        idx_diag += self.seq_len_diag // 100
                
                elif t_ece >= t_disrupt - self.dt * (self.seq_len_ece + self.dist) and t_ece <= tipminf - self.dt * self.seq_len_ece:
                    indx_ece = df_ece_shot.index.values[idx_ece]
                    indx_efit = df_efit_shot.index.values[idx_efit]
                    indx_diag = df_diag_shot.index.values[idx_diag]
                    
                    indices.append((indx_efit, indx_ece, indx_diag))
                    labels.append(1)
                    
                    pred_time = t_disrupt - t_ece - self.dt * self.seq_len_ece
                    dists.append(pred_time)

                    if self.dt == 0.01:
                        idx_ece += 1
                        idx_diag += 1
                        
                    elif self.dt == 0.001:
                        idx_ece += 2
                        idx_diag += 2
                
                elif t_ece < tftsrt + 1.0:
                    
                    if self.dt == 0.01:
                        idx_ece += self.seq_len_ece // 5
                        idx_diag += self.seq_len_diag // 5
                        
                    elif self.dt == 0.001:
                        idx_ece += self.seq_len_ece // 5
                        idx_diag += self.seq_len_diag // 5
                    
                elif t_ece > t_disrupt:
                    break
                
                else:
                    if self.dt == 0.01:
                        idx_ece += self.seq_len_ece // 5
                        idx_diag += self.seq_len_diag // 5
                        
                    elif self.dt == 0.001:
                        idx_ece += self.seq_len_ece // 5
                        idx_diag += self.seq_len_diag // 5
            
            self.shot_num.extend([shot for _ in range(len(indices))])
            self.t_warning_list.extend([t_disrupt - t_warning for _ in range(len(indices))])
            self.indices.extend(indices)
            self.labels.extend(labels)
            self.dist_list.extend(dists)
            
    def __getitem__(self, idx : int):
        
        indx_efit, indx_ece, indx_diag = self.indices[idx]
        label = np.array(self.labels[idx])
        
        data_efit = self.data_efit.loc[indx_efit + 1:indx_efit + self.seq_len_efit, config.EFIT + config.EFIT_FE].values.reshape(-1, self.seq_len_efit)
        data_ece = self.data_ece.loc[indx_ece + 1:indx_ece + self.seq_len_ece, config.ECE].values.reshape(-1, self.seq_len_ece)
        data_diag = self.data_diag.loc[indx_diag + 1:indx_diag + self.seq_len_diag, config.DIAG + config.DIAG_FE].values.reshape(-1, self.seq_len_diag)
 
        data_efit = torch.from_numpy(data_efit).float()
        data_ece = torch.from_numpy(data_ece).float()
        data_diag = torch.from_numpy(data_diag).float()
        label = torch.from_numpy(label).float()
        
        shot_num = self.shot_num[idx]
        dist = self.dist_list[idx]
        t_warning = self.t_warning_list[idx]
        
        data = {
            "efit":data_efit,
            "ece":data_ece,
            "diag":data_diag,
            "label":label,
            "shot_num":shot_num,
            'dist':dist,
            't_warning':t_warning
        }
  
        return data
    
    def __len__(self):
        return len(self.indices)
    
    def get_num_per_cls(self):
        self.num_per_cls_dict = dict()
        self.num_per_cls_dict[1] = np.sum(np.where(np.array(self.labels) > 0, 1, 0))
        self.num_per_cls_dict[0] = len(self.labels) - self.num_per_cls_dict[1]
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.n_classes):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list