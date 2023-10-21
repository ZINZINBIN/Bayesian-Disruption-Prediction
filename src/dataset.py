import numpy as np
import pandas as pd
import torch, os
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Literal
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
        dist:int = 4,
        dt:float = 0.01, 
        scaler_efit=None,
        scaler_ece=None,
        scaler_diag=None, 
        mode : Literal['TQ', 'CQ'] = 'CQ', 
        task : Literal['train', 'valid','test','eval'] = 'train'
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
            self.data_efit[config.EFIT + config.EFIT_FE] = self.scaler_efit.transform(self.data_efit[config.EFIT + config.EFIT_FE])
            
        if self.scaler_ece is not None:
            self.data_ece[config.ECE] = self.scaler_ece.transform(self.data_ece[config.ECE])
        
        if self.scaler_diag is not None:
            self.data_diag[config.DIAG] = self.scaler_diag.transform(self.data_diag[config.DIAG])
            
        print("# {} | preprocessing the data".format(self.task))
    
    def generate_indices(self):
        
        for shot in tqdm(self.shot_list, desc = "# {} | generate indice".format(self.task)):
            tTQend = self.data_disrupt[self.data_disrupt.shot == shot].t_tmq.values[0]
            tftsrt = self.data_disrupt[self.data_disrupt.shot == shot].t_flattop_start.values[0]
            tipminf = self.data_disrupt[self.data_disrupt.shot == shot].t_ip_min_fault.values[0]
            
            if self.mode == 'TQ':
                t_disrupt = tTQend
            elif self.mode =='CQ':
                t_disrupt = tipminf
            
            df_efit_shot = self.data_efit[self.data_efit.shot == shot]
            df_ece_shot = self.data_ece[self.data_ece.shot == shot]
            df_diag_shot = self.data_diag[self.data_diag.shot == shot]
            
            indices = []
            labels = []
            
            idx_efit = int(tftsrt / self.dt) // 5
            idx_ece = int(tftsrt / self.dt)
            idx_diag = int(tftsrt / self.dt)
            
            idx_efit_last = len(df_efit_shot.index)
            idx_ece_last = len(df_ece_shot.index)
            idx_diag_last = len(df_diag_shot.index)
            
            while(idx_ece < idx_ece_last or idx_diag < idx_diag_last):
                
                t_efit = df_efit_shot.iloc[idx_efit]['time']
                t_ece = df_ece_shot.iloc[idx_ece]['time']
                t_diag = df_diag_shot.iloc[idx_diag]['time']
                
                assert abs(t_diag - t_ece) <= 0.02, "Dignostic data and ECE data are not consistent"
                
                # end condition
                if idx_ece_last - idx_ece - self.seq_len_ece - self.dist < 0 or idx_efit_last - idx_efit - self.seq_len_efit - self.dist // 5 < 0:
                    break
                
                # synchronize t_efit and t_ece
                if t_efit <= t_ece - 0.05:
                    t_diff = t_ece - t_efit
                    idx_efit += int(round(t_diff / self.dt / 5, 1))
                
                if t_ece >= tftsrt and t_ece < t_disrupt - self.dt * (2.0 * self.seq_len_ece + self.dist):
                    indx_ece = df_ece_shot.index.values[idx_ece]
                    indx_efit = df_efit_shot.index.values[idx_efit]
                    indx_diag = df_diag_shot.index.values[idx_diag]
                    
                    indices.append((indx_efit, indx_ece, indx_diag))
                    labels.append(1)
        
                    idx_ece += self.seq_len_ece // 5
                    idx_diag += self.seq_len_diag // 5
                
                elif t_ece >=  t_disrupt - self.dt * (2.0 * self.seq_len_ece + self.dist) and t_ece < t_disrupt - self.dt * (self.seq_len_ece + self.dist):
                    indx_ece = df_ece_shot.index.values[idx_ece]
                    indx_efit = df_efit_shot.index.values[idx_efit]
                    indx_diag = df_diag_shot.index.values[idx_diag]
                    
                    indices.append((indx_efit, indx_ece, indx_diag))
                    labels.append(1)
                    
                    idx_ece += self.seq_len_ece // 10
                    idx_diag += self.seq_len_diag // 10
                    
                    # idx_ece += self.seq_len_ece // 25
                    # idx_diag += self.seq_len_diag // 25
                
                elif t_ece >= t_disrupt - self.dt * (self.seq_len_ece + self.dist) and t_ece <= t_disrupt - self.dt * self.seq_len_ece + 2 * self.dt:
                    indx_ece = df_ece_shot.index.values[idx_ece]
                    indx_efit = df_efit_shot.index.values[idx_efit]
                    indx_diag = df_diag_shot.index.values[idx_diag]
                    
                    indices.append((indx_efit, indx_ece, indx_diag))
                    labels.append(0)

                    idx_ece += 1
                    idx_diag += 1
                
                elif t_ece < tftsrt:
                    idx_ece += self.seq_len_ece // 5
                    idx_diag += self.seq_len_diag // 5
                    
                elif t_ece > t_disrupt:
                    break
                
                else:
                    idx_ece += self.seq_len_ece // 5
                    idx_diag += self.seq_len_diag // 5
            
            self.shot_num.extend([shot for _ in range(len(indices))])
            self.indices.extend(indices)
            self.labels.extend(labels)
        
        self.n_disrupt = np.sum(np.array(self.labels) == 0)
        self.n_normal = np.sum(np.array(self.labels) == 1)
        
        print("# {} | disruptive : {} | normal : {}".format(self.task, self.n_disrupt, self.n_normal))     
    
    def generate_indices_eval(self):
        
        # initialize 
        self.shot_num.clear()
        self.indices.clear()
        self.labels.clear()
        
        self.n_disrupt = None
        self.n_normal = None
        
        for shot in tqdm(self.shot_list, desc = "# {} | generate indice".format(self.task)):
            
            tTQend = self.data_disrupt[self.data_disrupt.shot == shot].t_tmq.values[0]
            tftsrt = self.data_disrupt[self.data_disrupt.shot == shot].t_flattop_start.values[0]
            tipminf = self.data_disrupt[self.data_disrupt.shot == shot].t_ip_min_fault.values[0]
            
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
            
            idx_efit = int(tftsrt / self.dt) // 5
            idx_ece = int(tftsrt / self.dt)
            idx_diag = int(tftsrt / self.dt)
            
            idx_efit_last = len(df_efit_shot.index)
            idx_ece_last = len(df_ece_shot.index)
            idx_diag_last = len(df_diag_shot.index)
            
            while(idx_ece < idx_ece_last or idx_diag < idx_diag_last):
                
                t_efit = df_efit_shot.iloc[idx_efit]['time']
                t_ece = df_ece_shot.iloc[idx_ece]['time']
                t_diag = df_diag_shot.iloc[idx_diag]['time']
                
                assert abs(t_diag - t_ece) <= 0.02, "Dignostic data and ECE data are not consistent"
                
                # end condition
                if idx_ece_last - idx_ece - self.seq_len_ece - self.dist < 0 or idx_efit_last - idx_efit - self.seq_len_efit - self.dist // 5 < 0:
                    break
                
                # synchronize t_efit and t_ece
                if t_efit <= t_ece - 0.05:
                    t_diff = t_ece - t_efit
                    idx_efit += int(round(t_diff / self.dt / 5, 1))
                
                if t_ece >= tftsrt and t_ece < t_disrupt - self.dt * (1.5 * self.seq_len_ece + self.dist):
                    indx_ece = df_ece_shot.index.values[idx_ece]
                    indx_efit = df_efit_shot.index.values[idx_efit]
                    indx_diag = df_diag_shot.index.values[idx_diag]
        
                    idx_ece += self.seq_len_ece // 5
                    idx_diag += self.seq_len_diag // 5
                
                elif t_ece >=  t_disrupt - self.dt * (1.5 * self.seq_len_ece + self.dist) and t_ece < t_disrupt - self.dt * (self.seq_len_ece + self.dist):
                    indx_ece = df_ece_shot.index.values[idx_ece]
                    indx_efit = df_efit_shot.index.values[idx_efit]
                    indx_diag = df_diag_shot.index.values[idx_diag]
                    
                    idx_ece += self.seq_len_ece // 10
                    idx_diag += self.seq_len_diag // 10
                
                elif t_ece >= t_disrupt - self.dt * (self.seq_len_ece + self.dist) and t_ece <= t_disrupt - self.dt * self.seq_len_ece + 2 * self.dt:
                    indx_ece = df_ece_shot.index.values[idx_ece]
                    indx_efit = df_efit_shot.index.values[idx_efit]
                    indx_diag = df_diag_shot.index.values[idx_diag]
                    
                    indices.append((indx_efit, indx_ece, indx_diag))
                    labels.append(0)
                    
                    pred_time = t_disrupt - t_ece - self.dt * self.seq_len_ece
                    dists.append(pred_time)

                    idx_ece += 1
                    idx_diag += 1
                
                elif t_ece < tftsrt:
                    idx_ece += self.seq_len_ece // 5
                    idx_diag += self.seq_len_diag // 5
                    
                elif t_ece > t_disrupt:
                    break
                
                else:
                    idx_ece += self.seq_len_ece // 5
                    idx_diag += self.seq_len_diag // 5
            
            self.shot_num.extend([shot for _ in range(len(indices))])
            self.indices.extend(indices)
            self.labels.extend(labels)
            self.dist_list.extend(dists)
            
    def __getitem__(self, idx : int):
        
        indx_efit, indx_ece, indx_diag = self.indices[idx]
        label = self.labels[idx]
        label = np.array(label)
        
        data_efit = self.data_efit.loc[indx_efit + 1:indx_efit + self.seq_len_efit, config.EFIT + config.EFIT_FE].values.reshape(-1, self.seq_len_efit)
        data_ece = self.data_ece.loc[indx_ece + 1:indx_ece + self.seq_len_ece, config.ECE].values.reshape(-1, self.seq_len_ece)
        data_diag = self.data_diag.loc[indx_diag + 1:indx_diag + self.seq_len_diag, config.DIAG].values.reshape(-1, self.seq_len_diag)
 
        data_efit = torch.from_numpy(data_efit).float()
        data_ece = torch.from_numpy(data_ece).float()
        data_diag = torch.from_numpy(data_diag).float()
        label = torch.from_numpy(label)
        
        if self.get_shot_num and self.task != 'eval':
            shot_num = self.shot_num[idx]
            data = {
                "efit":data_efit,
                "ece":data_ece,
                "diag":data_diag,
                "label":label,
                "shot_num":shot_num
            }
            
        elif self.task == 'eval':
            shot_num = self.shot_num[idx]
            dist = self.dist_list[idx]
            data = {
                "efit":data_efit,
                "ece":data_ece,
                "diag":data_diag,
                "label":label,
                "shot_num":shot_num,
                'dist':dist
            }
            
        else:
            data = {
                "efit":data_efit,
                "ece":data_ece,
                "diag":data_diag,
                "label":label,
            }
        
        return data
    
    def __len__(self):
        return len(self.indices)
    
    def get_num_per_cls(self):
        classes = np.unique(self.labels)
        self.num_per_cls_dict = dict()

        for cls in classes:
            num = np.sum(np.where(self.labels == cls, 1, 0))
            self.num_per_cls_dict[cls] = num
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.n_classes):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list