import numpy as np
import pandas as pd
import torch, os
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Optional, Dict, List, Union, Literal
from src.config import Config
from src.profile import get_profile

config = Config()

# Dataset for 0D data : different method used for label matching
class DatasetFor0D(Dataset):
    def __init__(self, ts_data : pd.DataFrame, disrupt_data : pd.DataFrame, seq_len : int = 21, cols : List = [], dist:int = 3, dt : float = 1.0 / 210 * 4, scaler = None, mode : Literal['TQ', 'CQ'] = 'TQ'):
        self.ts_data = ts_data
        self.disrupt_data = disrupt_data
        self.seq_len = seq_len
        self.dt = dt
        self.cols = cols
        self.dist = dist # distance

        self.indices = []
        self.labels = []
        self.shot_num = [] # shot list with respond to indices
        self.get_shot_num = False
        
        self.scaler = scaler
        
        self.n_classes = 2
        
        shot_list = np.unique(self.ts_data.shot.values).tolist()
        self.shot_list = [shot_num for shot_num in shot_list if shot_num in disrupt_data.shot.values]
        
        self.mode = mode
        
        self.preprocessing()
        self._generate_index()
        
    def preprocessing(self):
        # ignore shot which have too many nan values
        shot_ignore = []
        
        # filter : remove invalid shot
        for shot in tqdm(self.shot_list, desc = 'remove invalid data : null / measurement error'):
            # 1st filter : remove null data
            df_shot = self.ts_data[self.ts_data.shot == shot]
            null_check = df_shot[self.cols].isna().sum()
            
            is_null = False
            
            for c in null_check:
                if c > 0.5 * len(df_shot):
                    shot_ignore.append(shot)
                    is_null = True
                    break
            
            if is_null:
                continue
            
            # 2nd filter : measurement error
            for col in self.cols:
                if np.sum(df_shot[col] == 0) > 0.5 * len(df_shot):
                    shot_ignore.append(shot)
                    break

                # constant value
                if df_shot[col].max() - df_shot[col].min() < 1e-3:
                    shot_ignore.append(shot)
                    break
        
        # update shot list with ignoring the null data
        shot_list_new = [shot_num for shot_num in self.shot_list if shot_num not in shot_ignore]
        self.shot_list = shot_list_new
        
        # 0D parameter : NAN -> forward fill
        for shot in tqdm(self.shot_list, desc = 'replace nan value'):
            df_shot = self.ts_data[self.ts_data.shot == shot].copy()
            self.ts_data.loc[self.ts_data.shot == shot, self.cols] = df_shot[self.cols].fillna(0)
            
        if self.scaler is not None:
            self.ts_data[self.cols] = self.scaler.transform(self.ts_data[self.cols])

    def _generate_index(self):
        df_disruption = self.disrupt_data
        
        for shot in tqdm(self.shot_list, desc = 'generate index'):
            tTQend = df_disruption[df_disruption.shot == shot].t_tmq.values[0]
            tftsrt = df_disruption[df_disruption.shot == shot].t_flattop_start.values[0]
            tipminf = df_disruption[df_disruption.shot == shot].t_ip_min_fault.values[0]
            
            if self.mode == 'TQ':
                t_disrupt = tTQend
            elif self.mode =='CQ':
                t_disrupt = tipminf
            
            df_shot = self.ts_data[self.ts_data.shot == shot]
            
            indices = []
            labels = []

            idx = int(tftsrt / self.dt)
            idx_last = len(df_shot.index)

            while(idx < idx_last):
                
                row = df_shot.iloc[idx]
                t = row['time']

                if idx_last - idx - self.seq_len - self.dist < 0:
                    break

                if t >= tftsrt and t < t_disrupt - self.dt * (1.5 * self.seq_len + self.dist):
                    indx = df_shot.index.values[idx]
                    indices.append(indx)
                    labels.append(1)
                    idx += self.seq_len // 4
                    
                elif t >= t_disrupt - self.dt * (1.5 * self.seq_len + self.dist) and t < t_disrupt - self.dt * (self.seq_len + self.dist):
                    indx = df_shot.index.values[idx]
                    indices.append(indx)
                    labels.append(1)
                    idx += self.seq_len // 8
                
                elif t >= t_disrupt - self.dt * (self.seq_len + self.dist) and t <= t_disrupt - self.dt * self.seq_len + self.dt:
                    indx = df_shot.index.values[idx]
                    indices.append(indx)
                    labels.append(0)
                    idx += 1
                
                elif t < tftsrt:
                    idx += self.seq_len // 4
                
                elif t > t_disrupt:
                    break
                
                else:
                    idx += self.seq_len // 4
                    
            self.shot_num.extend([shot for _ in range(len(indices))])
            self.indices.extend(indices)
            self.labels.extend(labels)
            
        self.n_disrupt = np.sum(np.array(self.labels)==0)
        self.n_normal = np.sum(np.array(self.labels)==1)
        
    def __getitem__(self, idx:int):
        indx = self.indices[idx]
        label = self.labels[idx]
        label = np.array(label)
        label = torch.from_numpy(label)
        data = self.ts_data[self.cols].loc[indx+1:indx+self.seq_len].values
        data = torch.from_numpy(data).float()
        
        if self.get_shot_num:
            shot_num = self.shot_num[idx]
            return data, label, shot_num
        
        else:
            return data, label

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
    
    
# Dataset for 0D data : different method used for label matching
class DatasetForProfile(Dataset):
    def __init__(self, ts_data : pd.DataFrame, disrupt_data : pd.DataFrame, seq_len : int = 21, cols : List = [], dist:int = 3, dt : float = 1.0 / 210 * 4, scaler = None, mode : Literal['TQ', 'CQ'] = 'TQ', n_points : int = 128):
        self.ts_data = ts_data
        self.disrupt_data = disrupt_data
        self.seq_len = seq_len
        self.dt = dt
        self.cols = cols
        self.dist = dist # distance

        self.indices = []
        self.labels = []
        self.shot_num = [] # shot list with respond to indices
        self.get_shot_num = False
        
        self.scaler = scaler
        
        self.n_classes = 2
        
        shot_list = np.unique(self.ts_data.shot.values).tolist()
        self.shot_list = [shot_num for shot_num in shot_list if shot_num in disrupt_data.shot.values]
        
        self.mode = mode
        self.n_points = n_points
        
        self.preprocessing()
        self._generate_index()
        
    def preprocessing(self):
        # ignore shot which have too many nan values
        shot_ignore = []
        
        # filter : remove invalid shot
        for shot in tqdm(self.shot_list, desc = 'remove invalid data : null / measurement error'):
            # 1st filter : remove null data
            df_shot = self.ts_data[self.ts_data.shot == shot]
            null_check = df_shot[self.cols].isna().sum()
            
            is_null = False
            
            for c in null_check:
                if c > 0.5 * len(df_shot):
                    shot_ignore.append(shot)
                    is_null = True
                    break
            
            if is_null:
                continue
            
            # 2nd filter : measurement error
            for col in self.cols:
                if np.sum(df_shot[col] == 0) > 0.5 * len(df_shot):
                    shot_ignore.append(shot)
                    break

                # constant value
                if df_shot[col].max() - df_shot[col].min() < 1e-3:
                    shot_ignore.append(shot)
                    break        

        # update shot list with ignoring the null data
        shot_list_new = [shot_num for shot_num in self.shot_list if shot_num not in shot_ignore]
        shot_list_old = self.shot_list
        self.shot_list = shot_list_new
        
        # 0D parameter : NAN -> forward fill
        for shot in tqdm(self.shot_list, desc = 'replace nan value'):
            df_shot = self.ts_data[self.ts_data.shot == shot].copy()
            self.ts_data.loc[self.ts_data.shot == shot, self.cols] = df_shot[self.cols].fillna(0)
            
        if self.scaler is not None:
            self.ts_data[self.cols] = self.scaler.transform(self.ts_data[self.cols])
            
        # profile generation : Te and Ne
        self.ne_profile = np.zeros((len(self.ts_data), self.n_points))
        self.te_profile = np.zeros((len(self.ts_data), self.n_points))
        
        if os.path.exists("./dataset/profiles.npz"):
            profiles = np.load("./dataset/profiles.npz")
            self.ne_profile = profiles['ne']
            self.te_profile = profiles['te']
        else:
            idx = 0
            for shot in tqdm(shot_list_old, desc = "profile data generation"):
                df_shot = self.ts_data[self.ts_data.shot == shot].copy()
                tes = []
                nes = []
                for t in df_shot.time:
                    _, te = get_profile(df_shot, t, radius = config.RADIUS, cols_core = config.TS_TE_CORE_COLS, cols_edge = config.TS_TE_EDGE_COLS, n_points = self.n_points)
                    _, ne = get_profile(df_shot, t, radius = config.RADIUS, cols_core = config.TS_NE_CORE_COLS, cols_edge = config.TS_NE_EDGE_COLS, n_points = self.n_points)
                    tes.append(te.reshape(-1,1))
                    nes.append(ne.reshape(-1,1))
                    
                self.ne_profile[idx:idx + len(df_shot),:] = np.concatenate(nes, axis = 1).transpose(1,0)
                self.te_profile[idx:idx + len(df_shot),:] = np.concatenate(tes, axis = 1).transpose(1,0)
                
                idx += len(df_shot)    

    def _generate_index(self):
        df_disruption = self.disrupt_data
        
        for shot in tqdm(self.shot_list, desc = 'generate index'):
            tTQend = df_disruption[df_disruption.shot == shot].t_tmq.values[0]
            tftsrt = df_disruption[df_disruption.shot == shot].t_flattop_start.values[0]
            tipminf = df_disruption[df_disruption.shot == shot].t_ip_min_fault.values[0]
            
            if self.mode == 'TQ':
                t_disrupt = tTQend
            elif self.mode =='CQ':
                t_disrupt = tipminf
            
            df_shot = self.ts_data[self.ts_data.shot == shot]
            
            indices = []
            labels = []

            idx = int(tftsrt / self.dt)
            idx_last = len(df_shot.index)

            while(idx < idx_last):
                
                row = df_shot.iloc[idx]
                t = row['time']

                if idx_last - idx - self.seq_len - self.dist < 0:
                    break

                if t >= tftsrt and t < t_disrupt - self.dt * (1.5 * self.seq_len + self.dist):
                    indx = df_shot.index.values[idx]
                    indices.append(indx)
                    labels.append(1)
                    idx += self.seq_len // 4
                    
                elif t >= t_disrupt - self.dt * (1.5 * self.seq_len + self.dist) and t < t_disrupt - self.dt * (self.seq_len + self.dist):
                    indx = df_shot.index.values[idx]
                    indices.append(indx)
                    labels.append(1)
                    idx += self.seq_len // 8
                
                elif t >= t_disrupt - self.dt * (self.seq_len + self.dist) and t <= t_disrupt - self.dt * self.seq_len + self.dt:
                    indx = df_shot.index.values[idx]
                    indices.append(indx)
                    labels.append(0)
                    idx += 1
                
                elif t < tftsrt:
                    idx += self.seq_len // 4
                
                elif t > t_disrupt:
                    break
                
                else:
                    idx += self.seq_len // 4
                    
            self.shot_num.extend([shot for _ in range(len(indices))])
            self.indices.extend(indices)
            self.labels.extend(labels)
            
        self.n_disrupt = np.sum(np.array(self.labels)==0)
        self.n_normal = np.sum(np.array(self.labels)==1)
        
    def __getitem__(self, idx:int):
        indx = self.indices[idx]
        
        # label
        label = np.array(self.labels[idx])
        label = torch.from_numpy(label)
        
        # 0D data
        data_0D = self.ts_data[self.cols].loc[indx+1:indx+self.seq_len].values
        data_0D = torch.from_numpy(data_0D).float()
        
        # profile data
        te_profile = self.te_profile[indx:indx + self.seq_len,:]
        te_profile = torch.from_numpy(te_profile).float().unsqueeze(0)
        
        ne_profile = self.ne_profile[indx:indx + self.seq_len,:]
        ne_profile = torch.from_numpy(ne_profile).float().unsqueeze(0)
        
        data = {
            "0D" : data_0D,
            "te" : te_profile,
            "ne" : ne_profile,
        }
        
        if self.get_shot_num:
            shot_num = self.shot_num[idx]
            return data, label, shot_num
        
        else:
            return data, label

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