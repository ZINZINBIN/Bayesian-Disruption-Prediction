''' 
    Temperature scaling for optimizing calibration error
    See the github https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Union
from src.models.predictor import Predictor, BayesianPredictor

class TemperatureScaling(nn.Module):
    def __init__(self, model:Union[Predictor, BayesianPredictor], gamma : float = 1.0):
        super().__init__()
        self.model = model
        self.T = nn.Parameter(torch.ones(1) * gamma)
        
    def forward(self, x : Dict[str, torch.Tensor]):
        logits = self.model(x)
        return self.temperature_scale(logits)
    
    def temperature_scale(self, logits:torch.Tensor):
        T = self.T.to(self.model.device)
        return logits / T
    
    def fit(self, dataloader:DataLoader, loss_fn:nn.Module, lr : float  = 1e-3, max_iter : int = 32, n_bins : int = 15):
        
        total_logits = []
        total_labels = []
        
        for idx, (data) in enumerate(dataloader):
            with torch.no_grad():
                logits = self.model(data)
                total_logits.append(logits.view(-1))
                total_labels.append(data['label'].view(-1))
                
        total_logits = torch.cat(total_logits)
        total_labels = torch.cat(total_labels)
        
        loss_ece = ECE(n_bins)
        
        T_nll = loss_fn(total_logits, total_labels.to(total_logits.device)).item()
        T_ece = loss_ece(total_logits, total_labels.to(total_logits.device)).item()
        
        print("Before temperature scaling | Loss:{:.3f} | ECE:{:.3f}".format(T_nll, T_ece))
        
        optimizer = torch.optim.LBFGS([self.T], lr = lr, max_iter = max_iter)
        
        def eval():
            optimizer.zero_grad()
            loss = loss_fn(self.temperature_scale(total_logits), total_labels.to(total_logits.device))
            loss.backward()
            return loss

        optimizer.step(eval)
        
        T_nll = loss_fn(self.temperature_scale(total_logits), total_labels.to(total_logits.device)).item()
        T_ece = loss_ece(self.temperature_scale(total_logits), total_labels.to(total_logits.device)).item()
        
        print("After temperature scaling | Loss:{:.3f} | ECE:{:.3f}".format(T_nll, T_ece))
        print("Optimal temperature scale : {:.3f}".format(self.T.item()))
        
class ECE(nn.Module):
    def __init__(self, n_bins:int = 15):
        super().__init__()
        bins = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bins[:-1]
        self.bin_uppers = bins[1:]
        
    def forward(self, logits:torch.Tensor, labels:torch.Tensor):
        probs = torch.nn.functional.sigmoid(logits)
        acc = torch.where(probs>0.5,1,0).eq(labels)
        ece = torch.zeros(1,device=logits.device)
        
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = probs.gt(bin_lower.item()) * probs.le(bin_upper.item())
            prob_in_bin = in_bin.float().mean()
            
            if prob_in_bin.item() > 0:
                acc_in_bin = acc[in_bin].float().mean()
                avg_conf_in_bin = probs[in_bin].mean()
                ece += torch.abs(avg_conf_in_bin - acc_in_bin) * prob_in_bin
                
        return ece