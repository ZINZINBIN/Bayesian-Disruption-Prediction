# This code is based on LDAM
# reference : https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional, List
import warnings

warnings.filterwarnings(action="ignore")

# Focal Loss : https://arxiv.org/abs/1708.02002
class FocalLoss(nn.Module):
    def __init__(self, weight : Optional[torch.Tensor] = None, gamma : float = 2.0):
        super(FocalLoss, self).__init__()
        assert gamma >= 0, "gamma should be positive"
        self.model_type = "Focal"
        self.gamma = gamma
        self.weight = weight
    
    def update_weight(self, weight : Optional[torch.Tensor] = None):
        self.weight = weight

    def compute_focal_loss(self, inputs:torch.Tensor, gamma:float, alpha : torch.Tensor):
        p = torch.exp(-inputs)
        loss = alpha * (1-p) ** gamma * inputs
        return loss.mean()

    def forward(self, input : torch.Tensor, target : torch.Tensor):
        weight = self.weight.to(input.device)
        alpha = weight.gather(0, target.data.long())
        alpha = Variable(alpha)
        return self.compute_focal_loss(F.binary_cross_entropy(F.sigmoid(input).view(-1), target, reduction = 'none', weight = None), self.gamma, alpha)

# Label-Distribution-Aware Margin loss
class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list : Optional[List], max_m : float = 0.5, weight : Optional[torch.Tensor] = None, s : int = 30):
        super(LDAMLoss, self).__init__()
        assert s > 0, "s should be positive"
        self.model_type = "LDAM"
        self.s = s
        self.max_m = max_m
        self.weight = weight

        if cls_num_list:
            self.update_m_list(cls_num_list)
        
    def update_weight(self, weight : Optional[torch.Tensor] = None):
        self.weight = weight

    def update_m_list(self, cls_num_list : List):
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (self.max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list
    
    def forward(self, x : torch.Tensor, target : torch.Tensor):
        idx = torch.zeros_like(x, dtype = torch.uint8).to(x.device)
        idx.scatter_(1, torch.where(target.data > 0, 1, 0).view(-1,1).long(), 1)

        idx_float = idx.type(torch.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], idx_float.transpose(0,1))
        batch_m = batch_m.view((-1,1)).to(x.device)
        x_m = x - batch_m

        output = torch.where(idx, x_m, x)

        return F.binary_cross_entropy(self.s * F.sigmoid(output).view(-1), target, weight = self.weight, reduction = 'mean')

class CELoss(nn.Module):
    def __init__(self, weight : Optional[torch.Tensor] = None):
        super(CELoss, self).__init__()
        self.model_type = "CE"
        self.weight = weight
        
    def update_weight(self, weight : Optional[torch.Tensor] = None):
        self.weight = weight
    
    def forward(self, input : torch.Tensor, target : torch.Tensor):
        alpha = self.weight.gather(0, target.data.long())
        alpha = Variable(alpha)
        loss = alpha * F.binary_cross_entropy(F.sigmoid(input).view(-1), target, weight = None, reduction = 'none')   
        return loss.mean()
    
# Label smoothing
# Reference : https://cvml.tistory.com/9
class LabelSmoothingLoss(nn.Module):
    def __init__(self, original_loss : nn.Module, alpha : float = 0.01, kl_weight : float = 0.2, classes : int = 2, dim : int = -1):
        super().__init__()
        self.original_loss = original_loss
        self.confidence = 1.0 - alpha
        self.alpha = alpha
        self.cls = classes
        self.dim = dim
        self.kl_weight = kl_weight
        
    def forward(self, input : torch.Tensor, target : torch.Tensor):
        loss = self.original_loss(input, target)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(input)
            true_dist.fill_(self.alpha / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        log_p = input.log_softmax(dim = self.dim)
        kl_loss = torch.sum(true_dist * log_p, dim = self.dim).mean()
        return loss + kl_loss * self.kl_weight