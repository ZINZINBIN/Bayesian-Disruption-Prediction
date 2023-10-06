from typing import Optional, List, Literal, Union
from src.loss import LDAMLoss, FocalLoss, CELoss
import os
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from src.utils.EarlyStopping import EarlyStopping
from src.models.BNN import compute_uncertainty_per_data
from ray import tune
from ray.air import session
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from src.utils.EarlyStopping import EarlyStopping

import warnings

# remove warning
warnings.filterwarnings("ignore")

def train_per_epoch(
    train_loader : DataLoader, 
    model : Union[torch.nn.Module, torch.nn.DataParallel],
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    max_norm_grad : Optional[float] = None,
    ):

    model.train()
    train_loss = 0
    total_pred = []
    total_label = []
    total_size = 0
    
    for batch_idx, data in enumerate(train_loader):
        
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data['label'].to(device))
        
        if not torch.isfinite(loss):
            print("train_per_epoch | warning : loss nan occurs at batch_idx : {}".format(batch_idx))
            continue
        else:
            loss.backward()

        # use gradient clipping
        if max_norm_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_grad)

        optimizer.step()

        train_loss += loss.item()

        pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
        total_size += pred.size(0) 
        
        total_pred.append(pred.view(-1,1))
        total_label.append(data['label'].view(-1,1))
        
    if scheduler:
        scheduler.step()
        
    total_pred = torch.concat(total_pred, dim = 0).detach().view(-1,).cpu().numpy()
    total_label = torch.concat(total_label, dim = 0).detach().view(-1,).cpu().numpy()

    if total_size > 0:
        train_loss /= total_size
        train_f1 = f1_score(total_label, total_pred, average = "macro")
        
    else:
        train_loss = 0
        train_f1 = 0

    return train_loss, train_f1

def valid_per_epoch(
    valid_loader : DataLoader, 
    model : Union[torch.nn.Module, torch.nn.DataParallel],
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    ):

    model.eval()
    # model.to(device)
    valid_loss = 0

    total_pred = []
    total_label = []
    total_size = 0

    for batch_idx, data in enumerate(valid_loader):
        with torch.no_grad():
            
            optimizer.zero_grad()
            
            output = model(data)
          
            loss = loss_fn(output, data['label'].to(device))
    
            valid_loss += loss.item()
            pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
            total_size += pred.size(0)

            total_pred.append(pred.view(-1,1))
            total_label.append(data['label'].view(-1,1))

    total_pred = torch.concat(total_pred, dim = 0).detach().view(-1,).cpu().numpy()
    total_label = torch.concat(total_label, dim = 0).detach().view(-1,).cpu().numpy()

    valid_loss /= total_size
    valid_f1 = f1_score(total_label, total_pred, average = "macro")

    return valid_loss, valid_f1

def train(
    train_loader : DataLoader, 
    valid_loader : DataLoader,
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : Union[torch.nn.CrossEntropyLoss, LDAMLoss, FocalLoss],
    device : str = "cpu",
    num_epoch : int = 64,
    max_norm_grad : Optional[float] = None,
    checkpoint_dir : Optional[str] = None,
    is_early_stopping : bool = True,
    early_stopping_verbose : bool = True,
    early_stopping_patience : int = 12,
    early_stopping_delta : float = 1e-3
    ):

    train_loss_list = []
    valid_loss_list = []
    
    train_f1_list = []
    valid_f1_list = []
        
    if is_early_stopping:
        early_stopping = EarlyStopping(None, early_stopping_patience, early_stopping_verbose, early_stopping_delta)
    else:
        early_stopping = None

    for epoch in tqdm(range(num_epoch), desc = "training process for hyperparameter optimization"):

        train_loss, train_f1 = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            max_norm_grad,
        )

        valid_loss, valid_f1 = valid_per_epoch(
            valid_loader, 
            model,
            optimizer,
            loss_fn,
            device,
        )

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        train_f1_list.append(train_f1)
        valid_f1_list.append(valid_f1)
               
        # version 01
        # tune checkpoint and save
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        # tune report
        tune.report(loss=valid_loss, f1_score=valid_f1, training_iteration = epoch + 1)
        
        # version 02
        # session.report({"loss": valid_loss, "f1_score":valid_f1}, checkpoint=checkpoint_dir)
        
        if early_stopping:
            early_stopping(valid_f1, model)
            if early_stopping.early_stop:
                break
        
    return  train_loss_list, train_f1_list,  valid_loss_list, valid_f1_list

# Deferred Re-weighting with LDAM or Focal Loss
def train_DRW(
    train_loader : DataLoader,
    valid_loader : DataLoader,
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    loss_fn : Union[LDAMLoss, FocalLoss],
    device : str = "cpu",
    num_epoch : int = 64,
    max_norm_grad : Optional[float] = None,
    cls_num_list : Optional[List] = None,
    betas : List = [0, 0.25, 0.75, 0.9],
    checkpoint_dir : Optional[str] = None,
    is_early_stopping : bool = True,
    early_stopping_verbose : bool = True,
    early_stopping_patience : int = 12,
    early_stopping_delta : float = 1e-3
    ):

    train_loss_list = []
    valid_loss_list = []

    train_f1_list = []
    valid_f1_list = []

    # class per weight update
    def _update_per_cls_weights(epoch : int, betas : List, cls_num_list : List):
        idx = epoch // int(num_epoch / len(betas))
        
        if idx >= len(betas):
            idx = len(betas) - 1
        
        beta = betas[idx]
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
        return per_cls_weights    
        
    if is_early_stopping:
        early_stopping = EarlyStopping(None, early_stopping_patience, early_stopping_verbose, early_stopping_delta)
    else:
        early_stopping = None
    
    for epoch in tqdm(range(num_epoch), desc = "training process(DRW) for hyperparameter optimization"):

        per_cls_weights = _update_per_cls_weights(epoch, betas, cls_num_list)

        # FocalLoss / LDAMLoss update weight
        loss_fn.update_weight(per_cls_weights)

        train_loss, train_f1 = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            None,
            loss_fn,
            device,
            max_norm_grad,
        )

        valid_loss, valid_f1 = valid_per_epoch(
            valid_loader, 
            model,
            optimizer,
            loss_fn,
            device,
        )

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        train_f1_list.append(train_f1)
        valid_f1_list.append(valid_f1)
        
        # tune checkpoint and save
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        # tune report
        tune.report(loss=valid_loss, f1_score=valid_f1, training_iteration = epoch + 1)
        # session.report({"loss": valid_loss, "f1_score":valid_f1}, checkpoint=checkpoint_dir)
        
        if early_stopping:
            early_stopping(valid_f1, model)
            if early_stopping.early_stop:
                break

    return  train_loss_list, train_f1_list,  valid_loss_list, valid_f1_list

def evaluate(
    test_loader : DataLoader, 
    model : Union[torch.nn.Module, torch.nn.DataParallel],
    loss_fn : Optional[torch.nn.Module]= None,
    device : Optional[str] = "cpu",
    threshold : float = 0.5,
    ):

    test_loss = 0
    test_f1 = 0
    total_pred = []
    total_label = []

    if device is None:
        device = torch.device("cuda:0")
                              
    model.eval()
    
    total_size = 0
    
    for idx, data in enumerate(tqdm(test_loader, 'evaluation process')):
        with torch.no_grad():
            output = model(data)       
            loss = loss_fn(output, data['label'].to(device))
            test_loss += loss.item()
            
            pred = torch.nn.functional.softmax(output, dim = 1)[:,0]
            pred = torch.logical_not((pred > torch.FloatTensor([threshold]).to(device)))
            total_size += pred.size(0)
            
            pred_normal = torch.nn.functional.softmax(output, dim = 1)[:,1].detach()

            total_pred.append(pred_normal.view(-1,1))
            total_label.append(data['label'].view(-1,1))
            
    test_loss /= (idx + 1)
    
    total_pred = torch.concat(total_pred, dim = 0).detach().view(-1,).cpu().numpy()
    total_label = torch.concat(total_label, dim = 0).detach().view(-1,).cpu().numpy()

    # method 2 : compute f1, auc, roc and classification report
    # data clipping / postprocessing for ignoring nan, inf, too large data
    total_pred = np.nan_to_num(total_pred, copy = True, nan = 0, posinf = 1.0, neginf = 0)
    total_pred = np.where(total_pred > 1 - threshold, 1, 0)
   
    # f1 score
    test_f1 = f1_score(total_label, total_pred, average = "macro")
    
    model.cpu()
    
    return test_loss, test_f1