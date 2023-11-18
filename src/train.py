from typing import Optional, List, Literal, Union
from src.loss import LDAMLoss, FocalLoss
import os
import pdb
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from src.evaluate import evaluate_tensorboard
from src.utils.EarlyStopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from src.utils.utility import generate_model_performance
from src.config import Config

config = Config()

# anomaly detection from training process : backward process
torch.autograd.set_detect_anomaly(True)

def train_per_epoch(
    train_loader : DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    max_norm_grad : Optional[float] = None,
    ):

    model.train()
    model.to(device)

    train_loss = 0
    train_acc = 0

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
        pred = torch.nn.functional.sigmoid(output)
        pred = torch.where(pred > 0.5, 1, 0)
        train_acc += pred.eq(data['label'].to(device).view_as(pred)).sum().item()
        total_size += pred.size(0) 
        
        total_pred.append(pred.view(-1,1))
        total_label.append(torch.where(data['label'] > 0, 1, 0).view(-1,1))
        
    if scheduler:
        scheduler.step()
        
    total_pred = torch.concat(total_pred, dim = 0).detach().view(-1,).cpu().numpy()
    total_label = torch.concat(total_label, dim = 0).detach().view(-1,).cpu().numpy()

    if total_size > 0:
        # train_loss /= total_size
        train_loss /= (batch_idx + 1)
        train_acc /= total_size
        train_f1 = f1_score(total_label, total_pred, average = "macro")
        
    else:
        train_loss = 0
        train_acc = 0
        train_f1 = 0

    return train_loss, train_acc, train_f1

def valid_per_epoch(
    valid_loader : DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    ):

    model.eval()
    model.to(device)
    valid_loss = 0
    valid_acc = 0

    total_pred = []
    total_label = []
    total_shot = []
    total_dist = []
    total_t_warning = []
    
    dt = valid_loader.dataset.dt
    dist_warning = valid_loader.dataset.dist_warning
    dist_minimum = valid_loader.dataset.dist
    
    t_interval = dt * dist_warning
    t_minimum = dt * dist_minimum

    for batch_idx, data in enumerate(valid_loader):
        with torch.no_grad():
            optimizer.zero_grad()
            
            output = model(data)
            loss = loss_fn(output, data['label'].to(device))
    
            valid_loss += loss.item()
            pred = torch.nn.functional.sigmoid(output)
            pred = torch.where(pred > 0.5, 1, 0)
            
            total_pred.append(pred.view(-1,1))
            total_label.append(torch.where(data['label'] > 0, 1, 0).view(-1,1))
            total_shot.append(data['shot_num'].view(-1,1))
            total_dist.append(data['dist'].view(-1,1))
            total_t_warning.append(data['t_warning'].view(-1,1))

    total_pred = torch.concat(total_pred, dim = 0).detach().view(-1,).cpu().numpy()
    total_label = torch.concat(total_label, dim = 0).detach().view(-1,).cpu().numpy()
    
    total_dist = torch.concat(total_dist, dim = 0).detach().view(-1).cpu().numpy()
    total_shot = torch.concat(total_shot, dim = 0).detach().view(-1).cpu().numpy()
    total_t_warning = torch.concat(total_t_warning, dim = 0).detach().view(-1).cpu().numpy()
        
    indice = np.where((total_dist > total_t_warning + t_interval)|(total_dist<= t_minimum))[0]
    total_pred = total_pred[indice]
    total_label = total_label[indice]
    valid_acc = np.sum(np.where(total_pred == total_label,1,0)) / len(total_pred)

    valid_loss /= (batch_idx+1)
    valid_f1 = f1_score(total_label, total_pred, average = "macro")

    return valid_loss, valid_acc, valid_f1

def train(
    train_loader : DataLoader, 
    valid_loader : DataLoader,
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : Union[torch.nn.CrossEntropyLoss, LDAMLoss, FocalLoss],
    device : str = "cpu",
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best_dir : str = "./weights/best.pt",
    save_last_dir : str = "./weights/last.pt",
    exp_dir : Optional[str] = None,
    max_norm_grad : Optional[float] = None,
    test_for_check_per_epoch : Optional[DataLoader] = None,
    is_early_stopping : bool = False,
    early_stopping_verbose : bool = True,
    early_stopping_patience : int = 12,
    early_stopping_delta : float = 1e-3,
    scaler_type = None,
    dt : float = 0.01,
    ):

    train_loss_list = []
    valid_loss_list = []
    
    train_acc_list = []
    valid_acc_list = []

    train_f1_list = []
    valid_f1_list = []

    best_acc = 0
    best_epoch = 0
    best_f1 = 0
    best_loss = torch.inf
    
    
    # tensorboard
    if exp_dir is not None:
        if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)
        writer = SummaryWriter(exp_dir)
    else:
        writer = None
        
    if is_early_stopping:
        early_stopping = EarlyStopping(save_best_dir, early_stopping_patience, early_stopping_verbose, early_stopping_delta)
    else:
        early_stopping = None

    for epoch in tqdm(range(num_epoch), desc = "training process"):

        train_loss, train_acc, train_f1 = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            max_norm_grad,
        )

        valid_loss, valid_acc, valid_f1 = valid_per_epoch(
            valid_loader, 
            model,
            optimizer,
            loss_fn,
            device,
        )

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

        train_f1_list.append(train_f1)
        valid_f1_list.append(valid_f1)
        
        # tensorboard recording : loss and score
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/valid', valid_loss, epoch)
            
            writer.add_scalar('F1_score/train', train_f1, epoch)
            writer.add_scalar('F1_score/valid', valid_f1, epoch)
        
        if verbose:
            if epoch % verbose == 0:
                print("epoch : {}, train loss : {:.3f}, valid loss : {:.3f}, train acc : {:.3f}, valid acc : {:.3f}, train f1 : {:.3f}, valid f1 : {:.3f}".format(
                    epoch+1, train_loss, valid_loss, train_acc, valid_acc, train_f1, valid_f1
                ))
                
                if test_for_check_per_epoch and writer is not None:
                    
                    # General metrics for checking the precision and recall of the model
                    model.eval()
                    fig = evaluate_tensorboard(test_for_check_per_epoch, model, loss_fn, device, 0.5)
                    writer.add_figure('Model-performance', fig, epoch)
                     
                    # Checking the performance for continuous disruption prediciton 
                    _, _, fig_dis, _, fig_fi, _ = generate_model_performance(
                        filepath = config.filepath,
                        model = model, 
                        device = device,
                        save_dir = None,
                        shot_num = 30312,
                        seq_len_efit = test_for_check_per_epoch.dataset.seq_len_efit, 
                        seq_len_ece = test_for_check_per_epoch.dataset.seq_len_ece,
                        seq_len_diag = test_for_check_per_epoch.dataset.seq_len_diag, 
                        dist_warning=test_for_check_per_epoch.dataset.dist_warning,
                        dist = test_for_check_per_epoch.dataset.dist,
                        dt = dt,
                        mode = test_for_check_per_epoch.dataset.mode,
                        scaler_type=scaler_type,
                        is_plot_shot_info=False,
                        is_plot_uncertainty=False,
                        is_plot_feature_importance=False
                    )
                    
                    model.train()
                    
                    if fig_dis:
                        writer.add_figure('Continuous disruption prediction', fig_dis, epoch)
                    if fig_fi:
                        writer.add_figure('Feature importance', fig_fi, epoch)
                    
        # save the last parameters
        torch.save(model.state_dict(), save_last_dir)

        # save the best parameters
        if  best_f1 < valid_f1:
            best_acc = valid_acc
            best_f1 = valid_f1
            best_loss = valid_loss
            best_epoch  = epoch

        if early_stopping:
            early_stopping(valid_f1, model)
            if early_stopping.early_stop:
                print("Early stopping | epoch : {}, best f1 score : {:.3f}".format(epoch, best_f1))
                break
        else:
            torch.save(model.state_dict(), save_best_dir)

    print("training process finished, best loss : {:.3f} and best acc : {:.3f}, best f1 : {:.3f}, best epoch : {}".format(
        best_loss, best_acc, best_f1, best_epoch
    ))
    
    if writer:
        writer.close()

    return  train_loss_list, train_acc_list, train_f1_list,  valid_loss_list,  valid_acc_list, valid_f1_list