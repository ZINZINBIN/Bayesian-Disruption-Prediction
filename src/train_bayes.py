from typing import Optional, List, Literal, Union
from src.loss import LDAMLoss, FocalLoss
import os
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from src.evaluate import evaluate_tensorboard
from src.utils.EarlyStopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from src.models.BNN import BayesianModule, minibatch_weight
from src.utils.utility import generate_prob_curve_from_0D

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

    for batch_idx, (data, target) in enumerate(train_loader):
       
        optimizer.zero_grad()
        pi_weight = minibatch_weight(batch_idx, num_batches=len(train_loader))
        output = model(data.to(device))
        loss = model.elbo(
            inputs = data.to(device),
            targets = target.to(device),
            criterion = loss_fn,
            n_samples = 8,
            w_complexity = pi_weight
        )
        
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
        train_acc += pred.eq(target.to(device).view_as(pred)).sum().item()
        total_size += pred.size(0) 
        
        total_pred.append(pred.view(-1,1))
        total_label.append(target.view(-1,1))
        
    if scheduler:
        scheduler.step()
        
    total_pred = torch.concat(total_pred, dim = 0).detach().view(-1,).cpu().numpy()
    total_label = torch.concat(total_label, dim = 0).detach().view(-1,).cpu().numpy()

    if total_size > 0:
        train_loss /= total_size
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
    total_size = 0

    for batch_idx, (data, target) in enumerate(valid_loader):
        with torch.no_grad():
            
            optimizer.zero_grad()
            output = model(data.to(device))
            pi_weight = minibatch_weight(batch_idx, num_batches = len(valid_loader))
            loss = model.elbo(
                inputs = data.to(device),
                targets = target.to(device),
                criterion = loss_fn,
                n_samples = 8,
                w_complexity = pi_weight
            )
    
            valid_loss += loss.item()
            pred = torch.nn.functional.softmax(output, dim = 1).max(1, keepdim = True)[1]
            valid_acc += pred.eq(target.to(device).view_as(pred)).sum().item()
            total_size += pred.size(0)

            total_pred.append(pred.view(-1,1))
            total_label.append(target.view(-1,1))

    valid_loss /= total_size
    valid_acc /= total_size
    
    total_pred = torch.concat(total_pred, dim = 0).detach().view(-1,).cpu().numpy()
    total_label = torch.concat(total_label, dim = 0).detach().view(-1,).cpu().numpy()

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
    early_stopping_delta : float = 1e-3
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
                    fig = evaluate_tensorboard(test_for_check_per_epoch, model, optimizer, loss_fn, device, 0.5)
                    writer.add_figure('Model-performance', fig, epoch)
                    
                    # Checking the performance for continuous disruption prediciton 
                    fig, _, _ = generate_prob_curve_from_0D(
                        model = model, 
                        device = device, 
                        save_dir = None,
                        ts_data_dir = "./dataset/KSTAR_Disruption_ts_data_extend.csv",
                        ts_cols = test_for_check_per_epoch.dataset.cols,
                        shot_list_dir = './dataset/KSTAR_Disruption_Shot_List_2022.csv',
                        shot_num = 21310,
                        seq_len = test_for_check_per_epoch.dataset.seq_len,
                        dist = test_for_check_per_epoch.dataset.dist,
                        dt = test_for_check_per_epoch.dataset.dt,
                        scaler = test_for_check_per_epoch.dataset.scaler
                    )
                    model.train()
                    writer.add_figure('Continuous disruption prediction', fig, epoch)
                    
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