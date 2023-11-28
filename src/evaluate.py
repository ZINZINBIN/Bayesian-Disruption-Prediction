import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Optional, Literal
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from src.models.BNN import compute_uncertainty_per_data
from tqdm.auto import tqdm

def evaluate_prediction_performance(
    test_loader : DataLoader, 
    model : torch.nn.Module,
    device : Optional[str] = "cpu",
    save_log : Optional[str] = "./results/shot_log.csv",
    save_txt : Optional[str] = None,
    threshold : float = 0.5,
    t_warning : float = 0.4,
    t_minimum : float = 0.04
    ):
    
    total_pred = []
    total_label = []
    total_shot = []
    total_dist = []
    total_t_warning = []

    if device is None:
        device = torch.device("cuda:0")

    model.to(device)
    model.eval()

    total_size = 0

    for idx, data in enumerate(tqdm(test_loader, 'evaluation process: prediction performance')):
        with torch.no_grad():
            output = model(data)       
            probs = torch.nn.functional.sigmoid(output)
            total_pred.append(probs.view(-1,1))
            total_label.append(data['label'].view(-1,1))
            
            total_shot.append(data['shot_num'].view(-1,1))
            total_dist.append(data['dist'].view(-1,1))
            total_t_warning.append(data['t_warning'].view(-1,1))
            
    total_pred = torch.concat(total_pred, dim = 0).detach().view(-1,).cpu().numpy()
    total_label = torch.concat(total_label, dim = 0).detach().view(-1,).cpu().numpy()
    
    total_dist = torch.concat(total_dist, dim = 0).detach().view(-1,).cpu().numpy()
    total_shot = torch.concat(total_shot, dim = 0).detach().view(-1,).cpu().numpy()
    total_t_warning = torch.concat(total_t_warning, dim = 0).detach().view(-1).cpu().numpy()
    
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    
    log_missing = []
    log_true = []
    log_false = []
    log_warning_time = []
    
    for shot in np.unique(total_shot):
        indice = np.where(total_shot == shot)[0]
        
        # shot filtering
        pred = total_pred[indice]
        label = total_label[indice]
        dist = total_dist[indice]
        t_warning = total_t_warning[indice]

        # compute false positive case
        alarm = pred[np.where(dist > t_warning)[0]]
        alarm = np.where(alarm > threshold, 1, 0)
        n_alarm = len(alarm)
        
        non_disrupt = label[np.where(dist > t_warning)[0]]
        rate = np.sum(np.where(alarm == non_disrupt, 1, 0)) / n_alarm
        
        if rate > 0.9:
            TN += 1
            log_false.append(0)
        else:
            FP += 1
            log_false.append(1)
        
        # compute true positive case
        alarm = pred[np.where(dist < t_minimum)[0]]
        alarm = np.where(alarm > threshold, 1, 0)
        n_alarm = len(alarm)
        
        disrupt = label[np.where(dist < t_minimum)[0]]
        rate = np.sum(np.where(alarm == disrupt, 1, 0)) / n_alarm
     
        if rate > 0.9:
            TP += 1
            log_true.append(1)
            log_missing.append(0)
        else:
            FN += 1
            log_true.append(0)
            log_missing.append(1)
            
        # Alarm time vs accumulated detection rate
        indice_sort = np.argsort(dist)
        alarm = pred[indice_sort]
        alarm = np.where(alarm > threshold, 1, 0)
        dist_temp = []
        
        for idx in range(len(alarm)):
            
            rate = alarm[0:idx].sum() / (idx+1)
            
            if rate > 0.95:
                dist_temp.append(dist[indice_sort][idx])
            
        t_warning_strt = max(dist_temp) if len(dist_temp) > 0 else 0
        log_warning_time.append(t_warning_strt)
        
    TPR = TP / (TP+FN)
    FPR = FP / (FP+TN)
    
    PR = TP / (TP+FP)
    RC = TP / (TP+FN)
    ACC = (TP+TN) / (TP+FN+FP+TN)
    
    F1 = 2 * PR * RC / (PR + RC)
    
    shot_log = {
        "shot":np.unique(total_shot),
        "True":log_true,
        "Missing":log_missing,
        "False":log_false,
        "Warning":log_warning_time
    }
    
    shot_log = pd.DataFrame(shot_log)
    shot_log['shot'] = shot_log['shot'].astype(int)
    
    print("(Evaluation for shots)| TPR: {:.3f} | FPR:{:.3f} | Precision:{:.3f} | Recall:{:.3f} | F1 score:{:.3f} | Accuracy:{:.3f}".format(TPR,FPR,PR,RC,F1,ACC))
    
    if save_txt:
        with open(save_txt, 'w') as f:
            header = "="*20 +" Evaluation report for test shot "+ "="*20
            f.write(header)
            f.write("\nTPR: {:.3f} | FPR:{:.3f} | Precision:{:.3f} | Recall:{:.3f} | F1 score:{:.3f} | Accuracy:{:.3f}".format(TPR,FPR,PR,RC,F1,ACC))
            f.write("\n Total shot : {}".format(len(np.unique(total_shot))))
            f.write("\n True alarm : {}".format(TP))
            f.write("\n False alarm : {}".format(FP))
            f.write("\n Missing alarm : {}".format(FN))
    
    if save_log:
        shot_log.to_csv(save_log)
        
    return

def evaluate(
    test_loader : DataLoader, 
    model : torch.nn.Module,
    loss_fn : Optional[torch.nn.Module]= None,
    device : Optional[str] = "cpu",
    save_conf : Optional[str] = "./results/confusion_matrix.png",
    save_txt : Optional[str] = None,
    threshold : float = 0.5,
    use_uncertainty : bool = False,
    ):

    test_loss = 0
    test_acc = 0
    test_f1 = 0
    total_pred = []
    total_label = []
    total_t_warning = []
    total_au = []
    total_eu = []
    
    total_dist = []
    total_shot = []

    if device is None:
        device = torch.device("cuda:0")

    model.to(device)
    model.eval()

    for idx, data in enumerate(tqdm(test_loader, 'evaluation process')):
        with torch.no_grad():
            output = model(data)       
            loss = loss_fn(output, data['label'].to(device))
            test_loss += loss.item()
            
            pred = torch.nn.functional.sigmoid(output)
            
            total_shot.append(data['shot_num'].view(-1,1))
            total_dist.append(data['dist'].view(-1,1))
            total_t_warning.append(data['t_warning'].view(-1,1))
            
            if use_uncertainty:
                
                batch_size = data['label'].size()[0]
                aus = []
                eus = []
                
                for idx in range(batch_size):
                    input_data = {}
                    for key in data.keys():
                        input_data[key] = data[key][idx]
                    au, eu = compute_uncertainty_per_data(model, input_data, device, n_samples = 16)
                    aus.append(au)
                    eus.append(eu)
                        
                total_au.extend(aus)
                total_eu.extend(eus)
                
            total_pred.append(pred.view(-1,1))
            total_label.append(data['label'].view(-1,1))
            
    test_loss /= (idx + 1)
    
    total_pred = torch.concat(total_pred, dim = 0).detach().view(-1,).cpu().numpy()
    total_label = torch.concat(total_label, dim = 0).detach().view(-1,).cpu().numpy()
    
    total_au = np.array(total_au).reshape(-1,)
    total_eu = np.array(total_eu).reshape(-1,)
    
    total_dist = torch.concat(total_dist, dim = 0).detach().view(-1,).cpu().numpy()
    total_shot = torch.concat(total_shot, dim = 0).detach().view(-1,).cpu().numpy()
    t_warning = torch.concat(total_t_warning, dim = 0).detach().view(-1).cpu().numpy()
    
    # filtering data: non-disruptive data and disruptive data after TQ - 40ms for each shot
    dt = test_loader.dataset.dt 
    dist = test_loader.dataset.dist
    dist_warning = test_loader.dataset.dist_warning
    t_interval = dt * dist_warning
    t_minimum = dt * dist

    total_shot = total_shot[np.where(((total_dist >= t_warning + t_interval) | (total_dist <= t_minimum)))[0]]
    total_label = total_label[np.where(((total_dist >= t_warning + t_interval) | (total_dist <= t_minimum)))[0]]
    total_pred = total_pred[np.where(((total_dist >= t_warning + t_interval) | (total_dist <= t_minimum)))[0]]
    
    test_acc = np.sum(np.where(total_pred > threshold, 1, 0) == total_label) / len(total_label)
    
    # method 2 : compute f1, auc, roc and classification report
    # data clipping / postprocessing for ignoring nan, inf, too large data
    total_pred = np.nan_to_num(total_pred, copy = True, nan = 0, posinf = 1.0, neginf = 0)
    lr_probs = total_pred
    
    if not use_uncertainty:
        total_pred = np.where(total_pred > threshold, 1, 0)
    else:
        total_pred = np.where(total_pred > threshold, 1, 0)
    
    # f1 score
    test_f1 = f1_score(total_label, total_pred, average = "macro")
    
    # auc score
    test_auc = roc_auc_score(total_label, total_pred, average='macro')
    
    # roc score
    ns_probs = [0 for _ in range(len(total_label))]
    ns_auc = roc_auc_score(total_label, ns_probs, average="macro")
    lr_auc = roc_auc_score(total_label, lr_probs, average = "macro")
    
    fig, axes = plt.subplots(2,2, sharex = False, figsize = (15, 10))
    
    # confusion matrix
    conf_mat = confusion_matrix(total_label, total_pred, normalize = 'true')
    s = sns.heatmap(
        # conf_mat,
        conf_mat,
        annot = True,
        # fmt ='04d' ,
        fmt = '.2f',
        cmap = 'Blues',
        xticklabels=["disruption","normal"],
        yticklabels=["disruption","normal"],
        ax = axes[0,0]
    )

    s.set_xlabel("Prediction")
    s.set_ylabel("Actual")

    # roc curve
    ns_fpr, ns_tpr, _ = roc_curve(total_label, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(total_label, lr_probs)
    
    axes[0,1].plot(ns_fpr, ns_tpr, linestyle = '--', label = 'Random')
    axes[0,1].plot(lr_fpr, lr_tpr, marker = '.', label = 'Model')
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    
    lr_precision, lr_recall, _ = precision_recall_curve(total_label, lr_probs)
    axes[1,0].plot(lr_recall, lr_precision, marker = '.', label = 'Model')
    axes[1,0].set_xlabel("Recall")
    axes[1,0].set_ylabel("Precision")
    
    clf_report = classification_report(total_label, total_pred, labels = [0,1], target_names = ["Disrupt", "Normal"], output_dict = True)
    s2 = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot = True, ax = axes[1,1])
    fig.tight_layout()
    
    if save_conf:
        plt.savefig(save_conf)

    print("############### Classification Report ####################")
    print(classification_report(total_label, total_pred, labels = [0,1]))
    print("\n# test acc : {:.2f}, test f1 : {:.2f}, test AUC : {:.2f}, test loss : {:.3f}".format(test_acc, test_f1, test_auc, test_loss))

    if save_txt:
        with open(save_txt, 'w') as f:
            f.write(classification_report(total_label, total_pred, labels = [0,1]))
            summary = "\n# test score : {:.2f}, test loss : {:.3f}, test f1 : {:.3f}, test_auc : {:.3f}".format(test_acc, test_loss, test_f1, test_auc)
            f.write(summary)
    
    return test_loss, test_acc, test_f1

def evaluate_tensorboard(
    test_loader : DataLoader, 
    model : torch.nn.Module,
    loss_fn : Optional[torch.nn.Module]= None,
    device : Optional[str] = "cpu",
    threshold : float = 0.5,
    ):

    test_loss = 0
    total_pred = []
    total_label = []
    total_t_warning = []
    total_dist = []
    total_shot = []

    if device is None:
        device = torch.device("cuda:0")

    model.to(device)
    model.eval()

    total_size = 0

    for idx, data in enumerate(test_loader):
        with torch.no_grad():
            output = model(data)
                
            loss = loss_fn(output, data['label'].to(device))
            test_loss += loss.item()
            
            pred = torch.nn.functional.sigmoid(output)
            total_size += pred.size(0)
            
            total_shot.append(data['shot_num'].view(-1,1))
            total_dist.append(data['dist'].view(-1,1))
            total_pred.append(pred.view(-1,1))
            total_label.append(torch.where(data['label'] > 0, 1, 0).view(-1,1))
            total_t_warning.append(data['t_warning'].view(-1,1))

    total_pred = torch.concat(total_pred, dim = 0).detach().view(-1,).cpu().numpy()
    total_label = torch.concat(total_label, dim = 0).detach().view(-1,).cpu().numpy()
    total_dist = torch.concat(total_dist, dim = 0).detach().view(-1,).cpu().numpy()
    total_shot = torch.concat(total_shot, dim = 0).detach().view(-1,).cpu().numpy()
    t_warning = torch.concat(total_t_warning, dim = 0).detach().view(-1,).cpu().numpy()
    
    test_loss /= (idx + 1)
    
    # data clipping / postprocessing for ignoring nan, inf, too large data
    total_pred = np.nan_to_num(total_pred, copy = True, nan = 0, posinf = 1.0, neginf = 0)
    
    dt = test_loader.dataset.dt 
    dist = test_loader.dataset.dist
    dist_warning = test_loader.dataset.dist_warning
    t_interval = dt * dist_warning
    t_minimum = dt * dist
    
    total_shot = total_shot[np.where(((total_dist >= t_warning + t_interval) | (total_dist <= t_minimum)))[0]]
    total_label = total_label[np.where(((total_dist >= t_warning + t_interval) | (total_dist <= t_minimum)))[0]]
    total_pred = total_pred[np.where(((total_dist >= t_warning + t_interval) | (total_dist <= t_minimum)))[0]]
    
    lr_probs = total_pred
    total_pred = np.where(total_pred > threshold, 1, 0)
    
    # f1 score
    test_f1 = f1_score(total_label, total_pred, average = "macro")
    
    # roc score
    ns_probs = [0 for _ in range(len(total_label))]
    ns_auc = roc_auc_score(total_label, ns_probs, average="macro")
    lr_auc = roc_auc_score(total_label, lr_probs, average = "macro")
    
    fig, axes = plt.subplots(2,2, sharex = False, figsize = (15, 10))
    
    # confusion matrix
    conf_mat = confusion_matrix(total_label, total_pred)
    s = sns.heatmap(
        conf_mat, # conf_mat / np.sum(conf_mat),
        annot = True,
        fmt ='04d' ,# fmt = '.2f',
        cmap = 'Blues',
        xticklabels=["normal","disruption"],
        yticklabels=["normal","disruption"],
        ax = axes[0,0]
    )

    s.set_xlabel("Prediction")
    s.set_ylabel("Actual")

    # roc curve
    ns_fpr, ns_tpr, _ = roc_curve(total_label, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(total_label, lr_probs)
    
    axes[0,1].plot(ns_fpr, ns_tpr, linestyle = '--', label = 'Random')
    axes[0,1].plot(lr_fpr, lr_tpr, marker = '.', label = 'Model')
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    
    lr_precision, lr_recall, _ = precision_recall_curve(total_label, lr_probs)
    axes[1,0].plot(lr_recall, lr_precision, marker = '.', label = 'Model')
    axes[1,0].set_xlabel("Recall")
    axes[1,0].set_ylabel("Precision")
    
    clf_report = classification_report(total_label, total_pred, labels = [0,1], target_names = ["Normal", "Disruption"], output_dict = True)
    s2 = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot = True, ax = axes[1,1])
    fig.tight_layout()
    
    return fig

def evaluate_detail(
    train_loader : DataLoader,
    valid_loader : DataLoader,
    test_loader : DataLoader, 
    model : torch.nn.Module,
    device : Optional[str] = "cpu",
    save_csv : Optional[str] = None,
    tag : Optional[str] = None
    ):
    
    train_loader.dataset.eval()
    valid_loader.dataset.eval()
    test_loader.dataset.eval()

    total_shot = np.array([])
    total_pred = np.array([])
    total_label = np.array([])
    total_dist = np.array([])
    total_task = []

    if device is None and torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif device is None and not torch.cuda.is_available():
        device = 'cpu'

    model.to(device)
    model.eval()
    
    # evaluation for train dataset
    for idx, data in enumerate(tqdm(train_loader, 'evaluation process for train data')):
        with torch.no_grad():
            output = model(data)
            pred = torch.nn.functional.sigmoid(output)
            
            total_dist = np.concatenate((total_dist, data['dist'].cpu().numpy().reshape(-1,)))
            total_shot = np.concatenate((total_shot, data['shot_num'].cpu().numpy().reshape(-1,)))
            total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
            total_label = np.concatenate((total_label, data['label'].cpu().numpy().reshape(-1,)))
            
    total_task.extend(["train" for _ in range(train_loader.dataset.__len__())])
            
    model.eval()
    
    # evaluation for valid dataset
    for idx, data in enumerate(tqdm(valid_loader, 'evaluation process for valid data')):
        with torch.no_grad():
            output = model(data)
            pred = torch.nn.functional.sigmoid(output)
            
            total_dist = np.concatenate((total_dist, data['dist'].cpu().numpy().reshape(-1,)))
            total_shot = np.concatenate((total_shot, data['shot_num'].cpu().numpy().reshape(-1,)))
            total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
            total_label = np.concatenate((total_label, data['label'].cpu().numpy().reshape(-1,)))
            
    total_task.extend(["valid" for _ in range(valid_loader.dataset.__len__())])
            
    model.eval()
    # evaluation for test dataset
    for idx, data in enumerate(tqdm(test_loader, 'evaluation process for test data')):
        with torch.no_grad():
            output = model(data)    
            pred = torch.nn.functional.sigmoid(output)
            
            total_dist = np.concatenate((total_dist, data['dist'].cpu().numpy().reshape(-1,)))
            total_shot = np.concatenate((total_shot, data['shot_num'].cpu().numpy().reshape(-1,)))
            total_pred = np.concatenate((total_pred, pred.cpu().numpy().reshape(-1,)))
            total_label = np.concatenate((total_label, data['label'].cpu().numpy().reshape(-1,)))

    total_task.extend(["test" for _ in range(test_loader.dataset.__len__())])
    df = pd.DataFrame({})
    
    df['task'] = total_task
    df['label'] = total_label.astype(int)
    df['shot'] = total_shot.astype(int)
    df['prob'] = total_pred
    df['pred'] = np.where(total_pred > 0.5, 1, 0)
    df['tag'] = [tag for _ in range(len(total_pred))]
    df['success'] = df['label'] == df['pred']
    df['dist'] = total_dist
    
    df = df[df.dist >= 0.005]
    
    df.to_csv(save_csv, index = False)
    
    return

def plot_roc_curve(y_true : np.ndarray, y_pred : np.ndarray, save_dir : str, title : Optional[str] = None):
    auc = roc_auc_score(y_true, y_pred, average='macro')
    fpr, tpr, threshold = roc_curve(y_true, y_pred)

    lw = 2
    plt.figure()
    plt.plot(fpr, tpr, color = "darkorange", lw = lw, label = "ROC curve (area : {:.2f}".format(auc))
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    
    if title is not None:
        plt.title(title)
    else:
        plt.title("Receiver operating characteristic")
        
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(save_dir)