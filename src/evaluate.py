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

def evaluate(
    test_loader : DataLoader, 
    model : torch.nn.Module,
    optimizer : Optional[torch.optim.Optimizer],
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
    total_au = []
    total_eu = []

    if device is None:
        device = torch.device("cuda:0")

    model.to(device)
    model.eval()

    total_size = 0

    for idx, data in enumerate(tqdm(test_loader, 'evaluation process')):
        with torch.no_grad():
            optimizer.zero_grad()
        
            output = model(data)       
            loss = loss_fn(output, data['label'].to(device))
            test_loss += loss.item()
            
            pred = torch.nn.functional.softmax(output, dim = 1)[:,0]
            pred = torch.logical_not((pred > torch.FloatTensor([threshold]).to(device)))
            test_acc += pred.eq(data['label'].to(device).view_as(pred)).sum().item()
            total_size += pred.size(0)
            
            pred_normal = torch.nn.functional.softmax(output, dim = 1)[:,1].detach()
            
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
                
            total_pred.append(pred_normal.view(-1,1))
            total_label.append(data['label'].view(-1,1))
            
    test_loss /= (idx + 1)
    test_acc /= total_size
    
    total_pred = torch.concat(total_pred, dim = 0).detach().view(-1,).cpu().numpy()
    total_label = torch.concat(total_label, dim = 0).detach().view(-1,).cpu().numpy()
    
    total_au = np.array(total_au).reshape(-1,2)
    total_eu = np.array(total_eu).reshape(-1,2)
    
    # method 2 : compute f1, auc, roc and classification report
    # data clipping / postprocessing for ignoring nan, inf, too large data
    total_pred = np.nan_to_num(total_pred, copy = True, nan = 0, posinf = 1.0, neginf = 0)
    lr_probs = total_pred
    
    if not use_uncertainty:
        total_pred = np.where(total_pred > 1 - threshold, 1, 0)
    else:
        total_pred = np.where((total_pred > 1 - threshold) | ((total_pred < 1 - threshold) & (abs(2 * total_pred - 1) < 0.6) & (total_au[:,0] > 0.1)), 1, 0)
    
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
    conf_mat = confusion_matrix(total_label, total_pred)
    s = sns.heatmap(
        conf_mat, # conf_mat / np.sum(conf_mat),
        annot = True,
        fmt ='04d' ,# fmt = '.2f',
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
    optimizer : Optional[torch.optim.Optimizer],
    loss_fn : Optional[torch.nn.Module]= None,
    device : Optional[str] = "cpu",
    threshold : float = 0.5,
    ):

    test_loss = 0
    total_pred = []
    total_label = []

    if device is None:
        device = torch.device("cuda:0")

    model.to(device)
    model.eval()

    total_size = 0

    for idx, data in enumerate(test_loader):
        with torch.no_grad():
            optimizer.zero_grad()
            output = model(data)
                
            loss = loss_fn(output, data['label'].to(device))
            test_loss += loss.item()
            
            pred = torch.nn.functional.softmax(output, dim = 1)[:,1].detach()
            total_size += pred.size(0)
            
            total_pred.append(pred.view(-1,1))
            total_label.append(data['label'].view(-1,1))

    total_pred = torch.concat(total_pred, dim = 0).detach().view(-1,).cpu().numpy()
    total_label = torch.concat(total_label, dim = 0).detach().view(-1,).cpu().numpy()
    
    test_loss /= (idx + 1)
    
    # data clipping / postprocessing for ignoring nan, inf, too large data
    total_pred = np.nan_to_num(total_pred, copy = True, nan = 0, posinf = 1.0, neginf = 0)
    
    lr_probs = total_pred
    total_pred = np.where(total_pred > 1 - threshold, 1, 0)
    
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
            pred = torch.nn.functional.softmax(output, dim = 1)[:,0]
            
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
            pred = torch.nn.functional.softmax(output, dim = 1)[:,0]
            
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
            pred = torch.nn.functional.softmax(output, dim = 1)[:,0]
            
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
    df['pred'] = np.where(total_pred < 0.5, 1, 0)
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