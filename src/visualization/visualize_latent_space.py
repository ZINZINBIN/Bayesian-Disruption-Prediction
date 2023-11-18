import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
from scipy.interpolate import SmoothBivariateSpline
from typing import Literal

def visualize_2D_latent_space(model : nn.Module, dataloader : DataLoader, device : str = 'cpu', save_dir : str = './results/latent_2d_space.png', n_points_per_shot : int = 4, method : Literal['PCA', 'tSNE'] = 'tSNE'):
    model.to(device)
    model.eval()
    
    total_label = np.array([])
    total_latent = []
    total_dist = []
    total_shot = []
    total_t_warning = []
        
    for idx, data in enumerate(tqdm(dataloader, desc="visualize 2D latent space")):
        with torch.no_grad():
            latent = model.encode(data)   
            total_latent.append(latent.detach().cpu().numpy())
            total_label = np.concatenate((total_label, data['label'].detach().cpu().numpy().reshape(-1,)), axis = 0)
            
            total_shot.append(data['shot_num'].view(-1,1))
            total_dist.append(data['dist'].view(-1,1))
            total_t_warning.append(data['t_warning'].view(-1,1))
    
    total_latent = np.concatenate(total_latent, axis = 0)
    total_label = total_label.astype(int)
    total_dist = torch.concat(total_dist, dim = 0).detach().view(-1,).cpu().numpy()
    total_shot = torch.concat(total_shot, dim = 0).detach().view(-1,).cpu().numpy()
    t_warning = torch.concat(total_t_warning, dim = 0).detach().view(-1).cpu().numpy()
    
    # filtering data: non-disruptive data and disruptive data after TQ - 40ms for each shot
    dt = dataloader.dataset.dt 
    dist = dataloader.dataset.dist
    dist_warning = dataloader.dataset.dist_warning
    
    t_interval = dt * dist_warning
    t_minimum = dt * dist

    total_shot = total_shot[np.where(((total_dist >= t_warning + t_interval) | (total_dist <= t_minimum)))[0]].astype(int)
    total_label = total_label[np.where(((total_dist >= t_warning + t_interval) | (total_dist <= t_minimum)))[0]]
    total_latent = total_latent[np.where(((total_dist >= t_warning + t_interval) | (total_dist <= t_minimum)))[0]]
    
    new_indices = []
    
    for shot in np.unique(total_shot):
        idx_disrupt = np.where(((total_shot == shot)&(total_label == 1)))[0][0:n_points_per_shot].tolist()
        idx_normal = np.where(((total_shot == shot)&(total_label == 0)))[0][0:n_points_per_shot].tolist()
        
        new_indices.extend(idx_disrupt)
        new_indices.extend(idx_normal)
    
    new_indices = np.array(new_indices)
    total_shot = total_shot[new_indices]
    total_label = total_label[new_indices]
    total_latent = total_latent[new_indices]
    
    color = np.array(['#1f77b4', '#ff7f0e'])
    label  = np.array(['disruption','normal'])
    
    print("Dimension reduction process : start | latent vector : ({}, {})".format(total_latent.shape[0], total_latent.shape[1]))
    if method == 'PCA':
        # using PCA
        pca = IncrementalPCA(n_components=2)
        total_latent = pca.fit_transform(total_latent)
    else:
        # using t-SNE
        tSNE = TSNE(n_components=2, perplexity = 128)
        total_latent = tSNE.fit_transform(total_latent)   
        
    print("Dimension reduction process : complete")
    
    dis_idx = np.where(total_label == 1)
    normal_idx = np.where(total_label == 0)
    
    plt.figure(figsize = (8,6))
    plt.scatter(total_latent[normal_idx,0], total_latent[normal_idx,1], c = color[1], label = label[1])
    plt.scatter(total_latent[dis_idx,0], total_latent[dis_idx,1], c = color[0], label = label[0])
    plt.xlabel('z-0')
    plt.ylabel('z-1')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir)
    
    return
    
# revised : visualize 2D latent space and decision boundary at once
def visualize_2D_decision_boundary(model : nn.Module, dataloader : DataLoader, device : str = 'cpu', save_dir : str = './results/decision_boundary_2D_space.png', n_points_per_shot : int = 4, method : Literal['PCA', 'tSNE'] = 'tSNE'):
    model.to(device)
    model.eval()
    
    total_label = np.array([])
    total_t_warning = []
    total_probs = []
    total_latent = []
    total_dist = []
    total_shot = []
        
    for idx, data in enumerate(tqdm(dataloader, desc="visualize 2D latent space with decision boundary")):
        with torch.no_grad():
            
            latent = model.encode(data)
            probs = model(data)
  
            probs = torch.nn.functional.sigmoid(probs)
            probs = probs.cpu().detach().numpy().tolist()
            
            total_latent.append(latent.detach().cpu().numpy())
            total_label = np.concatenate((total_label, data['label'].detach().cpu().numpy().reshape(-1,)), axis = 0)
            total_probs.extend(probs)
            
            total_shot.append(data['shot_num'].view(-1,1))
            total_dist.append(data['dist'].view(-1,1))
            total_t_warning.append(data['t_warning'].view(-1,1))
    
    total_latent = np.concatenate(total_latent, axis = 0)
    total_label = total_label.astype(int)
    total_probs = np.array(total_probs)
    
    total_dist = torch.concat(total_dist, dim = 0).detach().view(-1,).cpu().numpy()
    total_shot = torch.concat(total_shot, dim = 0).detach().view(-1,).cpu().numpy()
    t_warning = torch.concat(total_t_warning, dim = 0).detach().view(-1).cpu().numpy()
    
    # filtering data: non-disruptive data and disruptive data after TQ - 40ms for each shot
    dt = dataloader.dataset.dt 
    dist = dataloader.dataset.dist
    dist_warning = dataloader.dataset.dist_warning
    
    t_interval = dt * dist_warning
    t_minimum = dt * dist

    total_shot = total_shot[np.where(((total_dist >= t_warning + t_interval) | (total_dist <= t_minimum)))[0]].astype(int)
    total_label = total_label[np.where(((total_dist >= t_warning + t_interval) | (total_dist <= t_minimum)))[0]]
    total_latent = total_latent[np.where(((total_dist >= t_warning + t_interval) | (total_dist <= t_minimum)))[0]]
    total_probs = total_probs[np.where(((total_dist >= t_warning + t_interval) | (total_dist <= t_minimum)))[0]]
    
    new_indices = []
    
    for shot in np.unique(total_shot):
        idx_disrupt = np.where(((total_shot == shot)&(total_label == 1)))[0][0:n_points_per_shot].tolist()
        idx_normal = np.where(((total_shot == shot)&(total_label == 0)))[0][0:n_points_per_shot].tolist()
        
        new_indices.extend(idx_disrupt)
        new_indices.extend(idx_normal)
    
    new_indices = np.array(new_indices)
    total_shot = total_shot[new_indices]
    total_label = total_label[new_indices]
    total_latent = total_latent[new_indices]
    total_probs = total_probs[new_indices]
    
    color = np.array(['#1f77b4', '#ff7f0e'])
    label  = np.array(['disruption','normal'])
    
    print("Dimension reduction process : start | latent vector : ({}, {})".format(total_latent.shape[0], total_latent.shape[1]))
    if method == 'PCA':
        # using PCA
        pca = IncrementalPCA(n_components=2)
        total_latent = pca.fit_transform(total_latent)
    else:
        # using t-SNE
        tSNE = TSNE(n_components=2, perplexity = 128)
        total_latent = tSNE.fit_transform(total_latent)   
    print("Dimension reduction process : complete")
    
    # meshgrid
    latent_x, latent_y = total_latent[:,0], total_latent[:,1]
    latent_x, latent_y = np.meshgrid(latent_x, latent_y)
    
    interpolate_fn = SmoothBivariateSpline(latent_x[0,:], latent_y[:,0], total_probs)
    probs_z = interpolate_fn(latent_x, latent_y, grid = False)
    probs_z = np.clip(probs_z, 0, 1)
    
    dis_idx = np.where(total_label == 1)
    normal_idx = np.where(total_label == 0)
    
    level = np.linspace(0,1.0,8)
    plt.figure(figsize = (8,6))
    plt.contourf(latent_x, latent_y, probs_z, level = level, cmap = plt.cm.coolwarm)
    
    mp = plt.cm.ScalarMappable(cmap = plt.cm.coolwarm)
    mp.set_array(probs_z)
    mp.set_clim(0, 1.0)
    
    plt.colorbar(mp, boundaries = np.linspace(0,1,5))
    plt.scatter(total_latent[normal_idx,0], total_latent[normal_idx,1], c = color[1], label = label[1])
    plt.scatter(total_latent[dis_idx,0], total_latent[dis_idx,1], c = color[0], label = label[0])
    plt.xlabel('z-0')
    plt.ylabel('z-1')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir)
    
    return