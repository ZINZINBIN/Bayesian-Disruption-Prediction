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

def visualize_2D_latent_space(model : nn.Module, dataloader : DataLoader, device : str = 'cpu', save_dir : str = './results/latent_2d_space.png', limit_iters : int = 2, method : Literal['PCA', 'tSNE'] = 'tSNE'):
    model.to(device)
    model.eval()
    
    total_label = np.array([])
    total_latent = []
        
    for idx, data in enumerate(tqdm(dataloader, desc="visualize 2D latent space")):
        with torch.no_grad():
            latent = model.encode(data)   
            total_latent.append(latent.detach().cpu().numpy())
            total_label = np.concatenate((total_label, data['label'].detach().cpu().numpy().reshape(-1,)), axis = 0)
            
        if limit_iters >0 and idx + 1 > limit_iters:
            break
    
    total_latent = np.concatenate(total_latent, axis = 0)
    total_label = total_label.astype(int)
    
    color = np.array(['#1f77b4', '#ff7f0e'])
    label  = np.array(['disruption','normal'])
    
    print("Dimension reduction process : start | latent vector : ({}, {})".format(total_latent.shape[0], total_latent.shape[1]))
    if method == 'PCA':
        # using PCA
        pca = IncrementalPCA(n_components=2)
        total_latent = pca.fit_transform(total_latent)
    else:
        # using t-SNE
        tSNE = TSNE(n_components=2, perplexity = 64)
        total_latent = tSNE.fit_transform(total_latent)   
    print("Dimension reduction process : complete")
    
    dis_idx = np.where(total_label == 0)
    normal_idx = np.where(total_label == 1)
    
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
def visualize_2D_decision_boundary(model : nn.Module, dataloader : DataLoader, device : str = 'cpu', save_dir : str = './results/decision_boundary_2D_space.png', limit_iters : int = 2, method : Literal['PCA', 'tSNE'] = 'tSNE'):
    model.to(device)
    model.eval()
    
    total_label = np.array([])
    total_probs = []
    total_latent = []
        
    for idx, data in enumerate(tqdm(dataloader, desc="visualize 2D latent space with decision boundary")):
        with torch.no_grad():
            
            latent = model.encode(data)
            probs = model(data)
  
            probs = torch.nn.functional.softmax(probs, dim = 1)[:,0]
            probs = probs.cpu().detach().numpy().tolist()
            
            total_latent.append(latent.detach().cpu().numpy())
            total_label = np.concatenate((total_label, data['label'].detach().cpu().numpy().reshape(-1,)), axis = 0)
            total_probs.extend(probs)
            
        if limit_iters >0 and idx + 1 > limit_iters:
            break
    
    total_latent = np.concatenate(total_latent, axis = 0)
    total_label = total_label.astype(int)
    
    color = np.array(['#1f77b4', '#ff7f0e'])
    label  = np.array(['disruption','normal'])
    
    print("Dimension reduction process : start | latent vector : ({}, {})".format(total_latent.shape[0], total_latent.shape[1]))
    if method == 'PCA':
        # using PCA
        pca = IncrementalPCA(n_components=2)
        total_latent = pca.fit_transform(total_latent)
    else:
        # using t-SNE
        tSNE = TSNE(n_components=2, perplexity = 64)
        total_latent = tSNE.fit_transform(total_latent)   
    print("Dimension reduction process : complete")
    
    # meshgrid
    latent_x, latent_y = total_latent[:,0], total_latent[:,1]
    latent_x, latent_y = np.meshgrid(latent_x, latent_y)
    
    interpolate_fn = SmoothBivariateSpline(latent_x[0,:], latent_y[:,0], np.array(total_probs))
    probs_z = interpolate_fn(latent_x, latent_y, grid = False)
    probs_z = np.clip(probs_z, 0, 1)
    
    dis_idx = np.where(total_label == 0)
    normal_idx = np.where(total_label == 1)
    
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
    
def visualize_3D_latent_space(model : nn.Module, dataloader : DataLoader, device : str = 'cpu', save_dir : str = './results/latent_2d_space.png', limit_iters : int = 8, method : Literal['PCA', 'tSNE'] = 'tSNE'):
    model.to(device)
    model.eval()
    
    total_label = np.array([])
    total_latent = []
        
    for idx, data in enumerate(tqdm(dataloader, desc = "visualize 3D latent space")):
        with torch.no_grad():
            latent = model.encode(data)
            total_latent.append(latent.detach().cpu().numpy())
            total_label = np.concatenate((total_label, data['label'].detach().cpu().numpy().reshape(-1,)), axis = 0)
            
        if limit_iters > 0 and idx + 1 > limit_iters:
            break
    
    total_latent = np.concatenate(total_latent, axis = 0)
    total_label = total_label.astype(int)
    
    color = np.array(['#1f77b4', '#ff7f0e'])
    label  = np.array(['disruption','normal'])
    
    print("Dimension reduction process : start | latent vector : ({}, {})".format(total_latent.shape[0], total_latent.shape[1]))
    if method == 'PCA':
        # using PCA
        pca = IncrementalPCA(n_components=3)
        total_latent = pca.fit_transform(total_latent)
    else:
        # using t-SNE
        tSNE = TSNE(n_components=3, perplexity = 64)
        total_latent = tSNE.fit_transform(total_latent)   
    print("Dimension reduction process : complete")
    
    dis_idx = np.where(total_label == 0)
    normal_idx = np.where(total_label == 1)
    
    fig = plt.figure(figsize = (8,6))
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(total_latent[normal_idx,0], total_latent[normal_idx,1], total_latent[normal_idx,2], c = color[1], label = label[1])
    ax.scatter(total_latent[dis_idx,0], total_latent[dis_idx,1], total_latent[dis_idx,2], c = color[0], label = label[0])
    ax.set_xlabel('z-0')
    ax.set_ylabel('z-1')
    ax.set_zlabel('z-2')
    ax.legend()
    fig.tight_layout()
    plt.savefig(save_dir)
    return