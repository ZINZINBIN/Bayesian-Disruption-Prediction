import numpy as np
import pandas as pd
import math
import torch

def compute_tau_e(df:pd.DataFrame):
    
    def _pow(x, p : float):
        return math.pow(x, p)
    
    Meff = 1.5
    
    tau_H_factors = {
        '\\ipmhd':0.85,
        '\\bcentr':0.91,
        '\\ne_inter01':0.17,
        '\\aminor':0.19 * (-1),
        '\\rsurf':2.3,
        '\\kappa':0.7,
        '\\PL':0.55 * (-1)
    }
    
    tau_L_factors = {
        '\\ipmhd':0.85,
        '\\bcentr':0.2,
        '\\ne_inter01':0.17,
        '\\aminor':0.3,
        '\\rsurf':1.2,
        '\\kappa':0.5,
        '\\PL':0.5 * (-1)
    }
    
    df['\\tau_H_92'] = 0.021 * _pow(Meff, 0.5)
    df['\\tau_L_89'] = 0.048 * _pow(Meff, 0.5)
    
    for key, value in tau_H_factors:
        df['\\tau_H_92'] *= _pow(df[key], value)
    
    for key, value in tau_L_factors:
        df['\\tau_L_89'] *= _pow(df[key], value)
    
def compute_GreenWald_density(df : pd.DataFrame):
    df['\\nG'] = df['\\ipmhd'].abs() / math.pi / df['\\aminor'] ** 2
      
def compute_Troyon_beta(df : pd.DataFrame):
    df['\\troyon'] = df['\\betan'] * df['\\ipmhd'].abs() / df['\\aminor'] / df['\\bcentr'] 