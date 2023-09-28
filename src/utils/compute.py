import numpy as np
import pandas as pd
import math
import torch

def compute_tau_e(df:pd.DataFrame):
    
    def _pow(x, p : float):
        return math.pow(x, p)
    
    tau_H_factors = {
        '\\ipmhd':0.93,
        '\\bcentr':0.15,
        '\\ne_inter01':0.41,
        '\\aminor':0.58,
        '\\rsurf':1.39,
        '\\kappa':0.78,
        '\\Pin':0.69 * (-1)
    }
    
    tau_L_factors = {
        '\\ipmhd':0.85,
        '\\bcentr':0.2,
        '\\ne_inter01':0.1,
        '\\aminor':0.3,
        '\\rsurf':1.2,
        '\\kappa':0.5,
        '\\Pin':0.5 * (-1)
    }
    
    df['\\tau_H_98'] = 0.056
    df['\\tau_L_89'] = 0.048
    
    for key, value in tau_H_factors:
        df['\\tau_H_98'] *= _pow(df[key], value)
    
    for key, value in tau_L_factors:
        df['\\tau_L_89'] *= _pow(df[key], value)
    
def compute_GreenWald_density(df : pd.DataFrame):
    df['\\nG'] = df['\\ipmhd'].abs() / math.pi / df['\\aminor'] ** 2
    # df['\\nG_tci01'] = df['\\ne_tci01'] / df['\\nG'] * 0.1
    # df['\\nG_tci02'] = df['\\ne_tci02'] / df['\\nG'] * 0.1
    # df['\\nG_tci03'] = df['\\ne_tci03'] / df['\\nG'] * 0.1
    # df['\\nG_tci04'] = df['\\ne_tci04'] / df['\\nG'] * 0.1
    # df['\\nG_tci05'] = df['\\ne_tci05'] / df['\\nG'] * 0.1