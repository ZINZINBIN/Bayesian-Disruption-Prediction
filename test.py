import torch
import os
import numpy as np
import pandas as pd
import argparse
from src.dataset import MultiSignalDataset
from torch.utils.data import DataLoader, RandomSampler
from src.utils.sampler import ImbalancedDatasetSampler
from src.utils.utility import preparing_0D_dataset, seed_everything
from src.config import Config

config = Config()

# argument parser
def parsing():
    parser = argparse.ArgumentParser()
    
    # random seed
    parser.add_argument("--random_seed", type = int, default = 42)
    
    # mode : predicting thermal quench vs current quench
    parser.add_argument("--mode", type = str, default = 'TQ', choices=['TQ','CQ'])

    # batch size / sequence length / epochs / distance / num workers / pin memory use
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--seq_len_efit", type = int, default = 10)
    parser.add_argument("--seq_len_ece", type = int, default = 50)
    parser.add_argument("--seq_len_diag", type = int, default = 50)
    parser.add_argument("--dist", type = int, default = 4)
    parser.add_argument("--num_workers", type = int, default = 8)
    parser.add_argument("--pin_memory", type = bool, default = True)
    
    # scaler type
    parser.add_argument("--scaler", type = str, choices=['Robust', 'Standard', 'MinMax', 'None'], default = "Robust")  
    args = vars(parser.parse_args())

    return args

if __name__ == "__main__":

    args = parsing()
    
    # seed initialize
    seed_everything(args['random_seed'], False)
    
    # dataset setup
    data_disrupt = pd.read_csv('./dataset/Bayesian_Disruption_Shot_List.csv', encoding = "euc-kr")
    data_efit = pd.read_csv("./dataset/Bayesian_Disruption_efit.csv")
    data_ece = pd.read_csv("./dataset/Bayesian_Disruption_ece.csv")
    data_diag = pd.read_csv("./dataset/Bayesian_Disruption_diag.csv")
    
    scaler_efit = None
    scaler_ece = None
    scaler_diag = None

    dataset = MultiSignalDataset(data_disrupt, data_efit, data_ece, data_diag, args['seq_len_efit'], args['seq_len_ece'], args['seq_len_diag'], args['dist'], 0.01, scaler_efit, scaler_ece, scaler_diag, args['mode'], 'train')
    dataloader = DataLoader(dataset, shuffle = True, batch_size = args['batch_size'], num_workers = args["num_workers"], pin_memory=args["pin_memory"], drop_last=True)

    for idx, data in enumerate(dataloader):
        print("batch:{} | data_efit:{} | data_ece:{} | data_diag:{} | label:{}".format(
            idx, data['efit'].size(), data['ece'].size(), data['diag'].size(), data['label'].size()
        ))