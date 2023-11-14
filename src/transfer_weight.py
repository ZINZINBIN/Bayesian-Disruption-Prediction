import torch
from src.models.predictor import BayesianPredictor, Predictor
from typing import Optional, Dict

def load_pretrained_weight(
    model:BayesianPredictor, 
    config:Dict,
    filepath:str, 
    device:str,
    freeze:bool = False,
    ):

    pretrained = Predictor(config.header_config, config.classifier_config, device)
    pretrained.to(device)
    pretrained.load_state_dict(torch.load(filepath))
    
    model.to(device)
    
    names_pretrained = []
    
    for name, param in pretrained.state_dict().items():
        
        if name not in model.state_dict().keys():
            continue
        
        names_pretrained.append(name)
        
        if isinstance(param, torch.nn.Parameter):
            param = param.data
            
        model.state_dict()[name].copy_(param)
                
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    pretrained.cpu()
    del pretrained
    
    print("Transfer pretrained model's weights: clear")