import torch, math
import torch.nn as nn
import numpy as np
from typing import Dict, Literal, Optional
from src.models.BNN import BayesLinear, variational_approximator
from src.models.tcn import TCN, calc_seq_length
from src.models.NoiseLayer import NoiseLayer
from pytorch_model_summary import summary

class Predictor(nn.Module):
    def __init__(self, header_config:Dict, classifier_config:Dict, device : Optional[str] = None):
        super(Predictor, self).__init__()
        self.header_config = header_config
        self.header_list = nn.ModuleDict([
            [key, self.create_base_model(header_config[key])] for key in header_config.keys()
        ])
        
        self.noise = NoiseLayer(mean = 0, std = 1.0)
        self.classifier_config = classifier_config
        self.device = device
          
        self.concat_feature_dim = 0
        
        for key in self.header_config.keys():
            config = self.header_config[key]
            self.concat_feature_dim += config['hidden_dim']
            
        self.connector = nn.Sequential(
            nn.Linear(self.concat_feature_dim, self.classifier_config['cls_dim']),
            nn.ReLU()
        )
    
        self.classifier = self.create_classifier()
        
    def create_base_model(self, config:Dict):
        tcn = TCN(config['num_inputs'],config['hidden_dim'],config['num_channels'], config['kernel_size'], config['dropout'], config['dilation_size'])
        return tcn
            
    def create_classifier(self):
        model = nn.Sequential(
            nn.Linear(self.classifier_config['cls_dim'], self.classifier_config['cls_dim']),
            nn.LayerNorm(self.classifier_config['cls_dim']),
            nn.ReLU(),
            nn.Linear(self.classifier_config['cls_dim'], self.classifier_config['n_classes']),
        )
        return model    
        
    def forward(self, data:Dict[str, torch.Tensor]):
        
        x = []

        for key in self.header_config.keys():
            signal = data[key]
            signal = self.noise(signal.to(self.device))
            signal_enc = self.header_list[key](signal)
            x.append(signal_enc)
            
        x = torch.cat(x, 1)
        x = self.connector(x)
        x = self.classifier(x)
        return x

    def encode(self, data:Dict[str, torch.Tensor]):
        with torch.no_grad():
            x = []
            for key in self.header_config.keys():
                signal = data[key]
                signal_enc = self.header_list[key](signal.to(self.device))
                x.append(signal_enc)
                
            x = torch.cat(x, 1)
            x = self.connector(x)
        return x
    
    def summary(self):
        
        sample = {
            "efit":torch.zeros((1, self.header_config['efit']['num_inputs'], self.header_config['efit']['seq_len'])),
            "ece":torch.zeros((1, self.header_config['ece']['num_inputs'], self.header_config['ece']['seq_len'])),
            "diag":torch.zeros((1, self.header_config['diag']['num_inputs'], self.header_config['diag']['seq_len'])),
        }

        summary(self, sample, batch_size = 1, show_input = True, print_summary=True, max_depth=2)
        
@variational_approximator
class BayesianPredictor(nn.Module):
    def __init__(self, header_config:Dict, classifier_config:Dict, device : str):
        super(BayesianPredictor, self).__init__()
        self.header_config = header_config
        self.header_list = nn.ModuleDict([
            [key, self.create_base_model(header_config[key])] for key in header_config.keys()
        ])
        
        self.noise = NoiseLayer(mean = 0, std = 1.0)
        
        self.classifier_config = classifier_config
        self.device = device
          
        self.concat_feature_dim = 0
        
        for key in self.header_config.keys():
            config = self.header_config[key]
            self.concat_feature_dim += config['hidden_dim']
            
        self.connector = nn.Sequential(
            nn.Linear(self.concat_feature_dim, self.classifier_config['cls_dim']),
            nn.ReLU()
        )
    
        self.classifier = self.create_classifier()
        
    def create_base_model(self, config:Dict):
        tcn = TCN(config['num_inputs'],config['hidden_dim'],config['num_channels'], config['kernel_size'], config['dropout'], config['dilation_size'])
        return tcn
            
    def create_classifier(self):
        model = nn.Sequential(
            BayesLinear(self.classifier_config['cls_dim'], self.classifier_config['cls_dim'], self.classifier_config['prior_pi'], self.classifier_config['prior_sigma1'], self.classifier_config['prior_sigma2']),
            nn.LayerNorm(self.classifier_config['cls_dim']),
            nn.ReLU(),
            BayesLinear(self.classifier_config['cls_dim'], self.classifier_config['n_classes'], self.classifier_config['prior_pi'], self.classifier_config['prior_sigma1'], self.classifier_config['prior_sigma2']),
        )
        return model    
        
    def forward(self, data:Dict[str, torch.Tensor]):
        
        x = []
        
        for key in self.header_config.keys():
            signal = data[key]
            signal = self.noise(signal.to(self.device))
            signal_enc = self.header_list[key](signal)
            x.append(signal_enc)
            
        x = torch.cat(x, 1)
        x = self.connector(x)
        x = self.classifier(x)
        return x

    def encode(self, data:Dict[str, torch.Tensor]):
        with torch.no_grad():
            x = []
            for key in self.header_config.keys():
                signal = data[key]
                signal_enc = self.header_list[key](signal.to(self.device))
                x.append(signal_enc)
                
            x = torch.cat(x, 1)
            x = self.connector(x)
        return x
    
    def summary(self):
        
        sample = {
            "efit":torch.zeros((1, self.header_config['efit']['num_inputs'], self.header_config['efit']['seq_len'])),
            "ece":torch.zeros((1, self.header_config['ece']['num_inputs'], self.header_config['ece']['seq_len'])),
            "diag":torch.zeros((1, self.header_config['diag']['num_inputs'], self.header_config['diag']['seq_len'])),
        }
        
        summary(self, sample, batch_size = 1, show_input = True, print_summary=True, max_depth=2)
        
    def predict_per_sample(self, x : Dict[str,torch.Tensor]):
        
        with torch.no_grad():
            h = self.encode(x)
            outputs = torch.zeros((h.size()[0], 2), device = self.device)
            
            for idx in range(h.size()[0]):
                outputs[idx, :] = self.classifier(h[idx, :].unsqueeze(0))
                
            return outputs
    
if __name__ == "__main__":
    
    # torch device state
    print("================= device setup =================")
    print("torch device avaliable : ", torch.cuda.is_available())
    print("torch current device : ", torch.cuda.current_device())
    print("torch device num : ", torch.cuda.device_count())

    # torch cuda initialize and clear cache
    torch.cuda.init()
    torch.cuda.empty_cache()
    
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(0)
    else:
        device = 'cpu'
    
    levels = 4
    kernel_size = 2
    dilation_size = 2
    nrecept = 1024
    channel_size = [128] * levels
    nrecepttotal = calc_seq_length(kernel_size, dilation_size, levels)
    nlastlevel = calc_seq_length(kernel_size,dilation_size,levels-1)
    last_dilation = int(np.ceil((nrecept - nlastlevel)/(2.*(kernel_size-1))))
    dilation_sizes = (dilation_size**np.arange(levels-1)).tolist() + [last_dilation]
    
    header_config = {
        "efit":{
            "num_inputs":12,
            "hidden_dim":64,
            "num_channels":channel_size,
            "kernel_size":2,
            "dropout":0.1,
            "dilation_size":dilation_size,
            "seq_len":20
        },
        "ece":{
            "num_inputs":8,
            "hidden_dim":64,
            "num_channels":channel_size,
            "kernel_size":2,
            "dropout":0.1,
            "dilation_size":dilation_size,
            "seq_len":100
        },
        "diag":{
            "num_inputs":16,
            "hidden_dim":64,
            "num_channels":channel_size,
            "kernel_size":2,
            "dropout":0.1,
            "dilation_size":dilation_size,
            "seq_len":100
        }
    }
    
    classifier_config = {
        "cls_dim" : 128,
        "n_classes":2,
        "prior_pi":0.5,
        "prior_sigma1":1.0,
        "prior_sigma2":0.0025,
    }
    
    data = {
        "efit":torch.zeros((1, 12, 20)),
        "ece":torch.zeros((1, 8, 100)),
        "diag":torch.zeros((1, 16, 100)),
    }
    
    model = Predictor(header_config, classifier_config, device)
    model.to(device)
    model.summary()
    output = model(data)
    print("output:",output.size())
    
    model = BayesianPredictor(header_config, classifier_config, device)
    model.to(device)
    model.summary()
    output = model(data)
    
    print("output:",output.size())