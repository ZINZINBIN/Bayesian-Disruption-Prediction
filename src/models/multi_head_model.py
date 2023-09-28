import torch
import torch.nn as nn
from src.models.transformer import TransformerEncoder, GELU
from pytorch_model_summary import summary

class ProfileEncoder(nn.Module):
    def __init__(self, width : int, height : int, feature_dim : int):
        super().__init__()
        self.width = width
        self.height = height
        self.layer = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 5, stride = 3, padding = 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        conv_feat_dim = self.compute_feature_dim()
        self.linear = nn.Linear(conv_feat_dim, feature_dim)
    
    def forward(self, x : torch.Tensor):
        x = self.layer(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x    
    
    def compute_feature_dim(self):
        sample_data = torch.zeros((2, 1, self.height, self.width))
        hidden = self.layer(sample_data).view(2,-1)
        return hidden.size()[-1]

class MultiHeadModel(nn.Module):
    def __init__(
        self, 
        n_features : int = 11, 
        kernel_size : int = 5,
        feature_dims : int = 256, 
        max_len : int = 128, 
        n_layers : int = 1, 
        n_heads : int = 8, 
        dim_feedforward : int = 1024, 
        dropout : float = 0.1, 
        cls_dims : int = 128, 
        n_points : int = 128,
        n_classes : int = 2
    ):
        super().__init__()
        self.n_classes = n_classes
        self.encoder_0D = TransformerEncoder(n_features, kernel_size, feature_dims, max_len, n_layers, n_heads, dim_feedforward, dropout)
        self.encoder_ne = ProfileEncoder(width = n_points, height = max_len, feature_dim = feature_dims)
        self.encoder_te = ProfileEncoder(width = n_points, height = max_len, feature_dim = feature_dims)
        
        self.connector = nn.Linear(feature_dims * 3, feature_dims)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dims, cls_dims),
            nn.LayerNorm(cls_dims),
            GELU(),
            nn.Linear(cls_dims, n_classes)
        )
        
    def forward(self, x_0D : torch.Tensor, ne_profile : torch.Tensor, te_profile : torch.Tensor):
        x_0D = self.encoder_0D(x_0D)
        x_ne = self.encoder_ne(ne_profile)
        x_te = self.encoder_te(te_profile)
        
        x = torch.concat([x_0D, x_ne, x_te], dim = 1)
        x = self.connector(x)
        x = self.classifier(x)
        
        return x
    
    def encode(self, x_0D : torch.Tensor, ne_profile : torch.Tensor, te_profile : torch.Tensor):
        with torch.no_grad():
            x_0D = self.encoder_0D(x_0D)
            x_ne = self.encoder_ne(ne_profile)
            x_te = self.encoder_te(te_profile)
            x = torch.concat([x_0D, x_ne, x_te], dim = 1)
            x = self.connector(x)
        return x
    
    def summary(self):
        sample_0D = torch.zeros((2, self.encoder_0D.max_len, self.encoder_0D.n_features))
        sample_ne = torch.zeros((2, 1, self.encoder_ne.height, self.encoder_ne.width))
        sample_te = torch.zeros((2, 1, self.encoder_te.height, self.encoder_te.width))
        summary(self, sample_0D, sample_ne, sample_te, batch_size = 2, show_input = True, print_summary=True)
        
if __name__ == "__model__":
    model = MultiHeadModel(
        11,
        5,
        128,
        20,
        4,
        8,
        512,
        0.1,
        128,
        128,
        2
    )
    model.summary()