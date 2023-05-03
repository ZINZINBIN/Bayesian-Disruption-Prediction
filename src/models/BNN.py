import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Optional, Dict
from src.models.transformer import TransformerEncoder, GELU
from pytorch_model_summary import summary

# abstract for bayesian module
class BayesianModule(nn.Module):
    def __init__(self):
        super().__init__()

    def compute_kld(self, *args):
        return NotImplementedError("BayesianModule::kld()")
    
    @property
    def kl_divergence(self):
        return NotImplementedError("BayesianModule::kl_divergence()")
    
    @property
    def elbo(self):
        return NotImplementedError("BayesianModule::elbo()")

# Gaussian Variational class
class GaussianVariational(nn.Module):
    def __init__(self, mu : torch.Tensor, rho : torch.Tensor):
        super().__init__()
        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)
        
        self.w = None
        self.sigma = None
        self.normal = torch.distributions.Normal(0,1)
        
    def sample(self):
        eps = self.normal.sample(self.mu.size()).to(self.mu.device)
        sig = torch.log(1 + torch.exp(self.rho)).to(self.rho.device)
        
        w = self.mu + sig * eps
        
        self.w = w
        self.sigma = sig
        
        return self.w

    def log_posterior(self):
        
        if self.w is None or self.sigma is None:
            raise ValueError("self.w must have a value..!")

        log_const = np.log(np.sqrt(2*np.pi))
        log_exp = ((self.w - self.mu)**2) / (2 * self.sigma ** 2)
        log_posterior = -log_const - torch.log(self.sigma) - log_exp
        
        return log_posterior.sum()
    
# Scale Mixture model
class ScaleMixture(nn.Module):
    def __init__(self, pi : float, sigma1 : float, sigma2 : float):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        
        self.norm1 = torch.distributions.Normal(0,sigma1)
        self.norm2 = torch.distributions.Normal(0,sigma2)
        
    def log_prior(self, w : torch.Tensor):
        li_n1 = torch.exp(self.norm1.log_prob(w))
        li_n2 = torch.exp(self.norm2.log_prob(w))
        
        p_mixture = self.pi * li_n1 + (1-self.pi) * li_n2
        log_prob = torch.log(p_mixture).sum()
        
        return log_prob
    
# minibatch weight for training process
def minibatch_weight(batch_idx : int, num_batches : int):
    return 2 ** (num_batches - batch_idx) / (2**num_batches - batch_idx)

# variational approximator for variantional inference
# automatically computing the kl divergence with forward process
def variational_approximator(model : nn.Module):
    
    def kl_divergence(self:nn.Module):
        kl = 0
        
        for module in self.modules():
            if isinstance(module, BayesianModule):
                kl += module.kl_divergence()
        return kl

    setattr(model, 'kl_divergence', kl_divergence)
    
    def elbo(self : BayesianModule, inputs : torch.Tensor, targets : torch.Tensor, criterion : Any, n_samples : int, w_complexity : Optional[float] = 1.0):
        loss = 0
        
        for sample in range(n_samples):
            outputs = self(inputs)
            loss += criterion(outputs, targets)
            loss += self.kl_divergence() * w_complexity
        
        return loss / n_samples

    setattr(model, 'elbo', elbo)

    return model

class BayesLinear(BayesianModule):
    def __init__(self, in_features : int, out_features : int, prior_pi : Optional[float] = 0.5, prior_sigma1 : Optional[float] = 1.0, prior_sigma2 : Optional[float] = 0.0025):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.prior_pi = prior_pi
        self.prior_sigma1 = prior_sigma1
        self.prior_sigma2 = prior_sigma2
        
        w_mu = torch.empty(out_features, in_features).uniform_(-0.2, 0.2)
        w_rho = torch.empty(out_features, in_features).uniform_(-5.0, -4.0)
        
        bias_mu = torch.empty(out_features).uniform_(-0.2, 0.2)
        bias_rho = torch.empty(out_features).uniform_(-5.0, -4.0)
        
        self.w_posterior = GaussianVariational(w_mu, w_rho)
        self.bias_posterior = GaussianVariational(bias_mu, bias_rho)
        
        self.w_prior = ScaleMixture(prior_pi, prior_sigma1, prior_sigma2)
        self.bias_prior = ScaleMixture(prior_pi, prior_sigma1, prior_sigma2)
        
        self._kl_divergence = 0
    
    def forward(self, x : torch.Tensor):
        
        w = self.w_posterior.sample()
        b = self.bias_posterior.sample()
        
        # compute posterior and prior of the p(w) and p(w|D)
        w_log_prior = self.w_prior.log_prior(w)
        b_log_prior = self.bias_prior.log_prior(b)
        
        w_log_posterior = self.w_posterior.log_posterior()
        b_log_posterior = self.bias_posterior.log_posterior()
        
        total_log_prior = w_log_prior + b_log_prior
        total_log_posterior = w_log_posterior + b_log_posterior
        
        # compute kl_divergence
        self._kl_divergence = self.compute_kld(total_log_prior, total_log_posterior)
        
        return F.linear(x,w,b)
    
    def kl_divergence(self):
        return self._kl_divergence
    
    def compute_kld(self, log_prior : torch.Tensor, log_posterior : torch.Tensor):
        return log_posterior - log_prior

class BayesDropout(nn.Module):
    def __init__(self, p : float = 0.5):
        super().__init__()
        assert p <= 1, "p must be smaller than 1"
        self.p = p
        self._multiplier = 1.0 / (1.0 - p)
        
    def forward(self, x : torch.Tensor):
        if not self.training:
            return x
        selected_ = torch.Tensor(x.size()).uniform_(0,1) > self.p
        selected_ = torch.autograd.Variable(selected_.type(torch.FloatTensor), requires_grad = False).to(x.device)
        res = torch.mul(selected_, x) * self._multiplier
        return res

@variational_approximator
class BayesianTransformer(nn.Module):
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
        n_classes : int = 2,
        prior_pi : Optional[float] = 0.5, 
        prior_sigma1 : Optional[float] = 1.0, 
        prior_sigma2 : Optional[float] = 0.0025
        ):
        super().__init__()
        self.encoder = TransformerEncoder(n_features, kernel_size, feature_dims, max_len, n_layers, n_heads, dim_feedforward, dropout)
        self.classifier = nn.Sequential(
            BayesLinear(feature_dims, cls_dims, prior_pi, prior_sigma1, prior_sigma2),
            nn.BatchNorm1d(cls_dims),
            GELU(),
            BayesLinear(cls_dims, n_classes, prior_pi, prior_sigma1, prior_sigma2),
        )
    
    def forward(self, x : torch.Tensor):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
    
    def encode(self, x: torch.Tensor):
        with torch.no_grad():
            x = self.encoder(x)
        return x

    def summary(self):
        sample_x = torch.zeros((2, self.encoder.max_len, self.encoder.n_features))
        summary(self, sample_x, batch_size = 2, show_input = True, print_summary=True)
    
@variational_approximator
class BayesianMultiHeadTransformer(nn.Module):
    def __init__(
        self,  
        feature_dims : int = 256, 
        max_len : int = 128, 
        n_layers : int = 1, 
        n_heads : int = 8, 
        dim_feedforward : int = 1024, 
        dropout : float = 0.1, 
        cls_dims : int = 128, 
        n_classes : int = 2,
        prior_pi : Optional[float] = 0.5, 
        prior_sigma1 : Optional[float] = 1.0, 
        prior_sigma2 : Optional[float] = 0.0025,
        args_enc : Dict = {}
        ):
        super().__init__()
        self.args_enc = args_enc
        self.feature_dims = feature_dims
        self.encoder = nn.ModuleDict([
            [key, TransformerEncoder(args_enc[key], feature_dims, max_len, n_layers, n_heads, dim_feedforward, dropout)] for key in args_enc.keys()]
        )
        self.connector = nn.Linear(feature_dims * len(args_enc.keys()), feature_dims)
        self.classifier = nn.Sequential(
            BayesLinear(feature_dims, cls_dims, prior_pi, prior_sigma1, prior_sigma2),
            nn.BatchNorm1d(cls_dims),
            GELU(),
            BayesLinear(cls_dims, n_classes, prior_pi, prior_sigma1, prior_sigma2),
        )
    
    def forward(self, x : Dict):
        
        device = next(self.parameters()).get_device()
        x_concat = []
        for key in self.args_enc.keys():
            x_ = self.encoder[key](x[key].to(device))
            x_concat.append(x_)
        
        x_concat = torch.concat(x_concat, dim = 1)
        x = self.connector(x_concat)
        x = self.classifier(x)
        
        return x
    
# Uncertaintiy computation
def compute_ensemble_probability(model : nn.Module, input : torch.Tensor, device : str = 'cpu', n_samples : int = 8):
    '''
        Compute the probability of each case with n_samples of the predictive conditional probability
        Input : (1,T,D)
        Output : Dict['disrupt':[], 'normal':[]]
    '''
    
    model.eval()
    model.to(device)
    
    if input.ndim == 2:
        input = input.unsqueeze(0)
        
    inputs = input.repeat(n_samples, 1, 1)
    outputs = model(inputs.to(device))
    probs = torch.nn.functional.softmax(outputs, dim = 1)
    probs = probs.detach().cpu().numpy() 
    return probs

def compute_uncertainty_per_data(model : nn.Module, input : torch.Tensor, device : str = "cpu", n_samples : int = 8, normalized : bool = False):
    
    if input.ndim == 2:
        input = input.unsqueeze(0)
        
    inputs = input.repeat(n_samples, 1, 1)
    
    model.eval()
    model.to(device)
    outputs = model(inputs.to(device))
    
    if normalized:
        probs = torch.nn.functional.softplus(outputs, dim = 1)
        probs = probs / torch.sum(probs, dim = 1).unsqueeze(1)
        
    else:
        probs = torch.nn.functional.softmax(outputs, dim = 1)
        
    probs = probs.detach().cpu().numpy()   
    probs_mean = np.mean(probs, axis = 0)
    probs_diff = probs = np.expand_dims(probs_mean, 0)
    epistemic_uncertainty = np.dot(probs_diff.T, probs_diff) / n_samples
    aleatoric_uncertainty = np.diag(probs_mean) - (np.dot(probs_diff.T, probs_diff) / n_samples)
    aleatoric_uncertainty = np.diag(aleatoric_uncertainty)
    
    return aleatoric_uncertainty, epistemic_uncertainty
