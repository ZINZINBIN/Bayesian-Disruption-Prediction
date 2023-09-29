import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Optional, Dict
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
    
# Uncertaintiy computation
def compute_ensemble_probability(model : nn.Module, input : Dict[str, torch.Tensor], device : str = 'cpu', n_samples : int = 8):
    '''
        Compute the probability of each case with n_samples of the predictive conditional probability
        Input : (1,T,D)
        Output : (n_samples, 2) : n_samples of [probs_normal, probs_disrupt]
    '''

    model.eval()
    model.to(device)
    
    
    # 배치 단위로 불확실성 계산하는 코드 : 추후 추가 예정
    input_n_repeat = {}
    for key in input.keys():
        if input[key].ndim == 2:
            input_n_repeat[key] = input[key].unsqueeze(0)
        input_n_repeat[key] = input[key].repeat(n_samples, 1, 1)
        
    outputs = model.predict_per_sample(input_n_repeat)
    probs = torch.nn.functional.softmax(outputs, dim = 1)
    
    '''
    # 일단 여러번 연산해서 얻는 구조로 진행
    for key in input.keys():
        if input[key].ndim == 2:
            input[key] = input[key].unsqueeze(0)
    
    probs = torch.zeros((n_samples, 2), device = device)
    
    for idx in range(n_samples):
        output = model(input)
        prob = torch.nn.functional.softmax(output, dim = 1)
        probs[idx,:] = prob
        
    '''
    probs = probs.detach().cpu().numpy() 
    
    return probs

def compute_uncertainty_per_data(model : nn.Module, input : Dict[str, torch.Tensor], device : str = "cpu", n_samples : int = 8, normalized : bool = False):
    
    model.eval()
    model.to(device)
    
    input_n_repeat = {}
    for key in input.keys():
        if input[key].ndim == 2:
            input_n_repeat[key] = input[key].unsqueeze(0)
        input_n_repeat[key] = input[key].repeat(n_samples, 1, 1)
        
    outputs = model.predict_per_sample(input_n_repeat)
    probs = torch.nn.functional.softmax(outputs, dim = 1)
    
    if normalized:
        probs = torch.nn.functional.softmax(outputs, dim = 1)
        probs = probs / torch.sum(probs, dim = 1).unsqueeze(1)
    else:
        probs = torch.nn.functional.softmax(outputs, dim = 1)
        
    probs = probs.detach().cpu().numpy()   
    probs_mean = np.mean(probs, axis = 0)
    probs_diff = probs - np.expand_dims(probs_mean, 0)
    epistemic_uncertainty = np.dot(probs_diff.T, probs_diff) / n_samples
    epistemic_uncertainty = np.diag(epistemic_uncertainty)
    
    aleatoric_uncertainty = np.diag(probs) - np.dot(probs.T, probs) / n_samples
    aleatoric_uncertainty = np.diag(aleatoric_uncertainty)
    
    return aleatoric_uncertainty, epistemic_uncertainty
