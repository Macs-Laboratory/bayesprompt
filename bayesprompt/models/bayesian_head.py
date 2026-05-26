import torch
import torch.nn as nn
from torch.func import functional_call
import math
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

def inverse_softplus(x, eps=1e-6):
    """Stable inverse softplus: rho = ln(exp(x) - 1)"""
    x = torch.clamp(x, min=eps)
    return x + torch.log(-torch.expm1(-x))

@dataclass
class GaussianDiagonal:
    mu: torch.Tensor
    rho: torch.Tensor
    min_sigma: float = 1e-5

    @property
    def sigma(self):
        return torch.nn.functional.softplus(self.rho) + self.min_sigma
        
    @property
    def variance(self):
        return self.sigma ** 2

    def sample(self):
        eps = torch.randn_like(self.mu)
        return self.mu + self.sigma * eps

    def kl_to(self, other: 'GaussianDiagonal'):
        """
        Computes KL(self || other)
        KL(q||p) = log(sigma_p/sigma_q) + (sigma_q^2 + (mu_q - mu_p)^2) / (2*sigma_p^2) - 0.5
        Returns a tensor of the same shape as mu (element-wise KL).
        """
        var_q = self.variance
        var_p = other.variance
        mu_q = self.mu
        mu_p = other.mu
        
        log_ratio = torch.log(other.sigma) - torch.log(self.sigma)
        term2 = (var_q + (mu_q - mu_p)**2) / (2 * var_p)
        return log_ratio + term2 - 0.5


class BayesianParameterModule(nn.Module):
    """
    Wraps standard PyTorch nn.Modules to make them Bayesian.
    Maintains variational posterior parameters (mu, rho) and prior parameters.
    Uses `torch.func.functional_call` for stateless reparameterized forward passes.
    """
    def __init__(self, module: nn.Module, min_sigma: float = 1e-5, init_sigma: float = 1e-3):
        super().__init__()
        self.module = module
        self.min_sigma = min_sigma
        
        self.num_params = sum(p.numel() for p in self.module.parameters())
        
        # Prior parameters (saved after source training)
        self.register_buffer("prior_mu", torch.zeros(self.num_params))
        
        # Initialize prior_rho to correspond to init_sigma
        init_rho_val = inverse_softplus(torch.tensor(init_sigma)).item()
        self.register_buffer("prior_rho", torch.full((self.num_params,), init_rho_val))
        
        # Variational posterior parameters
        self.mu = nn.Parameter(torch.zeros(self.num_params))
        self.rho = nn.Parameter(torch.full((self.num_params,), init_rho_val))
        
        self._init_params()

    def _init_params(self):
        # Initialize mu with the module's initial weights
        params = torch.cat([p.flatten() for p in self.module.parameters()]).detach()
        self.mu.data.copy_(params)
        
        # Disable requires_grad on the original module parameters
        for p in self.module.parameters():
            p.requires_grad = False

    def get_posterior(self) -> GaussianDiagonal:
        return GaussianDiagonal(self.mu, self.rho, self.min_sigma)
        
    def get_prior(self) -> GaussianDiagonal:
        return GaussianDiagonal(self.prior_mu, self.prior_rho, self.min_sigma)

    def load_source_prior(self):
        """Sets the prior to the current posterior (used after source training)."""
        self.prior_mu.copy_(self.mu.detach())
        self.prior_rho.copy_(self.rho.detach())
        
    def initialize_posterior_from_prior(self):
        """Used at the start of Bayesian Fine-Tuning."""
        self.mu.data.copy_(self.prior_mu)
        self.rho.data.copy_(self.prior_rho)

    def kl_to_prior(self):
        """Returns element-wise KL divergence tensor."""
        q = self.get_posterior()
        p = self.get_prior()
        return q.kl_to(p)

    def forward(self, *args, num_samples=1, deterministic=False, **kwargs):
        """
        Stateless forward pass using sampled weights.
        """
        outputs = []
        q = self.get_posterior()
        
        for _ in range(num_samples if not deterministic else 1):
            theta = q.mu if deterministic else q.sample()
            
            # Reconstruct state dict mapping for functional_call
            state_dict = {}
            offset = 0
            for name, param in self.module.named_parameters():
                numel = param.numel()
                state_dict[name] = theta[offset:offset+numel].view(param.shape)
                offset += numel
                
            out = functional_call(self.module, state_dict, args, kwargs)
            outputs.append(out)
            
        if num_samples == 1:
            return outputs[0]
        else:
            return torch.stack(outputs, dim=0)
