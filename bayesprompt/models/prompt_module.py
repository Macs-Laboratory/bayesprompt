import torch
import torch.nn as nn
from ..core.decorators import validate_shapes
import logging

logger = logging.getLogger(__name__)

class ProbabilisticPromptModule(nn.Module):
    def __init__(self, feature_dim: int, prompt_dim: int, uncertainty_alpha: float = 1.0, include_bg: bool = False):
        super().__init__()
        self.feature_dim = feature_dim
        self.prompt_dim = prompt_dim
        self.uncertainty_alpha = uncertainty_alpha
        self.include_bg = include_bg
        
        # Descriptor: [mu_c || sigma2_c || u_c]
        descriptor_dim = feature_dim * 2 + 1
        
        self.mlp = nn.Sequential(
            nn.Linear(descriptor_dim, prompt_dim),
            nn.ReLU(),
            nn.Linear(prompt_dim, prompt_dim)
        )

    def compute_predictive_entropy(self, probs: torch.Tensor):
        """
        probs: [B, C, H, W]
        H(p) = - sum( p * log(p) )
        Returns: [B, H, W]
        """
        eps = 1e-6
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
        return entropy

    @validate_shapes
    def forward(self, features: torch.Tensor, masks: torch.Tensor, pred_probs: torch.Tensor):
        """
        features: [B, C_f, H, W]
        masks: [B, C, H, W] (one-hot expected)
        pred_probs: [B, C, H, W] (mean MC probabilities)
        """
        B, C_f, H, W = features.shape
        _, C, _, _ = masks.shape
        
        entropy_map = self.compute_predictive_entropy(pred_probs) # [B, H, W]
        
        prompts = []
        
        for c in range(C):
            if not self.include_bg and c == 0:
                # Completely skip background class prompt generation if configured
                continue
                
            mask_c = masks[:, c, :, :].unsqueeze(1) # [B, 1, H, W]
            mask_sum = mask_c.sum(dim=(2, 3)) # [B, 1]
            
            # Robust empty mask handling
            valid = (mask_sum > 0).float()
            mask_sum_safe = torch.clamp(mask_sum, min=1.0)
            
            # Compute mu_c: [B, C_f]
            mu_c = (features * mask_c).sum(dim=(2, 3)) / mask_sum_safe
            mu_c = mu_c * valid # Zero out empty masks
            
            # Compute sigma2_c: [B, C_f]
            mu_c_spatial = mu_c.unsqueeze(-1).unsqueeze(-1)
            diff_sq = ((features - mu_c_spatial) ** 2) * mask_c
            sigma2_c = diff_sq.sum(dim=(2, 3)) / mask_sum_safe
            sigma2_c = sigma2_c * valid
            
            # Compute uncertainty u_c: [B, 1]
            u_c = (entropy_map.unsqueeze(1) * mask_c).sum(dim=(2, 3)) / mask_sum_safe
            u_c = u_c * valid
            
            # Uncertainty weighting
            weight = torch.exp(-self.uncertainty_alpha * u_c)
            
            # Form descriptor
            z_c = torch.cat([mu_c, sigma2_c, u_c], dim=1) # [B, 2*C_f + 1]
            
            p_c = self.mlp(z_c) # [B, D_prompt]
            p_c = p_c * weight # Modulate by reliability
            
            prompts.append(p_c)
            
        # Stack prompts: [B, C, D_prompt]
        prompts = torch.stack(prompts, dim=1)
        
        # Aggregate over batch (support set) -> [1, C, D_prompt]
        # Currently we do simple mean. 
        prompts = prompts.mean(dim=0, keepdim=True)
        return prompts
