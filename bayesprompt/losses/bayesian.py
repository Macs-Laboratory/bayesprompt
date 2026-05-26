import torch
import torch.nn as nn
from .segmentation import SegLoss
from ..core.decorators import register_loss
import logging

logger = logging.getLogger(__name__)

@register_loss("BMPALoss")
class BMPALoss(nn.Module):
    """
    Bayesian Meta-Prior Adaptation (BMPA) Loss.
    Combines Monte-Carlo expected segmentation loss with KL Divergence.
    """
    def __init__(self, lambda_kl=0.1, kl_norm_mode="mean", ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.kl_norm_mode = kl_norm_mode
        self.seg_loss = SegLoss(ce_weight=ce_weight, dice_weight=dice_weight)

    def forward(self, mc_logits, targets, kl_tensor):
        """
        mc_logits: [R, B, C, H, W] or [B, C, H, W]
        targets: [B, H, W]
        kl_tensor: element-wise KL from BayesianParameterModule
        """
        if mc_logits.dim() == 5:
            # [R, B, C, H, W]
            R = mc_logits.shape[0]
            l_seg = 0
            for r in range(R):
                l_seg += self.seg_loss(mc_logits[r], targets)
            l_seg = l_seg / R
        else:
            # Fallback to single sample
            l_seg = self.seg_loss(mc_logits, targets)

        # Normalize KL
        if self.kl_norm_mode == "mean":
            l_kl = kl_tensor.mean()
        elif self.kl_norm_mode == "sum":
            l_kl = kl_tensor.sum()
        else:
            raise ValueError(f"Unknown KL normalization mode: {self.kl_norm_mode}")

        total_loss = l_seg + self.lambda_kl * l_kl
        return total_loss, l_seg, l_kl
