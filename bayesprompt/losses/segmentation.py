import torch
import torch.nn as nn
from ..core.decorators import register_loss

@register_loss("DiceLoss")
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W]
        targets: [B, H, W] or [B, C, H, W]
        """
        probs = torch.softmax(logits, dim=1)
        
        if targets.dim() == 3:
            # One-hot encode
            targets = torch.nn.functional.one_hot(targets, num_classes=probs.shape[1]).permute(0, 3, 1, 2)
            
        targets = targets.float()
        
        dims = (2, 3)
        intersection = torch.sum(probs * targets, dim=dims)
        cardinality = torch.sum(probs + targets, dim=dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score.mean()

@register_loss("SegLoss")
class SegLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
    def forward(self, logits, targets):
        if targets.dim() == 4:
            # targets is one-hot, get argmax for CE
            targets_ce = targets.argmax(dim=1)
        else:
            targets_ce = targets
            
        l_ce = self.ce(logits, targets_ce)
        l_dice = self.dice(logits, targets)
        return self.ce_weight * l_ce + self.dice_weight * l_dice
