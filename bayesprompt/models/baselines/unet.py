import torch.nn as nn
from ...core.decorators import register_model

@register_model("UNet")
class UNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Lightweight UNet for structure
        self.conv = nn.Conv2d(3, num_classes, kernel_size=3, padding=1)
        
    def forward(self, x):
        return self.conv(x)
